#!/usr/bin/env python3
"""
MIL Attention One-vs-Rest Training Pipeline (Balanced, Binary per Class)

Overview:
- Trains one binary AttentionMIL model per class (k vs rest), yielding an OvR ensemble.
- Supports either class-weighted cross-entropy or oversampling with a WeightedRandomSampler.
- Performs a robust 80/10/10 split into train/validation/calibration on the TRAIN set per OvR classifier.
- Evaluates at each epoch with binary metrics (accuracy, F1, precision, recall) and, when possible,
  ROC AUC and average precision; saves PR/ROC curves and confusion matrices per class.
- Optionally calibrates each trained classifier via temperature scaling on its calibration split.

Inputs:
- --mil_pkl: A PKL containing TRAIN (and optional TEST) sets. Several common formats are supported
  (see load_train_test_from_pkl for details).

Outputs (per class k):
- Model: best model weights saved to model_{k}_{class}.pt based on validation F1.
- Metrics/plots: metrics JSON, PR and ROC curves, confusion matrix for validation.
- Debug JSON for splits and class imbalance (when --debug is enabled).
- Optional temperature parameters across classifiers saved to temperature_parameters.json.
- Global training summary saved to training_summary.json.

Reproducibility:
- Random seeds are applied to Python, NumPy, and PyTorch RNGs. DataLoader workers are seeded.

Example:
  python mil_attention_train_ovr_balanced.py \
    --mil_pkl <path.pkl> \
    --output_dir <dir> \
    --epochs 30 \
    --batch_size 8 \
    --hidden_dim 256 \
    --dropout_rate 0.0 \
    --learning_rate 1e-3 \
    --calibrate

Author: Konstantinos Papagoras
Date: 2025-09
"""
import os
import json
import pickle
import argparse
from typing import List, Tuple
import slideflow as sf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_curve, auc as sk_auc
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# -------------------- Split loader (train/test from single PKL) --------------------
def load_train_test_from_pkl(pkl_path):
    """
    Load TRAIN (and optional TEST) splits from a PKL with flexible schemas.

    Supported schema variants:
    - Keys: 'train_bags', 'train_labels' and optionally 'test_bags', 'test_labels'
    - Keys: 'X_train', 'y_train' and optionally 'X_test', 'y_test'
    - Nested dicts: {'train': {'bags', 'labels'}, 'test': {'bags', 'labels'}}
    - Optional: 'class_names' or 'label_encoder' with classes_

    Returns:
        X_train (list): List of bags (each bag is an array-like [n_tiles, feat_dim]).
        y_train (np.ndarray): Integer labels for TRAIN (shape [N_train]).
        X_test (list or None): Optional TEST bags if present, else None.
        y_test (np.ndarray or None): Optional TEST labels if present, else None.
        class_names (list or None): Class names if present, else None.

    Raises:
        KeyError: If the PKL format is not recognized.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    class_names = None
    if 'class_names' in data and isinstance(data['class_names'], (list, tuple)):
        class_names = list(data['class_names'])
    elif 'label_encoder' in data and hasattr(data['label_encoder'], 'classes_'):
        class_names = list(data['label_encoder'].classes_)

    # Case 1: train_bags/train_labels [+ optional test_bags/test_labels]
    if 'train_bags' in data and 'train_labels' in data:
        X_train, y_train = data['train_bags'], np.asarray(data['train_labels'], dtype=int)
        X_test  = data.get('test_bags')
        y_test  = np.asarray(data['test_labels'], dtype=int) if 'test_labels' in data else None
        return X_train, y_train, X_test, y_test, class_names

    # Case 2: X_train/y_train [+ X_test/y_test]
    if 'X_train' in data and 'y_train' in data:
        X_train, y_train = data['X_train'], np.asarray(data['y_train'], dtype=int)
        X_test  = data.get('X_test')
        y_test  = np.asarray(data['y_test'], dtype=int) if 'y_test' in data else None
        return X_train, y_train, X_test, y_test, class_names if class_names else None

    # Case 3: explicit dicts train/test
    if 'train' in data and 'test' in data and isinstance(data['train'], dict) and isinstance(data['test'], dict):
        X_train, y_train = data['train']['bags'], np.asarray(data['train']['labels'], dtype=int)
        X_test,  y_test  = data['test']['bags'],  np.asarray(data['test']['labels'], dtype=int)
        return X_train, y_train, X_test, y_test, class_names

    raise KeyError("Unrecognized PKL format. Expected keys like "
                   "('train_bags','train_labels') or ('X_train','y_train') or nested {'train':..., 'test':...}.")

# -------------------- Data + Model --------------------

class MILDataset(Dataset):
    """
    Simple Dataset wrapper for MIL:
    - Each item is a tuple (bag, label) where bag is a [n_tiles, feat_dim] tensor.
    - Labels are integers (0/1 for OvR binary training).
    """
    def __init__(self, bags, labels):
        """Initialize with lists/arrays of bags and labels."""
        self.bags = bags
        self.labels = np.asarray(labels, dtype=int)
    def __len__(self):
        """Return dataset size."""
        return len(self.bags)
    def __getitem__(self, idx):
        """Return (bag, label) for index idx as (FloatTensor, int)."""
        return torch.tensor(self.bags[idx], dtype=torch.float32), int(self.labels[idx])

def collate_fn(batch):
    """
    Collate function for variable-length MIL bags.
    Returns:
        bags (list[Tensor]): List of [n_tiles, feat_dim] tensors.
        labels (Tensor): LongTensor of shape [batch].
    """
    bags, labels = zip(*batch)
    return list(bags), torch.tensor(labels, dtype=torch.long)

class AttentionMIL(nn.Module):
    """
    Attention-based MIL model (binary head for OvR):
    - feature_extractor: Linear -> ReLU (+ optional Dropout) to get per-tile embeddings.
    - attention: Tanh-based attention producing a weight per tile.
    - classifier: Linear mapping from aggregated embedding to 2 logits (binary).
    Forward:
        Input: bag [n_tiles, input_dim]
        Output: logits [2], attention_weights [n_tiles]
    """
    def __init__(self, input_dim, hidden_dim, n_classes=2, dropout_rate=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*layers)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)
    def forward(self, bag):
        """
        Compute attention-pooled representation and logits.
        Args:
            bag (Tensor): [n_tiles, input_dim]
        Returns:
            logits (Tensor): [2] raw class scores.
            attn (Tensor): [n_tiles] attention weights summing to 1.
        """
        H = self.feature_extractor(bag)
        A = self.attention(H)
        A = torch.softmax(A, dim=0)
        M = torch.sum(A * H, dim=0)
        logits = self.classifier(M)
        return logits, A.squeeze(-1)


# -------------------- Training/Eval (binary) --------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """
    One training epoch over MIL bags for a binary OvR classifier.
    For each bag in a batch:
      - Forward bag, compute CE loss, and backprop.
      - Aggregate average loss per batch and running binary metrics.

    Returns:
        mean_loss (float): Average batch loss.
        acc (float): Accuracy over seen items.
        f1 (float): Binary F1 over seen items.
    """
    model.train()
    losses, all_preds, all_labels = [], [], []
    for bags, labels in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad()
        batch_loss = 0.0
        for bag, y in zip(bags, labels):
            bag = bag.to(device)
            y = y.to(device)
            logits, _ = model(bag)
            loss = criterion(logits.unsqueeze(0), y.unsqueeze(0))
            loss.backward()
            batch_loss += float(loss.item())
            all_preds.append(int(torch.argmax(logits).item()))
            all_labels.append(int(y.item()))
        optimizer.step()
        losses.append(batch_loss / max(1, len(bags)))
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0) if all_labels else 0.0
    return float(np.mean(losses) if losses else 0.0), float(acc), float(f1)

def eval_epoch(model, loader, criterion, device, eval_tag, out_dir, class_names=("Rest","Class"), verbose=False):
    """
    Evaluate a binary OvR classifier on MIL bags and persist plots/metrics.

    Saves:
        - Precision-Recall curve (PNG) with AP.
        - ROC curve (PNG) with AUC (if both classes present).
        - Confusion matrix (PNG) with counts.
        - metrics_{eval_tag}.json with accuracy, F1, precision, recall, AUC, AP, confusion matrix,
          classification_report, and misclassified indices.

    Returns:
        mean_loss (float), acc (float), f1 (float), auc (float or None)
    """
    model.eval()
    losses = []
    all_preds, all_labels, all_probs = [], [], []
    misclassified = []
    global_idx = 0
    with torch.no_grad():
        for bags, labels in tqdm(loader, desc=f"Eval[{eval_tag}]", leave=False):
            batch_loss = 0.0
            for bag, y in zip(bags, labels):
                bag = bag.to(device)
                y = y.to(device)
                logits, attn = model(bag)
                loss = criterion(logits.unsqueeze(0), y.unsqueeze(0))
                batch_loss += float(loss.item())
                probs = torch.softmax(logits, dim=0).cpu().numpy()
                pred = int(torch.argmax(logits).item())
                all_probs.append(probs)
                all_preds.append(pred)
                all_labels.append(int(y.item()))
                if pred != int(y.item()):
                    misclassified.append(global_idx)
                global_idx += 1
            losses.append(batch_loss / max(1, len(bags)))

    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0) if all_labels else 0.0
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0) if all_labels else 0.0
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0) if all_labels else 0.0

    auc = None
    ap = None
    fpr = tpr = precision_curve = recall_curve = np.array([])
    if len(np.unique(all_labels)) > 1:
        scores = np.array(all_probs)[:, 1]
        try:
            auc = roc_auc_score(all_labels, scores)
        except Exception:
            auc = None
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, scores)
        ap = average_precision_score(all_labels, scores)
        fpr, tpr, _ = roc_curve(all_labels, scores)

        # PR
        plt.figure(figsize=(7, 6))
        plt.plot(recall_curve, precision_curve, label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall - {eval_tag}")
        plt.legend(loc="lower left")
        plt.xlim(0, 1); plt.ylim(0, 1.02)
        plt.tight_layout()
        pr_path = os.path.join(out_dir, f"pr_curve_{eval_tag}.png")
        plt.savefig(pr_path, dpi=180); plt.close()

        # ROC
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}" if auc is not None else "AUC=N/A")
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {eval_tag}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(out_dir, f"roc_curve_{eval_tag}.png")
        plt.savefig(roc_path, dpi=180); plt.close()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap='Blues', interpolation='none', aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=12)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks([0,1]); ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks([0,1]); ax.set_yticklabels(class_names, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}", ha='center', va='center', fontsize=11, fontweight='bold')
    fig.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix_{eval_tag}.png")
    plt.savefig(cm_path, dpi=180); plt.close(fig)

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc) if auc is not None else None,
        "ap": float(ap) if ap is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "misclassified_indices": misclassified
    }
    with open(os.path.join(out_dir, f"metrics_{eval_tag}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"[Eval {eval_tag}] Acc={acc:.4f} F1={f1:.4f} Prec={precision:.4f} Rec={recall:.4f} "
              f"AUC={auc if auc is not None else 'NA'} AP={ap if ap is not None else 'NA'}")
        print("Misclassified indices:", misclassified)

    return float(np.mean(losses) if losses else 0.0), acc, f1, auc


# -------------------- Optional calibration --------------------

def calibrate_models(models: List[nn.Module], calib_datasets: List[Dataset], device: torch.device) -> List[float]:
    """
    Temperature scaling per binary classifier on its calibration set.

    For each OvR classifier:
      - Collect logits/labels on its calibration split.
      - Grid-search T in [0.1, 10.0] to minimize CrossEntropy on calibrated logits (logits/T).
      - If no calib data is available, default to T=1.0.

    Args:
        models: Trained OvR binary AttentionMIL models (n_classes elements; some may be skipped).
        calib_datasets: Matching list of calibration datasets or None where unavailable.
        device: torch.device for inference.

    Returns:
        List[float]: Temperature T per classifier (1.0 if not calibrated).
    """
    print("\n===== Calibrating models (temperature scaling) =====")
    temperatures = []
    for idx, (model, ds) in enumerate(zip(models, calib_datasets)):
        if ds is None or len(ds) == 0:
            print(f"  [WARN] Classifier {idx}: empty calib set; using T=1.0")
            temperatures.append(1.0)
            continue
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
        logits_list, labels_list = [], []
        model.eval()
        with torch.no_grad():
            for bags, labels in loader:
                for bag, y in zip(bags, labels):
                    out, _ = model(bag.to(device))
                    logits_list.append(out.unsqueeze(0).cpu())
                    labels_list.append(y.unsqueeze(0).cpu())
        if len(logits_list) == 0:
            print(f"  [WARN] Classifier {idx}: no samples; using T=1.0")
            temperatures.append(1.0)
            continue
        logits = torch.cat(logits_list, dim=0)   # [N, 2]
        labels = torch.cat(labels_list, dim=0)   # [N]
        def nll(T: float) -> float:
            Tt = torch.tensor(T, dtype=torch.float32)
            return nn.CrossEntropyLoss()(logits / Tt, labels).item()
        grid = np.linspace(0.1, 10.0, 200)
        losses = [nll(float(t)) for t in grid]
        T_opt = float(grid[int(np.argmin(losses))])
        temperatures.append(T_opt)
        print(f"  Classifier {idx}: T={T_opt:.4f}")
    return temperatures


# -------------------- Reproducibility --------------------
def set_seed(seed: int):
    """
    Set deterministic seeds across Python, NumPy, and PyTorch.
    Also configures cuDNN for determinism (may affect performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    Seed DataLoader workers for reproducible shuffling/augmentation.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -------------------- Robust 80/10/10 split (binary, stratified) --------------------
def stratified_train_val_cal_split(X, y, val_size=0.10, cal_size=0.10, seed=42):
    """
    Create stratified TRAIN/VAL/CAL splits for binary labels y in {0,1}.

    Strategy:
      - Target VAL and CAL fractions (val_size, cal_size) from positives and negatives separately.
      - Ensure both VAL and CAL contain both classes when possible.
      - If not feasible (e.g., too few positives/negatives), fall back to a stratified 80/20 TRAIN/VAL,
        with empty CAL (caller should skip calibration).

    Args:
        X (list): Bags (arrays) for the OvR classifier.
        y (array-like): Binary labels in {0,1}.
        val_size (float): Fraction for validation split (default 0.10).
        cal_size (float): Fraction for calibration split (default 0.10).
        seed (int): RNG seed.

    Returns:
        X_train, X_val, X_cal (lists of bags), y_train, y_val, y_cal (np.ndarray)

    Raises:
        AssertionError: If y is not binary.
        ValueError: If val_size/cal_size are out of (0,1) or their sum >= 1.
    """
    y = np.asarray(y, dtype=int)
    assert set(np.unique(y)).issubset({0, 1}), "y must be binary (0/1) for OvR split."
    holdout = val_size + cal_size
    if not (0.0 < val_size < 1.0 and 0.0 < cal_size < 1.0 and holdout < 1.0):
        raise ValueError("val_size and cal_size must be in (0,1) and val_size+cal_size < 1.")

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    # If we can't put at least one pos/neg in both val and calib, make calib empty
    if n_pos < 2 or n_neg < 2:
        # 80/20 train/val stratified
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        return X_tr, X_va, [], y_tr, y_va, np.array([], dtype=int)

    # Target counts per class
    n_pos_val = max(1, int(round(n_pos * val_size)))
    n_pos_cal = max(1, int(round(n_pos * cal_size)))
    # ensure we leave at least 1 pos for train
    if n_pos_val + n_pos_cal >= n_pos:
        n_pos_cal = max(1, n_pos - n_pos_val - 1)

    n_neg_val = max(1, int(round(n_neg * val_size)))
    n_neg_cal = max(1, int(round(n_neg * cal_size)))
    if n_neg_val + n_neg_cal >= n_neg:
        n_neg_cal = max(1, n_neg - n_neg_val - 1)

    # Slice indices
    pos_val = pos_idx[:n_pos_val]
    pos_cal = pos_idx[n_pos_val:n_pos_val + n_pos_cal]
    pos_train = pos_idx[n_pos_val + n_pos_cal:]

    neg_val = neg_idx[:n_neg_val]
    neg_cal = neg_idx[n_neg_val:n_neg_val + n_neg_cal]
    neg_train = neg_idx[n_neg_val + n_neg_cal:]

    # Assemble splits
    idx_train = np.concatenate([pos_train, neg_train])
    idx_val   = np.concatenate([pos_val,   neg_val])
    idx_cal   = np.concatenate([pos_cal,   neg_cal])

    rng.shuffle(idx_train); rng.shuffle(idx_val); rng.shuffle(idx_cal)

    X_train = [X[i] for i in idx_train]
    X_val   = [X[i] for i in idx_val]
    X_cal   = [X[i] for i in idx_cal]
    y_train = y[idx_train]
    y_val   = y[idx_val]
    y_cal   = y[idx_cal]

    # If any split lost a class due to rounding, make calib empty (skip calibration)
    if len(np.unique(y_val)) < 2 or len(np.unique(y_cal)) < 2:
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        return X_tr, X_va, [], y_tr, y_va, np.array([], dtype=int)

    return X_train, X_val, X_cal, y_train, y_val, y_cal


# -------------------- OvR training driver --------------------

def main():
    """
    Parse CLI arguments and run OvR training for AttentionMIL.
    Notes:
      - TRAIN split inside the PKL is used for fitting and internal VAL/CAL splits.
      - TEST (if present) is not used here; this script focuses on per-class training and validation.
    """
    parser = argparse.ArgumentParser(description="Train OvR AttentionMIL (binary per class, balanced).")
    parser.add_argument("--mil_pkl", type=str, required=True, help="Path to PKL containing explicit train (and optional test) splits")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save per-class models and metrics")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--calibrate", action="store_true", help="Run temperature scaling on calibration sets after training")
    parser.add_argument("--oversample", action="store_true", help="Use WeightedRandomSampler on the train split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for all RNGs")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--debug", action="store_true", help="Print and save split/imbalance diagnostics")
    parser.add_argument("--val_size", type=float, default=0.10, help="Fraction for validation (default 0.10)")
    parser.add_argument("--cal_size", type=float, default=0.10, help="Fraction for calibration (default 0.10)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"[INFO] Using device={device.type}, seed={args.seed}, oversample={args.oversample}")

    # Load only TRAIN from PKL; keep TEST untouched for later evaluation
    X_train_all, y_train_mc, X_test_all, y_test_mc, class_names_opt = load_train_test_from_pkl(args.mil_pkl)
    # Determine class names and n_classes
    if class_names_opt is not None:
        class_names = list(class_names_opt)
        n_classes = len(class_names)
    else:
        n_classes = int(max(y_train_mc.max(), (y_test_mc.max() if y_test_mc is not None else y_train_mc.max()))) + 1
        class_names = [f"class_{i}" for i in range(n_classes)]

    # Infer input dim from TRAIN only
    first_bag = X_train_all[0]
    input_dim = int(first_bag.shape[1] if hasattr(first_bag, "shape") else len(first_bag[0]))

    print(f"[INFO] Train N={len(X_train_all)} | Test N={(len(X_test_all) if X_test_all is not None else 0)} "
          f"| classes={n_classes} | input_dim={input_dim}")
    print(f"[INFO] Classes: {class_names}")

    saved_models = []
    calib_datasets = []
    per_class_summary = []

    # Use TRAIN only for fitting and internal val split
    bags = X_train_all
    labels_mc = np.asarray(y_train_mc, dtype=int)

    for k in range(n_classes):
        cls_name = class_names[k] if k < len(class_names) else f"class_{k}"
        print(f"\n===== Training classifier {k} ({cls_name}) vs Rest =====")
        y_bin = (labels_mc == k).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            print(f"[WARN] Class {cls_name}: not enough positive/negative samples in TRAIN; skipping.")
            continue

        # Overall TRAIN distribution for this OvR classifier
        total_counts = np.bincount(y_bin, minlength=2)
        total_neg, total_pos = int(total_counts[0]), int(total_counts[1])
        if args.debug:
            print(f"  [DEBUG] Overall TRAIN counts -> neg={total_neg}, pos={total_pos} "
                  f"(pos_rate={total_pos/(total_neg+total_pos):.3f})")

        # 80/10/10: train/val/calib with stratification and safeguards
        X_train, X_val, X_cal, y_train, y_val, y_cal = stratified_train_val_cal_split(
            bags, y_bin, val_size=args.val_size, cal_size=args.cal_size, seed=args.seed
        )

        train_ds = MILDataset(X_train, y_train)
        val_ds = MILDataset(X_val, y_val)
        calib_ds = MILDataset(X_cal, y_cal) if len(y_cal) > 0 else None

        # Sampler for reproducible oversampling
        gen = torch.Generator()
        gen.manual_seed(args.seed)
        if args.oversample:
            from torch.utils.data import WeightedRandomSampler
            y_arr = np.asarray(y_train, dtype=int)
            class_counts = np.bincount(y_arr, minlength=2).astype(float)
            # inverse-frequency sampling weights
            w_pos = 1.0 / max(class_counts[1], 1.0)
            w_neg = 1.0 / max(class_counts[0], 1.0)
            sample_weights = np.where(y_arr == 1, w_pos, w_neg).astype(np.float64)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(y_arr),
                replacement=True
            )
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, sampler=sampler,
                collate_fn=collate_fn, num_workers=args.num_workers,
                worker_init_fn=seed_worker, pin_memory=(device.type == "cuda")
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=args.num_workers,
                generator=gen, worker_init_fn=seed_worker, pin_memory=(device.type == "cuda")
            )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=args.num_workers,
            worker_init_fn=seed_worker, pin_memory=(device.type == "cuda")
        )

        counts = np.bincount(y_train, minlength=2); counts = np.clip(counts, 1, None)
        train_neg, train_pos = int(counts[0]), int(counts[1])
        val_counts = np.bincount(y_val, minlength=2)
        val_neg, val_pos = int(val_counts[0]), int(val_counts[1])

        if args.oversample:
            # Oversampling mode: unweighted CE
            criterion = nn.CrossEntropyLoss().to(device)
            # Expected class fraction under inverse-frequency sampling
            exp_pos = (w_pos * train_pos) / (w_pos * train_pos + w_neg * train_neg) if (train_neg + train_pos) > 0 else 0.5
            exp_neg = 1.0 - exp_pos
            print(f"  [INFO] Oversampling enabled -> unweighted CE")
            print(f"         Train split counts: neg={train_neg}, pos={train_pos} (pos_rate={train_pos/(train_neg+train_pos):.3f})")
            print(f"         Expected sampler mix per epoch: neg={exp_neg:.3f}, pos={exp_pos:.3f}")
        else:
            # Class-weighted CE (balanced by inverse frequency)
            weights = (len(y_train) / (2.0 * counts)).astype(np.float32)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
            print(f"  [INFO] Class-weighted CE enabled")
            print(f"         Train split counts: neg={train_neg}, pos={train_pos} (pos_rate={train_pos/(train_neg+train_pos):.3f})")
            print(f"         CE class weights: w_neg={weights[0]:.4f}, w_pos={weights[1]:.4f}")

        # Debug: show all three splits
        if args.debug:
            val_counts = np.bincount(y_val, minlength=2)
            calib_counts = np.bincount(y_cal, minlength=2) if len(y_cal) > 0 else np.array([0, 0])
            print(f"  [DEBUG] Val split counts:   neg={int(val_counts[0])}, pos={int(val_counts[1])} "
                  f"(pos_rate={(val_counts[1]/max(1, val_counts.sum())):.3f})")
            if calib_ds is None:
                print("  [DEBUG] Calib split counts: neg=0, pos=0 (no calib; T=1.0 will be used)")
            else:
                print(f"  [DEBUG] Calib split counts: neg={int(calib_counts[0])}, pos={int(calib_counts[1])} "
                      f"(pos_rate={(calib_counts[1]/max(1, calib_counts.sum())):.3f})")

            # Persist JSON
            cls_out_dbg = os.path.join(args.output_dir, f"class_{k}_{cls_name}")
            os.makedirs(cls_out_dbg, exist_ok=True)
            tr_counts = np.bincount(y_train, minlength=2)
            debug_json = {
                "overall_train_counts": {
                    "neg": int(total_counts[0]),
                    "pos": int(total_counts[1]),
                    "pos_rate": total_pos / (total_neg + total_pos)
                },
                "train_counts": {
                    "neg": int(tr_counts[0]),
                    "pos": int(tr_counts[1]),
                    "pos_rate": tr_counts[1] / max(1, tr_counts.sum())
                },
                "val_counts": {
                    "neg": int(val_counts[0]),
                    "pos": int(val_counts[1]),
                    "pos_rate": val_counts[1] / max(1, val_counts.sum())
                },
                "calib_counts": None if calib_ds is None else {
                    "neg": int(calib_counts[0]),
                    "pos": int(calib_counts[1]),
                    "pos_rate": calib_counts[1] / max(1, calib_counts.sum())
                },
                "mode": "oversample" if args.oversample else "class_weighted_ce",
                "seed": int(args.seed),
                "batch_size": int(args.batch_size)
            }
            with open(os.path.join(cls_out_dbg, "debug_splits.json"), "w") as fdbg:
                json.dump(debug_json, fdbg, indent=2)

        model = AttentionMIL(input_dim=input_dim, hidden_dim=args.hidden_dim, n_classes=2, dropout_rate=args.dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        cls_out = os.path.join(args.output_dir, f"class_{k}_{cls_name}")
        os.makedirs(cls_out, exist_ok=True)

        best_f1, best_path = -1.0, os.path.join(cls_out, f"model_{k}_{cls_name}.pt")
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1, val_auc = eval_epoch(
                model, val_loader, criterion, device,
                eval_tag=f"{cls_name}_val", out_dir=cls_out, class_names=("Rest", cls_name), verbose=False
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_path)
                print(f"  [INFO] Saved best model -> {best_path}")
            print(f"  Train: loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f}")
            print(f"  Val  : loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} auc={val_auc if val_auc is not None else 'NA'}")

        model.load_state_dict(torch.load(best_path, map_location=device))
        saved_models.append(model)
        calib_datasets.append(calib_ds)

        per_class_summary.append({
            "class_idx": k,
            "class_name": cls_name,
            "model_path": best_path,
            "train_counts": {"neg": int(np.bincount(y_train, minlength=2)[0]),
                             "pos": int(np.bincount(y_train, minlength=2)[1])},
            "val_counts": {"neg": int(val_counts[0]), "pos": int(val_counts[1])},
            "calib_counts": None if calib_ds is None else {
                "neg": int(np.bincount(y_cal, minlength=2)[0]),
                "pos": int(np.bincount(y_cal, minlength=2)[1])
            },
            "best_val_f1": float(best_f1)
        })

    temps = None
    if args.calibrate and saved_models and calib_datasets:
        temps = calibrate_models(saved_models, calib_datasets, device=torch.device(device.type))
        with open(os.path.join(args.output_dir, "temperature_parameters.json"), "w") as f:
            json.dump({"temperatures": [float(t) for t in temps]}, f, indent=2)
        print(f"[DONE] Saved temperatures -> {os.path.join(args.output_dir, 'temperature_parameters.json')}")

    summary = {
        "mil_pkl": args.mil_pkl,
        "output_dir": args.output_dir,
        "classes": class_names,
        "n_classes": n_classes,
        "input_dim": input_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "dropout_rate": args.dropout_rate,
        "calibrated": bool(args.calibrate),
        "temperatures": temps,
        "seed": int(args.seed),
        "oversample": bool(args.oversample),
        "train_size": int(len(X_train_all)),
        "test_size": int(len(X_test_all)) if X_test_all is not None else 0,
        "per_class": per_class_summary
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY] Saved to {os.path.join(args.output_dir, 'training_summary.json')}")


if __name__ == "__main__":
    main()