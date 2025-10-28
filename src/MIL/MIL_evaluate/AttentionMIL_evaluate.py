#!/usr/bin/env python3
"""
Evaluate_Mil.py

One-vs-Rest evaluation for Attention-based Multiple Instance Learning (MIL) models.

What it does:
- Loads test bags and labels from a PKL file (supports multiple formats).
- Loads per-class OvR models (2-way classifier per class) and optional temperature scaling.
- Computes per-class binary metrics (Acc, BalAcc, Precision, Recall, F1, AUROC, AP).
- Produces PR/ROC curves (per-class overlays), mean binary confusion matrices, and
  multiclass reference metrics by argmax across OvR scores.
- Exports a human-readable TXT summary and a machine-readable JSON with all details.
- Dumps attention weights for misclassified samples (per class) for analysis.

Usage:
  python Evaluate_Mil.py --models_dir path/to/models --mil_pkl path/to/test.pkl --output_dir results_eval

Expected inputs:
- models_dir:
    - training_summary.json (recommended) with keys: n_classes, hidden_dim, input_dim, dropout_rate, classes, per_class
      Each per_class entry may include model_path to the saved .pt file.
    - Alternatively, subfolders named "class_{k}_{name}" containing "model_{k}_{name}.pt"
    - Optional calibration file "temperature_parameters.json" with list "temperatures" (T per class)
- mil_pkl: PKL with a test split in any of the supported formats (see load_train_test_from_pkl).

Outputs:
- Curves: roc_curves_*.png, pr_curves_*.png
- Multiclass confusion matrices: confusion_matrix_multiclass_*.png
- Text report: metrics_test.txt
- JSON report: metrics_test.json
- Misclassified attention weights: misclassified_attention/*.json

Author: Konstantinos Papagoras
Date: 2025-09
"""

import os
import json
import pickle
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------- Logging --------------------
def _get_logger(name: str = "eval_mil") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(ch)
    return logger

LOGGER = _get_logger()


# -------------------- IO helpers --------------------
def load_train_test_from_pkl(pkl_path: str):
    """
    Load train/test bags and labels from a PKL in several supported formats.

    Returns:
        X_train, y_train, X_test, y_test, class_names, ids_train, ids_test
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    class_names = None
    if 'class_names' in data and isinstance(data['class_names'], (list, tuple)):
        class_names = list(data['class_names'])
    elif 'label_encoder' in data and hasattr(data['label_encoder'], 'classes_'):
        class_names = list(data['label_encoder'].classes_)

    ids_train = ids_test = None

    if 'train_bags' in data and 'train_labels' in data:
        X_train, y_train = data['train_bags'], np.asarray(data['train_labels'], dtype=int)
        X_test  = data.get('test_bags')
        y_test  = np.asarray(data['test_labels'], dtype=int) if 'test_labels' in data else None
        ids_train = data.get('train_ids') or data.get('ids_train')
        ids_test  = data.get('test_ids')  or data.get('ids_test')
        return X_train, y_train, X_test, y_test, class_names, ids_train, ids_test

    if 'X_train' in data and 'y_train' in data:
        X_train, y_train = data['X_train'], np.asarray(data['y_train'], dtype=int)
        X_test  = data.get('X_test')
        y_test  = np.asarray(data['y_test'], dtype=int) if 'y_test' in data else None
        ids_train = data.get('train_ids') or data.get('ids_train')
        ids_test  = data.get('test_ids')  or data.get('ids_test')
        return X_train, y_train, X_test, y_test, class_names, ids_train, ids_test

    if 'train' in data and 'test' in data and isinstance(data['train'], dict) and isinstance(data['test'], dict):
        X_train, y_train = data['train']['bags'], np.asarray(data['train']['labels'], dtype=int)
        X_test,  y_test  = data['test']['bags'],  np.asarray(data['test']['labels'], dtype=int)
        ids_train = data['train'].get('ids')
        ids_test  = data['test'].get('ids')
        return X_train, y_train, X_test, y_test, class_names, ids_train, ids_test

    raise KeyError("Unrecognized PKL format for test evaluation.")


# -------------------- Data + Model --------------------
class MILDataset(Dataset):
    """Simple dataset holding MIL bags, integer labels, and optional IDs."""
    def __init__(self, bags, labels, ids=None):
        self.bags = bags
        self.labels = np.asarray(labels, dtype=int)
        if ids is None:
            self.ids = [f"sample_{i:05d}" for i in range(len(self.bags))]
        else:
            self.ids = list(ids)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return torch.tensor(self.bags[idx], dtype=torch.float32), int(self.labels[idx]), self.ids[idx]


def collate_fn(batch):
    bags, labels, ids = zip(*batch)
    return list(bags), torch.tensor(labels, dtype=torch.long), list(ids)


class AttentionMIL(nn.Module):
    """
    Minimal Attention-based MIL head:
    - Linear -> ReLU (-> Dropout) as feature extractor
    - Attention MLP to compute per-instance weights
    - Weighted sum to bag embedding, then linear classifier (binary for OvR)
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
        H = self.feature_extractor(bag)              # [N, hidden]
        A = self.attention(H)                        # [N, 1]
        A = torch.softmax(A, dim=0)                  # instance weights
        M = torch.sum(A * H, dim=0)                  # bag embedding
        logits = self.classifier(M)                  # [2]
        return logits, A.squeeze(-1)                 # [N]


# -------------------- Eval helpers --------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_models(models_dir: str, n_classes: int, input_dim: int, hidden_dim: int, dropout: float, device: torch.device):
    """
    Load per-class OvR models. Prefers training_summary.json for paths and metadata.
    Fallback: infer subfolders 'class_{k}_{name}' -> 'model_{k}_{name}.pt'.
    """
    logger = logging.getLogger("eval_mil")
    summary_path = os.path.join(models_dir, "training_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        class_names = summary.get("classes", [f"class_{i}" for i in range(n_classes)])
        if n_classes != summary.get("n_classes", n_classes):
            n_classes = int(summary.get("n_classes", n_classes))
        hidden_dim = int(summary.get("hidden_dim", hidden_dim))
        dropout = float(summary.get("dropout_rate", dropout))
        # Collect model paths in class order if possible
        model_paths = [None] * n_classes
        for entry in summary.get("per_class", []):
            try:
                k = int(entry["class_idx"])
                model_paths[k] = entry.get("model_path")
            except Exception:
                continue
        # Fallback: scan directories
        for k in range(n_classes):
            if model_paths[k] is None:
                cname = class_names[k] if k < len(class_names) else f"class_{k}"
                cand = os.path.join(models_dir, f"class_{k}_{cname}", f"model_{k}_{cname}.pt")
                model_paths[k] = cand
    else:
        # Fallback: infer class_names and model paths
        class_names = []
        model_paths = []
        for k in range(n_classes):
            subdirs = [d for d in os.listdir(models_dir) if d.startswith(f"class_{k}_")]
            if subdirs:
                cname = subdirs[0].split(f"class_{k}_", 1)[1]
                class_names.append(cname)
                model_paths.append(os.path.join(models_dir, subdirs[0], f"model_{k}_{cname}.pt"))
            else:
                class_names.append(f"class_{k}")
                model_paths.append(None)

    models = []
    for k in range(n_classes):
        model = AttentionMIL(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=2, dropout_rate=dropout).to(device)
        mp = model_paths[k]
        if mp is None or not os.path.isfile(mp):
            logger.warning(f"Missing model for class {k} ('{class_names[k]}'): {mp}. Skipping.")
            models.append(None)
            continue
        state = torch.load(mp, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, class_names


def load_temperatures(models_dir: str, n_classes: int) -> Optional[List[float]]:
    """Load per-class temperature parameters if available."""
    tpath = os.path.join(models_dir, "temperature_parameters.json")
    if not os.path.isfile(tpath):
        return None
    with open(tpath, "r") as f:
        tdata = json.load(f)
    temps = tdata.get("temperatures") or tdata.get("T") or None
    if temps is None:
        return None
    # Normalize length
    if len(temps) < n_classes:
        temps = list(temps) + [1.0] * (n_classes - len(temps))
    return [float(t if t is not None else 1.0) for t in temps]


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: np.ndarray) -> Dict:
    """Compute standard binary metrics; returns dict with curves (for plotting) if possible."""
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["support_pos"] = int((y_true == 1).sum())
    metrics["support_total"] = int(len(y_true))
    # Curves
    if len(np.unique(y_true)) > 1:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob_pos))
        except Exception:
            metrics["auroc"] = None
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob_pos)
        metrics["ap"] = float(average_precision_score(y_true, y_prob_pos))
        metrics["_pr_curve"] = (pr_rec, pr_prec)  # store for plotting
        fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
        metrics["_roc_curve"] = (fpr, tpr)
    else:
        metrics["auroc"] = None
        metrics["ap"] = None
    return metrics


# Clean PR for plotting (dedupe/monotone/enforce endpoints)
def _pr_for_plot(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(np.asarray(y_true).astype(int), scores)
    precision = np.nan_to_num(precision, nan=0.0, posinf=1.0, neginf=0.0)
    recall = np.clip(recall, 0.0, 1.0)
    pairs = {}
    for r, p in zip(recall, precision):
        if (r not in pairs) or (p > pairs[r]):
            pairs[r] = p
    r = np.array(sorted(pairs.keys()), dtype=float)
    p = np.array([pairs[rr] for rr in r], dtype=float)
    p = np.maximum.accumulate(p[::-1])[::-1]
    prevalence = float(np.asarray(y_true).mean()) if len(y_true) else 0.0
    if r.size == 0 or r[0] > 0.0:
        r = np.insert(r, 0, 0.0); p = np.insert(p, 0, 1.0)
    else:
        p[0] = 1.0
    if r[-1] < 1.0:
        r = np.append(r, 1.0); p = np.append(p, prevalence)
    p = np.clip(p, 0.0, 1.0)
    return r, p


# Aggregate per-class metrics (macro and weighted by positive support)
def _average_per_class(per_class_dict: Dict[str, Dict], class_names: List[str]):
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f1_macro", "f1_weighted", "auroc", "ap"]
    rows = [per_class_dict.get(c, {}) for c in class_names]
    pos_weights = np.array([r.get("support_pos", 0) or 0 for r in rows], dtype=float)
    macro, weighted = {}, {}
    for k in keys:
        vals = np.array([
            (float(r[k]) if (r.get(k) is not None) else np.nan)
            for r in rows
        ], dtype=float)
        macro[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
        if pos_weights.sum() > 0 and np.isfinite(vals).any():
            mask = np.isfinite(vals)
            if mask.any():
                w = pos_weights[mask]
                v = vals[mask]
                w = w / (w.sum() if w.sum() > 0 else 1.0)
                weighted[k] = float(np.sum(v * w))
            else:
                weighted[k] = None
        else:
            weighted[k] = macro[k]
    return macro, weighted


# -------------------- Evaluation --------------------
def evaluate(models_dir: str, mil_pkl: str, output_dir: Optional[str], batch_size: int, device_str: Optional[str]):
    """
    Run OvR MIL evaluation.

    Args:
        models_dir: Directory containing trained per-class models and (optionally) training_summary.json
        mil_pkl:    PKL file with explicit test split (bags and integer labels)
        output_dir: Directory to store plots, reports, and JSON
        batch_size: DataLoader batch size (bags are variable-length; processing remains per-bag)
        device_str: 'cuda' or 'cpu' (auto-detected if None)

    Produces:
        - metrics_test.txt and metrics_test.json
        - PR/ROC plots and multiclass confusion matrices
        - Misclassified attention weights (JSON per sample/class)
    """
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    if output_dir is None:
        output_dir = os.path.join(models_dir, "eval_test")
    ensure_dir(models_dir)
    ensure_dir(output_dir)

    # add file logger
    fh = logging.FileHandler(os.path.join(output_dir, "evaluate.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    if all(not isinstance(h, logging.FileHandler) for h in LOGGER.handlers):
        LOGGER.addHandler(fh)

    # Load summary (for dims) early if present
    input_dim = None
    hidden_dim = 256
    dropout = 0.0
    n_classes_summary = None
    summary_path = os.path.join(models_dir, "training_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        hidden_dim = int(summary.get("hidden_dim", hidden_dim))
        dropout = float(summary.get("dropout_rate", dropout))
        n_classes_summary = int(summary.get("n_classes", 0))
        input_dim = int(summary.get("input_dim", 0)) or None

    # Load test data
    X_train, y_train, X_test, y_test, class_names_opt, ids_train, ids_test = load_train_test_from_pkl(mil_pkl)
    if X_test is None or y_test is None:
        raise ValueError("Test split not found in PKL. Provide a PKL with explicit test_bags/test_labels or X_test/y_test.")
    y_test_mc = np.asarray(y_test, dtype=int)
    n_classes = int(np.max(y_test_mc)) + 1
    if n_classes_summary is not None and n_classes_summary != n_classes:
        n_classes = n_classes_summary
    class_names = class_names_opt if class_names_opt else [f"class_{i}" for i in range(n_classes)]
    # infer input_dim if needed
    if input_dim is None:
        first_bag = X_test[0] if len(X_test) else X_train[0]
        input_dim = int(first_bag.shape[1] if hasattr(first_bag, "shape") else len(first_bag[0]))

    LOGGER.info(f"Device: {device.type} | Classes: {n_classes} | Hidden: {hidden_dim} | Dropout: {dropout} | Input dim: {input_dim}")
    LOGGER.info(f"Models dir: {models_dir}")
    LOGGER.info(f"PKL path: {mil_pkl}")
    LOGGER.info(f"Output dir: {output_dir}")

    # Load models and temperatures
    models, class_names = load_models(models_dir, n_classes, input_dim, hidden_dim, dropout, device)
    temps = load_temperatures(models_dir, n_classes)
    if temps is not None:
        LOGGER.info("Temperature parameters loaded.")

    # Build test loader
    test_ds = MILDataset(X_test, y_test_mc, ids=ids_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                             pin_memory=(device.type == "cuda"))

    # Storage
    per_class = { "uncal": {}, "cal": {} }
    binary_conf_mats_uncal = []
    binary_conf_mats_cal = []
    all_probs_uncal = []    # [N, n_classes]
    all_probs_cal = []      # [N, n_classes]
    all_labels = []         # [N]
    all_ids = []            # [N]
    # Per-class misclassified attention dump
    miscls_attn_dir = os.path.join(output_dir, "misclassified_attention")
    ensure_dir(miscls_attn_dir)

    # Inference loop: get per-class probs, predictions, and attn for misclassifications
    with torch.inference_mode():
        for bags, labels, ids in tqdm(test_loader, desc="Test", leave=False):
            # bags is a list of [Ti, D]
            for bag, y_true, sid in zip(bags, labels.tolist(), ids):
                y_true = int(y_true)
                all_labels.append(y_true)
                all_ids.append(sid)
                probs_row_uncal = np.zeros(n_classes, dtype=np.float32)
                probs_row_cal   = np.zeros(n_classes, dtype=np.float32)
                for k in range(n_classes):
                    model = models[k]
                    if model is None:
                        continue
                    logits, attn = model(bag.to(device))  # logits [2], attn [T]
                    # uncalibrated
                    p = torch.softmax(logits, dim=0).detach().cpu().numpy()  # [2]
                    prob_pos = float(p[1])
                    probs_row_uncal[k] = prob_pos
                    # calibrated (if temperatures)
                    if temps is not None:
                        T = float(temps[k]) if k < len(temps) else 1.0
                        logits_cal = logits / T
                        p_cal = torch.softmax(logits_cal, dim=0).detach().cpu().numpy()
                        prob_pos_cal = float(p_cal[1])
                        probs_row_cal[k] = prob_pos_cal
                    # binary misclassification for this class (uncalibrated)
                    y_bin = 1 if y_true == k else 0
                    y_pred_bin = int(np.argmax(p))  # 0 or 1
                    if y_pred_bin != y_bin:
                        # save attention weights for this sample and classifier
                        attn_np = attn.detach().cpu().numpy().astype(np.float32).tolist()
                        out_path = os.path.join(miscls_attn_dir, f"class_{k}_{class_names[k]}__{sid}.json")
                        with open(out_path, "w") as f:
                            json.dump({"id": sid, "class_idx": k, "class_name": class_names[k],
                                       "y_true_bin": y_bin, "y_pred_bin": y_pred_bin,
                                       "attn_weights": attn_np}, f, indent=2)
                all_probs_uncal.append(probs_row_uncal)
                if temps is not None:
                    all_probs_cal.append(probs_row_cal)

    all_probs_uncal = np.vstack(all_probs_uncal) if len(all_probs_uncal) else np.zeros((0, n_classes))
    all_probs_cal = np.vstack(all_probs_cal) if len(all_probs_cal) else None
    all_labels = np.asarray(all_labels, dtype=int)

    # Per-class binary metrics and curves (uncalibrated + calibrated)
    # Also build stacked curves for joint ROC/PR figures
    roc_curves_uncal, pr_curves_uncal = {}, {}
    roc_curves_cal, pr_curves_cal = {}, {}
    ap_uncal, ap_cal = {}, {}
    auc_uncal, auc_cal = {}, {}

    for k in range(n_classes):
        cname = class_names[k] if k < len(class_names) else f"class_{k}"
        y_true_bin = (all_labels == k).astype(int)
        # Uncal
        scores = all_probs_uncal[:, k]
        y_pred_bin = (scores >= 0.5).astype(int)  # threshold on positive prob
        m = compute_binary_metrics(y_true_bin, y_pred_bin, scores)
        per_class["uncal"][cname] = m
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1])
        binary_conf_mats_uncal.append(cm_bin.astype(np.int64))
        if len(np.unique(y_true_bin)) > 1:
            r_u, p_u = _pr_for_plot(y_true_bin, scores)
            pr_curves_uncal[cname] = (r_u, p_u)
            fpr_u, tpr_u, _ = roc_curve(y_true_bin, scores)
            roc_curves_uncal[cname] = (fpr_u, tpr_u)
        ap_uncal[cname] = m.get("ap", None)
        auc_uncal[cname] = m.get("auroc", None)

        # Calibrated
        if all_probs_cal is not None:
            scores_c = all_probs_cal[:, k]
            y_pred_bin_c = (scores_c >= 0.5).astype(int)
            m_c = compute_binary_metrics(y_true_bin, y_pred_bin_c, scores_c)
            per_class["cal"][cname] = m_c
            cm_bin_c = confusion_matrix(y_true_bin, y_pred_bin_c, labels=[0,1])
            binary_conf_mats_cal.append(cm_bin_c.astype(np.int64))
            if len(np.unique(y_true_bin)) > 1:
                r_c, p_c = _pr_for_plot(y_true_bin, scores_c)
                pr_curves_cal[cname] = (r_c, p_c)
                fpr_c, tpr_c, _ = roc_curve(y_true_bin, scores_c)
                roc_curves_cal[cname] = (fpr_c, tpr_c)
            ap_cal[cname] = m_c.get("ap", None)
            auc_cal[cname] = m_c.get("auroc", None)

    # Sanity check: AP/AUROC are rank-based, should be unchanged by temperature scaling
    if all_probs_cal is not None:
        for cname in class_names:
            if cname in ap_uncal and cname in ap_cal and ap_uncal[cname] is not None and ap_cal[cname] is not None:
                if not np.isclose(ap_uncal[cname], ap_cal[cname], atol=1e-12):
                    LOGGER.warning(f"AP changed after calibration for {cname}: {ap_uncal[cname]:.6f} -> {ap_cal[cname]:.6f}")
            if cname in auc_uncal and cname in auc_cal and auc_uncal[cname] is not None and auc_cal[cname] is not None:
                if not np.isclose(auc_uncal[cname], auc_cal[cname], atol=1e-12):
                    LOGGER.warning(f"AUROC changed after calibration for {cname}: {auc_uncal[cname]:.6f} -> {auc_cal[cname]:.6f}")

    # Mean binary confusion matrices across classes
    mean_cm_uncal = (np.stack(binary_conf_mats_uncal, axis=0).mean(axis=0)
                     if binary_conf_mats_uncal else np.zeros((2, 2)))
    mean_cm_cal = (np.stack(binary_conf_mats_cal, axis=0).mean(axis=0)
                   if binary_conf_mats_cal else None)

    # Plot all ROC/PR curves (one figure each), using cleaned PR points
    def plot_multi_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str, kind: str, tag: str,
                          label_scores: Optional[Dict[str, float]] = None):
        if not curves:
            return
        plt.figure(figsize=(7, 6))
        for cname, (x, y) in curves.items():
            lbl = cname
            if label_scores and cname in label_scores and label_scores[cname] is not None:
                lbl = f"{cname} ({'AUC' if kind=='roc' else 'AP'}={label_scores[cname]:.3f})"
            plt.plot(x, y, label=lbl)
        if kind == "roc":
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        else:
            plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"{title} ({tag})")
        plt.legend(loc="best", fontsize=9)
        plt.xlim(0, 1); plt.ylim(0, 1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{kind}_curves_{tag}.png"), dpi=180)
        plt.close()

    plot_multi_curves(roc_curves_uncal, "ROC Curves (OvR)", "roc", "uncal", label_scores=auc_uncal)
    plot_multi_curves(pr_curves_uncal,  "PR Curves (OvR)",  "pr",  "uncal", label_scores=ap_uncal)
    if all_probs_cal is not None:
        plot_multi_curves(roc_curves_cal, "ROC Curves (OvR)", "roc", "cal", label_scores=auc_cal)
        plot_multi_curves(pr_curves_cal,  "PR Curves (OvR)",  "pr",  "cal", label_scores=ap_cal)

    # -------- Multiclass (for reference only; confusion matrix + report) --------
    def multiclass_eval(scores: np.ndarray, tag: str):
        y_pred = np.argmax(scores, axis=1)
        report_txt = classification_report(all_labels, y_pred, target_names=class_names, digits=3, zero_division=0)
        cm_mc = confusion_matrix(all_labels, y_pred, labels=list(range(n_classes)))
        # Save confusion matrix figure
        fig, ax = plt.subplots(figsize=(5 + n_classes * 0.5, 4 + n_classes * 0.3))
        im = ax.imshow(cm_mc, cmap="Blues", interpolation="none")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Multiclass Confusion Matrix ({tag})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f"{cm_mc[i,j]}", ha='center', va='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_multiclass_{tag}.png"), dpi=180)
        plt.close(fig)
        return {"classification_report_txt": report_txt, "confusion_matrix": cm_mc.tolist()}

    mc_uncal = multiclass_eval(all_probs_uncal, tag="uncal")
    mc_cal = multiclass_eval(all_probs_cal, tag="cal") if all_probs_cal is not None else None

    # -------- Write TXT summary --------
    def fmt(x):
        if x is None:
            return "NA"
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.4f}"
        return str(x)

    txt_path = os.path.join(output_dir, "metrics_test.txt")
    with open(txt_path, "w") as f:
        f.write(f"Models dir: {models_dir}\n")
        f.write(f"PKL: {mil_pkl}\n")
        f.write(f"Device: {device.type}\n")
        f.write(f"Classes ({n_classes}): {', '.join(class_names)}\n")
        f.write(f"Temperatures loaded: {'yes' if temps is not None else 'no'}\n\n")

        # Per-class metrics
        f.write("Per-class metrics (uncalibrated):\n")
        for cname in class_names:
            m = per_class['uncal'].get(cname, {})
            f.write(f"- {cname}: Acc={fmt(m.get('accuracy'))}, BalAcc={fmt(m.get('balanced_accuracy'))}, "
                    f"Prec={fmt(m.get('precision'))}, Rec={fmt(m.get('recall'))}, "
                    f"F1={fmt(m.get('f1'))}, F1_macro={fmt(m.get('f1_macro'))}, F1_weighted={fmt(m.get('f1_weighted'))}, "
                    f"Support_pos={m.get('support_pos','NA')}/{m.get('support_total','NA')}, "
                    f"AUROC={fmt(m.get('auroc'))}, AP={fmt(m.get('ap'))}\n")
        f.write("\n")
        if all_probs_cal is not None:
            f.write("Per-class metrics (calibrated):\n")
            for cname in class_names:
                m = per_class['cal'].get(cname, {})
                f.write(f"- {cname}: Acc={fmt(m.get('accuracy'))}, BalAcc={fmt(m.get('balanced_accuracy'))}, "
                        f"Prec={fmt(m.get('precision'))}, Rec={fmt(m.get('recall'))}, "
                        f"F1={fmt(m.get('f1'))}, F1_macro={fmt(m.get('f1_macro'))}, F1_weighted={fmt(m.get('f1_weighted'))}, "
                        f"Support_pos={m.get('support_pos','NA')}/{m.get('support_total','NA')}, "
                        f"AUROC={fmt(m.get('auroc'))}, AP={fmt(m.get('ap'))}\n")
            f.write("\n")

        # Compute averaged per-class metrics (PRIMARY overall results)
        avg_uncal_macro, avg_uncal_weighted = _average_per_class(per_class["uncal"], class_names)
        avg_cal_macro = avg_cal_weighted = None
        if all_probs_cal is not None:
            avg_cal_macro, avg_cal_weighted = _average_per_class(per_class["cal"], class_names)

        # Averages of per-class metrics (PRIMARY overall results)
        f.write("Averaged per-class metrics (uncalibrated) — macro across classes:\n")
        f.write("  " + ", ".join([f"{k}={fmt(v)}" for k, v in avg_uncal_macro.items()]) + "\n")
        f.write("Averaged per-class metrics (uncalibrated) — weighted by positive support:\n")
        f.write("  " + ", ".join([f"{k}={fmt(v)}" for k, v in avg_uncal_weighted.items()]) + "\n\n")
        if avg_cal_macro is not None:
            f.write("Averaged per-class metrics (calibrated) — macro across classes:\n")
            f.write("  " + ", ".join([f"{k}={fmt(v)}" for k, v in avg_cal_macro.items()]) + "\n")
            f.write("Averaged per-class metrics (calibrated) — weighted by positive support:\n")
            f.write("  " + ", ".join([f"{k}={fmt(v)}" for k, v in avg_cal_weighted.items()]) + "\n\n")

        # Multiclass confusion matrices (reference)
        f.write("Multiclass (reference, OvR argmax) — confusion matrix and report (uncalibrated):\n")
        f.write(mc_uncal["classification_report_txt"] + "\n")
        if mc_cal is not None:
            f.write("Multiclass (reference, OvR argmax) — confusion matrix and report (calibrated):\n")
            f.write(mc_cal["classification_report_txt"] + "\n")

        # Mean binary confusion matrices
        f.write("\nMean binary confusion matrix across classes (uncalibrated):\n")
        f.write(np.array2string(mean_cm_uncal, formatter={'float_kind':lambda v: f'{v:.3f}'}))
        f.write("\n")
        if mean_cm_cal is not None:
            f.write("\nMean binary confusion matrix across classes (calibrated):\n")
            f.write(np.array2string(mean_cm_cal, formatter={'float_kind':lambda v: f'{v:.3f}'}))
            f.write("\n")

    # -------- JSON summary --------
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    results_json = {
        "config": {
            "models_dir": models_dir,
            "mil_pkl": mil_pkl,
            "device": device.type,
            "n_classes": n_classes,
            "classes": class_names,
            "hidden_dim": hidden_dim,
            "input_dim": input_dim,
            "dropout": dropout,
            "temperatures_loaded": temps is not None,
        },
        "per_class": {
            "uncal": per_class["uncal"],
            "cal": per_class["cal"] if all_probs_cal is not None else None
        },
        "averages": {
            "uncal_macro": _average_per_class(per_class["uncal"], class_names)[0],
            "uncal_weighted": _average_per_class(per_class["uncal"], class_names)[1],
            "cal_macro": _average_per_class(per_class["cal"], class_names)[0] if all_probs_cal is not None else None,
            "cal_weighted": _average_per_class(per_class["cal"], class_names)[1] if all_probs_cal is not None else None,
        },
        "multiclass": {
            "uncal": {
                "confusion_matrix": mc_uncal["confusion_matrix"],
                "classification_report_txt": mc_uncal["classification_report_txt"],
            },
            "cal": {
                "confusion_matrix": mc_cal["confusion_matrix"],
                "classification_report_txt": mc_cal["classification_report_txt"],
            } if mc_cal is not None else None
        },
        "mean_binary_confusion": {
            "uncal": np.asarray(mean_cm_uncal).tolist(),
            "cal": (np.asarray(mean_cm_cal).tolist() if mean_cm_cal is not None else None)
        }
    }
    with open(os.path.join(output_dir, "metrics_test.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    LOGGER.info(f"Saved evaluation artifacts to: {output_dir}")
    return 0


# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser("Evaluate OvR MIL Attention models on unseen test set")
    ap.add_argument("--models_dir", required=True, type=str, help="Directory with trained models (contains training_summary.json or class_* subdirs)")
    ap.add_argument("--mil_pkl", required=True, type=str, help="PKL with explicit test split")
    ap.add_argument("--output_dir", type=str, default=None, help="Where to save evaluation artifacts (default: models_dir/eval_test)")
    ap.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size (bags are variable length; processed per-bag)")
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (default: auto)")
    args = ap.parse_args()
    raise SystemExit(evaluate(args.models_dir, args.mil_pkl, args.output_dir, args.batch_size, args.device))


if __name__ == "__main__":
    main()