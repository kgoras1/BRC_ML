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
import csv
import json
import pickle
import argparse
import logging
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

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

# enforce Arial + manuscript base size
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,            # manuscript base text size
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


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


# -------------------- Confidence interval helpers --------------------

def delong_auroc_ci(
    y_true: np.ndarray, scores: np.ndarray, alpha: float = 0.05
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    DeLong 1988 AUROC with analytical 95% CI.
    Returns (auc, ci_lo, ci_hi).  Preferred over bootstrap for AUC.
    """
    from scipy import stats as _stats
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return None, None, None
    vx = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
    vy = np.array([np.mean(q < pos) + 0.5 * np.mean(q == pos) for q in neg])
    auc = float(vx.mean())
    se = float(np.sqrt(np.var(vx, ddof=1) / n1 + np.var(vy, ddof=1) / n0))
    z = float(_stats.norm.ppf(1.0 - alpha / 2.0))
    return auc, float(np.clip(auc - z * se, 0.0, 1.0)), float(np.clip(auc + z * se, 0.0, 1.0))


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, List]:
    """
    Non-parametric bootstrap 95% CIs for binary classification metrics.
    Returns {metric: [lo, hi]}.  [None, None] when too few valid resamples.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted", "precision", "recall", "auroc", "ap"]
    boots: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots["accuracy"].append(accuracy_score(yt, yp))
        boots["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        boots["f1"].append(f1_score(yt, yp, zero_division=0))
        boots["f1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
        boots["f1_weighted"].append(f1_score(yt, yp, average="weighted", zero_division=0))
        boots["precision"].append(precision_score(yt, yp, zero_division=0))
        boots["recall"].append(recall_score(yt, yp, zero_division=0))
        try:
            boots["auroc"].append(roc_auc_score(yt, ypr))
        except Exception:
            pass
        try:
            boots["ap"].append(average_precision_score(yt, ypr))
        except Exception:
            pass
    lo_p, hi_p = alpha / 2 * 100, (1 - alpha / 2) * 100
    return {
        k: [float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))] if len(v) >= 20 else [None, None]
        for k, v in boots.items()
    }


def bootstrap_macro_ci(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    n_classes: int,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, List]:
    """
    Bootstrap 95% CIs for macro-averaged OvR metrics (across all classes).
    Returns {metric: [lo, hi]}.
    """
    rng = np.random.default_rng(seed)
    n = len(all_labels)
    keys = ["f1", "precision", "recall", "balanced_accuracy", "auroc", "ap"]
    boots: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = all_labels[idx]
        yp_s = all_probs[idx]
        per_k: Dict[str, List[float]] = {k: [] for k in keys}
        for k in range(n_classes):
            ytb = (yt == k).astype(int)
            if len(np.unique(ytb)) < 2:
                continue
            sk = yp_s[:, k]
            ypb = (sk >= 0.5).astype(int)
            per_k["f1"].append(f1_score(ytb, ypb, zero_division=0))
            per_k["precision"].append(precision_score(ytb, ypb, zero_division=0))
            per_k["recall"].append(recall_score(ytb, ypb, zero_division=0))
            per_k["balanced_accuracy"].append(balanced_accuracy_score(ytb, ypb))
            try:
                per_k["auroc"].append(roc_auc_score(ytb, sk))
            except Exception:
                pass
            try:
                per_k["ap"].append(average_precision_score(ytb, sk))
            except Exception:
                pass
        for k in keys:
            if per_k[k]:
                boots[k].append(float(np.mean(per_k[k])))
    lo_p, hi_p = alpha / 2 * 100, (1 - alpha / 2) * 100
    return {
        k: [float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))] if len(v) >= 20 else [None, None]
        for k, v in boots.items()
    }


def bootstrap_case_ci(
    case_true: np.ndarray,
    case_scores: np.ndarray,
    n_classes: int,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[Dict[int, Dict[str, List]], Dict[str, List]]:
    """
    Bootstrap 95% CIs at the case/patient level by resampling cases (not WSIs).
    This is the statistically correct unit when multiple WSIs per patient exist,
    as WSIs from the same patient are not independent observations.
    Returns (per_class_ci, macro_ci) where per_class_ci[k] = {metric: [lo, hi]}.
    """
    rng = np.random.default_rng(seed)
    n = len(case_true)
    metric_keys = ["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted",
                   "precision", "recall", "auroc", "ap"]
    macro_keys  = ["f1", "f1_macro", "f1_weighted", "precision", "recall", "balanced_accuracy", "auroc", "ap"]

    per_class_boots: Dict[int, Dict[str, List[float]]] = {k: {m: [] for m in metric_keys} for k in range(n_classes)}
    macro_boots: Dict[str, List[float]] = {m: [] for m in macro_keys}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = case_true[idx]
        ys = case_scores[idx]
        per_iter: Dict[str, List[float]] = {m: [] for m in macro_keys}

        for k in range(n_classes):
            ytb = (yt == k).astype(int)
            if len(np.unique(ytb)) < 2:
                continue
            sk = ys[:, k]
            ypb = (sk >= 0.5).astype(int)

            per_class_boots[k]["accuracy"].append(accuracy_score(ytb, ypb))
            per_class_boots[k]["balanced_accuracy"].append(balanced_accuracy_score(ytb, ypb))
            per_class_boots[k]["f1"].append(f1_score(ytb, ypb, zero_division=0))
            per_class_boots[k]["f1_macro"].append(f1_score(ytb, ypb, average="macro", zero_division=0))
            per_class_boots[k]["f1_weighted"].append(f1_score(ytb, ypb, average="weighted", zero_division=0))
            per_class_boots[k]["precision"].append(precision_score(ytb, ypb, zero_division=0))
            per_class_boots[k]["recall"].append(recall_score(ytb, ypb, zero_division=0))
            per_iter["f1"].append(f1_score(ytb, ypb, zero_division=0))
            per_iter["f1_macro"].append(f1_score(ytb, ypb, average="macro", zero_division=0))
            per_iter["f1_weighted"].append(f1_score(ytb, ypb, average="weighted", zero_division=0))
            per_iter["precision"].append(precision_score(ytb, ypb, zero_division=0))
            per_iter["recall"].append(recall_score(ytb, ypb, zero_division=0))
            per_iter["balanced_accuracy"].append(balanced_accuracy_score(ytb, ypb))
            try:
                auc_k = roc_auc_score(ytb, sk)
                per_class_boots[k]["auroc"].append(auc_k)
                per_iter["auroc"].append(auc_k)
            except Exception:
                pass
            try:
                ap_k = average_precision_score(ytb, sk)
                per_class_boots[k]["ap"].append(ap_k)
                per_iter["ap"].append(ap_k)
            except Exception:
                pass

        for m in macro_keys:
            if per_iter[m]:
                macro_boots[m].append(float(np.mean(per_iter[m])))

    lo_p, hi_p = alpha / 2 * 100, (1 - alpha / 2) * 100
    per_class_ci: Dict[int, Dict[str, List]] = {
        k: {
            m: ([float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))]
                if len(v) >= 20 else [None, None])
            for m, v in per_class_boots[k].items()
        }
        for k in range(n_classes)
    }
    macro_ci: Dict[str, List] = {
        m: ([float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))]
            if len(v) >= 20 else [None, None])
        for m, v in macro_boots.items()
    }
    return per_class_ci, macro_ci


def bootstrap_roc_band(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    n_grid: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Bootstrap pointwise 95% CI band for a ROC curve. Returns (fpr_grid, tpr_lo, tpr_hi)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    fpr_grid = np.linspace(0, 1, n_grid)
    tpr_boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(yt, ys)
        tpr_boot.append(np.interp(fpr_grid, fpr_b, tpr_b))
    if len(tpr_boot) < 20:
        return fpr_grid, None, None
    arr = np.vstack(tpr_boot)
    lo_p, hi_p = alpha / 2 * 100, (1 - alpha / 2) * 100
    return fpr_grid, np.percentile(arr, lo_p, axis=0), np.percentile(arr, hi_p, axis=0)


def bootstrap_pr_band(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    n_grid: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Bootstrap pointwise 95% CI band for a PR curve. Returns (rec_grid, pre_lo, pre_hi)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rec_grid = np.linspace(0, 1, n_grid)
    pre_boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        pre_b, rec_b, _ = precision_recall_curve(yt, ys)
        pre_boot.append(np.interp(rec_grid, rec_b[::-1], pre_b[::-1]))
    if len(pre_boot) < 20:
        return rec_grid, None, None
    arr = np.vstack(pre_boot)
    lo_p, hi_p = alpha / 2 * 100, (1 - alpha / 2) * 100
    return rec_grid, np.percentile(arr, lo_p, axis=0), np.percentile(arr, hi_p, axis=0)


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


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_pos: np.ndarray,
    n_bootstrap: int = 2000,
) -> Dict:
    """Compute standard binary metrics + 95% CIs; returns dict with curves (for plotting) if possible."""
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
        # DeLong CI for AUROC (analytical, preferred for small N)
        _, ci_lo, ci_hi = delong_auroc_ci(y_true, y_prob_pos)
        metrics["auroc_ci"] = [ci_lo, ci_hi]
    else:
        metrics["auroc"] = None
        metrics["ap"] = None
        metrics["auroc_ci"] = [None, None]
    # Bootstrap CIs for threshold-based metrics
    ci = bootstrap_metric_ci(y_true, y_pred, y_prob_pos, n_bootstrap=n_bootstrap)
    for mn, pair in ci.items():
        metrics[f"{mn}_ci"] = pair
    return metrics


def parse_source_and_case_id(sample_id: str) -> Tuple[str, str]:
    """
    Infer cohort/source and case-level ID from a WSI/sample ID.
    This enables case-level aggregation for datasets that may include multiple WSIs per case.
    """
    sid = str(sample_id)

    if sid.startswith("TCGA-"):
        toks = sid.split("-")
        case_id = "-".join(toks[:3]) if len(toks) >= 3 else sid
        return "TCGA", case_id

    if sid.startswith("CPTAC_"):
        # Keep "CPTAC_<Subtype>_<SubjectCode>" as case-level key.
        m = re.match(r"^(CPTAC_[^_]+_[A-Za-z0-9]+)", sid)
        if m:
            return "CPTAC", m.group(1)
        return "CPTAC", sid.split("-")[0]

    if sid.startswith("HER2_Warwick_"):
        # Warwick IDs include score and may include training/testing tags.
        m = re.match(r"^(HER2_Warwick_(?:Training|Testing)_\d+)", sid)
        if m:
            return "Warwick", m.group(1)
        return "Warwick", re.sub(r"(_score_\d+)$", "", sid)

    return "Other", sid


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_stratified_ovr_metrics(
    y_true_mc: np.ndarray,
    y_pred_mc: np.ndarray,
    scores_mc: np.ndarray,
    sources: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Compute one-vs-rest metrics per source and per class.
    Useful for checking cohort shortcut effects (e.g., HER2 consistency across sources).
    """
    out = {}
    for src in sorted(set(sources.tolist())):
        idx = np.where(sources == src)[0]
        yt = y_true_mc[idx]
        yp = y_pred_mc[idx]
        sc = scores_mc[idx, :]
        src_dict = {"n_samples": int(len(idx)), "per_label": {}}

        for k, cname in enumerate(class_names):
            y_true_bin = (yt == k).astype(int)
            y_pred_bin = (yp == k).astype(int)
            m = {
                "support_pos": int((y_true_bin == 1).sum()),
                "support_total": int(len(y_true_bin)),
                "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
                "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
                "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
                "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
            }
            if len(np.unique(y_true_bin)) > 1:
                m["balanced_accuracy"] = float(balanced_accuracy_score(y_true_bin, y_pred_bin))
                try:
                    m["auroc"] = float(roc_auc_score(y_true_bin, sc[:, k]))
                except Exception:
                    m["auroc"] = None
                try:
                    m["ap"] = float(average_precision_score(y_true_bin, sc[:, k]))
                except Exception:
                    m["ap"] = None
            else:
                m["balanced_accuracy"] = None
                m["auroc"] = None
                m["ap"] = None

            src_dict["per_label"][cname] = m

        out[src] = src_dict
    return out


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


def _create_composite_figure3(output_dir: str,
                              class_names: List[str],
                              roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
                              pr_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
                              aucs: Dict[str, Optional[float]],
                              aps: Dict[str, Optional[float]],
                              cm_mc: np.ndarray,
                              roc_bands: Optional[Dict] = None,
                              pr_bands: Optional[Dict] = None,
                              aucs_ci: Optional[Dict] = None,
                              aps_ci: Optional[Dict] = None):
    import matplotlib.lines as mlines
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1])
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_cm = fig.add_subplot(gs[1, 1])

    panel_fs = 20
    axis_fs = 11
    tick_fs = 10
    cm_annot_fs = 16
    legend_fs = 10

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx_c, cname in enumerate(class_names):
        color = colors[idx_c % len(colors)]
        if cname in roc_curves:
            fpr, tpr = roc_curves[cname]
            ax_roc.plot(fpr, tpr, linewidth=2, color=color)
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
    ax_roc.set_xlabel("FPR", fontsize=axis_fs)
    ax_roc.set_ylabel("TPR", fontsize=axis_fs)
    ax_roc.tick_params(labelsize=tick_fs)
    ax_roc.text(-0.12, 1.05, "A", transform=ax_roc.transAxes,
                fontsize=panel_fs, fontweight="bold", va="top")

    for idx_c, cname in enumerate(class_names):
        color = colors[idx_c % len(colors)]
        if cname in pr_curves:
            recall, precision = pr_curves[cname]
            ax_pr.plot(recall, precision, linewidth=2, color=color)
    ax_pr.set_xlabel("Recall", fontsize=axis_fs)
    ax_pr.set_ylabel("Precision", fontsize=axis_fs)
    ax_pr.tick_params(labelsize=tick_fs)
    ax_pr.text(-0.12, 1.05, "B", transform=ax_pr.transAxes,
               fontsize=panel_fs, fontweight="bold", va="top")

    ax_legend.axis("off")
    handles, labels = [], []
    for idx_c, cname in enumerate(class_names):
        color = colors[idx_c % len(colors)]
        handles.append(mlines.Line2D([], [], color=color, linewidth=3))
        auc_val = aucs.get(cname, np.nan)
        ap_val  = aps.get(cname, np.nan)
        auc_ci  = (aucs_ci or {}).get(cname, [None, None])
        ap_ci   = (aps_ci  or {}).get(cname, [None, None])
        auc_str = (f"{auc_val:.3f} [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}]"
                   if (auc_ci[0] is not None) else f"{auc_val:.3f}")
        ap_str  = (f"{ap_val:.3f} [{ap_ci[0]:.3f}–{ap_ci[1]:.3f}]"
                   if (ap_ci[0]  is not None) else f"{ap_val:.3f}")
        labels.append(f"{cname}\n  AUC {auc_str}\n  AP  {ap_str}")
    ax_legend.legend(handles, labels, loc="center left", frameon=False, fontsize=legend_fs)

    ax_cm.imshow(cm_mc, cmap="Blues", interpolation="none")
    ax_cm.set_xticks(range(len(class_names)))
    ax_cm.set_yticks(range(len(class_names)))
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right", fontsize=tick_fs)
    ax_cm.set_yticklabels(class_names, fontsize=tick_fs)
    for i in range(cm_mc.shape[0]):
        for j in range(cm_mc.shape[1]):
            ax_cm.text(j, i, f"{int(cm_mc[i, j])}", ha="center", va="center",
                       fontsize=cm_annot_fs, color="black")
    ax_cm.set_xlabel("Predicted Class", fontsize=axis_fs)
    ax_cm.set_ylabel("True Class", fontsize=axis_fs)
    ax_cm.text(-0.12, 1.05, "C", transform=ax_cm.transAxes,
               fontsize=panel_fs, fontweight="bold", va="top")

    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    fig_path = os.path.join(output_dir, "figure3_composite.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# -------------------- Evaluation --------------------
def evaluate(models_dir: str, mil_pkl: str, output_dir: Optional[str], batch_size: int, device_str: Optional[str], n_bootstrap: int = 2000):
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
    roc_bands_uncal, pr_bands_uncal = {}, {}
    roc_bands_cal, pr_bands_cal = {}, {}
    ap_uncal, ap_cal = {}, {}
    auc_uncal, auc_cal = {}, {}
    # CI dicts for figure legend (DeLong for AUC, bootstrap for AP)
    aucs_ci_uncal: Dict[str, List] = {}
    aps_ci_uncal:  Dict[str, List] = {}
    aucs_ci_cal:   Dict[str, List] = {}
    aps_ci_cal:    Dict[str, List] = {}

    LOGGER.info(f"Computing metrics and bootstrap CIs (n_bootstrap={n_bootstrap}) ...")
    for k in range(n_classes):
        cname = class_names[k] if k < len(class_names) else f"class_{k}"
        y_true_bin = (all_labels == k).astype(int)
        # Uncal
        scores = all_probs_uncal[:, k]
        y_pred_bin = (scores >= 0.5).astype(int)  # threshold on positive prob
        m = compute_binary_metrics(y_true_bin, y_pred_bin, scores, n_bootstrap=n_bootstrap)
        per_class["uncal"][cname] = m
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1])
        binary_conf_mats_uncal.append(cm_bin.astype(np.int64))
        if len(np.unique(y_true_bin)) > 1:
            r_u, p_u = _pr_for_plot(y_true_bin, scores)
            pr_curves_uncal[cname] = (r_u, p_u)
            fpr_u, tpr_u, _ = roc_curve(y_true_bin, scores)
            roc_curves_uncal[cname] = (fpr_u, tpr_u)
            # Bootstrap CI bands for composite figure
            roc_bands_uncal[cname] = bootstrap_roc_band(y_true_bin, scores, n_bootstrap=n_bootstrap // 2)
            pr_bands_uncal[cname]  = bootstrap_pr_band(y_true_bin, scores,  n_bootstrap=n_bootstrap // 2)
        ap_uncal[cname]      = m.get("ap", None)
        auc_uncal[cname]     = m.get("auroc", None)
        aucs_ci_uncal[cname] = m.get("auroc_ci", [None, None])
        aps_ci_uncal[cname]  = m.get("ap_ci",    [None, None])

        # Calibrated
        if all_probs_cal is not None:
            scores_c = all_probs_cal[:, k]
            y_pred_bin_c = (scores_c >= 0.5).astype(int)
            m_c = compute_binary_metrics(y_true_bin, y_pred_bin_c, scores_c, n_bootstrap=n_bootstrap)
            per_class["cal"][cname] = m_c
            cm_bin_c = confusion_matrix(y_true_bin, y_pred_bin_c, labels=[0,1])
            binary_conf_mats_cal.append(cm_bin_c.astype(np.int64))
            if len(np.unique(y_true_bin)) > 1:
                r_c, p_c = _pr_for_plot(y_true_bin, scores_c)
                pr_curves_cal[cname] = (r_c, p_c)
                fpr_c, tpr_c, _ = roc_curve(y_true_bin, scores_c)
                roc_curves_cal[cname] = (fpr_c, tpr_c)
                roc_bands_cal[cname] = bootstrap_roc_band(y_true_bin, scores_c, n_bootstrap=n_bootstrap // 2)
                pr_bands_cal[cname]  = bootstrap_pr_band(y_true_bin, scores_c,  n_bootstrap=n_bootstrap // 2)
            ap_cal[cname]      = m_c.get("ap", None)
            auc_cal[cname]     = m_c.get("auroc", None)
            aucs_ci_cal[cname] = m_c.get("auroc_ci", [None, None])
            aps_ci_cal[cname]  = m_c.get("ap_ci",    [None, None])

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

    # -------- Final decision export (WSI-level) + case-level meta-analysis --------
    final_scores = all_probs_cal if all_probs_cal is not None else all_probs_uncal
    final_tag = "cal" if all_probs_cal is not None else "uncal"
    y_pred_final = np.argmax(final_scores, axis=1)

    # WSI-level export with final predicted class and per-class probabilities
    wsi_rows = []
    for i in range(len(all_ids)):
        source, case_id = parse_source_and_case_id(all_ids[i])
        row = {
            "wsi_id": all_ids[i],
            "source": source,
            "case_id": case_id,
            "true_label_idx": int(all_labels[i]),
            "true_label": class_names[int(all_labels[i])] if int(all_labels[i]) < len(class_names) else str(int(all_labels[i])),
            "pred_label_idx": int(y_pred_final[i]),
            "pred_label": class_names[int(y_pred_final[i])] if int(y_pred_final[i]) < len(class_names) else str(int(y_pred_final[i])),
            "correct": int(int(all_labels[i]) == int(y_pred_final[i])),
        }
        for k, cname in enumerate(class_names):
            row[f"prob_{cname}"] = float(final_scores[i, k])
        wsi_rows.append(row)

    prob_cols = [f"prob_{cname}" for cname in class_names]
    wsi_csv_fields = [
        "wsi_id", "source", "case_id",
        "true_label_idx", "true_label", "pred_label_idx", "pred_label", "correct"
    ] + prob_cols
    wsi_csv_path = os.path.join(output_dir, f"predictions_wsi_{final_tag}.csv")
    _write_csv(wsi_csv_path, wsi_rows, wsi_csv_fields)

    # Case-level aggregation by averaging WSI probabilities within each case
    grouped = defaultdict(list)
    for row in wsi_rows:
        grouped[(row["source"], row["case_id"])].append(row)

    case_rows = []
    case_sources = []
    case_true = []
    case_pred = []
    case_scores = []

    for (source, case_id), rows in grouped.items():
        probs = np.vstack([[float(r[f"prob_{cname}"]) for cname in class_names] for r in rows])
        mean_probs = probs.mean(axis=0)
        pred_idx = int(np.argmax(mean_probs))

        true_labels = [int(r["true_label_idx"]) for r in rows]
        true_counter = Counter(true_labels)
        true_idx = int(true_counter.most_common(1)[0][0])
        true_consistent = int(len(true_counter) == 1)

        case_row = {
            "source": source,
            "case_id": case_id,
            "n_wsis": int(len(rows)),
            "true_label_idx": true_idx,
            "true_label": class_names[true_idx] if true_idx < len(class_names) else str(true_idx),
            "true_label_consistent_across_wsis": true_consistent,
            "pred_label_idx": pred_idx,
            "pred_label": class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx),
            "correct": int(true_idx == pred_idx),
            "wsi_ids": "|".join([str(r["wsi_id"]) for r in rows]),
        }
        for k, cname in enumerate(class_names):
            case_row[f"mean_prob_{cname}"] = float(mean_probs[k])
        case_rows.append(case_row)

        case_sources.append(source)
        case_true.append(true_idx)
        case_pred.append(pred_idx)
        case_scores.append(mean_probs)

    case_sources = np.asarray(case_sources)
    case_true = np.asarray(case_true, dtype=int)
    case_pred = np.asarray(case_pred, dtype=int)
    case_scores = np.vstack(case_scores) if len(case_scores) else np.zeros((0, n_classes), dtype=float)

    # -------- Case-level per-class OvR metrics (PRIMARY — cases are the independent unit) --------
    LOGGER.info(f"Computing case-level per-class metrics and bootstrap CIs (resampling cases, n_bootstrap={n_bootstrap}) ...")
    case_per_class: Dict[str, Dict] = {}
    case_roc_curves: Dict[str, Tuple] = {}
    case_pr_curves: Dict[str, Tuple] = {}
    case_auc_map: Dict[str, Optional[float]] = {}
    case_ap_map:  Dict[str, Optional[float]] = {}
    case_aucs_ci_map: Dict[str, List] = {}
    case_aps_ci_map:  Dict[str, List] = {}
    case_ci_macro: Dict[str, List] = {}
    case_avg_macro: Dict = {}
    case_avg_weighted: Dict = {}

    if len(case_true) > 0:
        case_ci_per_class, case_ci_macro = bootstrap_case_ci(
            case_true, case_scores, n_classes, n_bootstrap=n_bootstrap
        )
        for k in range(n_classes):
            cname = class_names[k] if k < len(class_names) else f"class_{k}"
            y_true_bin_c = (case_true == k).astype(int)
            scores_k_c   = case_scores[:, k]
            y_pred_bin_c = (scores_k_c >= 0.5).astype(int)

            m_c: Dict = {
                "accuracy":          float(accuracy_score(y_true_bin_c, y_pred_bin_c)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true_bin_c, y_pred_bin_c)),
                "precision":         float(precision_score(y_true_bin_c, y_pred_bin_c, zero_division=0)),
                "recall":            float(recall_score(y_true_bin_c, y_pred_bin_c, zero_division=0)),
                "f1":                float(f1_score(y_true_bin_c, y_pred_bin_c, zero_division=0)),
                "f1_macro":          float(f1_score(y_true_bin_c, y_pred_bin_c, average="macro", zero_division=0)),
                "f1_weighted":       float(f1_score(y_true_bin_c, y_pred_bin_c, average="weighted", zero_division=0)),
                "support_pos":       int((y_true_bin_c == 1).sum()),
                "support_total":     int(len(y_true_bin_c)),
            }
            if len(np.unique(y_true_bin_c)) > 1:
                try:
                    m_c["auroc"] = float(roc_auc_score(y_true_bin_c, scores_k_c))
                except Exception:
                    m_c["auroc"] = None
                m_c["ap"] = float(average_precision_score(y_true_bin_c, scores_k_c))
                _, ci_lo, ci_hi = delong_auroc_ci(y_true_bin_c, scores_k_c)
                m_c["auroc_ci"] = [ci_lo, ci_hi]
                m_c["ap_ci"]    = case_ci_per_class[k].get("ap", [None, None])
                r_c, p_c = _pr_for_plot(y_true_bin_c, scores_k_c)
                case_pr_curves[cname]  = (r_c, p_c)
                fpr_c, tpr_c, _ = roc_curve(y_true_bin_c, scores_k_c)
                case_roc_curves[cname] = (fpr_c, tpr_c)
            else:
                m_c["auroc"]    = None
                m_c["ap"]       = None
                m_c["auroc_ci"] = [None, None]
                m_c["ap_ci"]    = [None, None]

            for metric in ["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted",
                           "precision", "recall"]:
                m_c[f"{metric}_ci"] = case_ci_per_class[k].get(metric, [None, None])

            case_per_class[cname]   = m_c
            case_auc_map[cname]     = m_c.get("auroc")
            case_ap_map[cname]      = m_c.get("ap")
            case_aucs_ci_map[cname] = m_c.get("auroc_ci", [None, None])
            case_aps_ci_map[cname]  = m_c.get("ap_ci",    [None, None])

        case_avg_macro, case_avg_weighted = _average_per_class(case_per_class, class_names)

    case_prob_cols = [f"mean_prob_{cname}" for cname in class_names]
    case_csv_fields = [
        "source", "case_id", "n_wsis",
        "true_label_idx", "true_label", "true_label_consistent_across_wsis",
        "pred_label_idx", "pred_label", "correct", "wsi_ids"
    ] + case_prob_cols
    case_csv_path = os.path.join(output_dir, f"predictions_case_{final_tag}.csv")
    _write_csv(case_csv_path, case_rows, case_csv_fields)

    # Case-level overall metrics — f1_macro/f1_weighted use OvR-averaged binary F1 (mean of per-class
    # OvR F1 scores), matching the reference evaluation approach rather than sklearn multiclass macro.
    case_level_summary = {
        "n_cases": int(len(case_true)),
        "accuracy": float(accuracy_score(case_true, case_pred)) if len(case_true) else None,
        "balanced_accuracy": float(balanced_accuracy_score(case_true, case_pred)) if len(case_true) else None,
        "f1_macro": float(case_avg_macro["f1"]) if case_avg_macro.get("f1") is not None else None,
        "f1_weighted": float(case_avg_weighted["f1"]) if case_avg_weighted.get("f1") is not None else None,
        "confusion_matrix": confusion_matrix(case_true, case_pred, labels=list(range(n_classes))).tolist() if len(case_true) else [],
        "classification_report_txt": classification_report(case_true, case_pred, target_names=class_names, digits=3, zero_division=0) if len(case_true) else "",
    }

    # Multi-WSI concordance: for cases with >1 WSI, track how many WSIs agree with case-level prediction
    multi_wsi_concordance = {}  # class_name -> list of per-case dicts
    for cname_idx, cname in enumerate(class_names):
        class_records = []
        for (src, cid), rows in grouped.items():
            if len(rows) <= 1:
                continue
            case_row = next((r for r in case_rows if r["case_id"] == cid and r["source"] == src), None)
            if case_row is None or int(case_row["true_label_idx"]) != cname_idx:
                continue
            case_pred_idx = int(case_row["pred_label_idx"])
            wsi_preds = [int(r["pred_label_idx"]) for r in rows]
            concordant = sum(1 for p in wsi_preds if p == case_pred_idx)
            class_records.append({
                "case_id": cid, "source": src,
                "n_wsis": len(rows),
                "concordant_wsis": concordant,
                "concordance_rate": concordant / len(rows),
            })
        multi_wsi_concordance[cname] = class_records

    # Source-stratified one-vs-rest metrics for reviewer analysis
    wsi_sources = np.asarray([parse_source_and_case_id(sid)[0] for sid in all_ids])
    stratified_wsi = _compute_stratified_ovr_metrics(all_labels, y_pred_final, final_scores, wsi_sources, class_names)
    stratified_case = _compute_stratified_ovr_metrics(case_true, case_pred, case_scores, case_sources, class_names)

    # Explicit HER2 consistency summary by source (if HER2 exists)
    her2_consistency = None
    if "HER2" in class_names:
        h = class_names.index("HER2")
        her2_consistency = {
            "wsi_level": {
                src: vals["per_label"]["HER2"]
                for src, vals in stratified_wsi.items()
                if "HER2" in vals.get("per_label", {})
            },
            "case_level": {
                src: vals["per_label"]["HER2"]
                for src, vals in stratified_case.items()
                if "HER2" in vals.get("per_label", {})
            }
        }

    try:
        use_cal = all_probs_cal is not None and roc_curves_cal and pr_curves_cal
        roc_sel = roc_curves_cal if use_cal else roc_curves_uncal
        pr_sel = pr_curves_cal if use_cal else pr_curves_uncal
        auc_sel = auc_cal if use_cal else auc_uncal
        ap_sel = ap_cal if use_cal else ap_uncal
        # Use case-level CM for the composite figure (primary analysis unit is the patient)
        cm_sel = (np.array(case_level_summary["confusion_matrix"])
                  if case_level_summary.get("confusion_matrix")
                  else (np.array(mc_cal["confusion_matrix"]) if (use_cal and mc_cal)
                        else np.array(mc_uncal["confusion_matrix"])))
        bands_roc_sel = roc_bands_cal if use_cal else roc_bands_uncal
        bands_pr_sel  = pr_bands_cal  if use_cal else pr_bands_uncal
        aucs_ci_sel   = aucs_ci_cal   if use_cal else aucs_ci_uncal
        aps_ci_sel    = aps_ci_cal    if use_cal else aps_ci_uncal
        fig3_path = _create_composite_figure3(
            output_dir, class_names, roc_sel, pr_sel, auc_sel, ap_sel, cm_sel,
            roc_bands=bands_roc_sel, pr_bands=bands_pr_sel,
            aucs_ci=aucs_ci_sel, aps_ci=aps_ci_sel,
        )
        LOGGER.info(f"Saved composite Figure 3: {fig3_path}")
    except Exception as exc:
        LOGGER.warning(f"Composite Figure 3 generation failed: {exc}")

    # -------- Write TXT summary --------
    def fmt(x):
        if x is None:
            return "NA"
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.4f}"
        return str(x)

    def fmt_ci(m: Dict, key: str, decimals: int = 3) -> str:
        """Format a metric with its 95% CI as 'value [lo–hi]'."""
        v = m.get(key)
        ci = m.get(f"{key}_ci", [None, None])
        if v is None:
            return "NA"
        s = f"{float(v):.{decimals}f}"
        if ci and ci[0] is not None:
            s += f" [{float(ci[0]):.{decimals}f}–{float(ci[1]):.{decimals}f}]"
        return s

    # Compute averaged per-class metrics and macro CIs (needed for TXT and JSON)
    avg_uncal_macro, avg_uncal_weighted = _average_per_class(per_class["uncal"], class_names)
    avg_cal_macro = avg_cal_weighted = None
    if all_probs_cal is not None:
        avg_cal_macro, avg_cal_weighted = _average_per_class(per_class["cal"], class_names)
    macro_ci_uncal = bootstrap_macro_ci(all_labels, all_probs_uncal, n_classes, n_bootstrap=n_bootstrap)
    macro_ci_cal = (
        bootstrap_macro_ci(all_labels, all_probs_cal, n_classes, n_bootstrap=n_bootstrap)
        if all_probs_cal is not None else None
    )

    txt_path = os.path.join(output_dir, "metrics_test.txt")
    with open(txt_path, "w") as f:
        f.write(f"Models dir: {models_dir}\n")
        f.write(f"PKL: {mil_pkl}\n")
        f.write(f"Device: {device.type}\n")
        f.write(f"Classes ({n_classes}): {', '.join(class_names)}\n")
        f.write(f"Temperatures loaded: {'yes' if temps is not None else 'no'}\n")
        f.write(f"Bootstrap iterations (CIs): {n_bootstrap}\n\n")

        # Macro averages helper (reused for both case and WSI sections)
        _macro_label_map = {
            "f1": "F1", "f1_macro": "F1_macro", "f1_weighted": "F1_weighted",
            "precision": "Precision", "recall": "Recall",
            "balanced_accuracy": "BalAcc", "auroc": "AUROC", "ap": "AP",
        }

        def _fmt_macro(avg_dict: Dict, ci_dict: Dict) -> str:
            parts = []
            for key in ["f1", "f1_macro", "precision", "recall", "balanced_accuracy", "auroc", "ap"]:
                v = avg_dict.get(key)
                ci = ci_dict.get(key, [None, None])
                if v is None:
                    continue
                label = _macro_label_map.get(key, key)
                s = f"{label}={float(v):.3f}"
                if ci and ci[0] is not None:
                    s += f" [{float(ci[0]):.3f}–{float(ci[1]):.3f}]"
                parts.append(s)
            return ", ".join(parts)

        # ============================================================
        # PRIMARY: Case/patient-level metrics (independent units)
        # CIs obtained by bootstrapping cases, not WSIs.
        # ============================================================
        f.write("=" * 70 + "\n")
        f.write("  PRIMARY RESULTS: CASE/PATIENT LEVEL\n")
        f.write("  (unit of analysis = patient; CIs bootstrapped over cases)\n")
        f.write("=" * 70 + "\n\n")

        if case_per_class:
            f.write(f"Case-level per-class OvR metrics with 95% CIs (n_cases={case_level_summary['n_cases']}):\n")
            for cname in class_names:
                m = case_per_class.get(cname, {})
                f.write(
                    f"- {cname} (n_pos={m.get('support_pos','?')}/{m.get('support_total','?')}): "
                    f"F1={fmt_ci(m,'f1')}, F1_macro={fmt_ci(m,'f1_macro')}, F1_weighted={fmt_ci(m,'f1_weighted')}, "
                    f"Prec={fmt_ci(m,'precision')}, Rec={fmt_ci(m,'recall')}, "
                    f"BalAcc={fmt_ci(m,'balanced_accuracy')}, "
                    f"AUROC={fmt_ci(m,'auroc')} [DeLong], AP={fmt_ci(m,'ap')} [bootstrap]\n"
                )
            f.write("\nCase-level macro-averaged metrics with 95% CIs (bootstrapped over cases):\n")
            f.write(f"  {_fmt_macro(case_avg_macro, case_ci_macro)}\n\n")
        else:
            f.write("(No cases available for case-level metrics.)\n\n")

        f.write("Case-level multiclass confusion matrix and classification report:\n")
        _f1m_ci = case_ci_macro.get("f1", [None, None])
        _f1m_str = fmt(case_level_summary["f1_macro"])
        if _f1m_ci and _f1m_ci[0] is not None:
            _f1m_str += f" [{_f1m_ci[0]:.3f}–{_f1m_ci[1]:.3f}]"
        f.write(f"  n_cases={case_level_summary['n_cases']}, Acc={fmt(case_level_summary['accuracy'])}, "
                f"BalAcc={fmt(case_level_summary['balanced_accuracy'])}, "
                f"F1_macro={_f1m_str}, "
                f"F1_weighted={fmt(case_level_summary['f1_weighted'])}\n")
        if case_level_summary.get("confusion_matrix"):
            _cm = np.array(case_level_summary["confusion_matrix"])
            _cw = max(len(cn) for cn in class_names) + 1
            f.write("  Confusion matrix (rows=true, cols=predicted):\n")
            f.write("  " + " " * (_cw + 2) + "  ".join(f"{cn:>{_cw}}" for cn in class_names) + "\n")
            for _i, _cn in enumerate(class_names):
                f.write(f"  {_cn:>{_cw}}: " + "  ".join(f"{int(_cm[_i, _j]):>{_cw}}" for _j in range(n_classes)) + "\n")
        f.write(case_level_summary["classification_report_txt"] + "\n")

        # ============================================================
        # SECONDARY: WSI-level metrics (supplementary)
        # CIs bootstrapped over WSIs; anti-conservative when multiple
        # WSIs per patient exist (pseudo-replication).
        # ============================================================
        f.write("=" * 70 + "\n")
        f.write("  SECONDARY RESULTS: WSI LEVEL (supplementary)\n")
        f.write("  (unit = slide; CIs bootstrapped over WSIs — treat as supplementary\n")
        f.write("   when patients have multiple slides)\n")
        f.write("=" * 70 + "\n\n")

        f.write("WSI-level per-class metrics with 95% CIs (uncalibrated):\n")
        for cname in class_names:
            m = per_class['uncal'].get(cname, {})
            f.write(
                f"- {cname} (n={m.get('support_pos','?')}/{m.get('support_total','?')}): "
                f"F1={fmt_ci(m,'f1')}, F1_macro={fmt_ci(m,'f1_macro')}, F1_weighted={fmt_ci(m,'f1_weighted')}, "
                f"Prec={fmt_ci(m,'precision')}, Rec={fmt_ci(m,'recall')}, "
                f"BalAcc={fmt_ci(m,'balanced_accuracy')}, "
                f"AUROC={fmt_ci(m,'auroc')} [DeLong], AP={fmt_ci(m,'ap')}\n"
            )
        f.write("\n")
        if all_probs_cal is not None:
            f.write("WSI-level per-class metrics with 95% CIs (calibrated):\n")
            for cname in class_names:
                m = per_class['cal'].get(cname, {})
                f.write(
                    f"- {cname} (n={m.get('support_pos','?')}/{m.get('support_total','?')}): "
                    f"F1={fmt_ci(m,'f1')}, F1_macro={fmt_ci(m,'f1_macro')}, F1_weighted={fmt_ci(m,'f1_weighted')}, "
                    f"Prec={fmt_ci(m,'precision')}, Rec={fmt_ci(m,'recall')}, "
                    f"BalAcc={fmt_ci(m,'balanced_accuracy')}, "
                    f"AUROC={fmt_ci(m,'auroc')} [DeLong], AP={fmt_ci(m,'ap')}\n"
                )
            f.write("\n")

        f.write("WSI-level macro-averaged metrics with 95% CIs (uncalibrated):\n")
        f.write(f"  {_fmt_macro(avg_uncal_macro, macro_ci_uncal)}\n\n")
        if avg_cal_macro is not None and macro_ci_cal is not None:
            f.write("WSI-level macro-averaged metrics with 95% CIs (calibrated):\n")
            f.write(f"  {_fmt_macro(avg_cal_macro, macro_ci_cal)}\n\n")

        # Multiclass confusion matrices (reference)
        f.write("WSI-level multiclass confusion matrix and report (uncalibrated):\n")
        f.write(mc_uncal["classification_report_txt"] + "\n")
        if mc_cal is not None:
            f.write("WSI-level multiclass confusion matrix and report (calibrated):\n")
            f.write(mc_cal["classification_report_txt"] + "\n")

        # Mean binary confusion matrices
        f.write("\nMean binary confusion matrix across classes — WSI-level (uncalibrated):\n")
        f.write(np.array2string(mean_cm_uncal, formatter={'float_kind':lambda v: f'{v:.3f}'}))
        f.write("\n")
        if mean_cm_cal is not None:
            f.write("\nMean binary confusion matrix across classes — WSI-level (calibrated):\n")
            f.write(np.array2string(mean_cm_cal, formatter={'float_kind':lambda v: f'{v:.3f}'}))
            f.write("\n")

        # ============================================================
        # Auxiliary: file paths, concordance, stratified analysis
        # ============================================================
        f.write("\n" + "=" * 70 + "\n")
        f.write("  AUXILIARY: OUTPUT FILES, CONCORDANCE, STRATIFICATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("Final decision outputs:\n")
        f.write(f"- Final score mode: {final_tag}\n")
        f.write(f"- WSI-level predictions CSV: {wsi_csv_path}\n")
        f.write(f"- Case-level aggregated predictions CSV: {case_csv_path}\n\n")

        # Multi-WSI concordance section
        total_multi = sum(len(v) for v in multi_wsi_concordance.values())
        f.write(f"Multi-WSI case concordance (cases with >1 WSI: {total_multi} total):\n")
        f.write("For each class, fraction of WSIs whose individual prediction matches\n")
        f.write("the case-level (probability-averaged) prediction.\n")
        for cname in class_names:
            records = multi_wsi_concordance[cname]
            if not records:
                f.write(f"- {cname}: 0 multi-WSI cases\n")
                continue
            n_cases_m = len(records)
            total_wsis = sum(r["n_wsis"] for r in records)
            total_concordant = sum(r["concordant_wsis"] for r in records)
            mean_rate = np.mean([r["concordance_rate"] for r in records])
            min_rate  = np.min([r["concordance_rate"] for r in records])
            max_rate  = np.max([r["concordance_rate"] for r in records])
            f.write(
                f"- {cname}: {n_cases_m} multi-WSI cases, {total_wsis} WSIs total; "
                f"concordant WSIs={total_concordant}/{total_wsis} ({100*total_concordant/total_wsis:.1f}%); "
                f"per-case rate: mean={mean_rate:.3f}, min={min_rate:.3f}, max={max_rate:.3f}\n"
            )
            for r in records:
                f.write(
                    f"    {r['source']} | {r['case_id']}: {r['concordant_wsis']}/{r['n_wsis']} WSIs concordant "
                    f"({100*r['concordance_rate']:.1f}%)\n"
                )
        f.write("\n")

        f.write("Source-stratified HER2 consistency (OvR, final decisions):\n")
        if her2_consistency is None:
            f.write("- HER2 class not present in class_names; skipped.\n")
        else:
            f.write("- WSI-level:\n")
            for src in sorted(her2_consistency["wsi_level"].keys()):
                m = her2_consistency["wsi_level"][src]
                f.write(
                    f"  {src}: support={m.get('support_pos')}/{m.get('support_total')}, "
                    f"Prec={fmt(m.get('precision'))}, Rec={fmt(m.get('recall'))}, "
                    f"F1={fmt(m.get('f1'))}, BalAcc={fmt(m.get('balanced_accuracy'))}, "
                    f"AUROC={fmt(m.get('auroc'))}, AP={fmt(m.get('ap'))}\n"
                )
            f.write("- Case-level:\n")
            for src in sorted(her2_consistency["case_level"].keys()):
                m = her2_consistency["case_level"][src]
                f.write(
                    f"  {src}: support={m.get('support_pos')}/{m.get('support_total')}, "
                    f"Prec={fmt(m.get('precision'))}, Rec={fmt(m.get('recall'))}, "
                    f"F1={fmt(m.get('f1'))}, BalAcc={fmt(m.get('balanced_accuracy'))}, "
                    f"AUROC={fmt(m.get('auroc'))}, AP={fmt(m.get('ap'))}\n"
                )

    # -------- Manuscript table CSV (case-level PRIMARY + WSI-level secondary, with 95% CIs) --------
    final_pc = per_class["cal"] if all_probs_cal is not None else per_class["uncal"]
    final_macro = avg_cal_macro if avg_cal_macro is not None else avg_uncal_macro
    final_macro_ci = macro_ci_cal if macro_ci_cal is not None else macro_ci_uncal
    table_rows = []

    def _mci(v_dict: Dict, ci_dict: Dict, key: str):
        v  = v_dict.get(key)
        ci = ci_dict.get(key, [None, None])
        if v is None:
            return "NA", "NA"
        vstr  = f"{float(v):.3f}"
        cistr = f"[{ci[0]:.3f}–{ci[1]:.3f}]" if (ci and ci[0] is not None) else "NA"
        return vstr, cistr

    def _row(m: Dict, level: str, label: str) -> Dict:
        def _f(k):   return f"{float(m[k]):.3f}" if m.get(k) is not None else "NA"
        def _ci(k):  ci = m.get(f"{k}_ci", [None, None]); return f"[{ci[0]:.3f}–{ci[1]:.3f}]" if (ci and ci[0] is not None) else "NA"
        return {
            "Level": level, "Class": label,
            "N": m.get("support_total", "-"),
            "F1": _f("f1"), "F1_95CI": _ci("f1"),
            "Precision": _f("precision"), "Precision_95CI": _ci("precision"),
            "Recall": _f("recall"), "Recall_95CI": _ci("recall"),
            "AUROC": _f("auroc"), "AUROC_95CI_DeLong": _ci("auroc"),
            "AP": _f("ap"), "AP_95CI": _ci("ap"),
        }

    # Case-level per-class rows (primary)
    for cname in class_names:
        table_rows.append(_row(case_per_class.get(cname, {}), "Case", cname))

    # Case-level macro row
    f1_v, f1_c  = _mci(case_avg_macro, case_ci_macro, "f1")
    p_v,  p_c   = _mci(case_avg_macro, case_ci_macro, "precision")
    r_v,  r_c   = _mci(case_avg_macro, case_ci_macro, "recall")
    auc_v,auc_c = _mci(case_avg_macro, case_ci_macro, "auroc")
    ap_v, ap_c  = _mci(case_avg_macro, case_ci_macro, "ap")
    table_rows.append({
        "Level": "Case", "Class": "Macro-average", "N": "-",
        "F1": f1_v, "F1_95CI": f1_c,
        "Precision": p_v, "Precision_95CI": p_c,
        "Recall": r_v, "Recall_95CI": r_c,
        "AUROC": auc_v, "AUROC_95CI_DeLong": auc_c,
        "AP": ap_v, "AP_95CI": ap_c,
    })

    # WSI-level per-class rows (secondary)
    for cname in class_names:
        table_rows.append(_row(final_pc.get(cname, {}), "WSI", cname))

    # WSI-level macro row
    f1_v, f1_c  = _mci(final_macro, final_macro_ci, "f1")
    p_v,  p_c   = _mci(final_macro, final_macro_ci, "precision")
    r_v,  r_c   = _mci(final_macro, final_macro_ci, "recall")
    auc_v,auc_c = _mci(final_macro, final_macro_ci, "auroc")
    ap_v, ap_c  = _mci(final_macro, final_macro_ci, "ap")
    table_rows.append({
        "Level": "WSI", "Class": "Macro-average", "N": "-",
        "F1": f1_v, "F1_95CI": f1_c,
        "Precision": p_v, "Precision_95CI": p_c,
        "Recall": r_v, "Recall_95CI": r_c,
        "AUROC": auc_v, "AUROC_95CI_DeLong": auc_c,
        "AP": ap_v, "AP_95CI": ap_c,
    })

    table_csv_path = os.path.join(output_dir, "manuscript_table_metrics_ci.csv")
    _write_csv(table_csv_path, table_rows,
               ["Level", "Class", "N",
                "F1", "F1_95CI", "Precision", "Precision_95CI", "Recall", "Recall_95CI",
                "AUROC", "AUROC_95CI_DeLong", "AP", "AP_95CI"])
    LOGGER.info(f"Saved manuscript table CSV: {table_csv_path}")

    # -------- JSON summary --------
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def _sanitize_per_class(per_class_dict):
        """Drop curve arrays and convert numpy scalars/arrays to JSON-safe values."""
        def _convert(val):
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (np.generic,)):
                return val.item()
            if isinstance(val, list):
                return [(_convert(x) if not isinstance(x, float) else x) for x in val]
            return val
        clean = {}
        for cname, metrics in per_class_dict.items():
            clean_metrics = {}
            for key, val in metrics.items():
                if key.startswith("_"):  # skip stored curves
                    continue
                clean_metrics[key] = _convert(val)
            clean[cname] = clean_metrics
        return clean

    per_class_uncal_json = _sanitize_per_class(per_class["uncal"])
    per_class_cal_json = _sanitize_per_class(per_class["cal"]) if per_class["cal"] else None
    case_per_class_json = _sanitize_per_class(case_per_class) if case_per_class else None

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
            "n_bootstrap": n_bootstrap,
        },
        "case_level_per_class": {
            "note": "PRIMARY results — bootstrapped by resampling cases (independent units)",
            "per_class": case_per_class_json,
            "macro": case_avg_macro,
            "weighted": case_avg_weighted,
            "macro_ci_bootstrap_case": case_ci_macro,
        },
        "wsi_level_per_class": {
            "note": "SECONDARY/supplementary — bootstrapped by resampling WSIs",
            "uncal": per_class_uncal_json,
            "cal": per_class_cal_json if all_probs_cal is not None else None,
        },
        "averages": {
            "uncal_macro": avg_uncal_macro,
            "uncal_macro_ci_bootstrap_wsi": macro_ci_uncal,
            "uncal_weighted": avg_uncal_weighted,
            "cal_macro": avg_cal_macro if all_probs_cal is not None else None,
            "cal_macro_ci_bootstrap_wsi": macro_ci_cal,
            "cal_weighted": avg_cal_weighted if all_probs_cal is not None else None,
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
        },
        "final_decision_exports": {
            "final_tag": final_tag,
            "wsi_csv": wsi_csv_path,
            "case_csv": case_csv_path,
        },
        "case_level_multiclass": case_level_summary,
        "stratified_by_source": {
            "wsi_level": stratified_wsi,
            "case_level": stratified_case,
        },
        "her2_consistency_by_source": her2_consistency,
        "multi_wsi_concordance": {
            cname: [
                {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                 for k, v in r.items()}
                for r in records
            ]
            for cname, records in multi_wsi_concordance.items()
        },
        "data_audit": {
            "n_wsi_rows": int(len(wsi_rows)),
            "n_case_rows": int(len(case_rows)),
            "cases_with_multiple_wsis": int(sum(1 for r in case_rows if int(r["n_wsis"]) > 1)),
            "case_true_label_inconsistencies": int(sum(1 for r in case_rows if int(r["true_label_consistent_across_wsis"]) == 0)),
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
    ap.add_argument("--n_bootstrap", type=int, default=2000,
                    help="Bootstrap iterations for 95%% CIs (default: 2000; use 500 for fast iteration)")
    args = ap.parse_args()
    raise SystemExit(evaluate(args.models_dir, args.mil_pkl, args.output_dir, args.batch_size, args.device, args.n_bootstrap))


if __name__ == "__main__":
    main()
