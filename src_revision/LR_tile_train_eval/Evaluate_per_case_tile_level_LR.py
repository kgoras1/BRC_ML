#!/usr/bin/env python3
"""
Evaluate_LR_tile.py

One-vs-Rest evaluation for tile-level Logistic Regression (OvR) models.

What it does:
- Loads trained OvR sklearn classifiers from a LR training results directory.
- Loads test tiles from a split_seed_{seed}.pkl file.
- Computes per-class binary metrics (Acc, BalAcc, Precision, Recall, F1, AUROC, AP)
  with 95% CIs (DeLong for AUROC, bootstrap for others).
- Evaluates at tile level (primary), slide level (averaged tile probs), and
  case/patient level (patient-aggregated slide probs).
- Produces PR/ROC curves, confusion matrices, and a composite Figure 3.
- Exports human-readable TXT, machine-readable JSON, and CSV prediction files.
- Source-stratified metrics (TCGA / CPTAC / Warwick / Other).

Usage:
  python Evaluate_LR_tile.py \\
      --models_dir /path/to/Enhanced_LogReg_OvR_seed_42_... \\
      --split_pkl  /path/to/split_seed_42.pkl \\
      --output_dir /path/to/eval_results

Expected inputs:
- models_dir: Directory containing:
    trained_models/classifier_{k}_{class}.pkl  (one OvR pipeline per class)
    enhanced_all_results.json                  (metadata: class names)
  If enhanced_all_results.json is absent, class names are inferred from filenames.

Author: Konstantinos Papagoras
Date: 2026-06
"""

from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# -------------------- Logging --------------------

def _get_logger(name: str = "eval_lr_tile") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(ch)
    return logger


LOGGER = _get_logger()


# -------------------- Patient / source helpers --------------------

def get_patient_id(slide_id: str) -> str:
    """
    Extract patient-level ID from a slide ID.

    TCGA   : TCGA-XX-XXXX-01Z-...  ->  TCGA-XX-XXXX
    CPTAC  : CPTAC_Label_PATID-UUID -> PATID
    Warwick: HER2_Warwick_Subset_N_score_M -> HER2_Warwick_Subset_N
    """
    sid = str(slide_id)
    if "TCGA" in sid:
        parts = sid.split("-")
        return "-".join(parts[:3]) if len(parts) >= 3 else sid
    if "CPTAC" in sid:
        parts = sid.split("_", 2)
        return parts[2].split("-")[0] if len(parts) >= 3 else sid
    if "Warwick" in sid:
        return sid.rsplit("_score_", 1)[0] if "_score_" in sid else sid
    return sid


def parse_source_and_case_id(sample_id: str) -> Tuple[str, str]:
    """Infer cohort/source and case-level ID from a WSI/sample ID."""
    sid = str(sample_id)
    if sid.startswith("TCGA-"):
        toks = sid.split("-")
        case_id = "-".join(toks[:3]) if len(toks) >= 3 else sid
        return "TCGA", case_id
    if sid.startswith("CPTAC_"):
        m = re.match(r"^(CPTAC_[^_]+_[A-Za-z0-9]+)", sid)
        if m:
            return "CPTAC", m.group(1)
        return "CPTAC", sid.split("-")[0]
    if sid.startswith("HER2_Warwick_"):
        m = re.match(r"^(HER2_Warwick_(?:Training|Testing)_\d+)", sid)
        if m:
            return "Warwick", m.group(1)
        return "Warwick", re.sub(r"(_score_\d+)$", "", sid)
    return "Other", sid


# -------------------- IO helpers --------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -------------------- Data loading --------------------

def _load_mil_pkl_flexible(pkl_path: str):
    """
    Load train/test bags from a MIL PKL (same schemas as Attention_based_MIL.py).
    Returns (X_train_bags, y_train, X_test_bags, y_test, class_names, ids_train, ids_test).
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    class_names = None
    if "class_names" in data and isinstance(data["class_names"], (list, tuple)):
        class_names = list(data["class_names"])
    elif "label_encoder" in data and hasattr(data["label_encoder"], "classes_"):
        class_names = list(data["label_encoder"].classes_)

    ids_tr = ids_te = None

    if "train_bags" in data and "train_labels" in data:
        X_tr = data["train_bags"]; y_tr = np.asarray(data["train_labels"], dtype=int)
        X_te = data.get("test_bags")
        y_te = np.asarray(data["test_labels"], dtype=int) if "test_labels" in data else None
        ids_tr = data.get("train_ids") or data.get("ids_train")
        ids_te = data.get("test_ids")  or data.get("ids_test")
        return X_tr, y_tr, X_te, y_te, class_names, ids_tr, ids_te

    if "X_train" in data and "y_train" in data:
        X_tr = data["X_train"]; y_tr = np.asarray(data["y_train"], dtype=int)
        X_te = data.get("X_test")
        y_te = np.asarray(data["y_test"], dtype=int) if "y_test" in data else None
        ids_tr = data.get("train_ids") or data.get("ids_train")
        ids_te = data.get("test_ids")  or data.get("ids_test")
        return X_tr, y_tr, X_te, y_te, class_names, ids_tr, ids_te

    if "train" in data and isinstance(data["train"], dict):
        X_tr = data["train"]["bags"]; y_tr = np.asarray(data["train"]["labels"], dtype=int)
        X_te = data["test"]["bags"];  y_te = np.asarray(data["test"]["labels"], dtype=int)
        ids_tr = data["train"].get("ids"); ids_te = data["test"].get("ids")
        return X_tr, y_tr, X_te, y_te, class_names, ids_tr, ids_te

    raise KeyError(f"Unrecognized MIL PKL format in {pkl_path}.")


def load_test_tiles(
    split_pkl: Optional[str] = None,
    mil_pkl: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[List[str]]]:
    """
    Load test tile features, labels, and slide IDs.

    Accepts either:
    - split_pkl: split_seed_{seed}.pkl with test_dict_paths / test_labels
    - mil_pkl:   MIL PKL (same file as Attention_based_MIL.py) with test_bags / test_labels

    Returns:
        X_test    : (N_tiles, feat_dim) float32 array
        y_test    : (N_tiles,) integer label array
        slide_ids : list of slide ID strings, one per tile
        class_names: list of class name strings (from MIL PKL) or None
    """
    if mil_pkl is not None:
        _, _, X_te_bags, y_te_bags, class_names, _, ids_te = _load_mil_pkl_flexible(mil_pkl)
        if X_te_bags is None or y_te_bags is None:
            raise ValueError("MIL PKL does not contain a test split.")
        X_list, y_list, ids_list = [], [], []
        for i, (bag, label) in enumerate(zip(X_te_bags, y_te_bags)):
            bag_arr = np.asarray(bag, dtype=np.float32)
            n_t = bag_arr.shape[0]
            X_list.append(bag_arr)
            y_list.extend([int(label)] * n_t)
            sid = str(ids_te[i]) if ids_te is not None else f"slide_{i:05d}"
            ids_list.extend([sid] * n_t)
        X = np.vstack(X_list) if X_list else np.zeros((0, 0), dtype=np.float32)
        return X, np.array(y_list, dtype=int), ids_list, class_names

    # dict-paths format
    with open(split_pkl, "rb") as f:
        split_data = pickle.load(f)
    test_dict_paths: List[str] = split_data["test_dict_paths"]
    test_labels = split_data["test_labels"]

    unique_labels = sorted(set(test_labels))
    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}

    X, y, slide_ids = [], [], []
    for dict_path, label in zip(test_dict_paths, test_labels):
        with open(dict_path, "rb") as f:
            tile_dict = pickle.load(f)
        slide_id = os.path.basename(dict_path).replace(".pkl", "")
        for tile in tile_dict.values():
            X.append(tile["feature"])
            y.append(label_to_int[label])
            slide_ids.append(slide_id)

    return np.array(X, dtype=np.float32), np.array(y, dtype=int), slide_ids, None


def load_models_and_classes(
    models_dir: str,
) -> Tuple[List, List[str], List[float]]:
    """
    Load OvR sklearn classifiers, class names, and calibration temperatures.

    Supports both model formats:
    - New format (dict): {"pipeline": ImbPipeline, "temperature": float, ...}
    - Old format (bare): ImbPipeline  (temperature defaults to 1.0 = no scaling)

    Prefers class names from enhanced_all_results.json; falls back to filename inference.
    Models are in models_dir/trained_models/classifier_{k}_{class}.pkl.
    Returns (classifiers, class_names, temperatures) — classifiers[k] may be None if missing.
    """
    class_names: Optional[List[str]] = None
    for meta_fname in ("enhanced_all_results.json",):
        meta_path = os.path.join(models_dir, meta_fname)
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            class_names = meta.get("class_names")
            break

    trained_dir = os.path.join(models_dir, "trained_models")
    if not os.path.isdir(trained_dir):
        raise FileNotFoundError(f"trained_models directory not found: {trained_dir}")

    clf_files = sorted(
        [fn for fn in os.listdir(trained_dir) if fn.startswith("classifier_") and fn.endswith(".pkl")],
        key=lambda fn: int(fn.split("_")[1]),
    )
    if not clf_files:
        raise FileNotFoundError(f"No classifier_*.pkl files found in {trained_dir}")

    if class_names is None:
        class_names = []
        for fn in clf_files:
            parts = fn.replace(".pkl", "").split("_", 2)
            class_names.append(parts[2] if len(parts) >= 3 else fn)

    n_classes = len(class_names)
    classifiers: List = [None] * n_classes
    temperatures: List[float] = [1.0] * n_classes

    for fn in clf_files:
        parts = fn.replace(".pkl", "").split("_", 2)
        try:
            k = int(parts[1])
        except (IndexError, ValueError):
            continue
        if k >= n_classes:
            continue
        clf_path = os.path.join(trained_dir, fn)
        with open(clf_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            classifiers[k] = obj["pipeline"]
            temperatures[k] = float(obj.get("temperature", 1.0))
        else:
            classifiers[k] = obj          # old bare-pipeline format
            temperatures[k] = 1.0
        LOGGER.info(
            f"  Loaded classifier {k} ({class_names[k]}): T={temperatures[k]:.4f}  {clf_path}"
        )

    missing = [class_names[k] for k in range(n_classes) if classifiers[k] is None]
    if missing:
        LOGGER.warning(f"Missing classifiers for classes: {missing}")

    return classifiers, class_names, temperatures


# -------------------- Confidence interval helpers --------------------

def delong_auroc_ci(
    y_true: np.ndarray, scores: np.ndarray, alpha: float = 0.05
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """DeLong 1988 AUROC with analytical 95% CI. Returns (auc, ci_lo, ci_hi)."""
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
    z = float(scipy_stats.norm.ppf(1.0 - alpha / 2.0))
    return auc, float(np.clip(auc - z * se, 0.0, 1.0)), float(np.clip(auc + z * se, 0.0, 1.0))


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, List]:
    """Non-parametric bootstrap 95% CIs for binary classification metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted",
            "precision", "recall", "auroc", "ap"]
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
        k: [float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))]
        if len(v) >= 20 else [None, None]
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
    """Bootstrap 95% CIs for macro-averaged OvR metrics."""
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
        k: [float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))]
        if len(v) >= 20 else [None, None]
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
    Bootstrap 95% CIs at the case/patient level by resampling cases (not tiles/slides).
    This is the statistically correct unit when multiple slides per patient exist,
    as slides from the same patient are not independent observations.
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
    y_true: np.ndarray, scores: np.ndarray,
    n_bootstrap: int = 1000, alpha: float = 0.05,
    n_grid: int = 100, seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Bootstrap pointwise 95% CI band for a ROC curve."""
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
    y_true: np.ndarray, scores: np.ndarray,
    n_bootstrap: int = 1000, alpha: float = 0.05,
    n_grid: int = 100, seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Bootstrap pointwise 95% CI band for a PR curve."""
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


# -------------------- Metric computation --------------------

def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_pos: np.ndarray,
    n_bootstrap: int = 2000,
) -> Dict:
    """Compute standard binary metrics + 95% CIs."""
    metrics: Dict = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["support_pos"] = int((y_true == 1).sum())
    metrics["support_total"] = int(len(y_true))
    if len(np.unique(y_true)) > 1:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob_pos))
        except Exception:
            metrics["auroc"] = None
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob_pos)
        metrics["ap"] = float(average_precision_score(y_true, y_prob_pos))
        metrics["_pr_curve"] = (pr_rec, pr_prec)
        fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
        metrics["_roc_curve"] = (fpr, tpr)
        _, ci_lo, ci_hi = delong_auroc_ci(y_true, y_prob_pos)
        metrics["auroc_ci"] = [ci_lo, ci_hi]
    else:
        metrics["auroc"] = None
        metrics["ap"] = None
        metrics["auroc_ci"] = [None, None]
    ci = bootstrap_metric_ci(y_true, y_pred, y_prob_pos, n_bootstrap=n_bootstrap)
    for mn, pair in ci.items():
        metrics[f"{mn}_ci"] = pair
    return metrics


def _pr_for_plot(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clean PR curve for plotting (dedupe, monotone, enforce endpoints)."""
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


def _average_per_class(per_class_dict: Dict[str, Dict], class_names: List[str]):
    """Compute macro and support-weighted averages of per-class metrics."""
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1",
            "f1_macro", "f1_weighted", "auroc", "ap"]
    rows = [per_class_dict.get(c, {}) for c in class_names]
    pos_weights = np.array([r.get("support_pos", 0) or 0 for r in rows], dtype=float)
    macro, weighted = {}, {}
    for k in keys:
        vals = np.array([
            (float(r[k]) if (r.get(k) is not None) else np.nan) for r in rows
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


def _compute_stratified_ovr_metrics(
    y_true_mc: np.ndarray,
    y_pred_mc: np.ndarray,
    scores_mc: np.ndarray,
    sources: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Per-source OvR metrics."""
    out = {}
    for src in sorted(set(sources.tolist())):
        idx = np.where(sources == src)[0]
        yt = y_true_mc[idx]; yp = y_pred_mc[idx]; sc = scores_mc[idx, :]
        src_dict = {"n_samples": int(len(idx)), "per_label": {}}
        for k, cname in enumerate(class_names):
            y_true_bin = (yt == k).astype(int)
            y_pred_bin = (yp == k).astype(int)
            m: Dict = {
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
                m["balanced_accuracy"] = None; m["auroc"] = None; m["ap"] = None
            src_dict["per_label"][cname] = m
        out[src] = src_dict
    return out


# -------------------- Plotting helpers --------------------

def _create_composite_figure(
    output_dir: str, class_names: List[str],
    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pr_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    aucs: Dict[str, Optional[float]], aps: Dict[str, Optional[float]],
    cm_mc: np.ndarray,
    aucs_ci: Optional[Dict] = None, aps_ci: Optional[Dict] = None,
    tag: str = "tile",
) -> str:
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1])
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_cm = fig.add_subplot(gs[1, 1])

    panel_fs, axis_fs, tick_fs, cm_annot_fs, legend_fs = 20, 11, 10, 16, 10
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx_c, cname in enumerate(class_names):
        color = colors[idx_c % len(colors)]
        if cname in roc_curves:
            fpr, tpr = roc_curves[cname]
            ax_roc.plot(fpr, tpr, linewidth=2, color=color)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax_roc.set_xlabel("FPR", fontsize=axis_fs)
    ax_roc.set_ylabel("TPR", fontsize=axis_fs)
    ax_roc.tick_params(labelsize=tick_fs)
    ax_roc.text(-0.12, 1.05, "A", transform=ax_roc.transAxes,
                fontsize=panel_fs, fontweight="bold", va="top")

    for idx_c, cname in enumerate(class_names):
        color = colors[idx_c % len(colors)]
        if cname in pr_curves:
            recall_arr, precision_arr = pr_curves[cname]
            ax_pr.plot(recall_arr, precision_arr, linewidth=2, color=color)
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
        ap_val = aps.get(cname, np.nan)
        auc_ci = (aucs_ci or {}).get(cname, [None, None])
        ap_ci = (aps_ci or {}).get(cname, [None, None])
        auc_str = (f"{auc_val:.3f} [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}]"
                   if (auc_val is not None and auc_ci[0] is not None) else
                   (f"{auc_val:.3f}" if auc_val is not None else "NA"))
        ap_str = (f"{ap_val:.3f} [{ap_ci[0]:.3f}–{ap_ci[1]:.3f}]"
                  if (ap_val is not None and ap_ci[0] is not None) else
                  (f"{ap_val:.3f}" if ap_val is not None else "NA"))
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
    fig_path = os.path.join(output_dir, f"figure3_composite_{tag}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _plot_multi_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str, kind: str, tag: str, output_dir: str,
    label_scores: Optional[Dict[str, float]] = None,
) -> None:
    if not curves:
        return
    plt.figure(figsize=(7, 6))
    for cname, (x, y) in curves.items():
        lbl = cname
        if label_scores and cname in label_scores and label_scores[cname] is not None:
            lbl = f"{cname} ({'AUC' if kind == 'roc' else 'AP'}={label_scores[cname]:.3f})"
        plt.plot(x, y, label=lbl)
    if kind == "roc":
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    else:
        plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"{title} ({tag})")
    plt.legend(loc="best", fontsize=9)
    plt.xlim(0, 1); plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{kind}_curves_{tag}.png"), dpi=180)
    plt.close()


def _plot_multiclass_cm(cm_mc: np.ndarray, class_names: List[str],
                        tag: str, output_dir: str) -> None:
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(5 + n_classes * 0.5, 4 + n_classes * 0.3))
    im = ax.imshow(cm_mc, cmap="Blues", interpolation="none")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Multiclass Confusion Matrix ({tag})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm_mc[i, j]}", ha="center", va="center",
                    fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_multiclass_{tag}.png"), dpi=180)
    plt.close(fig)


# -------------------- Aggregation helpers --------------------

def aggregate_to_slide(
    tile_probs: np.ndarray,
    tile_labels: np.ndarray,
    slide_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Average tile probabilities within each slide to get slide-level predictions.

    Returns:
        slide_probs  : (N_slides, n_classes)
        slide_preds  : (N_slides,)
        slide_labels : (N_slides,)
        unique_slides: list of slide IDs
    """
    grouped: Dict[str, List[int]] = defaultdict(list)
    for i, sid in enumerate(slide_ids):
        grouped[sid].append(i)

    unique_slides = sorted(grouped.keys())
    slide_probs_list, slide_labels_list = [], []
    for sid in unique_slides:
        idxs = grouped[sid]
        slide_probs_list.append(tile_probs[idxs].mean(axis=0))
        label_counts = Counter(tile_labels[idxs].tolist())
        slide_labels_list.append(label_counts.most_common(1)[0][0])

    slide_probs_arr = np.vstack(slide_probs_list)
    slide_labels_arr = np.array(slide_labels_list, dtype=int)
    slide_preds_arr = np.argmax(slide_probs_arr, axis=1)
    return slide_probs_arr, slide_preds_arr, slide_labels_arr, unique_slides


# -------------------- Main evaluation --------------------

def evaluate(
    models_dir: str,
    output_dir: Optional[str],
    n_bootstrap: int = 2000,
    split_pkl: Optional[str] = None,
    mil_pkl: Optional[str] = None,
) -> int:
    """
    Run tile-level LR OvR evaluation.

    Evaluates at three levels: tile, slide (tile-avg), case/patient (slide-avg).
    """
    if output_dir is None:
        output_dir = os.path.join(models_dir, "eval_test")
    ensure_dir(output_dir)

    fh = logging.FileHandler(os.path.join(output_dir, "evaluate_lr.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    if all(not isinstance(h, logging.FileHandler) for h in LOGGER.handlers):
        LOGGER.addHandler(fh)

    data_source = mil_pkl if mil_pkl is not None else split_pkl
    LOGGER.info(f"Models dir : {models_dir}")
    LOGGER.info(f"Data PKL   : {data_source}")
    LOGGER.info(f"Output dir : {output_dir}")

    # Load models (class names from metadata; may be overridden by PKL below)
    classifiers, class_names, temperatures = load_models_and_classes(models_dir)
    n_classes = len(class_names)
    LOGGER.info(f"Classes ({n_classes}): {class_names}")
    LOGGER.info(f"Temperatures (T): { {class_names[k]: temperatures[k] for k in range(n_classes)} }")

    # Load test tiles
    LOGGER.info("Loading test tiles ...")
    X_test, y_test, slide_ids, class_names_from_pkl = load_test_tiles(
        split_pkl=split_pkl, mil_pkl=mil_pkl
    )
    # PKL-provided class names take precedence (they are authoritative for MIL PKL)
    if class_names_from_pkl is not None:
        class_names = class_names_from_pkl
        n_classes = len(class_names)
    LOGGER.info(f"Test tiles: {X_test.shape[0]} | Slides: {len(set(slide_ids))}")

    # Tile-level inference: temperature-scaled calibrated probabilities.
    # sigmoid(logit / T) — identical to the MIL temperature scaling approach.
    # Slide-level predictions are obtained by mean-pooling these tile probabilities.
    LOGGER.info("Running tile-level inference (temperature-scaled) ...")
    tile_probs = np.zeros((X_test.shape[0], n_classes), dtype=np.float32)
    for k, (clf, T) in enumerate(zip(classifiers, temperatures)):
        if clf is None:
            continue
        logits = clf.decision_function(X_test)          # (n_tiles,) uncalibrated log-odds
        tile_probs[:, k] = (1.0 / (1.0 + np.exp(-logits / T))).astype(np.float32)
    tile_preds = np.argmax(tile_probs, axis=1)

    # ── TILE-LEVEL EVALUATION ────────────────────────────────────────────────
    # Tile-level bootstrap is capped: tiles within a slide are not independent,
    # and tile counts can reach millions making full n_bootstrap very slow.
    n_bootstrap_tile = min(n_bootstrap, 500)
    LOGGER.info(f"Computing tile-level metrics (n_bootstrap_tile={n_bootstrap_tile}, full n_bootstrap={n_bootstrap}) ...")
    per_class_tile: Dict[str, Dict] = {}
    roc_curves_tile, pr_curves_tile = {}, {}
    auc_tile, ap_tile = {}, {}
    aucs_ci_tile: Dict[str, List] = {}
    aps_ci_tile: Dict[str, List] = {}
    binary_conf_mats_tile = []

    for k in range(n_classes):
        cname = class_names[k]
        y_true_bin = (y_test == k).astype(int)
        scores = tile_probs[:, k]
        y_pred_bin = (scores >= 0.5).astype(int)

        m = compute_binary_metrics(y_true_bin, y_pred_bin, scores, n_bootstrap=n_bootstrap_tile)
        per_class_tile[cname] = m
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        binary_conf_mats_tile.append(cm_bin.astype(np.int64))

        if len(np.unique(y_true_bin)) > 1:
            r, p = _pr_for_plot(y_true_bin, scores)
            pr_curves_tile[cname] = (r, p)
            fpr, tpr, _ = roc_curve(y_true_bin, scores)
            roc_curves_tile[cname] = (fpr, tpr)

        auc_tile[cname] = m.get("auroc", None)
        ap_tile[cname] = m.get("ap", None)
        aucs_ci_tile[cname] = m.get("auroc_ci", [None, None])
        aps_ci_tile[cname] = m.get("ap_ci", [None, None])

    mean_cm_tile = (np.stack(binary_conf_mats_tile, axis=0).mean(axis=0)
                    if binary_conf_mats_tile else np.zeros((2, 2)))

    # Multiclass tile
    cm_mc_tile = confusion_matrix(y_test, tile_preds, labels=list(range(n_classes)))
    report_tile_txt = classification_report(y_test, tile_preds,
                                            target_names=class_names, digits=3, zero_division=0)

    avg_tile_macro, avg_tile_weighted = _average_per_class(per_class_tile, class_names)
    macro_ci_tile = bootstrap_macro_ci(y_test, tile_probs, n_classes, n_bootstrap=n_bootstrap_tile)

    # ── SLIDE-LEVEL EVALUATION ────────────────────────────────────────────────
    LOGGER.info("Aggregating to slide level ...")
    slide_probs, slide_preds, slide_labels, unique_slides = aggregate_to_slide(
        tile_probs, y_test, slide_ids
    )
    LOGGER.info(f"Slides: {len(unique_slides)}")

    per_class_slide: Dict[str, Dict] = {}
    roc_curves_slide, pr_curves_slide = {}, {}
    auc_slide, ap_slide = {}, {}
    aucs_ci_slide: Dict[str, List] = {}
    aps_ci_slide: Dict[str, List] = {}
    binary_conf_mats_slide = []

    LOGGER.info(f"Computing slide-level metrics (n_bootstrap={n_bootstrap}) ...")
    for k in range(n_classes):
        cname = class_names[k]
        y_true_bin = (slide_labels == k).astype(int)
        scores = slide_probs[:, k]
        y_pred_bin = (scores >= 0.5).astype(int)

        m = compute_binary_metrics(y_true_bin, y_pred_bin, scores, n_bootstrap=n_bootstrap)
        per_class_slide[cname] = m
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        binary_conf_mats_slide.append(cm_bin.astype(np.int64))

        if len(np.unique(y_true_bin)) > 1:
            r, p = _pr_for_plot(y_true_bin, scores)
            pr_curves_slide[cname] = (r, p)
            fpr, tpr, _ = roc_curve(y_true_bin, scores)
            roc_curves_slide[cname] = (fpr, tpr)

        auc_slide[cname] = m.get("auroc", None)
        ap_slide[cname] = m.get("ap", None)
        aucs_ci_slide[cname] = m.get("auroc_ci", [None, None])
        aps_ci_slide[cname] = m.get("ap_ci", [None, None])

    mean_cm_slide = (np.stack(binary_conf_mats_slide, axis=0).mean(axis=0)
                     if binary_conf_mats_slide else np.zeros((2, 2)))
    cm_mc_slide = confusion_matrix(slide_labels, slide_preds, labels=list(range(n_classes)))
    report_slide_txt = classification_report(slide_labels, slide_preds,
                                             target_names=class_names, digits=3, zero_division=0)

    avg_slide_macro, avg_slide_weighted = _average_per_class(per_class_slide, class_names)
    macro_ci_slide = bootstrap_macro_ci(slide_labels, slide_probs, n_classes, n_bootstrap=n_bootstrap)

    # ── CASE/PATIENT-LEVEL EVALUATION ────────────────────────────────────────
    LOGGER.info("Aggregating to case/patient level ...")
    slide_sources = np.array([parse_source_and_case_id(s)[0] for s in unique_slides])
    slide_case_ids = np.array([parse_source_and_case_id(s)[1] for s in unique_slides])

    grouped_cases: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, (src, cid) in enumerate(zip(slide_sources.tolist(), slide_case_ids.tolist())):
        grouped_cases[(src, cid)].append(i)

    case_rows, case_true_list, case_pred_list = [], [], []
    case_probs_list, case_sources_list = [], []

    for (src, cid), idxs in grouped_cases.items():
        probs_c = slide_probs[idxs].mean(axis=0)
        pred_idx = int(np.argmax(probs_c))
        true_counter = Counter(slide_labels[idxs].tolist())
        true_idx = int(true_counter.most_common(1)[0][0])
        true_consistent = int(len(true_counter) == 1)
        case_row: Dict = {
            "source": src,
            "case_id": cid,
            "n_slides": int(len(idxs)),
            "true_label_idx": true_idx,
            "true_label": class_names[true_idx] if true_idx < n_classes else str(true_idx),
            "true_label_consistent": true_consistent,
            "pred_label_idx": pred_idx,
            "pred_label": class_names[pred_idx] if pred_idx < n_classes else str(pred_idx),
            "correct": int(true_idx == pred_idx),
            "slide_ids": "|".join([unique_slides[i] for i in idxs]),
        }
        for k, cname in enumerate(class_names):
            case_row[f"mean_prob_{cname}"] = float(probs_c[k])
        case_rows.append(case_row)
        case_true_list.append(true_idx)
        case_pred_list.append(pred_idx)
        case_probs_list.append(probs_c)
        case_sources_list.append(src)

    case_true = np.array(case_true_list, dtype=int)
    case_pred = np.array(case_pred_list, dtype=int)
    case_probs = np.vstack(case_probs_list) if case_probs_list else np.zeros((0, n_classes))
    case_sources_arr = np.array(case_sources_list)

    case_summary = {
        "n_cases": int(len(case_true)),
        "accuracy": float(accuracy_score(case_true, case_pred)) if len(case_true) else None,
        "balanced_accuracy": float(balanced_accuracy_score(case_true, case_pred)) if len(case_true) else None,
        "f1_macro": float(f1_score(case_true, case_pred, average="macro", zero_division=0)) if len(case_true) else None,
        "f1_weighted": float(f1_score(case_true, case_pred, average="weighted", zero_division=0)) if len(case_true) else None,
        "confusion_matrix": confusion_matrix(case_true, case_pred, labels=list(range(n_classes))).tolist() if len(case_true) else [],
        "classification_report_txt": (classification_report(case_true, case_pred, target_names=class_names,
                                                             digits=3, zero_division=0) if len(case_true) else ""),
    }

    # ── CASE-LEVEL PER-CLASS OvR METRICS (PRIMARY) ───────────────────────────
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
            case_true, case_probs, n_classes, n_bootstrap=n_bootstrap
        )
        for k in range(n_classes):
            cname = class_names[k]
            y_true_bin_c = (case_true == k).astype(int)
            scores_k_c   = case_probs[:, k]
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

    # ── SOURCE-STRATIFIED METRICS ─────────────────────────────────────────────
    stratified_slide = _compute_stratified_ovr_metrics(
        slide_labels, slide_preds, slide_probs, slide_sources, class_names
    )
    stratified_case = _compute_stratified_ovr_metrics(
        case_true, case_pred, case_probs, case_sources_arr, class_names
    )

    # HER2 consistency by source
    her2_consistency = None
    if "HER2" in class_names:
        her2_consistency = {
            "slide_level": {
                src: vals["per_label"]["HER2"]
                for src, vals in stratified_slide.items()
                if "HER2" in vals.get("per_label", {})
            },
            "case_level": {
                src: vals["per_label"]["HER2"]
                for src, vals in stratified_case.items()
                if "HER2" in vals.get("per_label", {})
            },
        }

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    # Tile-level plots
    _plot_multi_curves(roc_curves_tile, "ROC Curves (OvR, Tile)", "roc", "tile",
                       output_dir, label_scores=auc_tile)
    _plot_multi_curves(pr_curves_tile, "PR Curves (OvR, Tile)", "pr", "tile",
                       output_dir, label_scores=ap_tile)
    _plot_multiclass_cm(cm_mc_tile, class_names, "tile", output_dir)

    # Slide-level plots
    _plot_multi_curves(roc_curves_slide, "ROC Curves (OvR, Slide)", "roc", "slide",
                       output_dir, label_scores=auc_slide)
    _plot_multi_curves(pr_curves_slide, "PR Curves (OvR, Slide)", "pr", "slide",
                       output_dir, label_scores=ap_slide)
    _plot_multiclass_cm(cm_mc_slide, class_names, "slide", output_dir)

    # Composite figures
    try:
        _create_composite_figure(
            output_dir, class_names, roc_curves_tile, pr_curves_tile,
            auc_tile, ap_tile, cm_mc_tile,
            aucs_ci=aucs_ci_tile, aps_ci=aps_ci_tile, tag="tile",
        )
        LOGGER.info("Saved composite figure (tile).")
    except Exception as exc:
        LOGGER.warning(f"Composite figure (tile) failed: {exc}")

    try:
        _create_composite_figure(
            output_dir, class_names, roc_curves_slide, pr_curves_slide,
            auc_slide, ap_slide, cm_mc_slide,
            aucs_ci=aucs_ci_slide, aps_ci=aps_ci_slide, tag="slide",
        )
        LOGGER.info("Saved composite figure (slide).")
    except Exception as exc:
        LOGGER.warning(f"Composite figure (slide) failed: {exc}")

    # Case-level plots and composite figure (PRIMARY)
    _plot_multi_curves(case_roc_curves, "ROC Curves (OvR, Case)", "roc", "case",
                       output_dir, label_scores=case_auc_map)
    _plot_multi_curves(case_pr_curves, "PR Curves (OvR, Case)", "pr", "case",
                       output_dir, label_scores=case_ap_map)
    cm_mc_case = confusion_matrix(case_true, case_pred, labels=list(range(n_classes))) if len(case_true) else np.zeros((n_classes, n_classes), dtype=int)
    _plot_multiclass_cm(cm_mc_case, class_names, "case", output_dir)
    try:
        _create_composite_figure(
            output_dir, class_names, case_roc_curves, case_pr_curves,
            case_auc_map, case_ap_map, cm_mc_case,
            aucs_ci=case_aucs_ci_map, aps_ci=case_aps_ci_map, tag="case",
        )
        LOGGER.info("Saved composite figure (case).")
    except Exception as exc:
        LOGGER.warning(f"Composite figure (case) failed: {exc}")

    # ── CSV EXPORTS ────────────────────────────────────────────────────────────
    # Tile-level predictions CSV
    prob_cols = [f"prob_{cname}" for cname in class_names]
    tile_rows = []
    for i in range(len(slide_ids)):
        source, case_id = parse_source_and_case_id(slide_ids[i])
        row: Dict = {
            "tile_idx": i,
            "slide_id": slide_ids[i],
            "source": source,
            "case_id": case_id,
            "true_label_idx": int(y_test[i]),
            "true_label": class_names[int(y_test[i])] if int(y_test[i]) < n_classes else str(int(y_test[i])),
            "pred_label_idx": int(tile_preds[i]),
            "pred_label": class_names[int(tile_preds[i])] if int(tile_preds[i]) < n_classes else str(int(tile_preds[i])),
            "correct": int(y_test[i] == tile_preds[i]),
        }
        for k, cname in enumerate(class_names):
            row[f"prob_{cname}"] = float(tile_probs[i, k])
        tile_rows.append(row)

    tile_csv_fields = ["tile_idx", "slide_id", "source", "case_id",
                       "true_label_idx", "true_label", "pred_label_idx", "pred_label", "correct"] + prob_cols
    tile_csv_path = os.path.join(output_dir, "predictions_tile.csv")
    _write_csv(tile_csv_path, tile_rows, tile_csv_fields)

    # Slide-level predictions CSV
    slide_prob_cols = [f"prob_{cname}" for cname in class_names]
    slide_rows_csv = []
    for i, sid in enumerate(unique_slides):
        source, case_id = parse_source_and_case_id(sid)
        row_s: Dict = {
            "slide_id": sid,
            "source": source,
            "case_id": case_id,
            "true_label_idx": int(slide_labels[i]),
            "true_label": class_names[int(slide_labels[i])] if int(slide_labels[i]) < n_classes else str(int(slide_labels[i])),
            "pred_label_idx": int(slide_preds[i]),
            "pred_label": class_names[int(slide_preds[i])] if int(slide_preds[i]) < n_classes else str(int(slide_preds[i])),
            "correct": int(slide_labels[i] == slide_preds[i]),
        }
        for k, cname in enumerate(class_names):
            row_s[f"prob_{cname}"] = float(slide_probs[i, k])
        slide_rows_csv.append(row_s)

    slide_csv_fields = ["slide_id", "source", "case_id",
                        "true_label_idx", "true_label", "pred_label_idx", "pred_label", "correct"] + slide_prob_cols
    slide_csv_path = os.path.join(output_dir, "predictions_slide.csv")
    _write_csv(slide_csv_path, slide_rows_csv, slide_csv_fields)

    # Case-level predictions CSV
    case_prob_cols = [f"mean_prob_{cname}" for cname in class_names]
    case_csv_fields = ["source", "case_id", "n_slides",
                       "true_label_idx", "true_label", "true_label_consistent",
                       "pred_label_idx", "pred_label", "correct", "slide_ids"] + case_prob_cols
    case_csv_path = os.path.join(output_dir, "predictions_case.csv")
    _write_csv(case_csv_path, case_rows, case_csv_fields)

    # ── MANUSCRIPT TABLE CSV (unified: Case=PRIMARY, Slide=secondary, Tile=supplementary) ──
    def _mci(key, m_dict, ci_dict):
        v = m_dict.get(key)
        ci = ci_dict.get(key, [None, None])
        if v is None:
            return "NA", "NA"
        vstr  = f"{float(v):.3f}"
        cistr = f"[{ci[0]:.3f}–{ci[1]:.3f}]" if (ci and ci[0] is not None) else "NA"
        return vstr, cistr

    def _row(m: Dict, level: str, label: str) -> Dict:
        def _f(k):  return f"{float(m[k]):.3f}" if m.get(k) is not None else "NA"
        def _ci(k): ci = m.get(f"{k}_ci", [None, None]); return f"[{ci[0]:.3f}–{ci[1]:.3f}]" if (ci and ci[0] is not None) else "NA"
        return {
            "Level": level, "Class": label,
            "N": m.get("support_total", "-"),
            "F1": _f("f1"), "F1_95CI": _ci("f1"),
            "Precision": _f("precision"), "Precision_95CI": _ci("precision"),
            "Recall": _f("recall"), "Recall_95CI": _ci("recall"),
            "AUROC": _f("auroc"), "AUROC_95CI_DeLong": _ci("auroc"),
            "AP": _f("ap"), "AP_95CI": _ci("ap"),
        }

    def _macro_row(level: str, mac_dict: Dict, ci_dict: Dict) -> Dict:
        f1_v, f1_c   = _mci("f1",       mac_dict, ci_dict)
        p_v,  p_c    = _mci("precision", mac_dict, ci_dict)
        r_v,  r_c    = _mci("recall",    mac_dict, ci_dict)
        auc_v,auc_c  = _mci("auroc",     mac_dict, ci_dict)
        ap_v, ap_c   = _mci("ap",        mac_dict, ci_dict)
        return {
            "Level": level, "Class": "Macro-average", "N": "-",
            "F1": f1_v, "F1_95CI": f1_c,
            "Precision": p_v, "Precision_95CI": p_c,
            "Recall": r_v, "Recall_95CI": r_c,
            "AUROC": auc_v, "AUROC_95CI_DeLong": auc_c,
            "AP": ap_v, "AP_95CI": ap_c,
        }

    table_rows_all = []
    for cname in class_names:
        table_rows_all.append(_row(case_per_class.get(cname, {}), "Case", cname))
    table_rows_all.append(_macro_row("Case", case_avg_macro, case_ci_macro))
    for cname in class_names:
        table_rows_all.append(_row(per_class_slide.get(cname, {}), "Slide", cname))
    table_rows_all.append(_macro_row("Slide", avg_slide_macro, macro_ci_slide))
    for cname in class_names:
        table_rows_all.append(_row(per_class_tile.get(cname, {}), "Tile", cname))
    table_rows_all.append(_macro_row("Tile", avg_tile_macro, macro_ci_tile))

    _write_csv(
        os.path.join(output_dir, "manuscript_table_metrics_ci.csv"),
        table_rows_all,
        ["Level", "Class", "N", "F1", "F1_95CI", "Precision", "Precision_95CI",
         "Recall", "Recall_95CI", "AUROC", "AUROC_95CI_DeLong", "AP", "AP_95CI"],
    )
    LOGGER.info("Saved unified manuscript table CSV.")

    # ── TXT SUMMARY ─────────────────────────────────────────────────────────────
    def fmt(x):
        if x is None:
            return "NA"
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.4f}"
        return str(x)

    def fmt_ci(m: Dict, key: str, decimals: int = 3) -> str:
        v = m.get(key)
        ci = m.get(f"{key}_ci", [None, None])
        if v is None:
            return "NA"
        s = f"{float(v):.{decimals}f}"
        if ci and ci[0] is not None:
            s += f" [{float(ci[0]):.{decimals}f}–{float(ci[1]):.{decimals}f}]"
        return s

    def _fmt_macro(avg_dict: Dict, ci_dict: Dict) -> str:
        parts = []
        for key in ["f1", "precision", "recall", "balanced_accuracy", "auroc", "ap"]:
            v = avg_dict.get(key)
            ci = ci_dict.get(key, [None, None])
            if v is None:
                continue
            s = f"{key}={float(v):.3f}"
            if ci and ci[0] is not None:
                s += f" [{float(ci[0]):.3f}–{float(ci[1]):.3f}]"
            parts.append(s)
        return ", ".join(parts)

    txt_path = os.path.join(output_dir, "metrics_test.txt")
    with open(txt_path, "w") as f:
        f.write(f"Models dir : {models_dir}\n")
        f.write(f"Data PKL   : {data_source}\n")
        f.write(f"Classes ({n_classes}): {', '.join(class_names)}\n")
        f.write(f"Bootstrap iterations (CIs): {n_bootstrap}\n\n")

        # ── PRIMARY: Case/patient level
        f.write("=" * 70 + "\n")
        f.write("  PRIMARY RESULTS: CASE/PATIENT LEVEL\n")
        f.write("  (unit of analysis = patient; CIs bootstrapped over cases)\n")
        f.write("=" * 70 + "\n\n")

        if case_per_class:
            f.write(f"Case-level per-class OvR metrics with 95% CIs (n_cases={case_summary['n_cases']}):\n")
            for cname in class_names:
                m = case_per_class.get(cname, {})
                f.write(
                    f"- {cname} (n_pos={m.get('support_pos','?')}/{m.get('support_total','?')}): "
                    f"F1={fmt_ci(m,'f1')}, Prec={fmt_ci(m,'precision')}, Rec={fmt_ci(m,'recall')}, "
                    f"BalAcc={fmt_ci(m,'balanced_accuracy')}, "
                    f"AUROC={fmt_ci(m,'auroc')} [DeLong], AP={fmt_ci(m,'ap')} [bootstrap]\n"
                )
            f.write("\nCase-level macro-averaged metrics with 95% CIs (bootstrapped over cases):\n")
            f.write(f"  {_fmt_macro(case_avg_macro, case_ci_macro)}\n\n")
        else:
            f.write("(No cases available for case-level metrics.)\n\n")

        f.write("Case-level multiclass confusion matrix and classification report:\n")
        f.write(f"  n_cases={case_summary['n_cases']}, "
                f"Acc={fmt(case_summary['accuracy'])}, "
                f"BalAcc={fmt(case_summary['balanced_accuracy'])}, "
                f"F1_macro={fmt(case_summary['f1_macro'])}, "
                f"F1_weighted={fmt(case_summary['f1_weighted'])}\n")
        f.write(case_summary["classification_report_txt"] + "\n")

        # ── SECONDARY: Slide level
        f.write("=" * 70 + "\n")
        f.write("  SECONDARY RESULTS: SLIDE/WSI LEVEL (supplementary)\n")
        f.write("  (unit = slide; CIs bootstrapped over slides)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total test slides : {len(unique_slides)}\n\n")
        f.write("Per-class binary metrics with 95% CIs (slide level):\n")
        for cname in class_names:
            m = per_class_slide.get(cname, {})
            f.write(
                f"- {cname} (n_pos={m.get('support_pos','?')}/{m.get('support_total','?')}): "
                f"F1={fmt_ci(m,'f1')}, Prec={fmt_ci(m,'precision')}, Rec={fmt_ci(m,'recall')}, "
                f"BalAcc={fmt_ci(m,'balanced_accuracy')}, "
                f"AUROC={fmt_ci(m,'auroc')} [DeLong], AP={fmt_ci(m,'ap')}\n"
            )
        f.write("\nMacro-averaged metrics with 95% CIs (slide level):\n")
        f.write(f"  {_fmt_macro(avg_slide_macro, macro_ci_slide)}\n\n")
        f.write("Multiclass confusion matrix (slide):\n")
        f.write(report_slide_txt + "\n")
        f.write("Mean binary confusion matrix (slide):\n")
        f.write(np.array2string(mean_cm_slide, formatter={"float_kind": lambda v: f"{v:.3f}"}))
        f.write("\n\n")

        # ── TERTIARY: Tile level
        f.write("=" * 70 + "\n")
        f.write("  TERTIARY RESULTS: TILE LEVEL (supplementary)\n")
        f.write("  (tiles within a slide are NOT independent observations)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total test tiles : {X_test.shape[0]}\n\n")
        f.write("Per-class binary metrics with 95% CIs (tile level):\n")
        for cname in class_names:
            m = per_class_tile.get(cname, {})
            f.write(
                f"- {cname} (n_pos={m.get('support_pos','?')}/{m.get('support_total','?')}): "
                f"F1={fmt_ci(m,'f1')}, Prec={fmt_ci(m,'precision')}, Rec={fmt_ci(m,'recall')}, "
                f"BalAcc={fmt_ci(m,'balanced_accuracy')}, "
                f"AUROC={fmt_ci(m,'auroc')} [DeLong], AP={fmt_ci(m,'ap')}\n"
            )
        f.write("\nMacro-averaged metrics with 95% CIs (tile level):\n")
        f.write(f"  {_fmt_macro(avg_tile_macro, macro_ci_tile)}\n\n")
        f.write("Multiclass confusion matrix (tile):\n")
        f.write(report_tile_txt + "\n")
        f.write("Mean binary confusion matrix (tile):\n")
        f.write(np.array2string(mean_cm_tile, formatter={"float_kind": lambda v: f"{v:.3f}"}))
        f.write("\n\n")

        # ── AUXILIARY: stratification & exports
        f.write("=" * 70 + "\n")
        f.write("  AUXILIARY: SOURCE STRATIFICATION & EXPORTS\n")
        f.write("=" * 70 + "\n")
        for src in sorted(stratified_slide.keys()):
            vals = stratified_slide[src]
            f.write(f"\n[{src}] n_slides={vals['n_samples']}\n")
            for cname in class_names:
                m = vals["per_label"].get(cname, {})
                f.write(
                    f"  {cname}: support={m.get('support_pos')}/{m.get('support_total')}, "
                    f"Prec={fmt(m.get('precision'))}, Rec={fmt(m.get('recall'))}, "
                    f"F1={fmt(m.get('f1'))}, BalAcc={fmt(m.get('balanced_accuracy'))}, "
                    f"AUROC={fmt(m.get('auroc'))}, AP={fmt(m.get('ap'))}\n"
                )

        if her2_consistency is not None:
            f.write("\nHER2 source-stratified consistency:\n")
            f.write("- Slide-level:\n")
            for src in sorted(her2_consistency["slide_level"].keys()):
                m = her2_consistency["slide_level"][src]
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

        f.write("\nPrediction exports:\n")
        f.write(f"- Tile-level CSV  : {tile_csv_path}\n")
        f.write(f"- Slide-level CSV : {slide_csv_path}\n")
        f.write(f"- Case-level CSV  : {case_csv_path}\n")

    LOGGER.info(f"Saved TXT summary: {txt_path}")

    # ── JSON SUMMARY ────────────────────────────────────────────────────────────
    def _sanitize(d):
        """Drop internal curve arrays; convert numpy scalars."""
        if isinstance(d, dict):
            return {k: _sanitize(v) for k, v in d.items() if not str(k).startswith("_")}
        if isinstance(d, np.ndarray):
            return d.tolist()
        if isinstance(d, (np.generic,)):
            return d.item()
        if isinstance(d, list):
            return [_sanitize(x) for x in d]
        return d

    results_json = {
        "config": {
            "models_dir": models_dir,
            "data_pkl": data_source,
            "n_classes": n_classes,
            "classes": class_names,
            "n_bootstrap": n_bootstrap,
        },
        "tile": {
            "per_class": _sanitize(per_class_tile),
            "averages": {
                "macro": avg_tile_macro,
                "macro_ci_bootstrap": macro_ci_tile,
                "weighted": avg_tile_weighted,
            },
            "multiclass": {
                "confusion_matrix": cm_mc_tile.tolist(),
                "classification_report_txt": report_tile_txt,
            },
            "mean_binary_confusion": mean_cm_tile.tolist(),
        },
        "slide": {
            "note": "SECONDARY — slide/WSI-level metrics; CIs bootstrapped over slides (not independent when multiple slides per patient)",
            "per_class": _sanitize(per_class_slide),
            "averages": {
                "macro": avg_slide_macro,
                "macro_ci_bootstrap": macro_ci_slide,
                "weighted": avg_slide_weighted,
            },
            "multiclass": {
                "confusion_matrix": cm_mc_slide.tolist(),
                "classification_report_txt": report_slide_txt,
            },
            "mean_binary_confusion": mean_cm_slide.tolist(),
        },
        "case": {
            "note": "PRIMARY — case/patient-level metrics; CIs bootstrapped over cases (statistically independent unit)",
            "summary": case_summary,
            "per_class": _sanitize(case_per_class) if case_per_class else {},
            "macro": case_avg_macro if case_per_class else {},
            "weighted": case_avg_weighted if case_per_class else {},
            "macro_ci_bootstrap_case": case_ci_macro if case_per_class else {},
        },
        "stratified_by_source": {
            "slide_level": _sanitize(stratified_slide),
            "case_level": _sanitize(stratified_case),
        },
        "her2_consistency_by_source": _sanitize(her2_consistency),
        "prediction_exports": {
            "tile_csv": tile_csv_path,
            "slide_csv": slide_csv_path,
            "case_csv": case_csv_path,
        },
        "data_audit": {
            "n_test_tiles": int(X_test.shape[0]),
            "n_test_slides": int(len(unique_slides)),
            "n_test_cases": int(len(case_rows)),
        },
    }

    json_path = os.path.join(output_dir, "metrics_test.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    LOGGER.info(f"Saved JSON summary: {json_path}")
    LOGGER.info(f"Evaluation complete. All artifacts in: {output_dir}")
    return 0


# -------------------- CLI --------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Evaluate OvR Logistic Regression tile-level models on held-out test set."
    )
    ap.add_argument("--models_dir", required=True, type=str,
                    help="LR training results directory (contains trained_models/ and enhanced_all_results.json)")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Where to save evaluation artifacts (default: models_dir/eval_test)")
    ap.add_argument("--n_bootstrap", type=int, default=2000,
                    help="Bootstrap iterations for 95%% CIs (default 2000; use 500 for fast iteration)")
    data_grp = ap.add_mutually_exclusive_group(required=True)
    data_grp.add_argument("--mil_pkl", type=str,
                          help="MIL PKL (same file as Attention_based_MIL.py) — preferred")
    data_grp.add_argument("--split_pkl", type=str,
                          help="Path to split_seed_{seed}.pkl with test_dict_paths/test_labels")
    args = ap.parse_args()
    raise SystemExit(evaluate(
        args.models_dir, args.output_dir, args.n_bootstrap,
        split_pkl=args.split_pkl, mil_pkl=args.mil_pkl,
    ))


if __name__ == "__main__":
    main()
