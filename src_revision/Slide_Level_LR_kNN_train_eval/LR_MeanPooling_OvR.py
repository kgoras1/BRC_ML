#!/usr/bin/env python3
"""
LR_MeanPooling_OvR.py

One-vs-Rest Logistic Regression on mean-pooled slide-level features.

Strategy (mirrors manuscript description):
- Slide-level embeddings obtained by mean-pooling tile features from MIL bags.
- Per-class OvR LR classifiers, hyperparameters tuned via GridSearchCV (5-fold CV)
  on the 80% training split; balanced accuracy is the scoring metric.
- Patient-level 80/10/10 train/val/cal split — no patient appears in more than one
  partition, preventing any form of data leakage.
- Optional temperature scaling: grid-searches T on the held-out calibration split;
  calibrated probabilities are L1-normalised across classes at inference.
- Optional class-imbalance strategies (all applied BEFORE CV fitting, on the train
  split only, never touching val/cal): random oversampling, random undersampling, SMOTE.
- Each imbalance strategy is supported with and without probability calibration.

Evaluation (identical to Evaluate_for_revision.py):
- Per-class binary metrics with 95% CIs (DeLong for AUROC, bootstrap for others).
- Composite figure: ROC overlay, PR overlay, multiclass confusion matrix, legend.
- WSI-level and case-level CSV exports.
- Source-stratified OvR metrics.
- Manuscript-table CSV with CIs.

Inputs:
  --mil_pkl       PKL with train/test splits in any supported format (same as
                  Attention_based_MIL.py; bags are mean-pooled internally).
  --output_dir    Where to write models and evaluation artefacts.

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
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
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


# ─────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────
def _get_logger(name: str = "lr_ovr") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(ch)
    return logger

LOGGER = _get_logger()


# ─────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────
#  PKL loader (identical schema support as Attention_based_MIL.py)
# ─────────────────────────────────────────────────────────────
def load_train_test_from_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    class_names = None
    if "class_names" in data and isinstance(data["class_names"], (list, tuple)):
        class_names = list(data["class_names"])
    elif "label_encoder" in data and hasattr(data["label_encoder"], "classes_"):
        class_names = list(data["label_encoder"].classes_)

    ids_train = ids_test = None

    # ── Format: pre-computed mean-pooled features (one vector per slide) ──────
    # Keys: train_features [N,D], test_features [M,D], train_labels/test_labels
    # (labels may be string class names; class_names list drives the mapping)
    if "train_features" in data and "train_labels" in data:
        raw_feat_tr = np.asarray(data["train_features"], dtype=np.float32)
        raw_lbl_tr  = data["train_labels"]
        raw_feat_te = data.get("test_features")
        raw_lbl_te  = data.get("test_labels")

        # If class_names not yet resolved, derive from sorted unique labels
        if class_names is None:
            class_names = sorted(set(str(l) for l in raw_lbl_tr))

        # Map string labels → integer indices via class_names order
        def _to_int(labels):
            if not labels:
                return np.array([], dtype=int)
            if isinstance(labels[0], str):
                lbl2idx = {c: i for i, c in enumerate(class_names)}
                return np.array([lbl2idx[str(l)] for l in labels], dtype=int)
            return np.asarray(labels, dtype=int)

        y_train = _to_int(raw_lbl_tr)
        y_test  = _to_int(raw_lbl_te) if raw_lbl_te else None

        # Return as list of 1-D vectors: mean_pool treats ndim==1 as pass-through,
        # and all split helpers work on plain Python lists.
        X_train = [raw_feat_tr[i] for i in range(len(raw_feat_tr))]
        if raw_feat_te is not None:
            raw_feat_te = np.asarray(raw_feat_te, dtype=np.float32)
            X_test = [raw_feat_te[i] for i in range(len(raw_feat_te))]
        else:
            X_test = None

        ids_train = data.get("train_ids") or data.get("ids_train")
        ids_test  = data.get("test_ids")  or data.get("ids_test")
        return X_train, y_train, X_test, y_test, class_names, ids_train, ids_test

    if "train_bags" in data and "train_labels" in data:
        X_train = data["train_bags"]
        y_train = np.asarray(data["train_labels"], dtype=int)
        X_test  = data.get("test_bags")
        y_test  = np.asarray(data["test_labels"], dtype=int) if "test_labels" in data else None
        ids_train = data.get("train_ids") or data.get("ids_train")
        ids_test  = data.get("test_ids")  or data.get("ids_test")
        return X_train, y_train, X_test, y_test, class_names, ids_train, ids_test

    if "X_train" in data and "y_train" in data:
        X_train = data["X_train"]
        y_train = np.asarray(data["y_train"], dtype=int)
        X_test  = data.get("X_test")
        y_test  = np.asarray(data["y_test"], dtype=int) if "y_test" in data else None
        ids_train = data.get("train_ids") or data.get("ids_train")
        ids_test  = data.get("test_ids")  or data.get("ids_test")
        return X_train, y_train, X_test, y_test, class_names, ids_train, ids_test

    if "train" in data and "test" in data and isinstance(data["train"], dict):
        X_train = data["train"]["bags"]
        y_train = np.asarray(data["train"]["labels"], dtype=int)
        X_test  = data["test"]["bags"]
        y_test  = np.asarray(data["test"]["labels"], dtype=int)
        ids_train = data["train"].get("ids")
        ids_test  = data["test"].get("ids")
        return X_train, y_train, X_test, y_test, class_names, ids_train, ids_test

    raise KeyError("Unrecognised PKL format.")


# ─────────────────────────────────────────────────────────────
#  Mean-pool bags → slide-level feature matrix
# ─────────────────────────────────────────────────────────────
def mean_pool(bags) -> np.ndarray:
    """Mean-pool a list of bags ([Ti, D] arrays) to shape [N, D]."""
    pooled = []
    for bag in bags:
        arr = np.asarray(bag, dtype=np.float32)
        if arr.ndim == 1:
            pooled.append(arr)
        else:
            pooled.append(arr.mean(axis=0))
    return np.vstack(pooled)


# ─────────────────────────────────────────────────────────────
#  Patient ID extraction (mirrors Attention_based_MIL.py)
# ─────────────────────────────────────────────────────────────
def get_patient_id(slide_id: str) -> str:
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


# ─────────────────────────────────────────────────────────────
#  Patient-level 80/10/10 split (identical logic to MIL script)
# ─────────────────────────────────────────────────────────────
def _slide_level_split(bags, y_bin, val_size=0.0, cal_size=0.20, seed=42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_bin == 1)[0]
    neg_idx = np.where(y_bin == 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos < 2 or n_neg < 2:
        X_tr, X_ca, y_tr, y_ca = train_test_split(
            bags, y_bin, test_size=cal_size, random_state=seed,
            stratify=y_bin if len(np.unique(y_bin)) > 1 else None
        )
        return X_tr, [], X_ca, y_tr, np.array([], dtype=int), y_ca

    # val slots: 0 when val_size=0 (no val split)
    n_pos_val = (max(1, int(round(n_pos * val_size))) if val_size > 0 else 0)
    n_pos_cal = max(1, int(round(n_pos * cal_size)))
    if n_pos_val + n_pos_cal >= n_pos:
        n_pos_cal = max(0, n_pos - n_pos_val - 1)

    n_neg_val = (max(1, int(round(n_neg * val_size))) if val_size > 0 else 0)
    n_neg_cal = max(1, int(round(n_neg * cal_size)))
    if n_neg_val + n_neg_cal >= n_neg:
        n_neg_cal = max(0, n_neg - n_neg_val - 1)

    pos_val   = pos_idx[:n_pos_val]
    pos_cal   = pos_idx[n_pos_val:n_pos_val + n_pos_cal]
    pos_train = pos_idx[n_pos_val + n_pos_cal:]
    neg_val   = neg_idx[:n_neg_val]
    neg_cal   = neg_idx[n_neg_val:n_neg_val + n_neg_cal]
    neg_train = neg_idx[n_neg_val + n_neg_cal:]

    idx_train = np.concatenate([pos_train, neg_train])
    idx_val   = np.concatenate([pos_val,   neg_val])
    idx_cal   = np.concatenate([pos_cal,   neg_cal])
    rng.shuffle(idx_train)
    if len(idx_val) > 0: rng.shuffle(idx_val)
    if len(idx_cal) > 0: rng.shuffle(idx_cal)

    X_train = [bags[i] for i in idx_train]
    X_val   = [bags[i] for i in idx_val]
    X_cal   = [bags[i] for i in idx_cal]
    y_train = y_bin[idx_train]
    y_val   = y_bin[idx_val] if len(idx_val) > 0 else np.array([], dtype=int)
    y_cal   = y_bin[idx_cal]

    # Degenerate check: only validate val quality when val is non-empty
    if (len(y_val) > 0 and len(np.unique(y_val)) < 2) or len(np.unique(y_cal)) < 2:
        X_tr, X_ca, y_tr, y_ca = train_test_split(
            bags, y_bin, test_size=cal_size, random_state=seed, stratify=y_bin
        )
        return X_tr, [], X_ca, y_tr, np.array([], dtype=int), y_ca

    return X_train, X_val, X_cal, y_train, y_val, y_cal


def patient_stratified_train_val_cal_split(bags, y_bin, slide_ids,
                                           val_size=0.0, cal_size=0.20, seed=42):
    y_bin = np.asarray(y_bin, dtype=int)

    if slide_ids is None:
        X_tr, X_va, X_ca, y_tr, y_va, y_ca = _slide_level_split(bags, y_bin, val_size, cal_size, seed)
        return X_tr, X_va, X_ca, y_tr, y_va, y_ca, None, None, None

    patient_slides = defaultdict(list)
    for idx, sid in enumerate(slide_ids):
        patient_slides[get_patient_id(str(sid))].append(idx)

    patients = sorted(patient_slides.keys())
    patient_label = {
        pid: Counter(y_bin[idxs].tolist()).most_common(1)[0][0]
        for pid, idxs in patient_slides.items()
    }
    pat_labels = np.array([patient_label[p] for p in patients])
    n_pos = int((pat_labels == 1).sum())
    n_neg = int((pat_labels == 0).sum())

    if n_pos < 3 or n_neg < 3:
        X_tr, X_va, X_ca, y_tr, y_va, y_ca = _slide_level_split(bags, y_bin, val_size, cal_size, seed)
        return X_tr, X_va, X_ca, y_tr, y_va, y_ca, None, None, None

    # Optional val split — skipped entirely when val_size=0
    if val_size > 0:
        try:
            train_cal_pats, val_pats = train_test_split(
                patients, test_size=val_size, random_state=seed, stratify=pat_labels
            )
        except ValueError:
            X_tr, X_va, X_ca, y_tr, y_va, y_ca = _slide_level_split(bags, y_bin, val_size, cal_size, seed)
            return X_tr, X_va, X_ca, y_tr, y_va, y_ca, None, None, None
        tc_labels    = np.array([patient_label[p] for p in train_cal_pats])
        cal_fraction = cal_size / (1.0 - val_size)
    else:
        train_cal_pats = patients
        val_pats       = []
        tc_labels      = pat_labels
        cal_fraction   = cal_size

    try:
        train_pats, cal_pats = train_test_split(
            train_cal_pats, test_size=cal_fraction, random_state=seed, stratify=tc_labels
        )
    except ValueError:
        train_pats = train_cal_pats
        cal_pats   = []

    train_idx = [i for p in train_pats for i in patient_slides[p]]
    val_idx   = [i for p in val_pats   for i in patient_slides[p]]
    cal_idx   = [i for p in cal_pats   for i in patient_slides[p]]

    X_train = [bags[i] for i in train_idx]
    X_val   = [bags[i] for i in val_idx]
    X_cal   = [bags[i] for i in cal_idx] if cal_idx else []
    y_train = y_bin[np.array(train_idx, dtype=int)]
    y_val   = y_bin[np.array(val_idx,   dtype=int)] if val_idx else np.array([], dtype=int)
    y_cal   = y_bin[np.array(cal_idx,   dtype=int)] if cal_idx else np.array([], dtype=int)

    # Degenerate check: only validate val quality when val_size > 0
    if (val_pats and len(np.unique(y_val)) < 2) or (len(y_cal) > 0 and len(np.unique(y_cal)) < 2):
        X_tr, X_va, X_ca, y_tr, y_va, y_ca = _slide_level_split(bags, y_bin, val_size, cal_size, seed)
        return X_tr, X_va, X_ca, y_tr, y_va, y_ca, None, None, None

    return (X_train, X_val, X_cal, y_train, y_val, y_cal,
            len(train_pats), len(val_pats), len(cal_pats))


# ─────────────────────────────────────────────────────────────
#  Temperature scaling (grid search on calibration set)
# ─────────────────────────────────────────────────────────────
def find_temperature(logits: np.ndarray, y_true: np.ndarray,
                     grid: np.ndarray = None) -> float:
    """
    Find optimal temperature T minimising NLL on calibration set.
    logits: [N, 2] actual logits — stack([-decision, decision], axis=1).
    """
    if grid is None:
        grid = np.linspace(0.05, 10.0, 400)
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        scaled = logits / T
        # log-sum-exp normalise
        mx = scaled.max(axis=1, keepdims=True)
        lse = np.log(np.exp(scaled - mx).sum(axis=1, keepdims=True)) + mx
        log_p = scaled - lse
        nll = -log_p[np.arange(len(y_true)), y_true].mean()
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T


def calibrate_lr_models(classifiers: list, X_cal_list: list,
                        y_cal_list: list) -> list:
    """Temperature-scale each binary OvR LR classifier on its cal split."""
    temperatures = []
    for k, (clf, X_cal, y_cal) in enumerate(zip(classifiers, X_cal_list, y_cal_list)):
        if clf is None or len(y_cal) == 0 or len(np.unique(y_cal)) < 2:
            LOGGER.warning(f"  Classifier {k}: empty/degenerate cal set; T=1.0")
            temperatures.append(1.0)
            continue
        raw = clf.decision_function(X_cal)          # [N,]
        logits = np.stack([-raw, raw], axis=1)       # [N, 2]
        T = find_temperature(logits, y_cal)
        temperatures.append(T)
        LOGGER.info(f"  Classifier {k}: T={T:.4f}")
    return temperatures


# ─────────────────────────────────────────────────────────────
#  Apply imbalance strategy
# ─────────────────────────────────────────────────────────────
def apply_imbalance_strategy(X_train: np.ndarray, y_train: np.ndarray,
                             strategy: str, seed: int):
    """
    Apply one of: 'none', 'oversample', 'undersample', 'smote'.
    Returns resampled (X, y).
    """
    if strategy == "none":
        return X_train, y_train

    if strategy == "oversample":
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=seed)
        return ros.fit_resample(X_train, y_train)

    if strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=seed)
        return rus.fit_resample(X_train, y_train)

    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        n_pos = int((y_train == 1).sum())
        k_neighbors = min(5, n_pos - 1)
        if k_neighbors < 1:
            LOGGER.warning("  Too few positives for SMOTE; falling back to oversample.")
            from imblearn.over_sampling import RandomOverSampler
            return RandomOverSampler(random_state=seed).fit_resample(X_train, y_train)
        sm = SMOTE(random_state=seed, k_neighbors=k_neighbors)
        return sm.fit_resample(X_train, y_train)

    raise ValueError(f"Unknown imbalance strategy: {strategy}")


# ─────────────────────────────────────────────────────────────
#  Evaluation helpers (identical to Evaluate_for_revision.py)
# ─────────────────────────────────────────────────────────────
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def delong_auroc_ci(y_true, scores, alpha=0.05):
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
    z = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))
    return auc, float(np.clip(auc - z * se, 0.0, 1.0)), float(np.clip(auc + z * se, 0.0, 1.0))


def bootstrap_metric_ci(y_true, y_pred, y_prob, n_bootstrap=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted",
            "precision", "recall", "auroc", "ap"]
    boots = {k: [] for k in keys}
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
        k: ([float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))]
            if len(v) >= 20 else [None, None])
        for k, v in boots.items()
    }


def bootstrap_macro_ci(all_labels, all_probs, n_classes, n_bootstrap=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(all_labels)
    keys = ["f1", "precision", "recall", "balanced_accuracy", "auroc", "ap"]
    boots = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = all_labels[idx]
        yp_s = all_probs[idx]
        yp_mc = np.argmax(yp_s, axis=1)   # multiclass argmax for F1/Prec/Rec/BalAcc
        per_k = {k: [] for k in keys}
        for k in range(n_classes):
            ytb = (yt == k).astype(int)
            if len(np.unique(ytb)) < 2:
                continue
            sk = yp_s[:, k]
            ypb = (yp_mc == k).astype(int)
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
        k: ([float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))]
            if len(v) >= 20 else [None, None])
        for k, v in boots.items()
    }


def bootstrap_slide_level_macro_ci(y_true_mc, y_pred_mc, n_bootstrap=2000, alpha=0.05, seed=42):
    """
    Bootstrap 95% CI for slide-level macro-F1, weighted-F1, balanced accuracy, and
    accuracy using the final argmax multiclass predictions (not per-class OvR thresholds).
    Each bootstrap iteration resamples slides with replacement and computes
    f1_score(..., average='macro') directly, so the CI captures sampling uncertainty
    in the multiclass macro-F1 without any distributional assumption.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true_mc)
    keys = ["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy"]
    boots = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true_mc[idx], y_pred_mc[idx]
        boots["f1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
        boots["f1_weighted"].append(f1_score(yt, yp, average="weighted", zero_division=0))
        boots["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        boots["accuracy"].append(accuracy_score(yt, yp))
    lo_p, hi_p = alpha / 2 * 100, (1 - alpha / 2) * 100
    return {
        k: ([float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))]
            if len(v) >= 20 else [None, None])
        for k, v in boots.items()
    }


def bootstrap_case_ci(case_true, case_scores, n_classes, n_bootstrap=2000, alpha=0.05, seed=42):
    """Bootstrap 95% CIs at the case/patient level by resampling cases (not WSIs)."""
    rng = np.random.default_rng(seed)
    n = len(case_true)
    metric_keys = ["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted",
                   "precision", "recall", "auroc", "ap"]
    macro_keys  = ["f1", "f1_macro", "f1_weighted", "precision", "recall", "balanced_accuracy", "auroc", "ap"]

    per_class_boots = {k: {m: [] for m in metric_keys} for k in range(n_classes)}
    macro_boots = {m: [] for m in macro_keys}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = case_true[idx]
        ys = case_scores[idx]
        per_iter = {m: [] for m in macro_keys}
        yp_mc = np.argmax(ys, axis=1)   # argmax-based, mirrors WSI-level and multiclass CM

        for k in range(n_classes):
            ytb = (yt == k).astype(int)
            if len(np.unique(ytb)) < 2:
                continue
            sk = ys[:, k]
            ypb = (yp_mc == k).astype(int)

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
    per_class_ci = {
        k: {
            m: ([float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))]
                if len(v) >= 20 else [None, None])
            for m, v in per_class_boots[k].items()
        }
        for k in range(n_classes)
    }
    macro_ci = {
        m: ([float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p))]
            if len(v) >= 20 else [None, None])
        for m, v in macro_boots.items()
    }
    return per_class_ci, macro_ci


def bootstrap_roc_band(y_true, scores, n_bootstrap=1000, alpha=0.05, n_grid=100, seed=42):
    rng = np.random.default_rng(seed)
    fpr_grid = np.linspace(0, 1, n_grid)
    tpr_boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
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


def bootstrap_pr_band(y_true, scores, n_bootstrap=1000, alpha=0.05, n_grid=100, seed=42):
    rng = np.random.default_rng(seed)
    rec_grid = np.linspace(0, 1, n_grid)
    pre_boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
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


def compute_binary_metrics(y_true, y_pred, y_prob_pos, n_bootstrap=2000):
    m = {}
    m["accuracy"]          = float(accuracy_score(y_true, y_pred))
    m["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    m["precision"]         = float(precision_score(y_true, y_pred, zero_division=0))
    m["recall"]            = float(recall_score(y_true, y_pred, zero_division=0))
    m["f1"]                = float(f1_score(y_true, y_pred, zero_division=0))
    m["f1_macro"]          = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    m["f1_weighted"]       = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    m["support_pos"]       = int((y_true == 1).sum())
    m["support_total"]     = int(len(y_true))

    if len(np.unique(y_true)) > 1:
        try:
            m["auroc"] = float(roc_auc_score(y_true, y_prob_pos))
        except Exception:
            m["auroc"] = None
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob_pos)
        m["ap"] = float(average_precision_score(y_true, y_prob_pos))
        m["_pr_curve"]  = (pr_rec, pr_prec)
        fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
        m["_roc_curve"] = (fpr, tpr)
        _, ci_lo, ci_hi = delong_auroc_ci(y_true, y_prob_pos)
        m["auroc_ci"] = [ci_lo, ci_hi]
    else:
        m["auroc"] = None
        m["ap"]    = None
        m["auroc_ci"] = [None, None]

    ci = bootstrap_metric_ci(y_true, y_pred, y_prob_pos, n_bootstrap=n_bootstrap)
    for mn, pair in ci.items():
        m[f"{mn}_ci"] = pair
    return m


def _pr_for_plot(y_true, scores):
    precision, recall, _ = precision_recall_curve(
        np.asarray(y_true).astype(int), scores
    )
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


def _average_per_class(per_class_dict, class_names):
    keys = ["accuracy", "balanced_accuracy", "precision", "recall",
            "f1", "f1_macro", "f1_weighted", "auroc", "ap"]
    rows = [per_class_dict.get(c, {}) for c in class_names]
    pos_weights = np.array([r.get("support_pos", 0) or 0 for r in rows], dtype=float)
    macro, weighted = {}, {}
    for k in keys:
        vals = np.array([
            float(r[k]) if r.get(k) is not None else np.nan
            for r in rows
        ], dtype=float)
        macro[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
        if pos_weights.sum() > 0 and np.isfinite(vals).any():
            mask = np.isfinite(vals)
            if mask.any():
                w = pos_weights[mask]; v = vals[mask]
                w = w / (w.sum() or 1.0)
                weighted[k] = float(np.sum(v * w))
            else:
                weighted[k] = None
        else:
            weighted[k] = macro[k]
    return macro, weighted


def parse_source_and_case_id(sample_id: str):
    sid = str(sample_id)
    if sid.startswith("TCGA-"):
        toks = sid.split("-")
        return "TCGA", ("-".join(toks[:3]) if len(toks) >= 3 else sid)
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


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_stratified_ovr_metrics(y_true_mc, y_pred_mc, scores_mc, sources, class_names):
    out = {}
    for src in sorted(set(sources.tolist())):
        idx = np.where(sources == src)[0]
        yt = y_true_mc[idx]; yp = y_pred_mc[idx]; sc = scores_mc[idx, :]
        src_dict = {"n_samples": int(len(idx)), "per_label": {}}
        for k, cname in enumerate(class_names):
            y_true_bin = (yt == k).astype(int)
            y_pred_bin = (yp == k).astype(int)
            mm = {
                "support_pos": int((y_true_bin == 1).sum()),
                "support_total": int(len(y_true_bin)),
                "accuracy":  float(accuracy_score(y_true_bin, y_pred_bin)),
                "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
                "recall":    float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
                "f1":        float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
            }
            if len(np.unique(y_true_bin)) > 1:
                mm["balanced_accuracy"] = float(balanced_accuracy_score(y_true_bin, y_pred_bin))
                try:
                    mm["auroc"] = float(roc_auc_score(y_true_bin, sc[:, k]))
                except Exception:
                    mm["auroc"] = None
                try:
                    mm["ap"] = float(average_precision_score(y_true_bin, sc[:, k]))
                except Exception:
                    mm["ap"] = None
            else:
                mm["balanced_accuracy"] = mm["auroc"] = mm["ap"] = None
            src_dict["per_label"][cname] = mm
        out[src] = src_dict
    return out


def _create_composite_figure(output_dir, class_names, roc_curves, pr_curves,
                             aucs, aps, cm_mc, aucs_ci=None, aps_ci=None):
    import matplotlib.lines as mlines
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1])
    ax_roc    = fig.add_subplot(gs[0, 0])
    ax_pr     = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_cm     = fig.add_subplot(gs[1, 1])

    panel_fs = 20; axis_fs = 11; tick_fs = 10; cm_annot_fs = 16; legend_fs = 10
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx_c, cname in enumerate(class_names):
        color = colors[idx_c % len(colors)]
        if cname in roc_curves:
            fpr, tpr = roc_curves[cname]
            ax_roc.plot(fpr, tpr, linewidth=2, color=color)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax_roc.set_xlabel("FPR", fontsize=axis_fs); ax_roc.set_ylabel("TPR", fontsize=axis_fs)
    ax_roc.tick_params(labelsize=tick_fs)
    ax_roc.text(-0.12, 1.05, "A", transform=ax_roc.transAxes,
                fontsize=panel_fs, fontweight="bold", va="top")

    for idx_c, cname in enumerate(class_names):
        color = colors[idx_c % len(colors)]
        if cname in pr_curves:
            recall, precision = pr_curves[cname]
            ax_pr.plot(recall, precision, linewidth=2, color=color)
    ax_pr.set_xlabel("Recall", fontsize=axis_fs); ax_pr.set_ylabel("Precision", fontsize=axis_fs)
    ax_pr.tick_params(labelsize=tick_fs)
    ax_pr.text(-0.12, 1.05, "B", transform=ax_pr.transAxes,
               fontsize=panel_fs, fontweight="bold", va="top")

    ax_legend.axis("off")
    handles, labels = [], []
    for idx_c, cname in enumerate(class_names):
        color = colors[idx_c % len(colors)]
        handles.append(mlines.Line2D([], [], color=color, linewidth=3))
        auc_val = aucs.get(cname, np.nan) or np.nan
        ap_val  = aps.get(cname, np.nan) or np.nan
        auc_ci  = (aucs_ci or {}).get(cname, [None, None])
        ap_ci   = (aps_ci  or {}).get(cname, [None, None])
        auc_str = (f"{auc_val:.3f} [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}]"
                   if (auc_ci[0] is not None) else f"{auc_val:.3f}")
        ap_str  = (f"{ap_val:.3f} [{ap_ci[0]:.3f}–{ap_ci[1]:.3f}]"
                   if (ap_ci[0]  is not None) else f"{ap_val:.3f}")
        labels.append(f"{cname}\n  AUC {auc_str}\n  AP  {ap_str}")
    ax_legend.legend(handles, labels, loc="center left", frameon=False, fontsize=legend_fs)

    n = len(class_names)
    ax_cm.imshow(cm_mc, cmap="Blues", interpolation="none")
    ax_cm.set_xticks(range(n)); ax_cm.set_yticks(range(n))
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
    fig_path = os.path.join(output_dir, "figure_composite.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# ─────────────────────────────────────────────────────────────
#  Full evaluation on the held-out test set
# ─────────────────────────────────────────────────────────────
def evaluate_on_test(classifiers, temperatures, X_test, y_test_mc,
                     ids_test, class_names, output_dir, n_bootstrap=2000):
    """
    Mirrors evaluate() in Evaluate_for_revision.py but for LR OvR models.
    classifiers: list of fitted sklearn estimators (one per class).
    temperatures: list of floats (one per class; 1.0 = uncalibrated).
    X_test: [N, D] mean-pooled feature matrix.
    """
    n_classes = len(class_names)
    ensure_dir(output_dir)

    all_probs_uncal = np.zeros((len(X_test), n_classes), dtype=np.float32)
    all_probs_cal   = np.zeros((len(X_test), n_classes), dtype=np.float32)

    for k, clf in enumerate(classifiers):
        if clf is None:
            continue
        raw_logits = clf.decision_function(X_test)      # [N,]
        all_probs_uncal[:, k] = (1.0 / (1.0 + np.exp(-raw_logits))).astype(np.float32)

        T = float(temperatures[k]) if temperatures else 1.0
        all_probs_cal[:, k] = (1.0 / (1.0 + np.exp(-raw_logits / T))).astype(np.float32)

    all_labels = np.asarray(y_test_mc, dtype=int)
    if ids_test is None:
        all_ids = [f"sample_{i:05d}" for i in range(len(all_labels))]
    else:
        all_ids = list(ids_test)

    # Check if calibration changed anything (temperature != 1 for any class)
    has_cal = any(abs(float(t) - 1.0) > 1e-6 for t in temperatures)

    per_class = {"uncal": {}, "cal": {}}
    roc_curves_uncal, pr_curves_uncal = {}, {}
    roc_curves_cal,   pr_curves_cal   = {}, {}
    auc_uncal, auc_cal = {}, {}
    ap_uncal,  ap_cal  = {}, {}
    aucs_ci_uncal, aps_ci_uncal = {}, {}
    aucs_ci_cal,   aps_ci_cal   = {}, {}

    # Multiclass argmax predictions — used for per-class binary F1/Prec/Rec/BalAcc so
    # they reflect actual classification decisions and show calibration effects.
    # AUROC and AP still use the raw per-class probabilities (rank-based, unchanged).
    y_pred_mc_uncal = np.argmax(all_probs_uncal, axis=1)
    y_pred_mc_cal   = np.argmax(all_probs_cal,   axis=1)

    LOGGER.info(f"Computing per-class metrics (n_bootstrap={n_bootstrap}) ...")
    for k in range(n_classes):
        cname = class_names[k]
        y_true_bin = (all_labels == k).astype(int)

        # Uncalibrated — argmax-based binary labels so calibration effect is visible
        scores = all_probs_uncal[:, k]
        y_pred_bin = (y_pred_mc_uncal == k).astype(int)
        m = compute_binary_metrics(y_true_bin, y_pred_bin, scores, n_bootstrap=n_bootstrap)
        per_class["uncal"][cname] = m
        if len(np.unique(y_true_bin)) > 1:
            pr_curves_uncal[cname]  = _pr_for_plot(y_true_bin, scores)
            fpr, tpr, _ = roc_curve(y_true_bin, scores)
            roc_curves_uncal[cname] = (fpr, tpr)
        auc_uncal[cname]      = m.get("auroc")
        ap_uncal[cname]       = m.get("ap")
        aucs_ci_uncal[cname]  = m.get("auroc_ci", [None, None])
        aps_ci_uncal[cname]   = m.get("ap_ci",    [None, None])

        # Calibrated
        scores_c = all_probs_cal[:, k]
        y_pred_bin_c = (y_pred_mc_cal == k).astype(int)
        m_c = compute_binary_metrics(y_true_bin, y_pred_bin_c, scores_c, n_bootstrap=n_bootstrap)
        per_class["cal"][cname] = m_c
        if len(np.unique(y_true_bin)) > 1:
            pr_curves_cal[cname]  = _pr_for_plot(y_true_bin, scores_c)
            fpr_c, tpr_c, _ = roc_curve(y_true_bin, scores_c)
            roc_curves_cal[cname] = (fpr_c, tpr_c)
        auc_cal[cname]      = m_c.get("auroc")
        ap_cal[cname]       = m_c.get("ap")
        aucs_ci_cal[cname]  = m_c.get("auroc_ci", [None, None])
        aps_ci_cal[cname]   = m_c.get("ap_ci",    [None, None])

    # Multiclass predictions
    def multiclass_eval(scores_mc, tag):
        y_pred = np.argmax(scores_mc, axis=1)
        report_txt = classification_report(
            all_labels, y_pred, target_names=class_names, digits=3, zero_division=0
        )
        cm_mc = confusion_matrix(all_labels, y_pred, labels=list(range(n_classes)))
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
                ax.text(j, i, f"{cm_mc[i,j]}", ha="center", va="center",
                        fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_multiclass_{tag}.png"), dpi=180)
        plt.close(fig)
        return {"classification_report_txt": report_txt, "confusion_matrix": cm_mc.tolist()}

    mc_uncal = multiclass_eval(all_probs_uncal, "uncal")
    mc_cal   = multiclass_eval(all_probs_cal,   "cal")

    # Multi-curve overlay plots
    def plot_multi_curves(curves, kind, tag, label_scores=None):
        if not curves:
            return
        plt.figure(figsize=(7, 6))
        for cname, (x, y) in curves.items():
            lbl = cname
            if label_scores and label_scores.get(cname) is not None:
                lbl = f"{cname} ({'AUC' if kind=='roc' else 'AP'}={label_scores[cname]:.3f})"
            plt.plot(x, y, label=lbl)
        if kind == "roc":
            plt.plot([0, 1], [0, 1], "k--", lw=1)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        else:
            plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.legend(loc="best", fontsize=9)
        plt.xlim(0, 1); plt.ylim(0, 1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{kind}_curves_{tag}.png"), dpi=180)
        plt.close()

    plot_multi_curves(roc_curves_uncal, "roc", "uncal", label_scores=auc_uncal)
    plot_multi_curves(pr_curves_uncal,  "pr",  "uncal", label_scores=ap_uncal)
    plot_multi_curves(roc_curves_cal,   "roc", "cal",   label_scores=auc_cal)
    plot_multi_curves(pr_curves_cal,    "pr",  "cal",   label_scores=ap_cal)

    # Prefer calibrated when temperatures differ from 1
    use_cal = has_cal and roc_curves_cal and pr_curves_cal

    # Final decision: calibrated if available
    final_scores = all_probs_cal if use_cal else all_probs_uncal
    final_tag    = "cal" if use_cal else "uncal"
    y_pred_final = np.argmax(final_scores, axis=1)

    # WSI-level CSV
    wsi_rows = []
    for i in range(len(all_ids)):
        source, case_id = parse_source_and_case_id(all_ids[i])
        row = {
            "wsi_id": all_ids[i], "source": source, "case_id": case_id,
            "true_label_idx": int(all_labels[i]),
            "true_label": class_names[int(all_labels[i])] if int(all_labels[i]) < n_classes else str(int(all_labels[i])),
            "pred_label_idx": int(y_pred_final[i]),
            "pred_label": class_names[int(y_pred_final[i])] if int(y_pred_final[i]) < n_classes else str(int(y_pred_final[i])),
            "correct": int(int(all_labels[i]) == int(y_pred_final[i])),
        }
        for k2, cn in enumerate(class_names):
            row[f"prob_{cn}"] = float(final_scores[i, k2])
        wsi_rows.append(row)

    prob_cols = [f"prob_{cn}" for cn in class_names]
    wsi_fields = ["wsi_id", "source", "case_id",
                  "true_label_idx", "true_label",
                  "pred_label_idx", "pred_label", "correct"] + prob_cols
    wsi_csv = os.path.join(output_dir, f"predictions_wsi_{final_tag}.csv")
    _write_csv(wsi_csv, wsi_rows, wsi_fields)

    # Case-level aggregation
    grouped = defaultdict(list)
    for row in wsi_rows:
        grouped[(row["source"], row["case_id"])].append(row)

    case_rows, case_sources, case_true, case_pred, case_scores_list = [], [], [], [], []
    for (source, case_id), rows in grouped.items():
        probs = np.vstack([[float(r[f"prob_{cn}"]) for cn in class_names] for r in rows])
        mean_probs = probs.mean(axis=0)
        pred_idx  = int(np.argmax(mean_probs))
        true_labels = [int(r["true_label_idx"]) for r in rows]
        tc = Counter(true_labels)
        true_idx = int(tc.most_common(1)[0][0])
        case_row = {
            "source": source, "case_id": case_id, "n_wsis": int(len(rows)),
            "true_label_idx": true_idx,
            "true_label": class_names[true_idx] if true_idx < n_classes else str(true_idx),
            "true_label_consistent_across_wsis": int(len(tc) == 1),
            "pred_label_idx": pred_idx,
            "pred_label": class_names[pred_idx] if pred_idx < n_classes else str(pred_idx),
            "correct": int(true_idx == pred_idx),
            "wsi_ids": "|".join([str(r["wsi_id"]) for r in rows]),
        }
        for k2, cn in enumerate(class_names):
            case_row[f"mean_prob_{cn}"] = float(mean_probs[k2])
        case_rows.append(case_row)
        case_sources.append(source); case_true.append(true_idx)
        case_pred.append(pred_idx); case_scores_list.append(mean_probs)

    case_sources = np.asarray(case_sources)
    case_true    = np.asarray(case_true, dtype=int)
    case_pred    = np.asarray(case_pred, dtype=int)
    case_scores  = np.vstack(case_scores_list) if case_scores_list else np.zeros((0, n_classes))

    case_prob_cols = [f"mean_prob_{cn}" for cn in class_names]
    case_fields = ["source", "case_id", "n_wsis",
                   "true_label_idx", "true_label", "true_label_consistent_across_wsis",
                   "pred_label_idx", "pred_label", "correct", "wsi_ids"] + case_prob_cols
    case_csv = os.path.join(output_dir, f"predictions_case_{final_tag}.csv")
    _write_csv(case_csv, case_rows, case_fields)

    # Case-level per-class OvR metrics (PRIMARY — cases are the independent unit)
    case_per_class: Dict = {}
    case_ci_macro: Dict = {}
    case_avg_macro: Dict = {}
    case_avg_weighted: Dict = {}

    if len(case_true) > 0:
        LOGGER.info(f"Computing case-level per-class metrics and bootstrap CIs (n_bootstrap={n_bootstrap}) ...")
        case_ci_per_class, case_ci_macro = bootstrap_case_ci(
            case_true, case_scores, n_classes, n_bootstrap=n_bootstrap
        )
        y_pred_mc_case = np.argmax(case_scores, axis=1)   # argmax-based, mirrors WSI-level
        for k in range(n_classes):
            cname = class_names[k]
            y_true_bin_c = (case_true == k).astype(int)
            scores_k_c   = case_scores[:, k]
            y_pred_bin_c = (y_pred_mc_case == k).astype(int)

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
            else:
                m_c["auroc"]    = None
                m_c["ap"]       = None
                m_c["auroc_ci"] = [None, None]
                m_c["ap_ci"]    = [None, None]

            for metric in ["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_weighted",
                           "precision", "recall"]:
                m_c[f"{metric}_ci"] = case_ci_per_class[k].get(metric, [None, None])

            case_per_class[cname] = m_c

        case_avg_macro, case_avg_weighted = _average_per_class(case_per_class, class_names)

    # f1_macro/f1_weighted use OvR-averaged binary F1 (mean of per-class OvR F1 scores),
    # matching the reference evaluation approach rather than sklearn multiclass macro.
    case_level_summary = {
        "n_cases": int(len(case_true)),
        "accuracy": float(accuracy_score(case_true, case_pred)) if len(case_true) else None,
        "balanced_accuracy": float(balanced_accuracy_score(case_true, case_pred)) if len(case_true) else None,
        "f1_macro": float(case_avg_macro["f1"]) if case_avg_macro.get("f1") is not None else None,
        "f1_weighted": float(case_avg_weighted["f1"]) if case_avg_weighted.get("f1") is not None else None,
        "confusion_matrix": confusion_matrix(case_true, case_pred, labels=list(range(n_classes))).tolist() if len(case_true) else [],
        "classification_report_txt": classification_report(case_true, case_pred, target_names=class_names, digits=3, zero_division=0) if len(case_true) else "",
    }

    # Composite figure (prefer calibrated; case-level CM as primary unit)
    try:
        roc_sel = roc_curves_cal if use_cal else roc_curves_uncal
        pr_sel  = pr_curves_cal  if use_cal else pr_curves_uncal
        auc_sel = auc_cal        if use_cal else auc_uncal
        ap_sel  = ap_cal         if use_cal else ap_uncal
        cm_sel  = (np.array(case_level_summary["confusion_matrix"])
                   if case_level_summary.get("confusion_matrix")
                   else np.array(mc_cal["confusion_matrix"] if use_cal else mc_uncal["confusion_matrix"]))
        aucs_ci_sel = aucs_ci_cal if use_cal else aucs_ci_uncal
        aps_ci_sel  = aps_ci_cal  if use_cal else aps_ci_uncal
        fig_path = _create_composite_figure(
            output_dir, class_names, roc_sel, pr_sel, auc_sel, ap_sel, cm_sel,
            aucs_ci=aucs_ci_sel, aps_ci=aps_ci_sel,
        )
        LOGGER.info(f"Composite figure saved: {fig_path}")
    except Exception as exc:
        LOGGER.warning(f"Composite figure failed: {exc}")

    wsi_sources = np.asarray([parse_source_and_case_id(sid)[0] for sid in all_ids])
    stratified_wsi  = _compute_stratified_ovr_metrics(all_labels, y_pred_final, final_scores, wsi_sources, class_names)
    stratified_case = _compute_stratified_ovr_metrics(case_true, case_pred, case_scores, case_sources, class_names)

    her2_consistency = None
    if "HER2" in class_names:
        her2_consistency = {
            "wsi_level":  {src: v["per_label"]["HER2"] for src, v in stratified_wsi.items()  if "HER2" in v.get("per_label", {})},
            "case_level": {src: v["per_label"]["HER2"] for src, v in stratified_case.items() if "HER2" in v.get("per_label", {})},
        }

    # Macro averages + CIs
    avg_uncal_macro, avg_uncal_weighted = _average_per_class(per_class["uncal"], class_names)
    avg_cal_macro,   avg_cal_weighted   = _average_per_class(per_class["cal"],   class_names)
    macro_ci_uncal = bootstrap_macro_ci(all_labels, all_probs_uncal, n_classes, n_bootstrap=n_bootstrap)
    macro_ci_cal   = bootstrap_macro_ci(all_labels, all_probs_cal,   n_classes, n_bootstrap=n_bootstrap)

    # Direct multiclass macro-F1 with CI (argmax-based; distinct from OvR-averaged macro above)
    LOGGER.info("Computing slide-level multiclass macro-F1 CI ...")
    slide_macro_vals = {
        "f1_macro":          float(f1_score(all_labels, y_pred_final, average="macro", zero_division=0)),
        "f1_weighted":       float(f1_score(all_labels, y_pred_final, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(all_labels, y_pred_final)),
        "accuracy":          float(accuracy_score(all_labels, y_pred_final)),
    }
    slide_macro_ci = bootstrap_slide_level_macro_ci(all_labels, y_pred_final, n_bootstrap=n_bootstrap)

    # Text summary
    def fmt(x):
        if x is None: return "NA"
        if isinstance(x, (float, np.floating)): return f"{float(x):.4f}"
        return str(x)

    def fmt_ci(m, key, decimals=3):
        v = m.get(key); ci = m.get(f"{key}_ci", [None, None])
        if v is None: return "NA"
        s = f"{float(v):.{decimals}f}"
        if ci and ci[0] is not None:
            s += f" [{float(ci[0]):.{decimals}f}–{float(ci[1]):.{decimals}f}]"
        return s

    _macro_label_map = {
        "f1": "F1", "f1_macro": "F1_macro", "f1_weighted": "F1_weighted",
        "precision": "Precision", "recall": "Recall",
        "balanced_accuracy": "BalAcc", "auroc": "AUROC", "ap": "AP",
    }

    def _fmt_macro(avg_d, ci_d):
        parts = []
        for key in ["f1", "f1_macro", "precision", "recall", "balanced_accuracy", "auroc", "ap"]:
            v = avg_d.get(key); ci = ci_d.get(key, [None, None])
            if v is None: continue
            label = _macro_label_map.get(key, key)
            s = f"{label}={float(v):.3f}"
            if ci and ci[0] is not None:
                s += f" [{float(ci[0]):.3f}–{float(ci[1]):.3f}]"
            parts.append(s)
        return ", ".join(parts)

    txt_path = os.path.join(output_dir, "metrics_test.txt")
    with open(txt_path, "w") as f:
        f.write(f"Output dir: {output_dir}\n")
        f.write(f"Classes ({n_classes}): {', '.join(class_names)}\n")
        f.write(f"Temperature calibration: {'yes' if has_cal else 'no'}\n")
        f.write(f"Bootstrap iterations: {n_bootstrap}\n\n")

        # ============================================================
        # PRIMARY: Case/patient-level metrics (independent units)
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
        f.write(f"  n_cases={case_level_summary['n_cases']}, "
                f"Acc={fmt(case_level_summary['accuracy'])}, "
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
        # ============================================================
        f.write("=" * 70 + "\n")
        f.write("  SECONDARY RESULTS: WSI LEVEL (supplementary)\n")
        f.write("  (unit = slide; CIs bootstrapped over WSIs — treat as supplementary\n")
        f.write("   when patients have multiple slides)\n")
        f.write("=" * 70 + "\n\n")

        for tag_, pc_ in [("uncalibrated", per_class["uncal"]), ("calibrated", per_class["cal"])]:
            f.write(f"WSI-level per-class metrics with 95% CIs ({tag_}):\n")
            for cname in class_names:
                m = pc_.get(cname, {})
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
        f.write("WSI-level macro-averaged metrics with 95% CIs (calibrated):\n")
        f.write(f"  {_fmt_macro(avg_cal_macro, macro_ci_cal)}\n\n")

        f.write("Slide-level Macro-F1 with 95% CIs (direct multiclass, argmax-based):\n")
        _label_map = {
            "f1_macro":          "Macro-F1",
            "f1_weighted":       "Weighted-F1",
            "balanced_accuracy": "Balanced-Accuracy",
            "accuracy":          "Accuracy",
        }
        for _key in ["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy"]:
            _v  = slide_macro_vals.get(_key)
            _ci = slide_macro_ci.get(_key, [None, None])
            _s  = f"  {_label_map[_key]}: {float(_v):.4f}" if _v is not None else f"  {_label_map[_key]}: NA"
            if _ci and _ci[0] is not None:
                _s += f" [95% CI: {float(_ci[0]):.4f}–{float(_ci[1]):.4f}]"
            f.write(_s + "\n")
        f.write("\n")

        f.write("WSI-level multiclass report (uncalibrated):\n")
        f.write(mc_uncal["classification_report_txt"] + "\n")
        f.write("WSI-level multiclass report (calibrated):\n")
        f.write(mc_cal["classification_report_txt"] + "\n")

        # ============================================================
        # AUXILIARY
        # ============================================================
        f.write("\n" + "=" * 70 + "\n")
        f.write("  AUXILIARY: OUTPUT FILES, STRATIFICATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Final decision mode: {final_tag}\n")
        f.write(f"WSI-level predictions: {wsi_csv}\n")
        f.write(f"Case-level predictions: {case_csv}\n\n")

        if her2_consistency:
            f.write("Source-stratified HER2 consistency (OvR, final decisions):\n")
            f.write("- WSI-level:\n")
            for src, mm in her2_consistency["wsi_level"].items():
                f.write(f"  {src}: Prec={fmt(mm.get('precision'))}, Rec={fmt(mm.get('recall'))}, "
                        f"F1={fmt(mm.get('f1'))}, AUROC={fmt(mm.get('auroc'))}\n")
            f.write("- Case-level:\n")
            for src, mm in her2_consistency["case_level"].items():
                f.write(f"  {src}: Prec={fmt(mm.get('precision'))}, Rec={fmt(mm.get('recall'))}, "
                        f"F1={fmt(mm.get('f1'))}, AUROC={fmt(mm.get('auroc'))}\n")

    # Manuscript-table CSV — Level column: Case (primary) then WSI (secondary)
    final_pc    = per_class["cal"]  if has_cal else per_class["uncal"]
    final_macro = avg_cal_macro     if has_cal else avg_uncal_macro
    final_ci    = macro_ci_cal      if has_cal else macro_ci_uncal

    def _mci(v_dict, ci_dict, key):
        v  = v_dict.get(key)
        ci = ci_dict.get(key, [None, None])
        if v is None: return "NA", "NA"
        return f"{float(v):.3f}", (f"[{ci[0]:.3f}–{ci[1]:.3f}]" if (ci and ci[0] is not None) else "NA")

    def _row(m, level, label):
        def _f(k):  return f"{float(m[k]):.3f}" if m.get(k) is not None else "NA"
        def _ci(k): ci = m.get(f"{k}_ci", [None, None]); return f"[{ci[0]:.3f}–{ci[1]:.3f}]" if (ci and ci[0] is not None) else "NA"
        return {
            "Level": level, "Class": label,
            "N": m.get("support_total", "-"),
            "F1": _f("f1"), "F1_95CI": _ci("f1"),
            "Precision": _f("precision"), "Precision_95CI": _ci("precision"),
            "Recall": _f("recall"), "Recall_95CI": _ci("recall"),
            "BalAcc": _f("balanced_accuracy"), "BalAcc_95CI": _ci("balanced_accuracy"),
            "AUROC": _f("auroc"), "AUROC_95CI_DeLong": _ci("auroc"),
            "AP": _f("ap"), "AP_95CI": _ci("ap"),
        }

    table_rows = []

    # Case-level per-class rows (primary)
    for cname in class_names:
        table_rows.append(_row(case_per_class.get(cname, {}), "Case", cname))

    # Case-level macro row
    f1_v, f1_c   = _mci(case_avg_macro, case_ci_macro, "f1")
    p_v,  p_c    = _mci(case_avg_macro, case_ci_macro, "precision")
    r_v,  r_c    = _mci(case_avg_macro, case_ci_macro, "recall")
    ba_v, ba_c   = _mci(case_avg_macro, case_ci_macro, "balanced_accuracy")
    auc_v, auc_c = _mci(case_avg_macro, case_ci_macro, "auroc")
    ap_v,  ap_c  = _mci(case_avg_macro, case_ci_macro, "ap")
    table_rows.append({
        "Level": "Case", "Class": "Macro-average", "N": "-",
        "F1": f1_v, "F1_95CI": f1_c,
        "Precision": p_v, "Precision_95CI": p_c,
        "Recall": r_v, "Recall_95CI": r_c,
        "BalAcc": ba_v, "BalAcc_95CI": ba_c,
        "AUROC": auc_v, "AUROC_95CI_DeLong": auc_c,
        "AP": ap_v, "AP_95CI": ap_c,
    })

    # WSI-level per-class rows (secondary)
    for cname in class_names:
        table_rows.append(_row(final_pc.get(cname, {}), "WSI", cname))

    # WSI-level macro row (OvR-averaged)
    f1_v, f1_c   = _mci(final_macro, final_ci, "f1")
    p_v,  p_c    = _mci(final_macro, final_ci, "precision")
    r_v,  r_c    = _mci(final_macro, final_ci, "recall")
    ba_v, ba_c   = _mci(final_macro, final_ci, "balanced_accuracy")
    auc_v, auc_c = _mci(final_macro, final_ci, "auroc")
    ap_v,  ap_c  = _mci(final_macro, final_ci, "ap")
    table_rows.append({
        "Level": "WSI", "Class": "Macro-average (OvR)", "N": "-",
        "F1": f1_v, "F1_95CI": f1_c,
        "Precision": p_v, "Precision_95CI": p_c,
        "Recall": r_v, "Recall_95CI": r_c,
        "BalAcc": ba_v, "BalAcc_95CI": ba_c,
        "AUROC": auc_v, "AUROC_95CI_DeLong": auc_c,
        "AP": ap_v, "AP_95CI": ap_c,
    })

    # WSI-level slide-level direct multiclass rows (argmax-based)
    _NA_cols = {"Precision": "NA", "Precision_95CI": "NA", "Recall": "NA", "Recall_95CI": "NA",
                "BalAcc": "NA", "BalAcc_95CI": "NA", "AUROC": "NA", "AUROC_95CI_DeLong": "NA",
                "AP": "NA", "AP_95CI": "NA"}
    for _smkey, _smname in [
        ("f1_macro",          "Macro-F1 (direct, argmax)"),
        ("f1_weighted",       "Weighted-F1 (direct, argmax)"),
        ("balanced_accuracy", "Balanced-Acc (direct, argmax)"),
        ("accuracy",          "Accuracy (direct, argmax)"),
    ]:
        _ci  = slide_macro_ci.get(_smkey, [None, None])
        _val = slide_macro_vals.get(_smkey)
        table_rows.append({
            "Level": "WSI", "Class": _smname, "N": str(len(all_labels)),
            "F1": f"{_val:.3f}" if _val is not None else "NA",
            "F1_95CI": (f"[{_ci[0]:.3f}–{_ci[1]:.3f}]" if _ci[0] is not None else "NA"),
            **_NA_cols,
        })

    table_csv = os.path.join(output_dir, "manuscript_table_metrics_ci.csv")
    _write_csv(table_csv, table_rows,
               ["Level", "Class", "N", "F1", "F1_95CI", "Precision", "Precision_95CI",
                "Recall", "Recall_95CI", "BalAcc", "BalAcc_95CI",
                "AUROC", "AUROC_95CI_DeLong", "AP", "AP_95CI"])
    LOGGER.info(f"Manuscript table CSV: {table_csv}")

    # JSON summary
    def _san(d):
        out = {}
        for key, val in d.items():
            if key.startswith("_"): continue
            if isinstance(val, np.ndarray): out[key] = val.tolist()
            elif isinstance(val, np.generic): out[key] = val.item()
            elif isinstance(val, list): out[key] = [(v.item() if isinstance(v, np.generic) else v) for v in val]
            else: out[key] = val
        return out

    results_json = {
        "case_level_per_class": {
            "note": "PRIMARY results — bootstrapped by resampling cases (independent units)",
            "per_class": {cn: _san(m) for cn, m in case_per_class.items()} if case_per_class else None,
            "macro": case_avg_macro,
            "weighted": case_avg_weighted,
            "macro_ci_bootstrap_case": case_ci_macro,
        },
        "wsi_level_per_class": {
            "note": "SECONDARY/supplementary — bootstrapped by resampling WSIs",
            "uncal": {cn: _san(m) for cn, m in per_class["uncal"].items()},
            "cal":   {cn: _san(m) for cn, m in per_class["cal"].items()},
        },
        "averages": {
            "uncal_macro": avg_uncal_macro, "uncal_macro_ci_bootstrap_wsi": macro_ci_uncal,
            "cal_macro":   avg_cal_macro,   "cal_macro_ci_bootstrap_wsi":   macro_ci_cal,
        },
        "multiclass": {
            "uncal": {"confusion_matrix": mc_uncal["confusion_matrix"],
                      "report": mc_uncal["classification_report_txt"]},
            "cal":   {"confusion_matrix": mc_cal["confusion_matrix"],
                      "report": mc_cal["classification_report_txt"]},
        },
        "slide_level_macro": {"values": slide_macro_vals, "ci_95": slide_macro_ci},
        "case_level_multiclass": case_level_summary,
        "stratified_by_source": {"wsi_level": stratified_wsi, "case_level": stratified_case},
        "her2_consistency_by_source": her2_consistency,
        "final_decision": {"final_tag": final_tag, "wsi_csv": wsi_csv, "case_csv": case_csv},
    }
    with open(os.path.join(output_dir, "metrics_test.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    LOGGER.info(f"Evaluation artefacts saved to: {output_dir}")


# ─────────────────────────────────────────────────────────────
#  Main training driver
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="OvR Logistic Regression on mean-pooled slide features (revised pipeline)."
    )
    parser.add_argument("--mil_pkl",    required=True,  help="PKL with train/test splits (same format as MIL scripts)")
    parser.add_argument("--output_dir", required=True,  help="Where to save models and evaluation artefacts")
    parser.add_argument("--calibrate",  action="store_true", help="Apply temperature scaling on calibration subset")
    parser.add_argument("--imbalance",  default="none",
                        choices=["none", "oversample", "undersample", "smote"],
                        help="Class-imbalance strategy applied to training split only")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--cal_size",   type=float, default=0.20,
                        help="Fraction of training patients held out for temperature calibration (default 0.20)")
    parser.add_argument("--cv_folds",   type=int,   default=5)
    parser.add_argument("--n_bootstrap",type=int,   default=2000)
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # File logger
    fh = logging.FileHandler(os.path.join(args.output_dir, "training.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(fh)

    LOGGER.info(f"Seed={args.seed}  imbalance={args.imbalance}  calibrate={args.calibrate}")

    # Load PKL
    bags_train, y_train_mc, bags_test, y_test_mc, class_names_opt, ids_train, ids_test = \
        load_train_test_from_pkl(args.mil_pkl)

    n_classes = int(y_train_mc.max()) + 1
    if class_names_opt is not None:
        class_names = list(class_names_opt)
        n_classes = len(class_names)
    else:
        class_names = [f"class_{i}" for i in range(n_classes)]

    LOGGER.info(f"Train N={len(bags_train)} | Test N={len(bags_test) if bags_test is not None else 0} | classes={class_names}")

    if ids_train is None:
        LOGGER.warning("No slide IDs in PKL; patient-level split will fall back to slide-level.")

    # Parameter grid for GridSearchCV
    param_grid = {
        "C":            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "solver":       ["lbfgs", "liblinear"],
        "class_weight": [None, "balanced"],
    }

    classifiers    = []
    temperatures   = []
    X_cal_per_cls  = []
    y_cal_per_cls  = []
    per_class_info = []

    slide_ids_train = list(ids_train) if ids_train is not None else None

    for k in range(n_classes):
        cls_name = class_names[k]
        print(f"\n===== Classifier {k} ({cls_name}) vs Rest =====")
        y_bin = (y_train_mc == k).astype(int)

        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            LOGGER.warning(f"  Class {cls_name}: degenerate; skipping.")
            classifiers.append(None); temperatures.append(1.0)
            X_cal_per_cls.append(np.zeros((0, 1))); y_cal_per_cls.append(np.array([], dtype=int))
            continue

        # Patient-level 80/20 split: train / cal (no val; CV handles internal validation)
        X_tr_bags, _, X_ca_bags, y_tr, _, y_ca, n_pat_tr, _, n_pat_ca = \
            patient_stratified_train_val_cal_split(
                list(bags_train), y_bin, slide_ids_train,
                cal_size=args.cal_size, seed=args.seed
            )

        # Mean-pool
        X_tr = mean_pool(X_tr_bags)
        X_ca = mean_pool(X_ca_bags) if len(X_ca_bags) > 0 else np.zeros((0, X_tr.shape[1]))
        y_ca = np.asarray(y_ca, dtype=int)

        split_strategy = "patient-level" if n_pat_tr is not None else "slide-level (fallback)"
        tr_counts = np.bincount(y_tr, minlength=2)
        ca_counts = np.bincount(y_ca, minlength=2) if len(y_ca) > 0 else np.array([0, 0])
        LOGGER.info(f"  Split: {split_strategy} | "
                    f"train neg={int(tr_counts[0])} pos={int(tr_counts[1])} | "
                    f"cal neg={int(ca_counts[0])} pos={int(ca_counts[1])}")

        # Apply imbalance strategy to training split only
        X_tr_fit, y_tr_fit = apply_imbalance_strategy(X_tr, y_tr, args.imbalance, args.seed)
        if args.imbalance != "none":
            LOGGER.info(f"  After {args.imbalance}: neg={int((y_tr_fit==0).sum())} pos={int((y_tr_fit==1).sum())}")

        # GridSearchCV with patient-stratified CV folds — use StratifiedKFold on the (resampled) train split
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        clf_base = LogisticRegression(max_iter=2000, random_state=args.seed, n_jobs=1)
        gs = GridSearchCV(
            clf_base, param_grid,
            cv=cv, scoring="balanced_accuracy",
            n_jobs=-1, verbose=0, return_train_score=True,
        )
        gs.fit(X_tr_fit, y_tr_fit)
        LOGGER.info(f"  Best params: {gs.best_params_}  CV balanced_acc={gs.best_score_:.4f}")

        # Refit best model on training split (without resampling artefacts leaking into val/cal)
        best_clf = LogisticRegression(
            max_iter=2000, random_state=args.seed, n_jobs=1, **gs.best_params_
        )
        best_clf.fit(X_tr_fit, y_tr_fit)
        classifiers.append(best_clf)
        X_cal_per_cls.append(X_ca)
        y_cal_per_cls.append(y_ca)

        per_class_info.append({
            "class_idx": k, "class_name": cls_name,
            "split_strategy": split_strategy,
            "best_params": gs.best_params_,
            "best_cv_score": float(gs.best_score_),
            "train_counts": {"neg": int(tr_counts[0]), "pos": int(tr_counts[1]), "n_patients": n_pat_tr},
            "cal_counts":   {"neg": int(ca_counts[0]), "pos": int(ca_counts[1]), "n_patients": n_pat_ca},
        })

        if args.debug:
            debug_path = os.path.join(args.output_dir, f"debug_class_{k}_{cls_name}.json")
            with open(debug_path, "w") as fd:
                json.dump({
                    "class_idx": k, "class_name": cls_name,
                    "split_strategy": split_strategy,
                    "imbalance": args.imbalance,
                    "best_params": gs.best_params_,
                    "best_cv_score": float(gs.best_score_),
                    "train_n": int(len(y_tr_fit)),
                    "train_pos": int((y_tr_fit == 1).sum()),
                    "cal_n": int(len(y_ca)),
                }, fd, indent=2)

    # Temperature scaling
    if args.calibrate:
        LOGGER.info("\n===== Temperature scaling (calibration) =====")
        temperatures = calibrate_lr_models(classifiers, X_cal_per_cls, y_cal_per_cls)
        with open(os.path.join(args.output_dir, "temperature_parameters.json"), "w") as f:
            json.dump({"temperatures": temperatures, "class_names": class_names}, f, indent=2)
        LOGGER.info(f"Temperatures: {temperatures}")
    else:
        temperatures = [1.0] * n_classes

    # Save classifiers
    models_pkl = os.path.join(args.output_dir, "lr_classifiers.pkl")
    with open(models_pkl, "wb") as f:
        pickle.dump({
            "classifiers": classifiers,
            "class_names": class_names,
            "temperatures": temperatures,
            "imbalance": args.imbalance,
            "seed": args.seed,
        }, f)
    LOGGER.info(f"Classifiers saved: {models_pkl}")

    # Save training summary
    summary = {
        "mil_pkl": args.mil_pkl,
        "output_dir": args.output_dir,
        "classes": class_names,
        "n_classes": n_classes,
        "calibrated": args.calibrate,
        "temperatures": temperatures,
        "imbalance": args.imbalance,
        "seed": args.seed,
        "cal_size": args.cal_size,
        "cv_folds": args.cv_folds,
        "param_grid": str(param_grid),
        "train_size": int(len(bags_train)),
        "test_size": int(len(bags_test)) if bags_test is not None else 0,
        "per_class": per_class_info,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info(f"Training summary saved.")

    # Evaluate on test set
    if bags_test is not None and y_test_mc is not None:
        LOGGER.info("\n===== Evaluating on test set =====")
        X_test = mean_pool(bags_test)
        eval_dir = os.path.join(args.output_dir, "eval_test")
        ensure_dir(eval_dir)
        evaluate_on_test(
            classifiers, temperatures,
            X_test, np.asarray(y_test_mc, dtype=int),
            ids_test, class_names,
            output_dir=eval_dir,
            n_bootstrap=args.n_bootstrap,
        )
    else:
        LOGGER.warning("No test split found in PKL; skipping evaluation.")

    print(f"\n[DONE] All artefacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
