#!/usr/bin/env python3
"""
CosineKNN_MeanPooling.py

k-Nearest Neighbors classifier with cosine similarity on mean-pooled
slide-level embeddings.

This script directly addresses reviewer comments (Section 2.3.1):

1. Terminology: This method IS k-nearest neighbors (kNN) classification.
   "Cosine similarity on slide-level embeddings" uses cosine similarity as
   the proximity measure, which is equivalent to using cosine distance as
   the distance metric in a standard kNN classifier. We therefore adopt the
   standard kNN terminology throughout.

2. K selection: K is selected by 5-fold stratified cross-validation on the
   training split, maximising balanced accuracy over K in {1,3,5,7,9,15,21}.
   This replaces the previously arbitrary K=3 choice.

3. Classification rule: For each test slide, the k training slides with the
   highest cosine similarity are identified. Each neighbor votes for its class
   with a weight equal to its cosine similarity score. The class with the
   highest total weighted vote is predicted. This is similarity-weighted
   voting — NOT simple majority voting. The distinction matters when K < the
   number of classes (e.g. K=3 with 4 classes): each of the 3 neighbours may
   belong to a different class, producing a 3-way tie under majority voting
   with no clear winner. Similarity-weighted scoring resolves this by
   selecting the class with the highest cosine similarity, which is a
   principled decision.

4. Softmax: Softmax IS required here. The class scores are sums of cosine
   similarities, which can be negative (cosine similarity range: [–1, +1]).
   Softmax converts any real-valued score vector into a valid probability
   distribution (all non-negative, summing to 1) regardless of the sign of
   the inputs. Simple L1 normalisation would produce invalid (negative)
   probabilities when scores are negative. Note: the argmax prediction
   (classification decision) is the same with or without softmax since
   softmax is order-preserving; softmax only affects the probability
   calibration used for AUROC/AP computation.

5. Tie-breaking: similarity-weighted scoring makes exact ties in class scores
   practically impossible in floating-point arithmetic. If they occurred,
   numpy's argmax would return the lowest class index. For neighbour
   retrieval, np.argpartition followed by argsort is fully deterministic.

Evaluation: identical to LR_MeanPooling_OvR.py — same metrics, same
confidence intervals (DeLong for AUROC, bootstrap for all others), same
output files (composite figure, ROC/PR overlay plots, WSI-level and
case-level CSV exports, manuscript-table CSV with CIs).

Author: Konstantinos Papagoras
Date: 2025
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
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

DEFAULT_K_VALUES = [1, 3, 5, 7, 9, 15, 21]


# ─────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────
def _get_logger(name: str = "cosine_knn") -> logging.Logger:
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
#  PKL loader (identical schema support as LR_MeanPooling_OvR.py)
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

    if "train_features" in data and "train_labels" in data:
        raw_feat_tr = np.asarray(data["train_features"], dtype=np.float32)
        raw_lbl_tr  = data["train_labels"]
        raw_feat_te = data.get("test_features")
        raw_lbl_te  = data.get("test_labels")

        if class_names is None:
            class_names = sorted(set(str(l) for l in raw_lbl_tr))

        def _to_int(labels):
            if not labels:
                return np.array([], dtype=int)
            if isinstance(labels[0], str):
                lbl2idx = {c: i for i, c in enumerate(class_names)}
                return np.array([lbl2idx[str(l)] for l in labels], dtype=int)
            return np.asarray(labels, dtype=int)

        y_train = _to_int(raw_lbl_tr)
        y_test  = _to_int(raw_lbl_te) if raw_lbl_te else None
        X_train = raw_feat_tr
        X_test  = np.asarray(raw_feat_te, dtype=np.float32) if raw_feat_te is not None else None
        ids_train = data.get("train_ids") or data.get("ids_train")
        ids_test  = data.get("test_ids")  or data.get("ids_test")
        return X_train, y_train, X_test, y_test, class_names, ids_train, ids_test

    if "X_train" in data and "y_train" in data:
        X_train = np.asarray(data["X_train"], dtype=np.float32)
        y_train = np.asarray(data["y_train"], dtype=int)
        X_test  = np.asarray(data["X_test"], dtype=np.float32) if "X_test" in data else None
        y_test  = np.asarray(data["y_test"], dtype=int) if "y_test" in data else None
        ids_train = data.get("train_ids") or data.get("ids_train")
        ids_test  = data.get("test_ids")  or data.get("ids_test")
        if class_names is None:
            class_names = [f"class_{i}" for i in range(int(y_train.max()) + 1)]
        return X_train, y_train, X_test, y_test, class_names, ids_train, ids_test

    raise KeyError("Unrecognised PKL format.")


# ─────────────────────────────────────────────────────────────
#  Patient ID extraction (mirrors LR_MeanPooling_OvR.py)
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


# ─────────────────────────────────────────────────────────────
#  kNN with cosine similarity
# ─────────────────────────────────────────────────────────────
def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


def knn_predict_proba(X_train_norm: np.ndarray,
                      y_train: np.ndarray,
                      X_test_norm: np.ndarray,
                      n_classes: int,
                      k: int,
                      eps: float = 1e-9,
                      temp: float = 1.0) -> np.ndarray:
    """
    kNN probability estimates via similarity-weighted voting with softmax.

    For each test slide:
      1. Find the k training slides with the highest cosine similarity
         (dot product of L2-normalised vectors).
      2. For each class, sum the cosine similarities of its representatives
         among the k neighbours (similarity-weighted voting).
      3. Add eps to all class scores (prevents zero scores for unrepresented
         classes — important when K < number of classes).
      4. Apply softmax to convert scores to a valid probability distribution.
         Softmax is necessary because cosine similarities can be negative
         (range [–1, +1]), so simple L1 normalisation could yield negative
         "probabilities". Softmax handles negative scores correctly.

    Note: the argmax prediction (classification decision) is identical
    regardless of whether softmax or L1 normalisation is used, because
    softmax is order-preserving. Softmax only affects AUROC/AP estimation.

    Tie-breaking: exact ties in class scores are practically impossible with
    floating-point cosine similarities. numpy's argmax breaks any tie by
    returning the lowest class index.

    Parameters
    ----------
    temp : softmax temperature (default 1.0; lower = sharper distribution)

    Returns
    -------
    probs : ndarray, shape (N_test, n_classes), rows sum to 1.
    """
    from scipy.special import softmax as _softmax

    sim    = X_test_norm @ X_train_norm.T          # [N_test, N_train]
    N_test = sim.shape[0]
    probs  = np.zeros((N_test, n_classes), dtype=np.float64)

    for i in range(N_test):
        row = sim[i]
        if k >= len(row):
            nn_idx = np.argsort(row)[::-1]
        else:
            # argpartition: O(N) then sort only the top-k: O(k log k)
            part   = np.argpartition(row, -k)[-k:]
            nn_idx = part[np.argsort(row[part])[::-1]]

        nn_sim    = row[nn_idx]
        nn_labels = y_train[nn_idx]

        class_scores = np.zeros(n_classes, dtype=np.float64)
        for j, lbl in enumerate(nn_labels):
            class_scores[int(lbl)] += nn_sim[j]

        class_scores += eps                        # smoothing for absent classes
        probs[i] = _softmax(class_scores / temp)

    return probs


def select_k_by_cv(X_train_norm: np.ndarray,
                   y_train: np.ndarray,
                   n_classes: int,
                   k_values: list,
                   n_folds: int = 5,
                   seed: int = 42) -> Tuple[int, dict, dict]:
    """
    Select K via stratified cross-validation on the training split.

    Scoring metric: balanced accuracy (consistent with LR_MeanPooling_OvR.py).

    Returns
    -------
    best_k      : int
    k_fold_scores : dict {k: list of per-fold balanced accuracy}
    k_mean_scores : dict {k: mean balanced accuracy across folds}
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    k_fold_scores = {k: [] for k in k_values}

    LOGGER.info(f"K selection: {n_folds}-fold CV over K={k_values}")
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train_norm, y_train)):
        X_tr, X_va = X_train_norm[tr_idx], X_train_norm[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        for k in k_values:
            probs = knn_predict_proba(X_tr, y_tr, X_va, n_classes, k)
            y_pred = np.argmax(probs, axis=1)
            score  = balanced_accuracy_score(y_va, y_pred)
            k_fold_scores[k].append(score)
        LOGGER.info(f"  Fold {fold_idx+1}/{n_folds} done")

    k_mean_scores = {k: float(np.mean(scores)) for k, scores in k_fold_scores.items()}
    best_k = max(k_mean_scores, key=k_mean_scores.get)

    LOGGER.info("K selection results (mean balanced accuracy):")
    for k in k_values:
        marker = " <-- selected" if k == best_k else ""
        LOGGER.info(f"  K={k:>3}: {k_mean_scores[k]:.4f}{marker}")

    return best_k, k_fold_scores, k_mean_scores


# ─────────────────────────────────────────────────────────────
#  Evaluation helpers (identical to LR_MeanPooling_OvR.py)
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
    se  = float(np.sqrt(np.var(vx, ddof=1) / n1 + np.var(vy, ddof=1) / n0))
    z   = float(sp_stats.norm.ppf(1.0 - alpha / 2.0))
    return auc, float(np.clip(auc - z * se, 0.0, 1.0)), float(np.clip(auc + z * se, 0.0, 1.0))


def bootstrap_metric_ci(y_true, y_pred, y_prob, n_bootstrap=2000, alpha=0.05, seed=42):
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
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
    rng   = np.random.default_rng(seed)
    n     = len(all_labels)
    keys  = ["f1", "precision", "recall", "balanced_accuracy", "auroc", "ap"]
    boots = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        idx    = rng.integers(0, n, size=n)
        yt     = all_labels[idx]
        yp_s   = all_probs[idx]
        yp_mc  = np.argmax(yp_s, axis=1)   # argmax-based, mirrors per-class point estimates
        per_k  = {k: [] for k in keys}
        for k in range(n_classes):
            ytb = (yt == k).astype(int)
            if len(np.unique(ytb)) < 2:
                continue
            sk  = yp_s[:, k]
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
    rng   = np.random.default_rng(seed)
    n     = len(y_true_mc)
    keys  = ["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy"]
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
        fpr, tpr, _     = roc_curve(y_true, y_prob_pos)
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
    precision  = np.nan_to_num(precision, nan=0.0, posinf=1.0, neginf=0.0)
    recall     = np.clip(recall, 0.0, 1.0)
    pairs      = {}
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
    keys       = ["accuracy", "balanced_accuracy", "precision", "recall",
                  "f1", "f1_macro", "f1_weighted", "auroc", "ap"]
    rows       = [per_class_dict.get(c, {}) for c in class_names]
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


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compute_stratified_ovr_metrics(y_true_mc, y_pred_mc, scores_mc, sources, class_names):
    out = {}
    for src in sorted(set(sources.tolist())):
        idx   = np.where(sources == src)[0]
        yt    = y_true_mc[idx]; yp = y_pred_mc[idx]; sc = scores_mc[idx, :]
        src_dict = {"n_samples": int(len(idx)), "per_label": {}}
        for k, cname in enumerate(class_names):
            y_true_bin = (yt == k).astype(int)
            y_pred_bin = (yp == k).astype(int)
            mm = {
                "support_pos":   int((y_true_bin == 1).sum()),
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
    gs  = gridspec.GridSpec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1])
    ax_roc    = fig.add_subplot(gs[0, 0])
    ax_pr     = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_cm     = fig.add_subplot(gs[1, 1])

    panel_fs = 20; axis_fs = 11; tick_fs = 10; cm_annot_fs = 16; legend_fs = 10
    colors   = plt.rcParams["axes.prop_cycle"].by_key()["color"]

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
        color   = colors[idx_c % len(colors)]
        handles.append(mlines.Line2D([], [], color=color, linewidth=3))
        auc_val = aucs.get(cname, np.nan) or np.nan
        ap_val  = aps.get(cname, np.nan)  or np.nan
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
#  Full evaluation on the held-out test set (kNN variant)
# ─────────────────────────────────────────────────────────────
def evaluate_on_test_knn(all_probs: np.ndarray,
                          y_test_mc: np.ndarray,
                          ids_test,
                          class_names: list,
                          output_dir: str,
                          best_k: int,
                          k_cv_scores: dict,
                          k_mean_scores: dict,
                          n_bootstrap: int = 2000):
    """
    Evaluation identical to LR_MeanPooling_OvR.evaluate_on_test but for kNN.

    Parameters
    ----------
    all_probs   : [N_test, n_classes] probability matrix from knn_predict_proba
    y_test_mc   : [N_test] integer ground-truth labels
    ids_test    : list of slide IDs
    class_names : list of class name strings
    output_dir  : directory to write all artefacts
    best_k      : selected K value
    k_cv_scores : per-fold balanced accuracy for each K (for the JSON summary)
    k_mean_scores : mean balanced accuracy per K
    n_bootstrap : number of bootstrap resamples for CIs
    """
    n_classes = len(class_names)
    ensure_dir(output_dir)

    all_labels = np.asarray(y_test_mc, dtype=int)
    if ids_test is None:
        all_ids = [f"sample_{i:05d}" for i in range(len(all_labels))]
    else:
        all_ids = list(ids_test)

    y_pred = np.argmax(all_probs, axis=1)

    per_class  = {}
    roc_curves = {}
    pr_curves  = {}
    aucs, aps  = {}, {}
    aucs_ci, aps_ci = {}, {}

    LOGGER.info(f"Computing per-class metrics (n_bootstrap={n_bootstrap}) ...")
    for k in range(n_classes):
        cname      = class_names[k]
        y_true_bin = (all_labels == k).astype(int)
        scores     = all_probs[:, k]
        y_pred_bin = (y_pred == k).astype(int)


        m = compute_binary_metrics(y_true_bin, y_pred_bin, scores, n_bootstrap=n_bootstrap)
        per_class[cname] = m

        if len(np.unique(y_true_bin)) > 1:
            pr_curves[cname]  = _pr_for_plot(y_true_bin, scores)
            fpr, tpr, _       = roc_curve(y_true_bin, scores)
            roc_curves[cname] = (fpr, tpr)

        aucs[cname]    = m.get("auroc")
        aps[cname]     = m.get("ap")
        aucs_ci[cname] = m.get("auroc_ci", [None, None])
        aps_ci[cname]  = m.get("ap_ci",    [None, None])

    # Multiclass confusion matrix
    cm_mc      = confusion_matrix(all_labels, y_pred, labels=list(range(n_classes)))
    report_txt = classification_report(
        all_labels, y_pred, target_names=class_names, digits=3, zero_division=0
    )

    # Stand-alone confusion matrix figure
    fig, ax = plt.subplots(figsize=(5 + n_classes * 0.5, 4 + n_classes * 0.3))
    im = ax.imshow(cm_mc, cmap="Blues", interpolation="none")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Cosine kNN (K={best_k}) — Multiclass Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm_mc[i,j]}", ha="center", va="center",
                    fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_multiclass.png"), dpi=180)
    plt.close(fig)

    # ROC and PR overlay plots
    def _plot_curves(curves, kind, label_scores=None):
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
        plt.savefig(os.path.join(output_dir, f"{kind}_curves.png"), dpi=180)
        plt.close()

    _plot_curves(roc_curves, "roc", label_scores=aucs)
    _plot_curves(pr_curves,  "pr",  label_scores=aps)

    # Composite figure
    try:
        _create_composite_figure(
            output_dir, class_names, roc_curves, pr_curves, aucs, aps, cm_mc,
            aucs_ci=aucs_ci, aps_ci=aps_ci,
        )
    except Exception as exc:
        LOGGER.warning(f"Composite figure failed: {exc}")

    # Macro averages + CIs
    avg_macro, avg_weighted = _average_per_class(per_class, class_names)
    macro_ci = bootstrap_macro_ci(all_labels, all_probs, n_classes, n_bootstrap=n_bootstrap)

    # Direct multiclass macro-F1 with CI
    LOGGER.info("Computing slide-level multiclass macro-F1 CI ...")
    slide_macro_vals = {
        "f1_macro":          float(f1_score(all_labels, y_pred, average="macro", zero_division=0)),
        "f1_weighted":       float(f1_score(all_labels, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(all_labels, y_pred)),
        "accuracy":          float(accuracy_score(all_labels, y_pred)),
    }
    slide_macro_ci = bootstrap_slide_level_macro_ci(all_labels, y_pred, n_bootstrap=n_bootstrap)

    # WSI-level CSV
    wsi_rows = []
    for i in range(len(all_ids)):
        source, case_id = parse_source_and_case_id(all_ids[i])
        row = {
            "wsi_id": all_ids[i], "source": source, "case_id": case_id,
            "true_label_idx": int(all_labels[i]),
            "true_label": class_names[int(all_labels[i])] if int(all_labels[i]) < n_classes else str(int(all_labels[i])),
            "pred_label_idx": int(y_pred[i]),
            "pred_label": class_names[int(y_pred[i])] if int(y_pred[i]) < n_classes else str(int(y_pred[i])),
            "correct": int(int(all_labels[i]) == int(y_pred[i])),
        }
        for k2, cn in enumerate(class_names):
            row[f"prob_{cn}"] = float(all_probs[i, k2])
        wsi_rows.append(row)

    prob_cols  = [f"prob_{cn}" for cn in class_names]
    wsi_fields = ["wsi_id", "source", "case_id",
                  "true_label_idx", "true_label",
                  "pred_label_idx", "pred_label", "correct"] + prob_cols
    wsi_csv = os.path.join(output_dir, "predictions_wsi.csv")
    _write_csv(wsi_csv, wsi_rows, wsi_fields)

    # Case-level aggregation
    grouped = defaultdict(list)
    for row in wsi_rows:
        grouped[(row["source"], row["case_id"])].append(row)

    case_rows, case_sources, case_true, case_pred, case_scores_list = [], [], [], [], []
    for (source, case_id), rows in grouped.items():
        probs     = np.vstack([[float(r[f"prob_{cn}"]) for cn in class_names] for r in rows])
        mean_probs = probs.mean(axis=0)
        pred_idx   = int(np.argmax(mean_probs))
        true_labels = [int(r["true_label_idx"]) for r in rows]
        tc         = Counter(true_labels)
        true_idx   = int(tc.most_common(1)[0][0])
        case_row   = {
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
    case_fields    = ["source", "case_id", "n_wsis",
                      "true_label_idx", "true_label", "true_label_consistent_across_wsis",
                      "pred_label_idx", "pred_label", "correct", "wsi_ids"] + case_prob_cols
    case_csv = os.path.join(output_dir, "predictions_case.csv")
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
        "n_cases":           int(len(case_true)),
        "accuracy":          float(accuracy_score(case_true, case_pred)) if len(case_true) else None,
        "balanced_accuracy": float(balanced_accuracy_score(case_true, case_pred)) if len(case_true) else None,
        "f1_macro":          float(case_avg_macro["f1"]) if case_avg_macro.get("f1") is not None else None,
        "f1_weighted":       float(case_avg_weighted["f1"]) if case_avg_weighted.get("f1") is not None else None,
        "confusion_matrix":  confusion_matrix(case_true, case_pred, labels=list(range(n_classes))).tolist() if len(case_true) else [],
        "classification_report_txt": classification_report(case_true, case_pred, target_names=class_names, digits=3, zero_division=0) if len(case_true) else "",
    }

    # Regenerate composite figure with case-level CM (primary analysis unit is the patient)
    if case_level_summary.get("confusion_matrix"):
        try:
            cm_case = np.array(case_level_summary["confusion_matrix"])
            _create_composite_figure(
                output_dir, class_names, roc_curves, pr_curves, aucs, aps, cm_case,
                aucs_ci=aucs_ci, aps_ci=aps_ci,
            )
        except Exception as exc:
            LOGGER.warning(f"Case-level composite figure update failed: {exc}")

    wsi_sources     = np.asarray([parse_source_and_case_id(sid)[0] for sid in all_ids])
    stratified_wsi  = _compute_stratified_ovr_metrics(all_labels, y_pred, all_probs, wsi_sources, class_names)
    stratified_case = _compute_stratified_ovr_metrics(case_true, case_pred, case_scores, case_sources, class_names)

    her2_consistency = None
    if "HER2" in class_names:
        her2_consistency = {
            "wsi_level":  {src: v["per_label"]["HER2"] for src, v in stratified_wsi.items()  if "HER2" in v.get("per_label", {})},
            "case_level": {src: v["per_label"]["HER2"] for src, v in stratified_case.items() if "HER2" in v.get("per_label", {})},
        }

    # ── Text metrics summary ──────────────────────────────────
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
        f.write(f"Classifier: cosine kNN (K={best_k}, selected by 5-fold CV)\n")
        f.write(f"Classes ({n_classes}): {', '.join(class_names)}\n")
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

        f.write("WSI-level per-class OvR metrics with 95% CIs:\n")
        for cname in class_names:
            m = per_class.get(cname, {})
            f.write(
                f"- {cname} (n={m.get('support_pos','?')}/{m.get('support_total','?')}): "
                f"F1={fmt_ci(m,'f1')}, F1_macro={fmt_ci(m,'f1_macro')}, F1_weighted={fmt_ci(m,'f1_weighted')}, "
                f"Prec={fmt_ci(m,'precision')}, Rec={fmt_ci(m,'recall')}, "
                f"BalAcc={fmt_ci(m,'balanced_accuracy')}, "
                f"AUROC={fmt_ci(m,'auroc')} [DeLong], AP={fmt_ci(m,'ap')}\n"
            )
        f.write("\n")

        f.write("WSI-level macro-averaged metrics with 95% CIs:\n")
        f.write(f"  {_fmt_macro(avg_macro, macro_ci)}\n\n")

        f.write("Slide-level Macro-F1 with 95% CIs (direct multiclass, argmax-based):\n")
        _label_map = {
            "f1_macro": "Macro-F1", "f1_weighted": "Weighted-F1",
            "balanced_accuracy": "Balanced-Accuracy", "accuracy": "Accuracy",
        }
        for _key in ["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy"]:
            _v  = slide_macro_vals.get(_key)
            _ci = slide_macro_ci.get(_key, [None, None])
            _s  = f"  {_label_map[_key]}: {float(_v):.4f}" if _v is not None else f"  {_label_map[_key]}: NA"
            if _ci and _ci[0] is not None:
                _s += f" [95% CI: {float(_ci[0]):.4f}–{float(_ci[1]):.4f}]"
            f.write(_s + "\n")
        f.write("\n")

        f.write("WSI-level multiclass classification report:\n")
        f.write(report_txt + "\n")

        # ============================================================
        # AUXILIARY
        # ============================================================
        f.write("\n" + "=" * 70 + "\n")
        f.write("  AUXILIARY: K SELECTION, OUTPUT FILES, STRATIFICATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("K selection (5-fold CV balanced accuracy):\n")
        for k_val in sorted(k_mean_scores.keys()):
            marker = " <-- selected" if k_val == best_k else ""
            f.write(f"  K={k_val:>3}: {k_mean_scores[k_val]:.4f}{marker}\n")
        f.write("\n")

        f.write(f"WSI-level predictions: {wsi_csv}\n")
        f.write(f"Case-level predictions: {case_csv}\n\n")

        if her2_consistency:
            f.write("Source-stratified HER2 consistency (OvR, kNN decisions):\n")
            f.write("- WSI-level:\n")
            for src, mm in her2_consistency["wsi_level"].items():
                f.write(f"  {src}: Prec={fmt(mm.get('precision'))}, Rec={fmt(mm.get('recall'))}, "
                        f"F1={fmt(mm.get('f1'))}, AUROC={fmt(mm.get('auroc'))}\n")
            f.write("- Case-level:\n")
            for src, mm in her2_consistency["case_level"].items():
                f.write(f"  {src}: Prec={fmt(mm.get('precision'))}, Rec={fmt(mm.get('recall'))}, "
                        f"F1={fmt(mm.get('f1'))}, AUROC={fmt(mm.get('auroc'))}\n")

    # ── Manuscript-table CSV — Level column: Case (primary) then WSI (secondary)
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
        table_rows.append(_row(per_class.get(cname, {}), "WSI", cname))

    # WSI-level macro row (OvR-averaged)
    f1_v, f1_c   = _mci(avg_macro, macro_ci, "f1")
    p_v,  p_c    = _mci(avg_macro, macro_ci, "precision")
    r_v,  r_c    = _mci(avg_macro, macro_ci, "recall")
    ba_v, ba_c   = _mci(avg_macro, macro_ci, "balanced_accuracy")
    auc_v, auc_c = _mci(avg_macro, macro_ci, "auroc")
    ap_v,  ap_c  = _mci(avg_macro, macro_ci, "ap")
    table_rows.append({
        "Level": "WSI", "Class": "Macro-average (OvR)", "N": "-",
        "F1": f1_v, "F1_95CI": f1_c,
        "Precision": p_v, "Precision_95CI": p_c,
        "Recall": r_v, "Recall_95CI": r_c,
        "BalAcc": ba_v, "BalAcc_95CI": ba_c,
        "AUROC": auc_v, "AUROC_95CI_DeLong": auc_c,
        "AP": ap_v, "AP_95CI": ap_c,
    })

    # WSI-level direct multiclass rows (argmax-based)
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

    # ── JSON summary ──────────────────────────────────────────
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
        "knn_config": {
            "best_k": best_k,
            "k_values_evaluated": sorted(k_mean_scores.keys()),
            "k_cv_mean_balanced_accuracy": k_mean_scores,
            "k_cv_fold_scores": {k: v for k, v in k_cv_scores.items()},
        },
        "case_level_per_class": {
            "note": "PRIMARY results — bootstrapped by resampling cases (independent units)",
            "per_class": {cn: _san(m) for cn, m in case_per_class.items()} if case_per_class else None,
            "macro": case_avg_macro,
            "weighted": case_avg_weighted,
            "macro_ci_bootstrap_case": case_ci_macro,
        },
        "wsi_level_per_class": {
            "note": "SECONDARY/supplementary — bootstrapped by resampling WSIs",
            "per_class": {cn: _san(m) for cn, m in per_class.items()},
        },
        "averages": {
            "macro": avg_macro, "macro_ci": macro_ci,
            "weighted": avg_weighted,
        },
        "multiclass": {
            "confusion_matrix": cm_mc.tolist(),
            "classification_report": report_txt,
        },
        "slide_level_macro": {"values": slide_macro_vals, "ci_95": slide_macro_ci},
        "case_level_multiclass": case_level_summary,
        "stratified_by_source": {"wsi_level": stratified_wsi, "case_level": stratified_case},
        "her2_consistency_by_source": her2_consistency,
        "output_files": {"wsi_csv": wsi_csv, "case_csv": case_csv},
    }
    with open(os.path.join(output_dir, "metrics_test.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    LOGGER.info(f"Evaluation artefacts saved to: {output_dir}")


# ─────────────────────────────────────────────────────────────
#  Main driver
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="kNN with cosine similarity on mean-pooled slide features."
    )
    parser.add_argument("--mil_pkl",      required=True,
                        help="PKL with train/test splits (same format as LR_MeanPooling_OvR.py)")
    parser.add_argument("--output_dir",   required=True,
                        help="Where to save evaluation artefacts")
    parser.add_argument("--k_values",     type=int, nargs="+", default=DEFAULT_K_VALUES,
                        help="K values to evaluate in CV (default: 1 3 5 7 9 15 21)")
    parser.add_argument("--cv_folds",     type=int, default=5)
    parser.add_argument("--n_bootstrap",  type=int, default=2000)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    fh = logging.FileHandler(os.path.join(args.output_dir, "training.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(fh)

    LOGGER.info(f"Seed={args.seed}  K candidates={args.k_values}  CV folds={args.cv_folds}")

    # Load data
    X_train, y_train, X_test, y_test, class_names, ids_train, ids_test = \
        load_train_test_from_pkl(args.mil_pkl)

    n_classes = len(class_names)
    LOGGER.info(f"Train N={len(X_train)} | Test N={len(X_test) if X_test is not None else 0} | classes={class_names}")

    # L2-normalise embeddings (cosine similarity = dot product of unit vectors)
    X_train_norm = l2_normalize(X_train)
    X_test_norm  = l2_normalize(X_test) if X_test is not None else None

    # Select K by cross-validation on the training split
    best_k, k_cv_scores, k_mean_scores = select_k_by_cv(
        X_train_norm, y_train, n_classes,
        k_values=args.k_values,
        n_folds=args.cv_folds,
        seed=args.seed,
    )
    LOGGER.info(f"Selected K={best_k} (CV balanced accuracy = {k_mean_scores[best_k]:.4f})")

    # Save K-selection summary
    with open(os.path.join(args.output_dir, "k_selection_summary.json"), "w") as f:
        json.dump({
            "best_k":             best_k,
            "k_values":           args.k_values,
            "k_mean_scores":      k_mean_scores,
            "k_fold_scores":      k_cv_scores,
        }, f, indent=2)

    # Predict on test set
    if X_test_norm is not None and y_test is not None:
        LOGGER.info(f"\nPredicting on test set with K={best_k} ...")
        all_probs = knn_predict_proba(
            X_train_norm, y_train, X_test_norm, n_classes, k=best_k
        )

        eval_dir = os.path.join(args.output_dir, "eval_test")
        ensure_dir(eval_dir)
        evaluate_on_test_knn(
            all_probs, np.asarray(y_test, dtype=int),
            ids_test, class_names,
            output_dir=eval_dir,
            best_k=best_k,
            k_cv_scores=k_cv_scores,
            k_mean_scores=k_mean_scores,
            n_bootstrap=args.n_bootstrap,
        )
    else:
        LOGGER.warning("No test split found in PKL; skipping evaluation.")

    print(f"\n[DONE] All artefacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
