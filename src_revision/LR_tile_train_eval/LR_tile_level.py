#!/usr/bin/env python3
"""
Tile-level Logistic Regression (One-vs-Rest) with Oversampling and Rich Evaluation.

Pipeline:
1. Load MIL PKL (same file used by Attention_based_MIL.py) — each bag is one slide
   with [n_tiles, feat_dim] features.
2. Flatten bags to tiles: every tile inherits its slide's class label (no tile-level
   labels exist; this is weakly-supervised training).
3. Build patient-level 5-fold CV splits: all tiles from the same patient stay in the
   same fold to prevent any patient-level data leakage.
4. For each class k train one binary OvR Logistic Regression inside an imblearn
   Pipeline with RandomOverSampler.  GridSearchCV tunes C, class_weight, solver.
5. Predict on test tiles: each OvR classifier scores tiles → argmax gives the class.
6. Optional slide-level aggregation: mean-pool tile probabilities per slide.

Key features:
- NO attention mechanism; pure sklearn LogisticRegression on tile features.
- Patient-level CV splits (TCGA/CPTAC/Warwick patient ID extraction).
- Oversampling via imblearn Pipeline applied only inside training folds.
- Comprehensive metrics: accuracy, balanced accuracy, macro/weighted F1, Cohen's kappa, MCC.
- ROC and PR curves, confusion matrix, CV stability/overfitting plots, confidence analysis.

Usage (MIL PKL — preferred, same split as Attention_based_MIL.py):
  python LR_tile_level.py --seed 42 --mil_pkl path/to/mil_data_seed_42.pkl \\
                           --output_dir results --eval_level both

Usage (legacy dict-paths format):
  python LR_tile_level.py --seed 42 --data_dir path/to/splits \\
                           --output_dir results --eval_level both

Author: Konstantinos Papagoras
Date: 2025-07 / updated 2026-06
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from collections import Counter, defaultdict

import warnings

warnings.filterwarnings("ignore")


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


def setup_logging(results_dir: str) -> logging.Logger:
    logger = logging.getLogger("logreg_ovr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    os.makedirs(results_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(results_dir, "run.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str, title: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_cv_error_analysis(cv_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract per-fold train/test scores for the best params and compute stability metrics.
    cv_results is sklearn GridSearchCV.cv_results_ dict.
    """
    # Best param index
    best_idx = int(np.argmin(cv_results["rank_test_score"]))

    # Collect split keys
    split_test_keys = sorted([k for k in cv_results if k.startswith("split") and k.endswith("_test_score")])
    split_train_keys = sorted([k for k in cv_results if k.startswith("split") and k.endswith("_train_score")])

    test_scores = [float(cv_results[k][best_idx]) for k in split_test_keys]
    train_scores = [float(cv_results[k][best_idx]) for k in split_train_keys] if split_train_keys else []

    cv_scores = np.array(test_scores, dtype=float)
    train_scores_arr = np.array(train_scores, dtype=float) if len(train_scores) > 0 else None

    mean_cv = float(np.mean(cv_scores))
    std_cv = float(np.std(cv_scores))
    stats = {
        "mean_cv_score": mean_cv,
        "std_cv_score": std_cv,
        "min_cv_score": float(np.min(cv_scores)),
        "max_cv_score": float(np.max(cv_scores)),
        "cv_score_range": float(np.max(cv_scores) - np.min(cv_scores)),
        "cv_scores": cv_scores.tolist(),
        "mean_train_score": float(np.mean(train_scores_arr)) if train_scores_arr is not None else None,
        "std_train_score": float(np.std(train_scores_arr)) if train_scores_arr is not None else None,
        "train_scores": train_scores_arr.tolist() if train_scores_arr is not None else [],
        "overfitting_gap": float(np.mean(train_scores_arr) - mean_cv) if train_scores_arr is not None else None,
        "stability_score": float(1 - (std_cv / mean_cv)) if mean_cv > 0 else 0.0,
    }
    return stats


def plot_cv_analysis(cv_analysis: Dict[str, Dict[str, Any]], class_names: List[str], out_path: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Boxplot of CV scores
    scores = [cv_analysis[c]["cv_scores"] for c in class_names]
    axes[0, 0].boxplot(scores, labels=class_names)
    axes[0, 0].set_title("CV Score Distribution by Class", fontweight="bold")
    axes[0, 0].set_ylabel("Balanced Accuracy")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # Train vs CV mean
    train_means = [cv_analysis[c].get("mean_train_score") for c in class_names]
    cv_means = [cv_analysis[c]["mean_cv_score"] for c in class_names]
    x = np.arange(len(class_names))
    w = 0.35
    axes[0, 1].bar(x - w / 2, train_means, w, label="Train", alpha=0.8)
    axes[0, 1].bar(x + w / 2, cv_means, w, label="CV", alpha=0.8)
    axes[0, 1].set_title("Train vs CV Scores", fontweight="bold")
    axes[0, 1].set_ylabel("Balanced Accuracy")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Overfitting gap
    gaps = [cv_analysis[c].get("overfitting_gap", 0.0) or 0.0 for c in class_names]
    colors = ["red" if g > 0.1 else "orange" if g > 0.05 else "green" for g in gaps]
    axes[1, 0].bar(class_names, gaps, color=colors, alpha=0.7)
    axes[1, 0].set_title("Overfitting Analysis (Train - CV)", fontweight="bold")
    axes[1, 0].set_ylabel("Score Gap")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].axhline(0.1, color="red", linestyle="--", alpha=0.5, label="High")
    axes[1, 0].axhline(0.05, color="orange", linestyle="--", alpha=0.5, label="Moderate")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Stability
    stabilities = [cv_analysis[c]["stability_score"] for c in class_names]
    stds = [cv_analysis[c]["std_cv_score"] for c in class_names]
    axes[1, 1].scatter(stds, stabilities, s=100, alpha=0.7)
    for i, c in enumerate(class_names):
        axes[1, 1].annotate(c, (stds[i], stabilities[i]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    axes[1, 1].set_title("Model Stability", fontweight="bold")
    axes[1, 1].set_xlabel("CV Std")
    axes[1, 1].set_ylabel("Stability Score")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    conf_analysis: Dict[str, Any] = {}
    for i, cname in enumerate(class_names):
        m = y_true == i
        if np.any(m):
            p = y_proba[m, i]
            conf_analysis[cname] = {
                "mean_confidence": float(np.mean(p)),
                "std_confidence": float(np.std(p)),
                "min_confidence": float(np.min(p)),
                "max_confidence": float(np.max(p)),
            }

    return {
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "per_class_metrics": report,
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "confidence_analysis": conf_analysis,
    }


def plot_confidence_analysis(conf_analysis: Dict[str, Any], out_path: str, title: str) -> None:
    class_names = list(conf_analysis.keys())
    means = [conf_analysis[c]["mean_confidence"] for c in class_names]
    stds = [conf_analysis[c]["std_confidence"] for c in class_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    bars = ax1.bar(class_names, means, yerr=stds, capsize=5, alpha=0.7)
    ax1.set_title("Mean Prediction Confidence by Class", fontweight="bold")
    ax1.set_ylabel("Mean Confidence")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)
    for bar, conf in zip(bars, means):
        ax1.annotate(f"{conf:.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom")

    for i, c in enumerate(class_names):
        ax2.errorbar(i, means[i], yerr=stds[i], fmt="o", capsize=5, capthick=2, label=c)
    ax2.set_title("Confidence Distribution by Class", fontweight="bold")
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Class")
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha="right")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_threshold_analysis(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, class_names: List[str]) -> Tuple[Dict[float, Any], Dict[str, Any]]:
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results: Dict[float, Any] = {}
    base_acc = accuracy_score(y_true, y_pred)

    max_conf = np.max(y_proba, axis=1)
    for t in thresholds:
        keep = max_conf >= t
        if np.sum(keep) == 0:
            continue
        acc_k = accuracy_score(y_true[keep], y_pred[keep])
        ba_k = balanced_accuracy_score(y_true[keep], y_pred[keep])
        results[t] = {
            "samples_kept": int(np.sum(keep)),
            "samples_rejected": int(len(y_true) - np.sum(keep)),
            "rejection_rate": float((len(y_true) - np.sum(keep)) / len(y_true)),
            "accuracy": float(acc_k),
            "balanced_accuracy": float(ba_k),
            "improvement": float(acc_k - base_acc),
        }

    per_class: Dict[str, Any] = {}
    for i, cname in enumerate(class_names):
        m = y_true == i
        if not np.any(m):
            continue
        p = y_proba[m, i]
        pred_is_i = (y_pred == i)[m]
        best_t, best_prec = 0.5, 0.0
        for t in np.arange(0.1, 1.0, 0.1):
            high = p >= t
            if np.sum(high) == 0:
                continue
            prec = float(np.mean(pred_is_i[high]))
            if prec > best_prec:
                best_prec, best_t = prec, float(t)
        per_class[cname] = {
            "optimal_threshold": best_t,
            "precision_at_threshold": best_prec,
            "samples_above_threshold": int(np.sum(p >= best_t)),
        }

    return results, per_class


def load_train_test_from_mil_pkl(pkl_path: str):
    """
    Load train/test bags from a MIL PKL (same format as Attention_based_MIL.py).
    Supports train_bags/train_labels, X_train/y_train, or nested train/test dict schemas.
    Returns (X_train_bags, y_train, X_test_bags, y_test, class_names, ids_train, ids_test).
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    class_names_pkl = None
    if "class_names" in data and isinstance(data["class_names"], (list, tuple)):
        class_names_pkl = list(data["class_names"])
    elif "label_encoder" in data and hasattr(data["label_encoder"], "classes_"):
        class_names_pkl = list(data["label_encoder"].classes_)

    ids_train = ids_test = None

    if "train_bags" in data and "train_labels" in data:
        X_tr = data["train_bags"]; y_tr = np.asarray(data["train_labels"], dtype=int)
        X_te = data.get("test_bags")
        y_te = np.asarray(data["test_labels"], dtype=int) if "test_labels" in data else None
        ids_train = data.get("train_ids") or data.get("ids_train")
        ids_test  = data.get("test_ids")  or data.get("ids_test")
        return X_tr, y_tr, X_te, y_te, class_names_pkl, ids_train, ids_test

    if "X_train" in data and "y_train" in data:
        X_tr = data["X_train"]; y_tr = np.asarray(data["y_train"], dtype=int)
        X_te = data.get("X_test")
        y_te = np.asarray(data["y_test"], dtype=int) if "y_test" in data else None
        ids_train = data.get("train_ids") or data.get("ids_train")
        ids_test  = data.get("test_ids")  or data.get("ids_test")
        return X_tr, y_tr, X_te, y_te, class_names_pkl, ids_train, ids_test

    if "train" in data and isinstance(data["train"], dict):
        X_tr = data["train"]["bags"]; y_tr = np.asarray(data["train"]["labels"], dtype=int)
        X_te = data["test"]["bags"];  y_te = np.asarray(data["test"]["labels"], dtype=int)
        ids_train = data["train"].get("ids"); ids_test = data["test"].get("ids")
        return X_tr, y_tr, X_te, y_te, class_names_pkl, ids_train, ids_test

    raise KeyError(f"Unrecognized MIL PKL format in {pkl_path}.")


def flatten_bags_to_tiles(
    bags, labels: np.ndarray, slide_ids=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten MIL bags (list of [n_tiles, feat_dim] arrays) to tile-level arrays.
    Each tile inherits the label and slide ID of its parent bag.
    Returns (X_tiles, y_tiles, tile_slide_ids).
    """
    X_list, y_list, ids_list = [], [], []
    for i, (bag, label) in enumerate(zip(bags, labels)):
        bag_arr = np.asarray(bag, dtype=np.float32)
        n_t = bag_arr.shape[0]
        X_list.append(bag_arr)
        y_list.extend([int(label)] * n_t)
        sid = str(slide_ids[i]) if slide_ids is not None else f"slide_{i:05d}"
        ids_list.extend([sid] * n_t)
    X = np.vstack(X_list) if X_list else np.zeros((0, 0), dtype=np.float32)
    return X, np.array(y_list, dtype=int), np.array(ids_list)


def aggregate_tile_predictions_to_slide(
    y_true_tiles: np.ndarray,
    y_proba_tiles: np.ndarray,
    tile_slide_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Aggregate pre-computed tile probabilities to slide level via mean pooling.
    Works with both dict-paths and MIL PKL formats (no re-inference required).
    Returns (y_true_slide, y_pred_slide, y_proba_slide, details_dict).
    """
    unique_slides = np.unique(tile_slide_ids)
    slide_true_list, slide_pred_list, slide_proba_list = [], [], []
    details: List[Dict[str, Any]] = []

    for sid in unique_slides:
        mask = tile_slide_ids == sid
        slide_proba = y_proba_tiles[mask].mean(axis=0)
        slide_pred = int(np.argmax(slide_proba))
        slide_true = int(Counter(y_true_tiles[mask].tolist()).most_common(1)[0][0])
        slide_true_list.append(slide_true)
        slide_pred_list.append(slide_pred)
        slide_proba_list.append(slide_proba)

        if slide_pred != slide_true:
            tile_preds_local = np.argmax(y_proba_tiles[mask], axis=1)
            mis_idx = np.where(tile_preds_local != slide_true)[0]
            details.append({
                "slide_id": str(sid),
                "n_tiles": int(mask.sum()),
                "n_misclassified_tiles": int(len(mis_idx)),
                "misclassified_tile_indices": mis_idx.tolist(),
            })

    y_true_s = np.array(slide_true_list, dtype=int)
    y_pred_s = np.array(slide_pred_list, dtype=int)
    y_proba_s = (np.vstack(slide_proba_list) if slide_proba_list
                 else np.zeros((0, y_proba_tiles.shape[1])))
    return y_true_s, y_pred_s, y_proba_s, {"misclassified_slides_details": details}


def load_tile_features(dict_paths: List[str], slide_labels: List[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y, slide_ids, tile_slide_idx = [], [], [], []
    for i, (dict_path, slide_label) in enumerate(zip(dict_paths, slide_labels)):
        with open(dict_path, "rb") as f:
            tile_dict = pickle.load(f)
        for tile in tile_dict.values():
            X.append(tile["feature"])
            y.append(slide_label)
            slide_ids.append(os.path.basename(dict_path).replace(".pkl", ""))
            tile_slide_idx.append(i)
    return np.array(X), np.array(y), np.array(slide_ids), np.array(tile_slide_idx)


def _patient_train_cal_split(
    patient_slides: Dict[str, List[str]],
    unique_patients: List[str],
    patient_labels: np.ndarray,
    cal_size: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Patient-level stratified split into fit (1-cal_size) and calibration (cal_size) sets.
    Mirrors the MIL 80/20 patient split: all slides from the same patient stay together.
    Falls back to unstratified split if any binary class has too few patients.
    Returns (fit_slide_ids, cal_slide_ids).
    """
    try:
        fit_pats, cal_pats = train_test_split(
            unique_patients, test_size=cal_size, random_state=seed, stratify=patient_labels
        )
    except ValueError:
        fit_pats, cal_pats = train_test_split(
            unique_patients, test_size=cal_size, random_state=seed
        )
    fit_slides = [s for p in fit_pats for s in patient_slides[p]]
    cal_slides  = [s for p in cal_pats  for s in patient_slides[p]]
    return fit_slides, cal_slides


def _find_temperature(
    pipe: ImbPipeline,
    X_cal: np.ndarray,
    tile_slide_ids_cal: np.ndarray,
    y_cal_bin: np.ndarray,
    n_grid: int = 200,
) -> float:
    """
    Temperature scaling on slide-level aggregated logits — identical to the MIL approach.

    For each calibration slide, tile decision-function logits are mean-aggregated.
    T is grid-searched in [0.1, 10.0] to minimise binary cross-entropy:
        p_cal = sigmoid(mean_logit / T).
    Returns T=1.0 (no scaling) if calibration is not possible.
    """
    if len(X_cal) == 0:
        return 1.0

    tile_logits = pipe.decision_function(X_cal)   # (n_cal_tiles,) uncalibrated log-odds

    slide_logits: List[float] = []
    slide_labels: List[int] = []
    for sid in np.unique(tile_slide_ids_cal):
        m = tile_slide_ids_cal == sid
        slide_logits.append(float(np.mean(tile_logits[m])))
        slide_labels.append(int(y_cal_bin[m][0]))

    z = np.array(slide_logits, dtype=np.float64)
    y = np.array(slide_labels, dtype=int)

    if len(np.unique(y)) < 2:
        return 1.0

    def _bce(T: float) -> float:
        p = np.clip(1.0 / (1.0 + np.exp(-z / T)), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    grid = np.linspace(0.1, 10.0, n_grid)
    T_opt = float(grid[int(np.argmin([_bce(float(T)) for T in grid]))])
    return T_opt


def aggregate_slide_predictions(
    dict_paths: List[str],
    slide_labels: List[Any],
    le: LabelEncoder,
    classifiers: List[ImbPipeline],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    class_count = len(le.classes_)
    slide_preds, slide_true = [], []
    details: List[Dict[str, Any]] = []

    for dict_path, slide_label in zip(dict_paths, slide_labels):
        with open(dict_path, "rb") as f:
            tile_dict = pickle.load(f)
        tile_feats = np.array([t["feature"] for t in tile_dict.values()])
        if tile_feats.size == 0:
            continue
        tile_probs = np.zeros((tile_feats.shape[0], class_count), dtype=float)
        for class_idx, clf in enumerate(classifiers):
            tile_probs[:, class_idx] = clf.predict_proba(tile_feats)[:, 1]
        tile_preds = np.argmax(tile_probs, axis=1)

        slide_pred = int(np.bincount(tile_preds).argmax())
        slide_true_label = int(le.transform([slide_label])[0])

        slide_id = os.path.basename(dict_path).replace(".pkl", "")
        slide_preds.append(slide_pred)
        slide_true.append(slide_true_label)

        if slide_pred != slide_true_label:
            mis_idx = np.where(tile_preds != slide_true_label)[0]
            details.append(
                {
                    "slide_id": slide_id,
                    "n_tiles": int(tile_feats.shape[0]),
                    "n_misclassified_tiles": int(len(mis_idx)),
                    "misclassified_tile_indices": mis_idx.tolist(),
                }
            )

    return np.array(slide_true), np.array(slide_preds), {"misclassified_slides_details": details}


def train_logistic_regression_ovr(
    seed: int,
    output_dir: str,
    eval_level: str = "tile",  # 'tile' | 'slide' | 'both'
    data_dir: Optional[str] = None,
    mil_pkl: Optional[str] = None,
    cal_size: float = 0.20,
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"Enhanced_LogReg_OvR_seed_{seed}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logging(results_dir)

    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    # ── Data loading: MIL PKL (same split as Attention_based_MIL) or dict-paths ──
    if mil_pkl is not None:
        logger.info(f"Loading from MIL PKL: {mil_pkl}")
        bags_tr, y_bags_tr, bags_te, y_bags_te, class_names_pkl, ids_tr, ids_te = \
            load_train_test_from_mil_pkl(mil_pkl)
        if bags_te is None or y_bags_te is None:
            logger.error("MIL PKL does not contain a test split.")
            sys.exit(1)
        X_train, y_train_enc, train_tile_slide_ids = flatten_bags_to_tiles(bags_tr, y_bags_tr, ids_tr)
        X_test,  y_test_enc,  test_tile_slide_ids  = flatten_bags_to_tiles(bags_te, y_bags_te, ids_te)
        # Free the raw PKL bags — X_train/X_test already hold float32 copies.
        del bags_tr, y_bags_tr, ids_tr, bags_te, y_bags_te, ids_te
        # Labels are already integer-encoded; build a minimal LabelEncoder for compatibility
        le = LabelEncoder()
        le.fit(np.arange(int(y_train_enc.max()) + 1))
        class_names = class_names_pkl if class_names_pkl is not None else [str(c) for c in le.classes_]
        n_classes = len(class_names)
        split_pkl_path = mil_pkl
    else:
        split_pkl_path = os.path.join(data_dir, f"split_seed_{seed}.pkl")
        if not os.path.exists(split_pkl_path):
            logger.error(f"Data file not found: {split_pkl_path}")
            sys.exit(1)
        logger.info(f"Loading split file: {split_pkl_path}")
        with open(split_pkl_path, "rb") as f:
            split_data = pickle.load(f)
        train_dict_paths: List[str] = split_data["train_dict_paths"]
        train_labels: List[Any] = split_data["train_labels"]
        test_dict_paths: List[str] = split_data["test_dict_paths"]
        test_labels: List[Any] = split_data["test_labels"]
        X_train, y_train, train_tile_slide_ids, _ = load_tile_features(train_dict_paths, train_labels)
        X_test,  y_test,  test_tile_slide_ids,  _ = load_tile_features(test_dict_paths,  test_labels)
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)
        class_names = list(le.classes_)
        n_classes   = len(class_names)

    # Ensure float32 throughout — lbfgs accepts float32 natively (no internal upcast),
    # halving working memory vs float64.  PKL bags may be stored as float64, so enforce here.
    if X_train.dtype != np.float32:
        logger.info(f"Casting X_train from {X_train.dtype} to float32")
        X_train = X_train.astype(np.float32)
    if X_test.dtype != np.float32:
        logger.info(f"Casting X_test from {X_test.dtype} to float32")
        X_test = X_test.astype(np.float32)
    logger.info(f"Feature dtype: {X_train.dtype} — X_train {X_train.nbytes/2**30:.1f} GB, X_test {X_test.nbytes/2**30:.1f} GB")

    # Build patient metadata — all slides from the same patient stay together.
    unique_slide_ids, first_tile_idx = np.unique(train_tile_slide_ids, return_index=True)
    slide_label_map = {sid: int(y_train_enc[first_tile_idx[i]]) for i, sid in enumerate(unique_slide_ids)}
    patient_slides: Dict[str, List[str]] = defaultdict(list)
    for sid in unique_slide_ids:
        patient_slides[get_patient_id(sid)].append(sid)
    unique_patients: List[str] = sorted(patient_slides.keys())

    # C=0.01 is consistent across multiple seeds and datasets; CV on tile-level data
    # yields ~0.50 balanced accuracy for all C values (weak labels → noise), so fixing
    # the parameter is more reliable than noisy grid search.
    FIXED_C: float = 0.01

    logger.info(f"Training tiles: {X_train.shape}, Test tiles: {X_test.shape}, Classes: {class_names}")
    logger.info(
        f"Fixed params: C={FIXED_C}, class_weight='balanced', solver='lbfgs' | "
        f"Calibration: {cal_size:.0%} patient-level holdout per OvR class (temperature scaling)"
    )

    best_classifiers: List[ImbPipeline] = []
    temperatures: List[float] = []
    cal_info_per_class: List[Dict[str, Any]] = []

    # Create models_dir early so each classifier can be saved immediately after training.
    # This ensures partial results survive an OOM kill mid-loop.
    models_dir = os.path.join(results_dir, "trained_models")
    os.makedirs(models_dir, exist_ok=True)

    # One-vs-rest training loop
    for class_idx, class_name in enumerate(class_names):
        logger.info(f"Training OvR classifier for class: {class_name}")
        y_train_bin = (y_train_enc == class_idx).astype(int)

        # Patient-level 80/20 fit/calibration split, stratified by binary OvR label.
        # Mirrors the MIL 80/10/10 approach (no val needed for LR — no early stopping).
        patient_bin_labels = np.array([
            Counter([int(slide_label_map[s] == class_idx) for s in patient_slides[p]]
                    ).most_common(1)[0][0]
            for p in unique_patients
        ])
        fit_slides, cal_slides = _patient_train_cal_split(
            patient_slides, unique_patients, patient_bin_labels, cal_size, seed
        )

        fit_mask = np.isin(train_tile_slide_ids, fit_slides)
        cal_mask = np.isin(train_tile_slide_ids, cal_slides)
        X_fit    = X_train[fit_mask]
        y_fit_bin = y_train_bin[fit_mask]

        original_dist = np.bincount(y_fit_bin).tolist()
        logger.info(
            f"  [{class_name}] fit={fit_mask.sum():,} tiles ({len(fit_slides)} slides), "
            f"cal={cal_mask.sum():,} tiles ({len(cal_slides)} slides) | "
            f"class dist (fit): {original_dist}"
        )

        pipe = ImbPipeline(
            steps=[
                ("sampler", RandomOverSampler(random_state=seed)),
                ("clf", LogisticRegression(
                    C=FIXED_C, class_weight="balanced", solver="lbfgs",
                    max_iter=2000, random_state=seed, n_jobs=1,
                )),
            ]
        )
        pipe.fit(X_fit, y_fit_bin)

        # Temperature scaling on calibration slides — identical to MIL approach.
        # Tile logits are mean-aggregated per slide; T minimises binary cross-entropy.
        T = _find_temperature(
            pipe,
            X_train[cal_mask],
            train_tile_slide_ids[cal_mask],
            y_train_bin[cal_mask],
        )
        logger.info(f"  [{class_name}] Temperature T={T:.4f}")

        best_classifiers.append(pipe)
        temperatures.append(T)
        cal_info_per_class.append({
            "class_name": class_name,
            "temperature": T,
            "fixed_C": FIXED_C,
            "n_fit_tiles": int(fit_mask.sum()),
            "n_cal_tiles": int(cal_mask.sum()),
            "n_fit_slides": len(fit_slides),
            "n_cal_slides": len(cal_slides),
            "class_dist_fit": original_dist,
        })

        # Save immediately so a crash on a later class doesn't lose this one.
        model_path = os.path.join(models_dir, f"classifier_{class_idx}_{class_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"pipeline": pipe, "temperature": T, "class_name": class_name}, f)
        logger.info(f"  Saved [{class_name}] -> {model_path}")

    # Predict on test tiles with temperature-scaled calibrated probabilities.
    # sigmoid(logit / T) — identical to the MIL temperature scaling.
    y_proba_tile = np.zeros((X_test.shape[0], n_classes), dtype=float)
    for class_idx, (clf, T) in enumerate(zip(best_classifiers, temperatures)):
        logits = clf.decision_function(X_test)           # (n_test_tiles,)
        y_proba_tile[:, class_idx] = 1.0 / (1.0 + np.exp(-logits / T))
    y_pred_tile = np.argmax(y_proba_tile, axis=1)

    # Tile-level metrics
    tile_metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test_enc, y_pred_tile)),
        "macro_f1": float(f1_score(y_test_enc, y_pred_tile, average="macro")),
        "weighted_f1": float(f1_score(y_test_enc, y_pred_tile, average="weighted")),
        "accuracy": float(accuracy_score(y_test_enc, y_pred_tile)),
    }
    add_tile = calculate_additional_metrics(y_test_enc, y_pred_tile, y_proba_tile, class_names)

    # ROC and PR per class (tile-level)
    y_test_bin = label_binarize(y_test_enc, classes=list(range(n_classes)))
    roc_auc_per_class: Dict[str, float] = {}
    avg_precision_per_class: Dict[str, float] = {}
    for i in range(n_classes):
        roc_auc_per_class[class_names[i]] = float(roc_auc_score(y_test_bin[:, i], y_proba_tile[:, i]))
        avg_precision_per_class[class_names[i]] = float(average_precision_score(y_test_bin[:, i], y_proba_tile[:, i]))

    cm_tile = confusion_matrix(y_test_enc, y_pred_tile)

    # Optional slide-level aggregation (mean-pool tile probs; works for both data formats)
    slide_results: Dict[str, Any] = {}
    if eval_level in ("slide", "both"):
        y_true_slide, y_pred_slide, y_proba_slide, slide_details = \
            aggregate_tile_predictions_to_slide(y_test_enc, y_proba_tile, test_tile_slide_ids)
        if len(y_true_slide) > 0:
            slide_metrics = {
                "balanced_accuracy": float(balanced_accuracy_score(y_true_slide, y_pred_slide)),
                "macro_f1": float(f1_score(y_true_slide, y_pred_slide, average="macro")),
                "weighted_f1": float(f1_score(y_true_slide, y_pred_slide, average="weighted")),
                "accuracy": float(accuracy_score(y_true_slide, y_pred_slide)),
            }
            add_slide = {
                "cohen_kappa": float(cohen_kappa_score(y_true_slide, y_pred_slide)),
                "matthews_corrcoef": float(matthews_corrcoef(y_true_slide, y_pred_slide)),
                "classification_report": classification_report(y_true_slide, y_pred_slide, target_names=class_names, output_dict=True),
            }
            cm_slide = confusion_matrix(y_true_slide, y_pred_slide)
            slide_results = {
                "y_true": y_true_slide.tolist(),
                "y_pred": y_pred_slide.tolist(),
                "metrics": slide_metrics,
                "additional": add_slide,
                "confusion_matrix": cm_slide.tolist(),
                "details": slide_details,
            }
        else:
            logging.warning("No slide-level data available for aggregation.")

    # Threshold analysis on tile-level
    threshold_analysis, class_thresholds = calculate_threshold_analysis(y_test_enc, y_pred_tile, y_proba_tile, class_names)

    # Plots
    plot_confusion_matrix(
        cm_tile,
        class_names,
        out_path=os.path.join(results_dir, "confusion_matrix_tile.png"),
        title=f"Confusion Matrix (Tile) - Seed {seed}",
    )
    plot_confidence_analysis(
        add_tile["confidence_analysis"],
        out_path=os.path.join(results_dir, "confidence_analysis_tile.png"),
        title=f"Confidence Analysis (Tile) - Seed {seed}",
    )

    # ROC curves (tile)
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_tile[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc_per_class[class_names[i]]:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (Tile, OvR) - Seed {seed}")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "roc_auc_tile.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Precision-Recall curves (tile)
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba_tile[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={avg_precision_per_class[class_names[i]]:.3f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves (Tile, OvR) - Seed {seed}")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "precision_recall_tile.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save report (text)
    results_txt_path = os.path.join(results_dir, "comprehensive_results.txt")
    with open(results_txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ENHANCED LOGISTIC REGRESSION (OvR) - SEED {seed}\n")
        f.write("=" * 80 + "\n\n")
        f.write("EXPERIMENT INFO:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Data split file: {split_pkl_path}\n")
        f.write(f"Results directory: {results_dir}\n\n")

        f.write("DATA INFO:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Training tiles: {len(X_train)}\n")
        f.write(f"Test tiles: {len(X_test)}\n")
        f.write(f"Features per tile: {X_train.shape[1]}\n")
        f.write(f"Classes ({n_classes}): {class_names}\n\n")

        f.write("TILE-LEVEL METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Balanced Accuracy: {tile_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1: {tile_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {tile_metrics['weighted_f1']:.4f}\n")
        f.write(f"Accuracy: {tile_metrics['accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa: {add_tile['cohen_kappa']:.4f}\n")
        f.write(f"Matthews Corrcoef: {add_tile['matthews_corrcoef']:.4f}\n\n")

        f.write("PER-CLASS (Tile) REPORT:\n")
        f.write("-" * 40 + "\n")
        rep = add_tile["per_class_metrics"]
        f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}\n")
        f.write("-" * 70 + "\n")
        for cname in class_names:
            if cname in rep:
                m = rep[cname]
                f.write(f"{cname:<20} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1-score']:<10.4f} {m['support']:<10.0f}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Macro Avg':<20} {rep['macro avg']['precision']:<10.4f} {rep['macro avg']['recall']:<10.4f} {rep['macro avg']['f1-score']:<10.4f} {rep['macro avg']['support']:<10.0f}\n")
        f.write(f"{'Weighted Avg':<20} {rep['weighted avg']['precision']:<10.4f} {rep['weighted avg']['recall']:<10.4f} {rep['weighted avg']['f1-score']:<10.4f} {rep['weighted avg']['support']:<10.0f}\n\n")

        f.write("ROC AUC (Tile) PER CLASS:\n")
        f.write("-" * 40 + "\n")
        for cname in class_names:
            f.write(f"{cname:<20}: {roc_auc_per_class[cname]:.4f}\n")
        f.write("\n")

        f.write("AVERAGE PRECISION (Tile) PER CLASS:\n")
        f.write("-" * 40 + "\n")
        for cname in class_names:
            f.write(f"{cname:<20}: {avg_precision_per_class[cname]:.4f}\n")
        f.write("\n")

        f.write("CONFUSION MATRIX (Tile):\n")
        f.write("-" * 40 + "\n")
        f.write("Rows=True, Cols=Pred\n")
        f.write(f"{'':>15}" + "".join([f"{c:>10}" for c in class_names]) + "\n")
        for i, cname in enumerate(class_names):
            f.write(f"{cname:>15}" + "".join([f"{cm_tile[i][j]:>10}" for j in range(n_classes)]) + "\n")
        f.write("\n")

        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  C (fixed)      : {FIXED_C}\n")
        f.write(f"  class_weight   : balanced\n")
        f.write(f"  solver         : lbfgs\n")
        f.write(f"  calibration    : temperature scaling (MIL-identical)\n")
        f.write(f"  cal_size       : {cal_size:.0%} of training patients (patient-level holdout)\n\n")

        f.write("TEMPERATURE CALIBRATION (per OvR class):\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<20} {'T':<8} {'Fit tiles':<14} {'Cal tiles':<14} {'Cal slides':<12}\n")
        f.write("-" * 70 + "\n")
        for c in cal_info_per_class:
            f.write(
                f"{c['class_name']:<20} {c['temperature']:<8.4f} "
                f"{c['n_fit_tiles']:<14,} {c['n_cal_tiles']:<14,} {c['n_cal_slides']:<12}\n"
            )
        f.write("\n")

        f.write("CONFIDENCE THRESHOLD ANALYSIS (Tile):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Threshold':<12} {'Kept':<8} {'Rejected':<10} {'Reject%':<10} {'Acc':<10} {'Improve':<10}\n")
        f.write("-" * 80 + "\n")
        for t, an in threshold_analysis.items():
            f.write(
                f"{t:<12.1f} {an['samples_kept']:<8} {an['samples_rejected']:<10} "
                f"{an['rejection_rate']*100:<10.1f} {an['accuracy']:<10.4f} {an['improvement']:<10.4f}\n"
            )
        f.write("\n")
        f.write("OPTIMAL THRESHOLDS PER CLASS (Tile):\n")
        f.write("-" * 40 + "\n")
        for cname, th in class_thresholds.items():
            f.write(f"{cname}:\n")
            f.write(f"  Optimal threshold: {th['optimal_threshold']:.2f}\n")
            f.write(f"  Precision at threshold: {th['precision_at_threshold']:.4f}\n")
            f.write(f"  Samples above threshold: {th['samples_above_threshold']}\n\n")

        if eval_level in ("slide", "both") and slide_results:
            f.write("SLIDE-LEVEL METRICS (Majority Vote):\n")
            f.write("-" * 40 + "\n")
            sm = slide_results["metrics"]
            ad = slide_results["additional"]
            f.write(f"Balanced Accuracy: {sm['balanced_accuracy']:.4f}\n")
            f.write(f"Macro F1: {sm['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {sm['weighted_f1']:.4f}\n")
            f.write(f"Accuracy: {sm['accuracy']:.4f}\n")
            f.write(f"Cohen's Kappa: {ad['cohen_kappa']:.4f}\n")
            f.write(f"Matthews Corrcoef: {ad['matthews_corrcoef']:.4f}\n\n")
            f.write("CONFUSION MATRIX (Slide):\n")
            f.write("-" * 40 + "\n")
            cm_slide = np.array(slide_results["confusion_matrix"])
            f.write(f"{'':>15}" + "".join([f"{c:>10}" for c in class_names]) + "\n")
            for i, cname in enumerate(class_names):
                f.write(f"{cname:>15}" + "".join([f"{cm_slide[i][j]:>10}" for j in range(n_classes)]) + "\n")
            f.write("\n")
            f.write("MISCLASSIFIED SLIDE DETAILS:\n")
            f.write("-" * 40 + "\n")
            details = slide_results["details"]["misclassified_slides_details"]
            f.write(f"Total misclassified slides: {len(details)}\n")
            for d in details:
                f.write(
                    f"Slide {d['slide_id']}: n_tiles={d['n_tiles']}, n_misclassified_tiles={d['n_misclassified_tiles']}\n"
                )

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    # Save programmatic results
    results_pkl_path = os.path.join(results_dir, "enhanced_all_results.pkl")
    results_json_path = os.path.join(results_dir, "enhanced_all_results.json")
    results_dict: Dict[str, Any] = {
        "seed": seed,
        "timestamp": timestamp,
        "class_names": class_names,
        "tile": {
            "metrics": tile_metrics,
            "additional_metrics": add_tile,
            "confusion_matrix": cm_tile.tolist(),
            "roc_auc_per_class": roc_auc_per_class,
            "avg_precision_per_class": avg_precision_per_class,
            "predictions": {
                "y_true": y_test_enc.tolist(),
                "y_pred": y_pred_tile.tolist(),
                "y_proba": y_proba_tile.tolist(),
                "slide_ids_per_tile": test_tile_slide_ids.tolist(),
            },
            "threshold_analysis": threshold_analysis,
            "class_thresholds": class_thresholds,
        },
        "calibration": {
            "fixed_C": FIXED_C,
            "cal_size": cal_size,
            "temperatures": temperatures,
            "per_class": cal_info_per_class,
        },
        "paths": {"results_dir": results_dir, "split_file": split_pkl_path, "report_txt": results_txt_path},
    }
    if eval_level in ("slide", "both") and slide_results:
        results_dict["slide"] = slide_results

    with open(results_pkl_path, "wb") as f:
        pickle.dump(results_dict, f)
    with open(results_json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Report: {results_txt_path}")
    logger.info(f"Results (pkl): {results_pkl_path}")
    logger.info(f"Results (json): {results_json_path}")
    logger.info(f"Models: {models_dir}")

    return results_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Logistic Regression (OvR) with Oversampling")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--eval_level",
        type=str,
        default="tile",
        choices=["tile", "slide", "both"],
        help="Evaluation level to compute and save",
    )
    parser.add_argument(
        "--cal_size",
        type=float,
        default=0.20,
        help="Fraction of training patients held out for temperature calibration (default 0.20).",
    )
    # Mutually exclusive data sources
    data_grp = parser.add_mutually_exclusive_group(required=True)
    data_grp.add_argument("--mil_pkl", type=str,
                          help="Path to MIL PKL (same split as Attention_based_MIL.py)")
    data_grp.add_argument("--data_dir", type=str,
                          help="Directory containing split_seed_{seed}.pkl")
    args = parser.parse_args()

    train_logistic_regression_ovr(
        seed=args.seed,
        output_dir=args.output_dir,
        eval_level=args.eval_level,
        data_dir=args.data_dir,
        mil_pkl=args.mil_pkl,
        cal_size=args.cal_size,
    )


if __name__ == "__main__":
    main()
