#!/usr/bin/env python3
"""
Logistic Regression Calibration for WSI Classification

This script trains and calibrates One-vs-Rest (OvR) logistic regression classifiers
for whole slide image classification with proper handling of calibration data to avoid
data leakage.

Key Features:
- OvR (One-vs-Rest) multi-class classification
- Isotonic or Sigmoid calibration methods
- Held-out calibration split from training data
- Optional model retraining or loading pre-trained models
- SMOTE or RandomOverSampler for handling class imbalance
- Comprehensive metrics (accuracy, F1, AUC, calibration metrics)
- Detailed visualizations (ROC, PR curves, reliability diagrams)
- Bootstrap confidence intervals for metrics

The script ensures no data leakage by:
1. Splitting training data into fit/calibration sets
2. Training base models only on fit set
3. Calibrating on held-out calibration set
4. Evaluating on completely separate test set

Usage:
    # Retrain models with calibration
    python LR_callibration.py --seed 42 --data_dir ./data_splits --output_dir ./results --retrain
    
    # Load pre-trained models and calibrate
    python LR_callibration.py --seed 42 --data_dir ./data_splits --models_dir ./models --output_dir ./results

Author: Konstantnos Papagoras
Date: 2025-07
"""

import os
import sys
import re
import json
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    log_loss,
    roc_curve,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef
)
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline


# Configuration Constants
DEFAULT_CALIBRATION_SIZE = 0.2
DEFAULT_CALIBRATION_METHOD = "isotonic"
DEFAULT_N_BINS = 15
DEFAULT_N_BOOTSTRAP = 500
DEFAULT_CONFIDENCE_ALPHA = 0.05
DEFAULT_CV_FOLDS = 5

# ==================== Calibration Metrics ====================

def multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute multiclass Brier score.
    
    The Brier score measures the mean squared difference between predicted
    probabilities and the actual outcomes (one-hot encoded).
    
    Args:
        y_true: True class labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes)
        
    Returns:
        Brier score (lower is better, range [0, 2])
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    K = y_proba.shape[1]
    onehot = np.eye(K)[y_true]
    return float(np.mean(np.sum((y_proba - onehot) ** 2, axis=1)))

def top1_ece(y_true: np.ndarray, y_proba: np.ndarray, 
             n_bins: int = DEFAULT_N_BINS) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE) using top-1 predictions.
    
    ECE measures the difference between prediction confidence and actual accuracy,
    binned by confidence levels.
    
    Args:
        y_true: True class labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes)
        n_bins: Number of bins for calibration curve
        
    Returns:
        Tuple of (ece_score, bin_accuracies, bin_confidences, bin_counts, bin_edges)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    correct = (pred == y_true).astype(int)
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.digitize(conf, edges[1:-1], right=False)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = (ids == b)
        if m.any():
            bin_cnt[b] = m.sum()
            bin_acc[b] = correct[m].mean()
            bin_conf[b] = conf[m].mean()
    N = len(y_true)
    ece = np.sum(np.abs(bin_acc - bin_conf) * (bin_cnt / max(N, 1)))
    return float(ece), bin_acc, bin_conf, bin_cnt, edges

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba_uncal: np.ndarray,
    y_proba_cal: np.ndarray,
    out_path: str,
    n_bins: int = DEFAULT_N_BINS,
    title: str = "Reliability (Top-1)"
) -> Tuple[float, float]:
    """
    Plot reliability diagrams comparing uncalibrated vs calibrated predictions.
    
    Shows calibration curves with confidence vs accuracy for both uncalibrated
    and calibrated probability estimates.
    
    Args:
        y_true: True class labels
        y_proba_uncal: Uncalibrated predicted probabilities
        y_proba_cal: Calibrated predicted probabilities
        out_path: Output path for the plot
        n_bins: Number of bins for calibration
        title: Plot title
        
    Returns:
        Tuple of (uncalibrated_ece, calibrated_ece)
    """
    e_u, acc_u, conf_u, cnt_u, edges = top1_ece(y_true, y_proba_uncal, n_bins=n_bins)
    e_c, acc_c, conf_c, cnt_c, _     = top1_ece(y_true, y_proba_cal,  n_bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = (edges[1] - edges[0]) * 0.9

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(centers, acc_u, width=w, alpha=0.6, label='Empirical acc')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.plot(centers, conf_u, color='C1', lw=2, label='Mean conf')
    plt.title(f"Uncalibrated (ECE={e_u:.3f})"); plt.ylim(0, 1); plt.xlim(0, 1)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.legend(loc='lower right', fontsize=9)

    plt.subplot(1, 2, 2)
    plt.bar(centers, acc_c, width=w, alpha=0.6, label='Empirical acc')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.plot(centers, conf_c, color='C1', lw=2, label='Mean conf')
    plt.title(f"Calibrated (ECE={e_c:.3f})"); plt.ylim(0, 1); plt.xlim(0, 1)
    plt.xlabel("Confidence"); plt.legend(loc='lower right', fontsize=9)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()
    return float(e_u), float(e_c)

def macro_auc_ap(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Compute macro-averaged ROC AUC and Average Precision.
    
    Args:
        y_true: True class labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        
    Returns:
        Tuple of (macro_roc_auc, macro_average_precision)
    """
    y_bin = label_binarize(y_true, classes=range(y_proba.shape[1]))
    aucs, aps = [], []
    for i in range(y_proba.shape[1]):
        try:
            aucs.append(roc_auc_score(y_bin[:, i], y_proba[:, i]))
        except Exception:
            pass
        try:
            aps.append(average_precision_score(y_bin[:, i], y_proba[:, i]))
        except Exception:
            pass
    return (float(np.mean(aucs)) if aucs else float('nan'),
            float(np.mean(aps)) if aps else float('nan'))


# ==================== Bootstrap Confidence Intervals ====================

def _bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate bootstrap sample indices."""
    return rng.integers(0, n, size=n)


def bootstrap_ci_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metric_fn: Callable,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_CONFIDENCE_ALPHA,
    random_state: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for a metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        metric_fn: Metric function taking (y_true, y_pred, y_proba)
        n_boot: Number of bootstrap samples
        alpha: Significance level (default 0.05 for 95% CI)
        random_state: Random seed
        
    Returns:
        Tuple of (mean_value, (lower_ci, upper_ci))
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    vals = []
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    pr = np.asarray(y_proba)
    
    for _ in range(n_boot):
        idx = _bootstrap_indices(n, rng)
        vals.append(metric_fn(yt[idx], yp[idx], pr[idx]))
    
    lo = float(np.percentile(vals, 100 * (alpha / 2)))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return float(np.mean(vals)), (lo, hi)


# Metric wrapper functions for bootstrap
def _ba_fn(y_true: np.ndarray, y_pred: np.ndarray, _: np.ndarray) -> float:
    """Balanced accuracy wrapper."""
    return balanced_accuracy_score(y_true, y_pred)


def _mf1_fn(y_true: np.ndarray, y_pred: np.ndarray, _: np.ndarray) -> float:
    """Macro F1 wrapper."""
    return f1_score(y_true, y_pred, average='macro')


def _macro_auc_fn(y_true: np.ndarray, _: np.ndarray, y_proba: np.ndarray) -> float:
    """Macro AUC wrapper."""
    return macro_auc_ap(y_true, y_proba)[0]


def _macro_ap_fn(y_true: np.ndarray, _: np.ndarray, y_proba: np.ndarray) -> float:
    """Macro AP wrapper."""
    return macro_auc_ap(y_true, y_proba)[1]


def _brier_fn(y_true: np.ndarray, _: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score wrapper."""
    return multiclass_brier_score(y_true, y_proba)


def _logloss_fn(y_true: np.ndarray, _: np.ndarray, y_proba: np.ndarray) -> float:
    """Log loss wrapper."""
    return log_loss(y_true, y_proba, labels=range(y_proba.shape[1]))


def _ece_fn(y_true: np.ndarray, _: np.ndarray, y_proba: np.ndarray) -> float:
    """ECE wrapper."""
    return top1_ece(y_true, y_proba, n_bins=DEFAULT_N_BINS)[0]


# ==================== Visualization and Analysis ====================
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    results_dir: str,
    seed: int,
    title_suffix: str = ""
) -> str:
    """
    Create and save confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        results_dir: Output directory
        seed: Random seed (for filename)
        title_suffix: Optional suffix for title and filename
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.title(f'Confusion Matrix - Seed {seed}{title_suffix}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    total_samples = np.sum(cm)
    correct_predictions = np.sum(np.diag(cm))
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.3f}',
             transform=plt.gca().transAxes, ha='center', fontsize=10)
    plt.tight_layout()
    cm_plot_path = os.path.join(results_dir, f"confusion_matrix{title_suffix.lower().replace(' ', '_')}.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return cm_plot_path

def calculate_additional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Calculate additional classification metrics beyond basic accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        class_names: List of class names
        
    Returns:
        Dictionary containing additional metrics including:
            - cohen_kappa
            - matthews_corrcoef
            - per_class_metrics
            - macro/weighted precision/recall
            - confidence_analysis per class
    """
    additional_metrics = {}
    additional_metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    additional_metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    additional_metrics['per_class_metrics'] = report
    additional_metrics['macro_precision'] = report['macro avg']['precision']
    additional_metrics['macro_recall'] = report['macro avg']['recall']
    additional_metrics['weighted_precision'] = report['weighted avg']['precision']
    additional_metrics['weighted_recall'] = report['weighted avg']['recall']
    confidence_analysis = {}
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.any(class_mask):
            class_proba = y_proba[class_mask, i]
            confidence_analysis[class_name] = {
                'mean_confidence': float(np.mean(class_proba)),
                'std_confidence': float(np.std(class_proba)),
                'min_confidence': float(np.min(class_proba)),
                'max_confidence': float(np.max(class_proba))
            }
    additional_metrics['confidence_analysis'] = confidence_analysis
    return additional_metrics

def calculate_threshold_analysis(y_true, y_pred, y_proba, class_names, ids_test):
    threshold_analysis = {}
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        high_conf_mask = np.max(y_proba, axis=1) >= threshold
        if np.sum(high_conf_mask) > 0:
            y_true_filtered = y_true[high_conf_mask]
            y_pred_filtered = y_pred[high_conf_mask]
            accuracy_filtered = accuracy_score(y_true_filtered, y_pred_filtered)
            balanced_acc_filtered = balanced_accuracy_score(y_true_filtered, y_pred_filtered)
            threshold_analysis[threshold] = {
                'samples_kept': int(np.sum(high_conf_mask)),
                'samples_rejected': int(len(y_true) - np.sum(high_conf_mask)),
                'rejection_rate': float((len(y_true) - np.sum(high_conf_mask)) / len(y_true)),
                'accuracy': float(accuracy_filtered),
                'balanced_accuracy': float(balanced_acc_filtered),
                'improvement': float(accuracy_filtered - accuracy_score(y_true, y_pred))
            }
    class_thresholds = {}
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.any(class_mask):
            class_proba = y_proba[class_mask, i]
            class_pred = (y_pred == i)[class_mask]
            best_threshold = 0.5
            best_precision = 0.0
            for thresh in np.arange(0.1, 1.0, 0.1):
                high_conf = class_proba >= thresh
                if np.sum(high_conf) > 0:
                    precision = np.mean(class_pred[high_conf])
                    if precision > best_precision:
                        best_precision = float(precision)
                        best_threshold = float(thresh)
            class_thresholds[class_name] = {
                'optimal_threshold': float(best_threshold),
                'precision_at_threshold': float(best_precision),
                'samples_above_threshold': int(np.sum(class_proba >= best_threshold))
            }
    return threshold_analysis, class_thresholds
# ---------------------------------------------------------------------------

def _discover_data_file(seed, data_dir):
    data_file = os.path.join(data_dir, f"seed_{seed}_TCGA_Warwick_Clean.pkl")
    if not os.path.exists(data_file):
        data_file = os.path.join(data_dir, "data_splitsV3", f"seed_{seed}_TCGA_Warwick_Clean.pkl")
    if not os.path.exists(data_file):
        data_file = f"/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_mean_features/data_splits/data_splitsV3/seed_{seed}_TCGA_Warwick_Clean.pkl"
    if not os.path.exists(data_file):
        data_file = os.path.join(data_dir, f"seed_{seed}_data.pkl")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Could not find a data file for seed={seed} under {data_dir}")
    return data_file

def _load_models_by_index(models_dir, n_classes):
    # Expect filenames: classifier_{class_name}_{i}.pkl; we place by index i
    model_paths = [None] * n_classes
    pat = re.compile(r"_(\d+)\.pkl$")
    for fname in os.listdir(models_dir):
        if not fname.endswith(".pkl"):
            continue
        m = pat.search(fname)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < n_classes:
                model_paths[idx] = os.path.join(models_dir, fname)
    missing = [i for i, p in enumerate(model_paths) if p is None]
    if missing:
        raise RuntimeError(f"Missing model files for class indices: {missing} in {models_dir}")
    models = []
    for p in model_paths:
        with open(p, "rb") as f:
            models.append(pickle.load(f))
    return models

def evaluate_with_calibration(seed, data_dir, models_dir, output_dir,
                              method="isotonic", cal_size=0.2, retrain=False, debug=False, sampler="ros"):
    print("=" * 70)
    print(f"CALIBRATED EVALUATION (OvR coupling) - SEED {seed}")
    print("=" * 70)
    def dprint(msg):
        if debug:
            print(msg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"Calibrated_OvR_seed_{seed}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results directory: {results_dir}")

    # Load data
    print(f"\n📂 Loading data for seed {seed}...")
    data_file = _discover_data_file(seed, data_dir)
    print(f"   Using data file: {data_file}")
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_test  = data['X_test']
    y_train = data['y_train']
    y_test  = data['y_test']
    class_names = data['class_names']
    ids_test = data['ids_test']
    cv_splits = data.get('cv_splits', None)
    n_classes = len(class_names)

    print(f"📊 Training: {X_train.shape}, Test: {X_test.shape}")
    print(f"📊 Classes: {class_names}")

    # Global true class counts (train/test)
    train_true_counts = np.bincount(y_train, minlength=len(class_names))
    test_true_counts  = np.bincount(y_test,  minlength=len(class_names))
    dprint(f"[GLOBAL] Train true class counts: {dict(zip(class_names, train_true_counts.tolist()))}")
    dprint(f"[GLOBAL] Test  true class counts: {dict(zip(class_names, test_true_counts.tolist()))}")

    # Split a held-out calibration subset from original training distribution
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cal_size, random_state=seed)
    fit_idx, cal_idx = next(sss.split(X_train, y_train))
    X_fit_base, y_fit_base = X_train[fit_idx], y_train[fit_idx]
    X_cal, y_cal_full = X_train[cal_idx], y_train[cal_idx]
    print(f"📏 Fit base: {X_fit_base.shape}, Calibration: {X_cal.shape}")

    # Report true class counts in FIT/CAL splits
    fit_true_counts = np.bincount(y_fit_base, minlength=len(class_names))
    cal_true_counts = np.bincount(y_cal_full, minlength=len(class_names))
    dprint(f"[SPLITS] FIT true counts: {dict(zip(class_names, fit_true_counts.tolist()))}")
    dprint(f"[SPLITS] CAL true counts: {dict(zip(class_names, cal_true_counts.tolist()))}")

    calibrated_clfs = []
    base_clfs = []
    debug_info = {
        "seed": seed,
        "timestamp": timestamp,
        "method": method,
        "retrain": retrain,
        "global": {
            "train_true_counts": dict(zip(class_names, train_true_counts.tolist())),
            "test_true_counts": dict(zip(class_names, test_true_counts.tolist()))
        },
        "splits": {
            "fit_true_counts": dict(zip(class_names, fit_true_counts.tolist())),
            "cal_true_counts": dict(zip(class_names, cal_true_counts.tolist())),
        },
        "per_class": []
    }

    if retrain:
        print("\n🚀 Retraining base OvR models on FIT split only (to avoid leakage)...")
        # Grid over the classifier hyperparameters (pipeline step name: 'clf')
        param_grid = {
            'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'clf__class_weight': [None],
            'clf__solver': ['liblinear', 'lbfgs']
        }

        # CV is defined on the (non-resampled) fit split; ROS happens inside each fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        if cv_splits is not None:
            print("ℹ️ Using StratifiedKFold on the FIT split; ignoring provided cv_splits to avoid index mismatch.")

        for class_idx, class_name in enumerate(class_names):
            # Binarize labels for this class on the FIT and CAL splits
            y_fit_bin = (y_fit_base == class_idx).astype(int)
            y_cal_bin = (y_cal_full == class_idx).astype(int)

            fit_bin_counts = np.bincount(y_fit_bin, minlength=2).tolist()
            cal_bin_counts = np.bincount(y_cal_bin, minlength=2).tolist()
            dprint(f"[{class_name}] FIT bin counts (neg,pos): {fit_bin_counts}, CAL bin counts: {cal_bin_counts}")

            # Choose sampler
            if sampler == "smote":
                sampler_obj = SMOTE(random_state=seed)
            else:
                sampler_obj = RandomOverSampler(random_state=seed)

            # Pipeline: oversample within each CV fold, then fit LR
            pipe = Pipeline(steps=[
                ('sampler', sampler_obj),
                ('clf', LogisticRegression(max_iter=2000, random_state=seed, n_jobs=1))
            ])

            gs = GridSearchCV(
                pipe,
                param_grid=param_grid,
                cv=cv,
                scoring='balanced_accuracy',
                n_jobs=1,
                verbose=0,
                return_train_score=False
            )

            # Fit on the FIT split (no manual resampling here)
            gs.fit(X_fit_base, y_fit_bin)
            best_params = gs.best_params_
            best_score = float(gs.best_score_)
            print(f"   ✅ Best params for '{class_name}': {best_params}")
            print(f"   ✅ Best CV BA: {best_score:.4f}")

            tuned = gs.best_estimator_

            # DEBUG: show what ROS would do on full FIT split
            ros_resampled_counts = None
            if hasattr(tuned, "named_steps") and "sampler" in tuned.named_steps:
                X_res, y_res = tuned.named_steps["sampler"].fit_resample(X_fit_base, y_fit_bin)
                ros_resampled_counts = np.bincount(y_res, minlength=2).tolist()
                dprint(f"[{class_name}] Sampler resampled counts on FIT (neg,pos): {ros_resampled_counts}")
            else:
                dprint(f"[{class_name}] WARNING: Best estimator has no sampler step.")

            debug_info["per_class"].append({
                "class": class_name,
                "fit_bin_counts_neg_pos": fit_bin_counts,
                "cal_bin_counts_neg_pos": cal_bin_counts,
                "best_params": best_params,
                "best_cv_balanced_accuracy": best_score,
                "has_sampler_in_best_estimator": bool(hasattr(tuned, "named_steps") and "sampler" in tuned.named_steps),
                "ros_resampled_counts_neg_pos": ros_resampled_counts
            })

            base_clfs.append(tuned)
    else:
        print("\n📦 Loading pre-trained OvR models...")
        base_clfs = _load_models_by_index(models_dir, n_classes)
        # Log per-class FIT/CAL bin counts even when loading
        for class_idx, class_name in enumerate(class_names):
            y_fit_bin = (y_fit_base == class_idx).astype(int)
            y_cal_bin = (y_cal_full == class_idx).astype(int)
            fit_bin_counts = np.bincount(y_fit_bin, minlength=2).tolist()
            cal_bin_counts = np.bincount(y_cal_bin, minlength=2).tolist()
            debug_info["per_class"].append({
                "class": class_name,
                "fit_bin_counts_neg_pos": fit_bin_counts,
                "cal_bin_counts_neg_pos": cal_bin_counts,
                "loaded_model_type": type(base_clfs[class_idx]).__name__
            })
            dprint(f"[{class_name}] (Loaded) FIT bin counts (neg,pos): {fit_bin_counts}, CAL: {cal_bin_counts}")

    # -------- Pass 1: UNCALIBRATED probabilities (OvR) --------
    print("\n🔎 Pass 1: Computing UNCALIBRATED probabilities (OvR) and metrics...")
    y_proba_uncal_ovr = np.zeros((X_test.shape[0], n_classes))
    for class_idx, base in enumerate(base_clfs):
        y_proba_uncal_ovr[:, class_idx] = base.predict_proba(X_test)[:, 1]
    # couple to multiclass by row-normalization
    sums_u = y_proba_uncal_ovr.sum(axis=1, keepdims=True)
    y_proba_uncal = np.divide(y_proba_uncal_ovr, sums_u, out=np.zeros_like(y_proba_uncal_ovr), where=sums_u > 0)
    y_pred_uncal = np.argmax(y_proba_uncal, axis=1)
    uncal_metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_uncal),
        "macro_f1": f1_score(y_test, y_pred_uncal, average='macro'),
        "weighted_f1": f1_score(y_test, y_pred_uncal, average='weighted'),
        "accuracy": accuracy_score(y_test, y_pred_uncal),
        "macro_roc_auc": macro_auc_ap(y_test, y_proba_uncal_ovr)[0],  # use uncoupled scores for ranking
        "macro_ap": macro_auc_ap(y_test, y_proba_uncal_ovr)[1],
        "brier": multiclass_brier_score(y_test, y_proba_uncal),
        "log_loss": log_loss(y_test, y_proba_uncal, labels=range(n_classes)),
        "ece_top1": top1_ece(y_test, y_proba_uncal, n_bins=15)[0],
    }
    print(f"   Uncalibrated -> BA {uncal_metrics['balanced_accuracy']:.4f}, MacroF1 {uncal_metrics['macro_f1']:.4f}, "
          f"AUC {uncal_metrics['macro_roc_auc']:.4f}, AP {uncal_metrics['macro_ap']:.4f}, "
          f"Brier {uncal_metrics['brier']:.4f}, LogLoss {uncal_metrics['log_loss']:.4f}, ECE {uncal_metrics['ece_top1']:.4f}")

    print("\n🧪 Calibrating per-class probabilities on held-out ORIGINAL distribution (OvR)...")
    for class_idx, base in enumerate(base_clfs):
        y_cal_bin = (y_cal_full == class_idx).astype(int)
        pos = int(y_cal_bin.sum())
        neg = int(len(y_cal_bin) - pos)
        if pos == 0 or neg == 0:
            print(f"⚠️  Skip calibration for class '{class_names[class_idx]}' (pos={pos}, neg={neg}). Using base model.")
            calibrated_clfs.append(None)
            continue
        calibrator = CalibratedClassifierCV(base, method=method, cv='prefit')
        calibrator.fit(X_cal, y_cal_bin)
        calibrated_clfs.append(calibrator)

    # Inference with calibrated probabilities
    print("\n🎯 Predicting on test set with calibrated & coupled probabilities...")
    y_proba_cal_ovr = np.zeros((X_test.shape[0], n_classes))
    for class_idx, clf in enumerate(calibrated_clfs):
        if clf is None:
            y_proba_cal_ovr[:, class_idx] = base_clfs[class_idx].predict_proba(X_test)[:, 1]
        else:
            y_proba_cal_ovr[:, class_idx] = clf.predict_proba(X_test)[:, 1]

    # Couple to a proper multiclass distribution (row-normalize)
    row_sums = y_proba_cal_ovr.sum(axis=1, keepdims=True)
    y_proba_coupled = np.divide(y_proba_cal_ovr, row_sums,
                                out=np.zeros_like(y_proba_cal_ovr), where=row_sums > 0)

    y_pred_ovr = np.argmax(y_proba_coupled, axis=1)

    # Test metrics (same keys/structure as LR_training.py)
    test_metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_ovr),
        "macro_f1": f1_score(y_test, y_pred_ovr, average='macro'),
        "weighted_f1": f1_score(y_test, y_pred_ovr, average='weighted'),
        "accuracy": accuracy_score(y_test, y_pred_ovr),
        "macro_roc_auc": macro_auc_ap(y_test, y_proba_cal_ovr)[0],   # use uncoupled calibrated scores for ranking
        "macro_ap": macro_auc_ap(y_test, y_proba_cal_ovr)[1],
        "brier": multiclass_brier_score(y_test, y_proba_coupled),
        "log_loss": log_loss(y_test, y_proba_coupled, labels=range(n_classes)),
        "ece_top1": top1_ece(y_test, y_proba_coupled, n_bins=15)[0],
    }
    additional_metrics = calculate_additional_metrics(y_test, y_pred_ovr, y_proba_coupled, class_names)

    print(f"\n📈 TEST METRICS (calibrated & coupled):")
    print("-" * 40)
    print(f"   Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"   Macro F1:         {test_metrics['macro_f1']:.4f}")
    print(f"   Weighted F1:      {test_metrics['weighted_f1']:.4f}")
    print(f"   Accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"   Cohen's Kappa:    {additional_metrics['cohen_kappa']:.4f}")
    print(f"   Matthews Corr.:   {additional_metrics['matthews_corrcoef']:.4f}")
    print(f"   Macro ROC AUC:    {test_metrics['macro_roc_auc']:.4f}")
    print(f"   Macro AP:         {test_metrics['macro_ap']:.4f}")
    print(f"   Brier score:      {test_metrics['brier']:.4f}")
    print(f"   Log-loss:         {test_metrics['log_loss']:.4f}")
    print(f"   ECE (top-1):      {test_metrics['ece_top1']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_ovr)
    print(f"\n📊 Confusion matrix:\n{cm}")

    # Misclassified analysis
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_test, y_pred_ovr)):
        if true != pred:
            confidence = y_proba_coupled[i, pred]
            misclassified.append({
                'index': i,
                'slide_id': ids_test[i],  # FIXED to use ids_test
                'true_class': class_names[true],
                'pred_class': class_names[pred],
                'confidence': float(confidence),
                'true_idx': int(true),
                'pred_idx': int(pred)
            })
    print(f"\n📊 Misclassified slides: {len(misclassified)}")

    # Plots
    print(f"\n📊 Generating visualizations")
    cm_plot_path = plot_confusion_matrix(cm, class_names, results_dir, seed)
    print(f"✅ Confusion matrix plot saved: {cm_plot_path}")

    # ROC/PR (UNCALIBRATED, from uncoupled OvR scores)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    roc_auc_per_class_uncal = {}
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_uncal_ovr[:, i])
        auc_score = roc_auc_score(y_test_bin[:, i], y_proba_uncal_ovr[:, i])
        roc_auc_per_class_uncal[class_names[i]] = float(auc_score)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title(f"ROC (OvR, UNCAL) - Seed {seed}"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3); plt.tight_layout()
    roc_plot_uncal = os.path.join(results_dir, "roc_auc_ovr_testset_uncalibrated.png")
    plt.savefig(roc_plot_uncal, dpi=300, bbox_inches='tight'); plt.close()

    avg_precision_per_class_uncal = {}
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        p, r, _ = precision_recall_curve(y_test_bin[:, i], y_proba_uncal_ovr[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_proba_uncal_ovr[:, i])
        avg_precision_per_class_uncal[class_names[i]] = float(ap)
        plt.plot(r, p, label=f"{class_names[i]} (AP={ap:.3f})", linewidth=2)
    plt.title(f"PR (OvR, UNCAL) - Seed {seed}"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3); plt.tight_layout()
    pr_plot_uncal = os.path.join(results_dir, "precision_recall_ovr_testset_uncalibrated.png")
    plt.savefig(pr_plot_uncal, dpi=300, bbox_inches='tight'); plt.close()

    # ROC/PR (CALIBRATED, from uncoupled calibrated scores)
    roc_auc_per_class = {}
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_cal_ovr[:, i])
        auc_score = roc_auc_score(y_test_bin[:, i], y_proba_cal_ovr[:, i])
        roc_auc_per_class[class_names[i]] = float(auc_score)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title(f"ROC (OvR, CAL) - Seed {seed}"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3); plt.tight_layout()
    roc_plot_path = os.path.join(results_dir, "roc_auc_ovr_testset_calibrated.png")
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight'); plt.close()

    avg_precision_per_class = {}
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba_cal_ovr[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_proba_cal_ovr[:, i])
        avg_precision_per_class[class_names[i]] = float(ap)
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.3f})", linewidth=2)
    plt.title(f"PR (OvR, CAL) - Seed {seed}"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3); plt.tight_layout()
    pr_plot_path = os.path.join(results_dir, "precision_recall_ovr_testset_calibrated.png")
    plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight'); plt.close()

    print(f"✅ ROC curves saved: {roc_plot_uncal}, {roc_plot_path}")
    print(f"✅ PR curves saved: {pr_plot_uncal}, {pr_plot_path}")

    # Reliability (top-1) comparing coupled distributions
    reliab_path = os.path.join(results_dir, "reliability_top1_uncal_vs_cal.png")
    ece_u, ece_c = plot_reliability_diagram(
        y_test, y_proba_uncal, y_proba_coupled, reliab_path, n_bins=15,
        title=f"Reliability (Top-1) — Seed {seed}"
    )
    print(f"✅ Reliability diagram saved: {reliab_path} (ECE uncal={ece_u:.3f}, cal={ece_c:.3f})")

    # Threshold analysis (same structure)
    threshold_analysis, class_thresholds = calculate_threshold_analysis(
        y_test, y_pred_ovr, y_proba_coupled, class_names, ids_test
    )

    # Save comprehensive text report (matching structure)
    results_txt_path = os.path.join(results_dir, "comprehensive_results.txt")
    with open(results_txt_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"CALIBRATED OVR RESULTS - SEED {seed}\n")
        f.write("="*80 + "\n\n")
        f.write("EXPERIMENT INFO:\n")
        f.write("-"*40 + "\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Data file: {data_file}\n")
        f.write(f"Results directory: {results_dir}\n")
        f.write(f"Calibration method: {method}\n")
        f.write(f"Retrain base models: {retrain}\n\n")

        f.write("DATA INFO:\n")
        f.write("-"*40 + "\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Number of features: {X_train.shape[1]}\n")
        f.write(f"Number of classes: {n_classes}\n")
        f.write(f"Class names: {class_names}\n\n")

        f.write("OVERALL TEST SET METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1: {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {test_metrics['weighted_f1']:.4f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Macro ROC AUC (uncoupled): {test_metrics['macro_roc_auc']:.4f}\n")
        f.write(f"Macro AP (uncoupled): {test_metrics['macro_ap']:.4f}\n")
        f.write(f"Brier (multiclass): {test_metrics['brier']:.4f}\n")
        f.write(f"Log-loss: {test_metrics['log_loss']:.4f}\n")
        f.write(f"ECE (top-1): {test_metrics['ece_top1']:.4f}\n\n")

        f.write("UNALIBRATED OVERALL METRICS (for comparison):\n")
        f.write("-"*40 + "\n")
        f.write(f"Balanced Accuracy: {uncal_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1: {uncal_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {uncal_metrics['weighted_f1']:.4f}\n")
        f.write(f"Accuracy: {uncal_metrics['accuracy']:.4f}\n")
        f.write(f"Macro ROC AUC (uncoupled): {uncal_metrics['macro_roc_auc']:.4f}\n")
        f.write(f"Macro AP (uncoupled): {uncal_metrics['macro_ap']:.4f}\n")
        f.write(f"Brier (multiclass): {uncal_metrics['brier']:.4f}\n")
        f.write(f"Log-loss: {uncal_metrics['log_loss']:.4f}\n")
        f.write(f"ECE (top-1): {uncal_metrics['ece_top1']:.4f}\n\n")

        f.write("PER-CLASS TEST SET METRICS:\n")
        f.write("-"*40 + "\n")
        report_dict = additional_metrics['per_class_metrics']
        f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-"*70 + "\n")
        for class_name in class_names:
            if class_name in report_dict:
                metrics = report_dict[class_name]
                f.write(f"{class_name:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                        f"{metrics['f1-score']:<10.4f} {metrics['support']:<10.0f}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Macro Avg':<20} {report_dict['macro avg']['precision']:<10.4f} "
                f"{report_dict['macro avg']['recall']:<10.4f} {report_dict['macro avg']['f1-score']:<10.4f} "
                f"{report_dict['macro avg']['support']:<10.0f}\n")
        f.write(f"{'Weighted Avg':<20} {report_dict['weighted avg']['precision']:<10.4f} "
                f"{report_dict['weighted avg']['recall']:<10.4f} {report_dict['weighted avg']['f1-score']:<10.4f} "
                f"{report_dict['weighted avg']['support']:<10.0f}\n\n")

        f.write("ROC AUC SCORES PER CLASS (CALIBRATED, uncoupled):\n")
        f.write("-"*40 + "\n")
        for class_name in class_names:
            if class_name in roc_auc_per_class:
                f.write(f"{class_name:<20}: {roc_auc_per_class[class_name]:.4f}\n")
        f.write("\n")

        f.write("AVERAGE PRECISION SCORES PER CLASS (CALIBRATED, uncoupled):\n")
        f.write("-"*40 + "\n")
        for class_name in class_names:
            if class_name in avg_precision_per_class:
                f.write(f"{class_name:<20}: {avg_precision_per_class[class_name]:.4f}\n")
        f.write("\n")

        f.write("ROC AUC SCORES PER CLASS (UNCALIBRATED, uncoupled):\n")
        f.write("-"*40 + "\n")
        for class_name in class_names:
            if class_name in roc_auc_per_class_uncal:
                f.write(f"{class_name:<20}: {roc_auc_per_class_uncal[class_name]:.4f}\n")
        f.write("\n")

        f.write("AVERAGE PRECISION SCORES PER CLASS (UNCALIBRATED, uncoupled):\n")
        f.write("-"*40 + "\n")
        for class_name in class_names:
            if class_name in avg_precision_per_class_uncal:
                f.write(f"{class_name:<20}: {avg_precision_per_class_uncal[class_name]:.4f}\n")
        f.write("\n")

        f.write("CONFUSION MATRIX:\n")
        f.write("-"*40 + "\n")
        f.write("Rows = True, Columns = Predicted\n")
        f.write(f"{'':>15}")
        for class_name in class_names:
            f.write(f"{class_name:>10}")
        f.write("\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:>15}")
            for j in range(n_classes):
                f.write(f"{cm[i][j]:>10}")
            f.write("\n")
        f.write("\n")

        f.write("CONFIDENCE THRESHOLD ANALYSIS:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Threshold':<12} {'Kept':<8} {'Rejected':<10} {'Reject%':<10} {'Accuracy':<10} {'Improvement':<12}\n")
        f.write("-"*80 + "\n")
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            if threshold in threshold_analysis:
                analysis = threshold_analysis[threshold]
                f.write(f"{threshold:<12.1f} {analysis['samples_kept']:<8} {analysis['samples_rejected']:<10} "
                        f"{analysis['rejection_rate']*100:<10.1f} {analysis['accuracy']:<10.4f} "
                        f"{analysis['improvement']:<12.4f}\n")
        f.write("\n")

        f.write("OPTIMAL THRESHOLDS PER CLASS:\n")
        f.write("-"*40 + "\n")
        for class_name in class_names:
            if class_name in class_thresholds:
                info = class_thresholds[class_name]
                f.write(f"{class_name}:\n")
                f.write(f"  Optimal threshold: {info['optimal_threshold']:.2f}\n")
                f.write(f"  Precision at threshold: {info['precision_at_threshold']:.4f}\n")
                f.write(f"  Samples above threshold: {info['samples_above_threshold']}\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    # Save results pickle (same keys)
    results_pkl_path = os.path.join(results_dir, "enhanced_all_results.pkl")
    results_dict = {
        "seed": seed,
        "timestamp": timestamp,
        "test_metrics_calibrated": test_metrics,
        "test_metrics_uncalibrated": uncal_metrics,
        "additional_metrics": additional_metrics,
        "confusion_matrix": cm,
        "roc_auc_per_class": roc_auc_per_class,
        "avg_precision_per_class": avg_precision_per_class,
        "roc_auc_per_class_uncalibrated": roc_auc_per_class_uncal,
        "avg_precision_per_class_uncalibrated": avg_precision_per_class_uncal,
        "misclassified": misclassified,
        "best_params_per_class": None,
        "cv_analysis": None,
        "predictions": {
            "y_pred_calibrated": y_pred_ovr,
            "y_pred_uncalibrated": y_pred_uncal,
            "y_true": y_test,
            "y_proba_calibrated": y_proba_coupled,
            "y_proba_uncalibrated": y_proba_uncal,
            "y_proba_uncalibrated_ovr": y_proba_uncal_ovr,
            "y_proba_calibrated_ovr": y_proba_cal_ovr
         }
    }
    with open(results_pkl_path, "wb") as f:
        pickle.dump(results_dict, f)

    # Save calibrated models
    models_out = os.path.join(results_dir, "calibrated_models")
    os.makedirs(models_out, exist_ok=True)
    for i, (clf, cname) in enumerate(zip(calibrated_clfs, class_names)):
        with open(os.path.join(models_out, f"calibrated_{cname}_{i}.pkl"), "wb") as f:
            pickle.dump(clf, f)

    print(f"\n✅ Comprehensive text report saved: {results_txt_path}")
    print(f"✅ Results pickle saved: {results_pkl_path}")
    print(f"✅ Calibrated models saved: {models_out}")
    print(f"\n🎉 Calibrated evaluation completed.")
    return {
        "results_dir": results_dir,
        "test_metrics": test_metrics
    }

def main():
    ap = argparse.ArgumentParser(description="Calibrated evaluation for OvR classifiers")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_dir", type=str, default="data_splits")
    ap.add_argument("--models_dir", type=str, required=False,
                    help="Directory containing classifier_{class}_{i}.pkl (ignored with --retrain)")
    ap.add_argument("--output_dir", type=str, default="results")
    ap.add_argument("--method", type=str, choices=["isotonic", "sigmoid"], default="isotonic")
    ap.add_argument("--cal_size", type=float, default=0.2, help="Calibration split size from original train")
    ap.add_argument("--retrain", action="store_true", help="Retrain base models on fit split to avoid leakage")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug logging")  # NEW
    ap.add_argument("--sampler", type=str, choices=["ros", "smote"], default="ros",
                help="Which sampler to use in the pipeline: 'ros' (RandomOverSampler) or 'smote' (SMOTE)")
    args = ap.parse_args()

    if not args.retrain and not args.models_dir:
        print("Error: --models_dir is required unless you pass --retrain", file=sys.stderr)
        sys.exit(2)

    evaluate_with_calibration(
        seed=args.seed,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        method=args.method,
        cal_size=args.cal_size,
        retrain=args.retrain,
        debug=args.debug,
        sampler=args.sampler  # NEW
    )

if __name__ == "__main__":
    main()