#!/usr/bin/env python3
"""
Evaluation-only for pre-calibrated OvR classifiers.
- Loads saved calibrated models (one per class).
- Predicts on the unseen test set.
- Computes same metrics/plots as LR_callibration.py.
- Generates ranking CSVs (confidence, ambiguity, per-class) and includes summaries in the report.

Author: Konstantnos Papagoras
Date: 2025-07
"""
import os
import re
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import csv

from sklearn.metrics import (
    roc_curve, roc_auc_score, balanced_accuracy_score, f1_score,
    accuracy_score, confusion_matrix, precision_recall_curve,
    average_precision_score, classification_report, cohen_kappa_score,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize

# ---------- Helpers (same behavior as in LR_callibration.py) ----------
def plot_confusion_matrix(cm, class_names, results_dir, seed, title_suffix=""):
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

def calculate_additional_metrics(y_true, y_pred, y_proba, class_names):
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
    # Accept filenames like calibrated_{class}_{i}.pkl or classifier_{class}_{i}.pkl
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

def evaluate_loaded_calibrated(seed, data_dir, models_dir, output_dir):
    print("=" * 70)
    print(f"EVALUATION-ONLY (pre-calibrated OvR) - SEED {seed}")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"Calibrated_OvR_EvalOnly_seed_{seed}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results directory: {results_dir}")

    # Load data
    print(f"\n📂 Loading data for seed {seed}...")
    data_file = _discover_data_file(seed, data_dir)
    print(f"   Using data file: {data_file}")
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    X_test  = data['X_test']
    y_test  = data['y_test']
    class_names = data['class_names']
    ids_test = data['ids_test']
    n_classes = len(class_names)

    print(f"📊 Test: {X_test.shape}")
    print(f"📊 Classes: {class_names}")

    # Load pre-calibrated per-class models (no retraining, no recalibration)
    print("\n📦 Loading pre-calibrated OvR models...")
    calibrated_clfs = _load_models_by_index(models_dir, n_classes)

    # Predict calibrated per-class probabilities on test
    print("\n🎯 Predicting on test set (calibrated probabilities) ...")
    y_proba_ovr = np.zeros((X_test.shape[0], n_classes))
    for class_idx, clf in enumerate(calibrated_clfs):
        # Each clf is a CalibratedClassifierCV; take positive class probability
        y_proba_ovr[:, class_idx] = clf.predict_proba(X_test)[:, 1]

    # Couple to a proper multiclass distribution (row-normalize)
    row_sums = y_proba_ovr.sum(axis=1, keepdims=True)
    y_proba_coupled = np.divide(y_proba_ovr, row_sums,
                                out=np.zeros_like(y_proba_ovr), where=row_sums > 0)

    y_pred_ovr = np.argmax(y_proba_coupled, axis=1)

    # Test metrics (same keys/structure)
    test_metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_ovr),
        "macro_f1": f1_score(y_test, y_pred_ovr, average='macro'),
        "weighted_f1": f1_score(y_test, y_pred_ovr, average='weighted'),
        "accuracy": accuracy_score(y_test, y_pred_ovr)
    }
    additional_metrics = calculate_additional_metrics(y_test, y_pred_ovr, y_proba_coupled, class_names)

    print(f"\n📈 TEST METRICS (pre-calibrated & coupled):")
    print("-" * 40)
    print(f"   Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"   Macro F1:         {test_metrics['macro_f1']:.4f}")
    print(f"   Weighted F1:      {test_metrics['weighted_f1']:.4f}")
    print(f"   Accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"   Cohen's Kappa:    {additional_metrics['cohen_kappa']:.4f}")
    print(f"   Matthews Corr.:   {additional_metrics['matthews_corrcoef']:.4f}")

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
                'slide_id': ids_test[i],
                'true_class': class_names[true],
                'pred_class': class_names[pred],
                'confidence': float(confidence),
                'true_idx': int(true),
                'pred_idx': int(pred)
            })
    print(f"\n📊 Misclassified slides: {len(misclassified)}")

    # Plots: ROC/PR using calibrated probabilities (OvR)
    print("\n📊 Generating visualizations ...")
    cm_plot_path = plot_confusion_matrix(cm, class_names, results_dir, seed)
    print(f"✅ Confusion matrix plot saved: {cm_plot_path}")

    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    roc_auc_per_class = {}
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_coupled[:, i])
        auc_score = roc_auc_score(y_test_bin[:, i], y_proba_coupled[:, i])
        roc_auc_per_class[class_names[i]] = float(auc_score)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves (OvR, Test Set) - Seed {seed}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(results_dir, "roc_auc_ovr_testset.png")
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    avg_precision_per_class = {}
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba_coupled[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_proba_coupled[:, i])
        avg_precision_per_class[class_names[i]] = float(ap)
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={ap:.3f})", linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curves (OvR, Test Set) - Seed {seed}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_plot_path = os.path.join(results_dir, "precision_recall_ovr_testset.png")
    plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ ROC curves saved: {roc_plot_path}")
    print(f"✅ PR curves saved: {pr_plot_path}")

    # Threshold analysis
    threshold_analysis, class_thresholds = calculate_threshold_analysis(
        y_test, y_pred_ovr, y_proba_coupled, class_names, ids_test
    )

    # ===================== Ranking analysis =====================
    print("\n📊 Building ranking outputs (confidence, margin, per-class) ...")
    eps = 1e-12
    n = len(y_test)

    top1_idx = np.argmax(y_proba_coupled, axis=1)
    top1_prob = np.max(y_proba_coupled, axis=1)
    idx_sorted = np.argsort(y_proba_coupled, axis=1)
    top2_idx = idx_sorted[:, -2]
    top2_prob = y_proba_coupled[np.arange(n), top2_idx]
    margin = top1_prob - top2_prob
    entropy = -np.sum(y_proba_coupled * np.log(y_proba_coupled + eps), axis=1) / np.log(y_proba_coupled.shape[1])
    correct = (y_pred_ovr == y_test)

    ranked_by_confidence = np.argsort(-top1_prob)
    ranked_by_ambiguity = np.argsort(margin)
    high_conf_mask = (top1_prob >= 0.8) & (~correct)
    ranked_high_conf_errors = np.where(high_conf_mask)[0]
    ranked_high_conf_errors = ranked_high_conf_errors[np.argsort(-top1_prob[ranked_high_conf_errors])]

    def _write_csv(path, header, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)

    # Overall ranked by confidence
    conf_rows = []
    for i in ranked_by_confidence:
        conf_rows.append([
            int(i), ids_test[i],
            class_names[int(y_test[i])],
            class_names[int(y_pred_ovr[i])],
            class_names[int(top1_idx[i])], float(top1_prob[i]),
            class_names[int(top2_idx[i])], float(top2_prob[i]),
            float(margin[i]), float(entropy[i]),
            bool(correct[i])
        ])
    conf_path = os.path.join(results_dir, "ranking_ranked_by_confidence.csv")
    _write_csv(
        conf_path,
        ["index","slide_id","true_class","pred_class","top1_label","top1_prob",
         "top2_label","top2_prob","margin","entropy","correct"],
        conf_rows
    )

    # Overall ranked by ambiguity (smallest margin)
    amb_rows = []
    for i in ranked_by_ambiguity:
        amb_rows.append([
            int(i), ids_test[i],
            class_names[int(y_test[i])],
            class_names[int(y_pred_ovr[i])],
            class_names[int(top1_idx[i])], float(top1_prob[i]),
            class_names[int(top2_idx[i])], float(top2_prob[i]),
            float(margin[i]), float(entropy[i]),
            bool(correct[i])
        ])
    amb_path = os.path.join(results_dir, "ranking_ranked_by_ambiguity.csv")
    _write_csv(
        amb_path,
        ["index","slide_id","true_class","pred_class","top1_label","top1_prob",
         "top2_label","top2_prob","margin","entropy","correct"],
        amb_rows
    )

    # High-confidence misclassifications
    hc_err_rows = []
    for i in ranked_high_conf_errors:
        hc_err_rows.append([
            int(i), ids_test[i],
            class_names[int(y_test[i])],
            class_names[int(y_pred_ovr[i])],
            class_names[int(top1_idx[i])], float(top1_prob[i]),
            class_names[int(top2_idx[i])], float(top2_prob[i]),
            float(margin[i]), float(entropy[i])
        ])
    hc_err_path = os.path.join(results_dir, "ranking_high_conf_misclassifications.csv")
    _write_csv(
        hc_err_path,
        ["index","slide_id","true_class","pred_class","top1_label","top1_prob",
         "top2_label","top2_prob","margin","entropy"],
        hc_err_rows
    )

    # Per-class rankings
    per_class_dir = os.path.join(results_dir, "ranking_per_class")
    os.makedirs(per_class_dir, exist_ok=True)
    for c, cname in enumerate(class_names):
        order = np.argsort(-y_proba_coupled[:, c])
        rows = []
        for i in order:
            rows.append([
                int(i), ids_test[i],
                class_names[int(y_test[i])],
                class_names[int(y_pred_ovr[i])],
                cname, float(y_proba_coupled[i, c]),
                float(top1_prob[i]), float(margin[i]),
                bool(correct[i])
            ])
        _write_csv(
            os.path.join(per_class_dir, f"ranking_by_prob_{cname}.csv"),
            ["index","slide_id","true_class","pred_class","class","p_class",
             "top1_prob","margin","correct"],
            rows
        )

    print(f"✅ Rankings saved:\n   - {conf_path}\n   - {amb_path}\n   - {hc_err_path}\n   - Per-class: {per_class_dir}")

    # Save comprehensive text report
    results_txt_path = os.path.join(results_dir, "comprehensive_results.txt")
    with open(results_txt_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"CALIBRATED OVR EVALUATION-ONLY - SEED {seed}\n")
        f.write("="*80 + "\n\n")

        f.write("METHOD SUMMARY:\n")
        f.write("-"*40 + "\n")
        f.write("Using pre-calibrated OvR classifiers (one per class). No retraining or recalibration.\n")
        f.write("Per-class calibrated probabilities are row-normalized; final class is argmax.\n")
        f.write("Metrics computed on the unseen test set.\n\n")

        f.write("EXPERIMENT INFO:\n")
        f.write("-"*40 + "\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Data file: {data_file}\n")
        f.write(f"Results directory: {results_dir}\n")
        f.write(f"Models directory: {models_dir}\n\n")

        f.write("OVERALL TEST SET METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1: {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {test_metrics['weighted_f1']:.4f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa: {additional_metrics['cohen_kappa']:.4f}\n")
        f.write(f"Matthews Correlation: {additional_metrics['matthews_corrcoef']:.4f}\n\n")

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

        f.write("ROC AUC SCORES PER CLASS:\n")
        f.write("-"*40 + "\n")
        for class_name in class_names:
            if class_name in report_dict:
                f.write(f"{class_name:<20}: {roc_auc_score(y_test_bin[:, list(class_names).index(class_name)], y_proba_coupled[:, list(class_names).index(class_name)]):.4f}\n")
        f.write("\n")

        f.write("AVERAGE PRECISION SCORES PER CLASS:\n")
        f.write("-"*40 + "\n")
        for class_name in class_names:
            idx = list(class_names).index(class_name)
            ap = average_precision_score(y_test_bin[:, idx], y_proba_coupled[:, idx])
            f.write(f"{class_name:<20}: {ap:.4f}\n")
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
        for threshold, analysis in calculate_threshold_analysis(y_test, y_pred_ovr, y_proba_coupled, class_names, ids_test).items() if False else []:
            pass  # placeholder to indicate section present; actual values are printed below

        # Print the actual threshold analysis (already computed)
        ths, cls_th = threshold_analysis, class_thresholds
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            if threshold in ths:
                analysis = ths[threshold]
                f.write(f"{threshold:<12.1f} {analysis['samples_kept']:<8} {analysis['samples_rejected']:<10} "
                        f"{analysis['rejection_rate']*100:<10.1f} {analysis['accuracy']:<10.4f} "
                        f"{analysis['improvement']:<12.4f}\n")
        f.write("\n")

        f.write("OPTIMAL THRESHOLDS PER CLASS:\n")
        f.write("-"*40 + "\n")
        for class_name in class_names:
            if class_name in cls_th:
                info = cls_th[class_name]
                f.write(f"{class_name}:\n")
                f.write(f"  Optimal threshold: {info['optimal_threshold']:.2f}\n")
                f.write(f"  Precision at threshold: {info['precision_at_threshold']:.4f}\n")
                f.write(f"  Samples above threshold: {info['samples_above_threshold']}\n\n")

        f.write("RANKING SUMMARY:\n")
        f.write("-"*40 + "\n")
        f.write(f"Top-1 confidence CSV: {conf_path}\n")
        f.write(f"Ambiguity (margin) CSV: {amb_path}\n")
        f.write(f"High-confidence misclassifications CSV: {hc_err_path}\n")
        f.write(f"Per-class rankings folder: {per_class_dir}\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    # Save results pickle
    results_pkl_path = os.path.join(results_dir, "enhanced_all_results.pkl")
    results_dict = {
        "seed": seed,
        "timestamp": timestamp,
        "test_metrics": test_metrics,
        "additional_metrics": additional_metrics,
        "confusion_matrix": cm,
        "roc_auc_per_class": {k: float(v) for k, v in roc_auc_per_class.items()},
        "avg_precision_per_class": avg_precision_per_class,
        "misclassified": misclassified,
        "best_params_per_class": None,
        "cv_analysis": None,
        "predictions": {
            "y_pred": y_pred_ovr,
            "y_true": y_test,
            "y_proba": y_proba_coupled
        }
    }
    with open(results_pkl_path, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"\n✅ Comprehensive text report saved: {results_txt_path}")
    print(f"✅ Results pickle saved: {results_pkl_path}")
    print(f"\n🎉 Evaluation-only completed.")
    return {"results_dir": results_dir, "test_metrics": test_metrics}

def main():
    ap = argparse.ArgumentParser(description="Evaluation-only for pre-calibrated OvR classifiers")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_dir", type=str, default="data_splits")
    ap.add_argument("--models_dir", type=str, required=True,
                    help="Directory containing calibrated_{class}_{i}.pkl")
    ap.add_argument("--output_dir", type=str, default="results")
    args = ap.parse_args()

    evaluate_loaded_calibrated(
        seed=args.seed,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()