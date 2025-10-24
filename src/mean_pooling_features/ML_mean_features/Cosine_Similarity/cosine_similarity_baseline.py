#!/usr/bin/env python3
"""
Cosine Similarity Baseline for WSI Classification

This script implements a k-nearest neighbors baseline using cosine similarity
for whole slide image (WSI) classification. It evaluates multiple k values across
different random seeds and generates comprehensive metrics and visualizations.

Features:
- GPU-accelerated similarity computation (optional)
- Multiple k-value evaluation
- Comprehensive metrics (balanced accuracy, F1, ROC AUC, etc.)
- Detailed visualizations (confusion matrices, ROC curves, PR curves)
- Per-seed analysis with JSON summaries

Author: Konstantinos Papagoras
Date: 2024
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

# GPU support (optional)
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


# Configuration Constants
DEFAULT_K_VALUES = [1, 3, 5, 7, 9]
DEFAULT_KNN_TEMP = 1.0
GPU_MEMORY_THRESHOLD = 5000  # Maximum test samples for GPU computation

def load_data_split(split_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                               np.ndarray, np.ndarray, np.ndarray, 
                                               np.ndarray]:
    """
    Load features from a data split file.
    
    Args:
        split_path: Path to the pickle file containing split data
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, ids_train, ids_test, class_names)
    """
    print(f"Loading data from: {split_path}")
    
    with open(split_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    ids_train = data['ids_train']
    ids_test = data['ids_test']
    class_names = data.get('class_names', None)
    
    print(f"Loaded {len(ids_train)} training slides and {len(ids_test)} test slides")
    print(f"Classes: {class_names}")
    
    return X_train, X_test, y_train, y_test, ids_train, ids_test, class_names

def compute_and_save_similarities(X_test, X_train, ids_test, ids_train, output_dir):
    """Compute pairwise similarities and save them"""
    print(f"Computing cosine similarities... ({X_test.shape[0]} test × {X_train.shape[0]} train)")
    
    try:
        if HAS_GPU and X_test.shape[0] <= 5000:  # Reduce GPU memory threshold
            print("🚀 Using GPU acceleration for similarity computation")
            X_test_gpu = cp.asarray(X_test)
            X_train_gpu = cp.asarray(X_train)
            
            # Normalize feature vectors
            X_test_norm = X_test_gpu / cp.sqrt(cp.sum(X_test_gpu ** 2, axis=1, keepdims=True))
            X_train_norm = X_train_gpu / cp.sqrt(cp.sum(X_train_gpu ** 2, axis=1, keepdims=True))
            
            # Compute cosine similarity
            similarities = cp.matmul(X_test_norm, X_train_norm.T)
            similarities = cp.asnumpy(similarities)
        else:
            print("🖥️ Using CPU for similarity computation")
            similarities = cosine_similarity(X_test, X_train)
    except Exception as e:
        print(f"⚠️ GPU computation failed ({e}), falling back to CPU")
        similarities = cosine_similarity(X_test, X_train)
    
    # Save similarity matrix
    similarities_path = os.path.join(output_dir, "test_train_similarities.npy")
    np.save(similarities_path, similarities)
    print(f"✅ Similarity matrix saved: {similarities_path}")
    
    # Save as DataFrame with slide IDs
    similarity_df = pd.DataFrame(similarities, index=ids_test, columns=ids_train)
    similarity_csv_path = os.path.join(output_dir, "test_train_similarities.csv")
    similarity_df.to_csv(similarity_csv_path)
    print(f"✅ Similarity CSV saved: {similarity_csv_path}")
    
    print(f"📊 Similarity matrix shape: {similarities.shape}")
    print(f"📊 Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
    
    return similarities

def predict_with_knn_proba(X_train, y_train, X_test, class_names, k=3, use_softmax=True, temp=1.0, eps=1e-12):
    """
    KNN-style class probabilities using cosine similarity.
    - For each test sample: find k nearest training samples (by cosine similarity).
    - Aggregate similarities per class (sum), then normalize across classes to get probabilities.
    - Optional: apply softmax(temp) to class scores before normalization.
    Returns: probs (n_test, n_classes)
    """
    n_test = X_test.shape[0]
    n_classes = len(class_names)
    probs = np.zeros((n_test, n_classes), dtype=float)
    
    # Normalize feature vectors (important for consistent cosine similarity values)
    X_train_norm = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_test_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    
    # Compute full cosine similarity matrix (n_test x n_train)
    sim = X_test_norm @ X_train_norm.T

    # First add tiny random noise to break ties (important for ROC curves)
    # This prevents the "bag" effect by making all probability values unique
    np.random.seed(42)  # For reproducibility
    noise = np.random.uniform(0, 1e-8, sim.shape)
    sim = sim + noise
    
    # For tracking probability distributions
    all_class_scores = []
    
    for i in range(n_test):
        # Get top-k neighbors by similarity
        neighbor_indices = np.argsort(sim[i])[::-1][:k]
        neighbor_similarities = sim[i, neighbor_indices]
        neighbor_labels = y_train[neighbor_indices]
        
        # Aggregate similarity scores per class
        class_scores = np.zeros(n_classes)
        for idx, label in enumerate(neighbor_labels):
            class_scores[int(label)] += neighbor_similarities[idx]
        
        # Ensure smoothness with small addition to all scores
        class_scores += eps
        all_class_scores.append(class_scores.copy())
            
        # Apply softmax with temperature to enhance class separation
        if use_softmax:
            class_scores = class_scores / temp
            probs[i] = softmax(class_scores)
        else:
            # Simple normalization
            probs[i] = class_scores / np.sum(class_scores)
            
    # Optional: Save probability histograms to analyze distribution
    all_class_scores = np.vstack(all_class_scores)
    np.save("debug_class_scores.npy", all_class_scores)
    
    return probs

def calculate_comprehensive_metrics(y_true, y_pred, y_proba, class_names, ids_test):
    """Calculate all metrics matching the LogisticRegression evaluation"""
    n_classes = len(class_names)
    
    # Basic metrics
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
        "accuracy": accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred)
    }
    
    # Detailed classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics["classification_report"] = report
    metrics["per_class_metrics"] = report
    
    # Macro and weighted averages
    metrics['macro_precision'] = report['macro avg']['precision']
    metrics['macro_recall'] = report['macro avg']['recall']
    metrics['weighted_precision'] = report['weighted avg']['precision']
    metrics['weighted_recall'] = report['weighted avg']['recall']
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    
    # ROC AUC scores
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    roc_auc_per_class = {}
    for i, class_name in enumerate(class_names):
        try:
            roc_auc_per_class[class_name] = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        except:
            roc_auc_per_class[class_name] = 0.5
    metrics["roc_auc_per_class"] = roc_auc_per_class
    
    # Average precision scores
    avg_precision_per_class = {}
    for i, class_name in enumerate(class_names):
        try:
            avg_precision_per_class[class_name] = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        except:
            avg_precision_per_class[class_name] = 0.0
    metrics["avg_precision_per_class"] = avg_precision_per_class
    
    # Per-class confidence analysis
    confidence_analysis = {}
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.any(class_mask):
            class_proba = y_proba[class_mask, i]
            confidence_analysis[class_name] = {
                'mean_confidence': np.mean(class_proba),
                'std_confidence': np.std(class_proba),
                'min_confidence': np.min(class_proba),
                'max_confidence': np.max(class_proba)
            }
    metrics['confidence_analysis'] = confidence_analysis
    
    # Misclassified slides
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            confidence = y_proba[i, pred]
            misclassified.append({
                'slide_id': ids_test[i],
                'true_class': class_names[true],
                'pred_class': class_names[pred],
                'confidence': float(confidence),
                'true_idx': int(true),
                'pred_idx': int(pred)
            })
    metrics["misclassified"] = misclassified
    
    return metrics

def create_visualizations(metrics, class_names, results_dir, seed, k, y_true, y_proba):
    """Create comprehensive visualizations"""
    n_classes = len(class_names)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = metrics["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title(f'Confusion Matrix - Cosine Similarity (k={k}) - Seed {seed}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    total_samples = np.sum(cm)
    correct_predictions = np.sum(np.diag(cm))
    accuracy = correct_predictions / total_samples
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.3f}', 
             transform=plt.gca().transAxes, ha='center', fontsize=10)
    
    plt.tight_layout()
    cm_plot_path = os.path.join(results_dir, f"confusion_matrix_seed_{seed}_k_{k}.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curves
    plt.figure(figsize=(10, 8))
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc_score = metrics["roc_auc_per_class"][class_name]
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_score:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curves - Cosine Similarity (k={k}) - Seed {seed}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(results_dir, f"roc_curves_seed_{seed}_k_{k}.png")
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap_score = metrics["avg_precision_per_class"][class_name]
        plt.plot(recall, precision, label=f"{class_name} (AP={ap_score:.3f})", linewidth=2)
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curves - Cosine Similarity (k={k}) - Seed {seed}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_plot_path = os.path.join(results_dir, f"precision_recall_curves_seed_{seed}_k_{k}.png")
    plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Confidence Analysis
    if 'confidence_analysis' in metrics:
        plt.figure(figsize=(12, 6))
        confidence_data = metrics['confidence_analysis']
        
        class_names_conf = list(confidence_data.keys())
        mean_confidences = [confidence_data[cn]['mean_confidence'] for cn in class_names_conf]
        std_confidences = [confidence_data[cn]['std_confidence'] for cn in class_names_conf]
        
        bars = plt.bar(class_names_conf, mean_confidences, yerr=std_confidences, 
                      capsize=5, alpha=0.7)
        plt.title(f'Mean Prediction Confidence by Class - Seed {seed}, k={k}', fontweight='bold')
        plt.ylabel('Mean Confidence')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, conf in zip(bars, mean_confidences):
            height = bar.get_height()
            plt.annotate(f'{conf:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        conf_plot_path = os.path.join(results_dir, f"confidence_analysis_seed_{seed}_k_{k}.png")
        plt.savefig(conf_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return cm_plot_path, roc_plot_path, pr_plot_path

def save_comprehensive_results(metrics, seed, k, results_dir, similarities, y_true, y_pred, y_proba):
    """Save comprehensive results in multiple formats"""
    
    # Save as pickle
    results_dict = {
        "seed": seed,
        "k": k,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metrics": metrics,
        "similarities": similarities,
        "predictions": {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
    }
    
    results_pkl_path = os.path.join(results_dir, f"cosine_similarity_results_seed_{seed}_k_{k}.pkl")
    with open(results_pkl_path, 'wb') as f:
        pickle.dump(results_dict, f)
    
    # Save detailed text report
    report_path = os.path.join(results_dir, f"cosine_similarity_report_seed_{seed}_k_{k}.txt")
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"COSINE SIMILARITY BASELINE RESULTS - SEED {seed}, k={k}\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXPERIMENT INFO:\n")
        f.write("-"*40 + "\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"k value: {k}\n")
        f.write(f"Timestamp: {results_dict['timestamp']}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n")
        f.write(f"Matthews Correlation: {metrics['matthews_corrcoef']:.4f}\n")
        f.write(f"Macro Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {metrics['macro_recall']:.4f}\n")
        f.write(f"Weighted Precision: {metrics['weighted_precision']:.4f}\n")
        f.write(f"Weighted Recall: {metrics['weighted_recall']:.4f}\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS:\n")
        f.write("-"*40 + "\n")
        if 'per_class_metrics' in metrics:
            report_dict = metrics['per_class_metrics']
            f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-"*70 + "\n")
            
            class_names = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
            for class_name in class_names:
                if class_name in report_dict:
                    class_metrics = report_dict[class_name]
                    f.write(f"{class_name:<20} {class_metrics['precision']:<10.4f} {class_metrics['recall']:<10.4f} "
                           f"{class_metrics['f1-score']:<10.4f} {class_metrics['support']:<10.0f}\n")
        
        # ROC AUC and AP scores
        f.write("\nROC AUC SCORES:\n")
        f.write("-"*30 + "\n")
        for class_name, auc in metrics["roc_auc_per_class"].items():
            f.write(f"{class_name:<20}: {auc:.4f}\n")
        
        f.write("\nAVERAGE PRECISION SCORES:\n")
        f.write("-"*30 + "\n")
        for class_name, ap in metrics["avg_precision_per_class"].items():
            f.write(f"{class_name:<20}: {ap:.4f}\n")
        
        # Misclassified slides
        f.write(f"\nMISCLASSIFIED SLIDES: {len(metrics['misclassified'])}\n")
        f.write("-"*30 + "\n")
        
        if metrics['misclassified']:
            f.write(f"{'Slide ID':<25} {'True Class':<20} {'Predicted Class':<20} {'Confidence':<12}\n")
            f.write("-"*80 + "\n")
            for error in metrics['misclassified']:
                f.write(f"{error['slide_id']:<25} {error['true_class']:<20} "
                       f"{error['pred_class']:<20} {error['confidence']:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    return results_pkl_path, report_path

def run_cosine_similarity_for_seed(seed, data_dir, output_dir, k_values=[1, 3, 5, 7, 9]):
    """Run cosine similarity baseline for a specific seed"""
    
    # Find the data split file
    split_file = None
    candidates = [
        os.path.join(data_dir, f"seed_{seed}_TCGA_Warwick_Clean.pkl"),
        os.path.join(data_dir, "data_splitsV2", f"seed_{seed}_TCGA_Warwick_Clean.pkl"),
        f"/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_mean_features/data_splits/data_splitsV3/seed_{seed}_TCGA_Warwick_Clean.pkl"
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            split_file = candidate
            break
    
    if not split_file:
        print(f"❌ Could not find split file for seed {seed}")
        return None
    
    print(f"\n{'='*70}")
    print(f"COSINE SIMILARITY BASELINE - SEED {seed}")
    print(f"{'='*70}")
    
    # Create output directory for this seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_results_dir = os.path.join(output_dir, f"CosineSimilarity_Baseline_seed_{seed}_{timestamp}")
    os.makedirs(seed_results_dir, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test, ids_train, ids_test, class_names = load_data_split(split_file)
    n_classes = len(class_names)
    
    # Compute and save similarities
    similarities = compute_and_save_similarities(X_test, X_train, ids_test, ids_train, seed_results_dir)
    
    # Test different k values
    best_k = None
    best_balanced_acc = 0
    all_results = {}
    
    for k in k_values:
        print(f"\n{'='*40}")
        print(f"Evaluating k={k}...")
        print(f"{'='*40}")
        
        # Predict with current k
        y_proba = predict_with_knn_proba(X_train, y_train, X_test, class_names, k=k)
        y_pred = np.argmax(y_proba, axis=1)
        
        # Calculate comprehensive metrics
        print("Calculating metrics...")
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba, class_names, ids_test)
        
        # Track best k
        if metrics["balanced_accuracy"] > best_balanced_acc:
            best_balanced_acc = metrics["balanced_accuracy"]
            best_k = k
        
        # Create visualizations
        print("Creating visualizations...")
        cm_plot, roc_plot, pr_plot = create_visualizations(metrics, class_names, seed_results_dir, seed, k, y_test, y_proba)
        
        # Save results
        print("Saving results...")
        pkl_path, report_path = save_comprehensive_results(
            metrics, seed, k, seed_results_dir, similarities, y_test, y_pred, y_proba
        )
        
        all_results[k] = {
            'metrics': metrics,
            'plots': {
                'confusion_matrix': cm_plot,
                'roc_curves': roc_plot,
                'precision_recall_curves': pr_plot
            },
            'files': {
                'results_pkl': pkl_path,
                'report_txt': report_path
            }
        }
        
        print(f"✅ Results for k={k}:")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        print(f"  Misclassified: {len(metrics['misclassified'])}/{len(y_test)}")
    
    # Save summary for this seed
    summary_path = os.path.join(seed_results_dir, f"seed_{seed}_summary.json")
    summary = {
        'seed': seed,
        'best_k': best_k,
        'best_balanced_accuracy': best_balanced_acc,
        'k_values_tested': k_values,
        'results_by_k': {k: {'balanced_accuracy': all_results[k]['metrics']['balanced_accuracy'],
                            'macro_f1': all_results[k]['metrics']['macro_f1'],
                            'cohen_kappa': all_results[k]['metrics']['cohen_kappa']} 
                        for k in k_values}
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Seed {seed} completed. Best k={best_k} (Balanced Accuracy: {best_balanced_acc:.4f})")
    print(f"📁 Results saved to: {seed_results_dir}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Cosine Similarity Baseline')
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                        help='List of seeds to process (e.g., --seeds 42 123 456)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory containing split files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory name')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 3, 5, 7, 9],
                        help='k values to test')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Cosine Similarity Baseline evaluation")
    print(f"📊 Seeds: {args.seeds}")
    print(f"📊 k values: {args.k_values}")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Run for each seed
    for seed in args.seeds:
        try:
            print(f"\n🔄 Processing seed {seed}...")
            results = run_cosine_similarity_for_seed(
                seed, args.data_dir, args.output_dir, args.k_values
            )
            if results:
                print(f"✅ Successfully processed seed {seed}")
            else:
                print(f"❌ Failed to process seed {seed}")
        except Exception as e:
            print(f"❌ Error processing seed {seed}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Cosine Similarity Baseline completed!")

if __name__ == "__main__":
    main()