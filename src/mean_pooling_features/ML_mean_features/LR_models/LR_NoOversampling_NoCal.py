#!/usr/bin/env python3
"""
LogisticRegression OvR Baseline (NO Oversampling, NO Calibration)
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_curve, roc_auc_score, balanced_accuracy_score, 
                           f1_score, accuracy_score, confusion_matrix,
                           precision_recall_curve, average_precision_score,
                           classification_report, cohen_kappa_score,
                           matthews_corrcoef)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')
import argparse

def plot_confusion_matrix(cm, class_names, results_dir, seed, title_suffix=""):
    """
    Create a beautiful confusion matrix visualization
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title(f'Confusion Matrix - Seed {seed}{title_suffix}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy text
    total_samples = np.sum(cm)
    correct_predictions = np.sum(np.diag(cm))
    accuracy = correct_predictions / total_samples
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.3f}', 
             transform=plt.gca().transAxes, ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    cm_plot_path = os.path.join(results_dir, f"confusion_matrix{title_suffix.lower().replace(' ', '_')}.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm_plot_path

def calculate_cv_error_analysis(cv_scores_per_class, class_names):
    """
    Calculate comprehensive CV error analysis
    """
    cv_analysis = {}
    
    for class_data in cv_scores_per_class:
        class_name = class_data['class']
        cv_results = class_data['cv_results']
        
        # Extract CV scores - GridSearchCV uses different key names
        # Find the best parameter combination index
        best_idx = cv_results['rank_test_score'].argmin()
        
        # Get all CV scores for the best parameter combination
        cv_scores = []
        cv_train_scores = []
        
        # Extract scores for each fold
        for fold in range(len(cv_results['split0_test_score'])):
            test_key = f'split{fold}_test_score'
            train_key = f'split{fold}_train_score'
            
            if test_key in cv_results and train_key in cv_results:
                cv_scores.append(cv_results[test_key][best_idx])
                cv_train_scores.append(cv_results[train_key][best_idx])
        
        # Convert to numpy arrays
        cv_scores = np.array(cv_scores)
        cv_train_scores = np.array(cv_train_scores)
        
        # Calculate statistics
        cv_stats = {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'min_cv_score': np.min(cv_scores),
            'max_cv_score': np.max(cv_scores),
            'cv_score_range': np.max(cv_scores) - np.min(cv_scores),
            'cv_scores': cv_scores.tolist(),
            'mean_train_score': np.mean(cv_train_scores),
            'std_train_score': np.std(cv_train_scores),
            'train_scores': cv_train_scores.tolist(),
            'overfitting_gap': np.mean(cv_train_scores) - np.mean(cv_scores),
            'stability_score': 1 - (np.std(cv_scores) / np.mean(cv_scores)) if np.mean(cv_scores) > 0 else 0
        }
        
        cv_analysis[class_name] = cv_stats
    
    return cv_analysis

def plot_cv_analysis(cv_analysis, class_names, results_dir, seed):
    """
    Create comprehensive CV analysis plots
    """
    n_classes = len(class_names)
    
    # 1. CV Score Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # CV scores boxplot
    cv_scores_data = [cv_analysis[class_name]['cv_scores'] for class_name in class_names]
    axes[0, 0].boxplot(cv_scores_data, labels=class_names)
    axes[0, 0].set_title('CV Score Distribution by Class', fontweight='bold')
    axes[0, 0].set_ylabel('CV Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Train vs CV scores
    train_scores = [cv_analysis[class_name]['mean_train_score'] for class_name in class_names]
    cv_scores = [cv_analysis[class_name]['mean_cv_score'] for class_name in class_names]
    x_pos = np.arange(len(class_names))
    width = 0.35
    
    axes[0, 1].bar(x_pos - width/2, train_scores, width, label='Train', alpha=0.8)
    axes[0, 1].bar(x_pos + width/2, cv_scores, width, label='CV', alpha=0.8)
    axes[0, 1].set_title('Train vs CV Scores', fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overfitting analysis
    overfitting_gaps = [cv_analysis[class_name]['overfitting_gap'] for class_name in class_names]
    colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in overfitting_gaps]
    axes[1, 0].bar(class_names, overfitting_gaps, color=colors, alpha=0.7)
    axes[1, 0].set_title('Overfitting Analysis (Train - CV)', fontweight='bold')
    axes[1, 0].set_ylabel('Score Gap')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='High overfitting')
    axes[1, 0].axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Moderate overfitting')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Stability analysis
    stability_scores = [cv_analysis[class_name]['stability_score'] for class_name in class_names]
    cv_stds = [cv_analysis[class_name]['std_cv_score'] for class_name in class_names]
    axes[1, 1].scatter(cv_stds, stability_scores, s=100, alpha=0.7)
    for i, class_name in enumerate(class_names):
        axes[1, 1].annotate(class_name, (cv_stds[i], stability_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1, 1].set_title('Model Stability Analysis', fontweight='bold')
    axes[1, 1].set_xlabel('CV Standard Deviation')
    axes[1, 1].set_ylabel('Stability Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    cv_analysis_path = os.path.join(results_dir, f"cv_analysis_seed_{seed}.png")
    plt.savefig(cv_analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cv_analysis_path

def calculate_additional_metrics(y_true, y_pred, y_proba, class_names):
    """
    Calculate additional useful metrics
    """
    additional_metrics = {}
    
    # Multi-class metrics
    additional_metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    additional_metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    additional_metrics['per_class_metrics'] = report
    
    # Macro and weighted averages
    additional_metrics['macro_precision'] = report['macro avg']['precision']
    additional_metrics['macro_recall'] = report['macro avg']['recall']
    additional_metrics['weighted_precision'] = report['weighted avg']['precision']
    additional_metrics['weighted_recall'] = report['weighted avg']['recall']
    
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
    
    additional_metrics['confidence_analysis'] = confidence_analysis
    
    return additional_metrics

def plot_confidence_analysis(confidence_analysis, results_dir, seed):
    """
    Plot confidence analysis
    """
    class_names = list(confidence_analysis.keys())
    mean_confidences = [confidence_analysis[class_name]['mean_confidence'] for class_name in class_names]
    std_confidences = [confidence_analysis[class_name]['std_confidence'] for class_name in class_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mean confidence by class
    bars = ax1.bar(class_names, mean_confidences, yerr=std_confidences, capsize=5, alpha=0.7)
    ax1.set_title('Mean Prediction Confidence by Class', fontweight='bold')
    ax1.set_ylabel('Mean Confidence')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, conf in zip(bars, mean_confidences):
        height = bar.get_height()
        ax1.annotate(f'{conf:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # Confidence distribution
    for i, class_name in enumerate(class_names):
        mean_conf = confidence_analysis[class_name]['mean_confidence']
        std_conf = confidence_analysis[class_name]['std_confidence']
        ax2.errorbar(i, mean_conf, yerr=std_conf, fmt='o', capsize=5, capthick=2, label=class_name)
    
    ax2.set_title('Confidence Distribution by Class', fontweight='bold')
    ax2.set_ylabel('Confidence')
    ax2.set_xlabel('Class')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    confidence_plot_path = os.path.join(results_dir, f"confidence_analysis_seed_{seed}.png")
    plt.savefig(confidence_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return confidence_plot_path

def calculate_threshold_analysis(y_true, y_pred, y_proba, class_names, ids_test):
    """
    Analyze optimal confidence thresholds and rejection strategies
    """
    threshold_analysis = {}
    
    # Test different confidence thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        # Calculate metrics when rejecting low-confidence predictions
        high_conf_mask = np.max(y_proba, axis=1) >= threshold
        
        if np.sum(high_conf_mask) > 0:
            y_true_filtered = y_true[high_conf_mask]
            y_pred_filtered = y_pred[high_conf_mask]
            
            accuracy_filtered = accuracy_score(y_true_filtered, y_pred_filtered)
            balanced_acc_filtered = balanced_accuracy_score(y_true_filtered, y_pred_filtered)
            
            threshold_analysis[threshold] = {
                'samples_kept': np.sum(high_conf_mask),
                'samples_rejected': len(y_true) - np.sum(high_conf_mask),
                'rejection_rate': (len(y_true) - np.sum(high_conf_mask)) / len(y_true),
                'accuracy': accuracy_filtered,
                'balanced_accuracy': balanced_acc_filtered,
                'improvement': accuracy_filtered - accuracy_score(y_true, y_pred)
            }
    
    # Per-class confidence thresholds
    class_thresholds = {}
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.any(class_mask):
            class_proba = y_proba[class_mask, i]
            class_pred = (y_pred == i)[class_mask]
            
            # Find optimal threshold for this class
            best_threshold = 0.5
            best_precision = 0
            
            for thresh in np.arange(0.1, 1.0, 0.1):
                high_conf = class_proba >= thresh
                if np.sum(high_conf) > 0:
                    precision = np.mean(class_pred[high_conf])
                    if precision > best_precision:
                        best_precision = precision
                        best_threshold = thresh
            
            class_thresholds[class_name] = {
                'optimal_threshold': best_threshold,
                'precision_at_threshold': best_precision,
                'samples_above_threshold': np.sum(class_proba >= best_threshold)
            }
    
    return threshold_analysis, class_thresholds

def train_logistic_regression_ovr_baseline(seed, data_dir="data_splits", output_dir="results"):
    """
    Baseline LogisticRegression training without oversampling or calibration
    """
    
    print(f"="*70)
    print(f"BASELINE LOGISTIC REGRESSION (NO OVERSAMPLING, NO CALIBRATION) - SEED {seed}")
    print(f"="*70)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"Baseline_LogReg_OvR_NoOversampling_seed_{seed}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results directory: {results_dir}")
    
    # Load seed-specific data
    print(f"\n📂 Loading data for seed {seed}...")
    try:
        data_file = os.path.join(data_dir, f"seed_{seed}_TCGA_Warwick_Clean.pkl")
        if not os.path.exists(data_file):
            data_file = os.path.join(data_dir, "data_splitsV3", f"seed_{seed}_TCGA_Warwick_Clean.pkl")
        if not os.path.exists(data_file):
            data_file = f"/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_mean_features/data_splits/data_splitsV3/seed_{seed}_TCGA_Warwick_Clean.pkl"
        if not os.path.exists(data_file):
            data_file = os.path.join(data_dir, f"seed_{seed}_data.pkl")
        print(f"   Trying to load: {data_file}")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Data loaded successfully from: {data_file}")
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_file}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Data directory parameter: {data_dir}")
        sys.exit(1)
    
    # Extract data
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    cv_splits = data['cv_splits']
    class_names = data['class_names']
    ids_test = data['ids_test']
    n_classes = len(class_names)
    
    print(f"📊 Training: {X_train.shape}, Test: {X_test.shape}")
    print(f"📊 Classes: {class_names}")
    print(f"📊 CV folds: {len(cv_splits)}")
    
    # Parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'class_weight': [None],
        'solver': ['liblinear', 'lbfgs']
    }
    print(f"\n🔧 Parameter grid: {param_grid}")
    
    print(f"\n🚀 TRAINING ONE-VS-REST CLASSIFIERS (NO OVERSAMPLING, NO CALIBRATION)")
    print("-" * 50)
    
    best_classifiers = []
    best_params_per_class = []
    cv_scores_per_class = []
    
    for class_idx, class_name in enumerate(class_names):
        print(f"\n🔎 Tuning classifier for class '{class_name}' (OvR)...")
        y_train_bin = (y_train == class_idx).astype(int)
        print(f"   Class distribution: {np.bincount(y_train_bin)}")
        clf = LogisticRegression(max_iter=2000, random_state=seed, n_jobs=1)
        gs = GridSearchCV(
            clf, param_grid, 
            cv=cv_splits,
            scoring='balanced_accuracy', 
            n_jobs=1, 
            verbose=0, 
            return_train_score=True
        )
        gs.fit(X_train, y_train_bin)
        print(f"   ✅ Best params: {gs.best_params_}")
        print(f"   ✅ Best CV BA: {gs.best_score_:.4f}")
        cv_scores_per_class.append({
            'class': class_name,
            'class_idx': class_idx,
            'cv_results': gs.cv_results_,
            'best_params': gs.best_params_,
            'best_score': gs.best_score_,
            'original_distribution': np.bincount(y_train_bin).tolist(),
            'oversampled_distribution': None
        })
        best_clf = LogisticRegression(
            max_iter=2000, 
            random_state=seed, 
            n_jobs=1, 
            **gs.best_params_
        )
        best_clf.fit(X_train, y_train_bin)
        best_classifiers.append(best_clf)
        best_params_per_class.append(gs.best_params_)

    # Calculate CV error analysis
    print(f"\n📊 CALCULATING CV ERROR ANALYSIS")
    print("-" * 40)
    cv_analysis = calculate_cv_error_analysis(cv_scores_per_class, class_names)
    
    # Print CV analysis summary
    print("\nCV Analysis Summary:")
    for class_name in class_names:
        analysis = cv_analysis[class_name]
        print(f"  {class_name:15s}: CV={analysis['mean_cv_score']:.4f}±{analysis['std_cv_score']:.4f}, "
              f"Overfitting={analysis['overfitting_gap']:.4f}, Stability={analysis['stability_score']:.4f}")
    
    # Predict on test set (OvR)
    print(f"\n🎯 PREDICTING ON TEST SET")
    print("-" * 30)
    
    y_proba_ovr = np.zeros((X_test.shape[0], n_classes))
    for class_idx, clf in enumerate(best_classifiers):
        y_proba_ovr[:, class_idx] = clf.predict_proba(X_test)[:, 1]
    
    y_pred_ovr = np.argmax(y_proba_ovr, axis=1)
    
    # Basic test metrics
    test_metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_ovr),
        "macro_f1": f1_score(y_test, y_pred_ovr, average='macro'),
        "weighted_f1": f1_score(y_test, y_pred_ovr, average='weighted'),
        "accuracy": accuracy_score(y_test, y_pred_ovr)
    }
    
    # Calculate additional metrics
    additional_metrics = calculate_additional_metrics(y_test, y_pred_ovr, y_proba_ovr, class_names)
    
    # Combine all metrics
    all_metrics = {**test_metrics, **additional_metrics}
    
    print(f"\n📈 ENHANCED TEST SET METRICS:")
    print("-" * 35)
    print(f"   Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"   Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"   Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Cohen's Kappa: {additional_metrics['cohen_kappa']:.4f}")
    print(f"   Matthews Correlation: {additional_metrics['matthews_corrcoef']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_ovr)
    print(f"\n📊 Confusion matrix:\n{cm}")
    
    # Misclassified analysis
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_test, y_pred_ovr)):
        if true != pred:
            confidence = y_proba_ovr[i, pred]
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
    
    # VISUALIZATIONS
    print(f"\n📊 GENERATING ENHANCED VISUALIZATIONS")
    print("-" * 45)
    
    # 1. Confusion Matrix
    cm_plot_path = plot_confusion_matrix(cm, class_names, results_dir, seed)
    print(f"✅ Confusion matrix plot saved: {cm_plot_path}")
    
    # 2. CV Analysis
    cv_plot_path = plot_cv_analysis(cv_analysis, class_names, results_dir, seed)
    print(f"✅ CV analysis plots saved: {cv_plot_path}")
    
    # 3. Confidence Analysis
    confidence_plot_path = plot_confidence_analysis(additional_metrics['confidence_analysis'], results_dir, seed)
    print(f"✅ Confidence analysis saved: {confidence_plot_path}")
    
    # 4. ROC curves (existing)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    roc_auc_per_class = {}
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_ovr[:, i])
        auc_score = roc_auc_score(y_test_bin[:, i], y_proba_ovr[:, i])
        roc_auc_per_class[class_names[i]] = auc_score
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
    
    # 5. Precision-Recall curves (existing)
    avg_precision_per_class = {}
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba_ovr[:, i])
        avg_precision = average_precision_score(y_test_bin[:, i], y_proba_ovr[:, i])
        avg_precision_per_class[class_names[i]] = avg_precision
        plt.plot(recall, precision, label=f"{class_names[i]} (AP={avg_precision:.3f})", linewidth=2)
    
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
    
    # SAVE RESULTS
    print(f"\n💾 SAVING ENHANCED RESULTS")
    print("-" * 30)

    # Save comprehensive text report
    results_txt_path = os.path.join(results_dir, "comprehensive_results.txt")
    with open(results_txt_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"ENHANCED LOGISTIC REGRESSION RESULTS - SEED {seed}\n")
        f.write("="*80 + "\n\n")
        
        # Experiment info
        f.write("EXPERIMENT INFO:\n")
        f.write("-"*40 + "\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Data file: {data_file}\n")
        f.write(f"Results directory: {results_dir}\n\n")
        
        # Data info
        f.write("DATA INFO:\n")
        f.write("-"*40 + "\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Number of features: {X_train.shape[1]}\n")
        f.write(f"Number of classes: {n_classes}\n")
        f.write(f"Class names: {class_names}\n\n")
        
        # Overall test metrics
        f.write("OVERALL TEST SET METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Macro F1: {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {test_metrics['weighted_f1']:.4f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa: {additional_metrics['cohen_kappa']:.4f}\n")
        f.write(f"Matthews Correlation: {additional_metrics['matthews_corrcoef']:.4f}\n\n")
        
        # Per-class test metrics
        f.write("PER-CLASS TEST SET METRICS:\n")
        f.write("-"*40 + "\n")
        report_dict = additional_metrics['per_class_metrics']
        f.write(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-"*70 + "\n")
        for class_name in class_names:
            if class_name in report_dict:
                metrics = report_dict[class_name]
                f.write(f"{class_name:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10.0f}\n")
        
        # Add macro and weighted averages
        f.write("-"*70 + "\n")
        f.write(f"{'Macro Avg':<20} {report_dict['macro avg']['precision']:<10.4f} {report_dict['macro avg']['recall']:<10.4f} {report_dict['macro avg']['f1-score']:<10.4f} {report_dict['macro avg']['support']:<10.0f}\n")
        f.write(f"{'Weighted Avg':<20} {report_dict['weighted avg']['precision']:<10.4f} {report_dict['weighted avg']['recall']:<10.4f} {report_dict['weighted avg']['f1-score']:<10.4f} {report_dict['weighted avg']['support']:<10.0f}\n\n")
        
        # ROC AUC per class
        f.write("ROC AUC SCORES PER CLASS:\n")
        f.write("-"*40 + "\n")
        for class_name in class_names:
            if class_name in roc_auc_per_class:
                f.write(f"{class_name:<20}: {roc_auc_per_class[class_name]:.4f}\n")
        f.write("\n")
        
        # Average Precision per class
        f.write("AVERAGE PRECISION SCORES PER CLASS:\n")
        f.write("-"*40 + "\n")
        for class_name in class_names:
            if class_name in avg_precision_per_class:
                f.write(f"{class_name:<20}: {avg_precision_per_class[class_name]:.4f}\n")
        f.write("\n")
        
        # Confusion Matrix
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
        
        # CV Analysis Summary
        f.write("CROSS-VALIDATION ANALYSIS:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Class':<20} {'Mean CV':<10} {'Std CV':<10} {'Overfitting':<12} {'Stability':<10}\n")
        f.write("-"*70 + "\n")
        for class_name in class_names:
            if class_name in cv_analysis:
                analysis = cv_analysis[class_name]
                f.write(f"{class_name:<20} {analysis['mean_cv_score']:<10.4f} {analysis['std_cv_score']:<10.4f} {analysis['overfitting_gap']:<12.4f} {analysis['stability_score']:<10.4f}\n")
        f.write("\n")
        
        # Best parameters per class
        f.write("BEST PARAMETERS PER CLASS:\n")
        f.write("-"*40 + "\n")
        for class_name, params in zip(class_names, best_params_per_class):
            f.write(f"{class_name}:\n")
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
        
        # Confidence analysis
        f.write("CONFIDENCE ANALYSIS:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Class':<20} {'Mean Conf':<12} {'Std Conf':<12} {'Min Conf':<12} {'Max Conf':<12}\n")
        f.write("-"*80 + "\n")
        for class_name in class_names:
            if class_name in additional_metrics['confidence_analysis']:
                conf = additional_metrics['confidence_analysis'][class_name]
                f.write(f"{class_name:<20} {conf['mean_confidence']:<12.4f} {conf['std_confidence']:<12.4f} {conf['min_confidence']:<12.4f} {conf['max_confidence']:<12.4f}\n")
        f.write("\n")
        
        # Misclassified slides
        f.write("MISCLASSIFIED SLIDES:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total misclassified: {len(misclassified)}\n\n")
        
        if misclassified:
            f.write(f"{'Slide ID':<25} {'True Class':<20} {'Predicted Class':<20} {'Confidence':<12}\n")
            f.write("-"*80 + "\n")
            for error in misclassified:
                f.write(f"{error['slide_id']:<25} {error['true_class']:<20} {error['pred_class']:<20} {error['confidence']:<12.4f}\n")
        else:
            f.write("No misclassified slides!\n")
        
        f.write("\n")
        
        # Detailed classification report
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write("-"*40 + "\n")
        f.write(classification_report(y_test, y_pred_ovr, target_names=class_names))
        
        # THRESHOLD ANALYSIS (NEW)
        threshold_analysis, class_thresholds = calculate_threshold_analysis(
            y_test, y_pred_ovr, y_proba_ovr, class_names, ids_test)
        
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
                thresh_info = class_thresholds[class_name]
                f.write(f"{class_name}:\n")
                f.write(f"  Optimal threshold: {thresh_info['optimal_threshold']:.2f}\n")
                f.write(f"  Precision at threshold: {thresh_info['precision_at_threshold']:.4f}\n")
                f.write(f"  Samples above threshold: {thresh_info['samples_above_threshold']}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    # Save as pickle for programmatic access
    results_pkl_path = os.path.join(results_dir, "enhanced_all_results.pkl")
    results_dict = {
        "seed": seed,
        "timestamp": timestamp,
        "test_metrics": test_metrics,
        "additional_metrics": additional_metrics,
        "confusion_matrix": cm,
        "roc_auc_per_class": roc_auc_per_class,
        "avg_precision_per_class": avg_precision_per_class,
        "misclassified": misclassified,
        "best_params_per_class": best_params_per_class,
        "cv_analysis": cv_analysis,
        "predictions": {
            "y_pred": y_pred_ovr,
            "y_true": y_test,
            "y_proba": y_proba_ovr
        }
    }

    with open(results_pkl_path, "wb") as f:
        pickle.dump(results_dict, f)

    # Save trained models
    models_dir = os.path.join(results_dir, "trained_models")
    os.makedirs(models_dir, exist_ok=True)

    for i, (clf, class_name) in enumerate(zip(best_classifiers, class_names)):
        model_path = os.path.join(models_dir, f"classifier_{class_name}_{i}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)

    print(f"✅ Comprehensive text report saved: {results_txt_path}")
    print(f"✅ Results pickle saved: {results_pkl_path}")
    print(f"✅ Trained models saved: {models_dir}")
    
    print(f"\n🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Seed: {seed}")
    print(f"📁 Results directory: {results_dir}")
    print(f"🏆 Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"🏆 Test F1-Macro: {test_metrics['macro_f1']:.4f}")
    print(f"🏆 Cohen's Kappa: {additional_metrics['cohen_kappa']:.4f}")
    
    # ...existing code...
    results = {
        'y_true': y_test,
        'y_pred': y_pred_ovr,
        'y_proba': y_proba_ovr,
        # Add more if needed, but only use defined variables
    }
    return results

def main():
    parser = argparse.ArgumentParser(description='Baseline LogisticRegression OvR (no oversampling, no calibration)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='data_splits', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    print(f"🚀 Starting Baseline LogisticRegression training with seed={args.seed}")
    print(f"📂 Data dir: {args.data_dir}")
    print(f"📁 Output dir: {args.output_dir}")
    train_logistic_regression_ovr_baseline(
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()