"""
Linear Discriminant Analysis (LDA) for WSI Feature Visualization

This script performs comprehensive LDA analysis on mean-pooled WSI features to:
- Reduce high-dimensional features to interpretable components
- Visualize class separation in lower-dimensional space
- Analyze discriminative power of each LDA component
- Generate detailed visualizations and statistical summaries

LDA is particularly effective for WSI classification because it:
- Maximizes between-class variance while minimizing within-class variance
- Reduces dimensionality from 1536D to (n_classes - 1)D
- Creates linearly separable representations optimized for classification

Author: Konstantinos Papagoras
Date: 2025-06
"""

import os
import pickle
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# Configuration Constants
DEFAULT_LABEL_MAPPING = {
    'LumA': 'LumA',
    'BRCA_LumA': 'LumA',
    'LumB': 'LumB',
    'BRCA_LumB': 'LumB',
    'Basal': 'Basal',
    'BRCA_Basal': 'Basal',
    'HER2': 'HER2',
    'BRCA_Her2': 'HER2',
    'nan': 'exclude',
    'unknown': 'exclude',
    'BRCA_Normal': 'exclude',
    'Normal': 'exclude',
    'NORMAL': 'exclude',
    'Benign': 'exclude',
    'benign': 'exclude'
}

DEFAULT_OUTPUT_FILENAME = 'lda_detailed_analysis.png'
DEFAULT_DPI = 300


def load_and_clean_data(
    data_file: str,
    label_mapping: Dict[str, str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and clean WSI feature data with label standardization.
    
    Args:
        data_file: Path to pickle file containing features
        label_mapping: Dictionary mapping original labels to standardized labels
        
    Returns:
        Tuple of (features_clean, labels_clean, names_clean, unique_labels)
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        KeyError: If required keys missing from data
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    if label_mapping is None:
        label_mapping = DEFAULT_LABEL_MAPPING
    
    print("="*70)
    print("LOADING AND CLEANING DATA")
    print("="*70)
    
    # Load data
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    # Validate required keys
    required_keys = ['slide_features', 'slide_labels', 'slide_names']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in data file")
    
    slide_features = data['slide_features']
    slide_labels_raw = data['slide_labels']
    slide_names = data['slide_names']
    
    print(f"Loaded data from: {os.path.basename(data_file)}")
    print(f"Total samples: {len(slide_labels_raw)}")
    print(f"Feature dimension: {slide_features.shape[1]}")
    
    # Apply label mapping
    slide_labels = np.array([
        label_mapping.get(label, 'exclude') for label in slide_labels_raw
    ])
    
    # Get unique valid labels
    unique_labels = np.array([
        label for label in np.unique(slide_labels) if label != 'exclude'
    ])
    
    print(f"\nLabel standardization:")
    print(f"  Unique labels after mapping: {unique_labels}")
    
    # Filter valid samples
    valid_mask = slide_labels != 'exclude'
    features_clean = slide_features[valid_mask]
    labels_clean = slide_labels[valid_mask]
    names_clean = slide_names[valid_mask]
    
    # Print statistics
    excluded_count = len(slide_labels) - len(labels_clean)
    print(f"\nData cleaning:")
    print(f"  Original samples: {len(slide_labels)}")
    print(f"  Excluded samples: {excluded_count}")
    print(f"  Valid samples: {len(labels_clean)}")
    
    print(f"\nClass distribution:")
    for label in sorted(unique_labels):
        count = np.sum(labels_clean == label)
        percentage = count / len(labels_clean) * 100
        print(f"  • {label:15s}: {count:4d} samples ({percentage:5.1f}%)")
    
    return features_clean, labels_clean, names_clean, unique_labels


def perform_lda_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    unique_labels: np.ndarray
) -> Tuple[np.ndarray, LinearDiscriminantAnalysis, StandardScaler]:
    """
    Perform LDA transformation and detailed analysis.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Label array (n_samples,)
        unique_labels: Array of unique class labels
        
    Returns:
        Tuple of (lda_features, lda_model, scaler)
    """
    print("\n" + "="*70)
    print("PERFORMING LDA TRANSFORMATION")
    print("="*70)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply LDA
    lda = LinearDiscriminantAnalysis()
    features_lda = lda.fit_transform(features_scaled, labels)
    
    # Print transformation summary
    n_components = features_lda.shape[1]
    compression_ratio = features_scaled.shape[1] / n_components
    
    print(f"\nDimensionality Reduction:")
    print(f"  Original features: {features_scaled.shape[1]}D")
    print(f"  LDA components: {n_components}D")
    print(f"  Compression ratio: {compression_ratio:.1f}x")
    
    # Explained variance
    print(f"\nLDA Explained Variance Ratio:")
    for i, var_ratio in enumerate(lda.explained_variance_ratio_):
        print(f"  Component {i+1}: {var_ratio:.4f} ({var_ratio*100:.1f}%)")
    
    # Cumulative variance
    cumulative_variance = np.cumsum(lda.explained_variance_ratio_)
    print(f"\nCumulative Variance Explained:")
    for i, cum_var in enumerate(cumulative_variance):
        print(f"  First {i+1} component(s): {cum_var:.4f} ({cum_var*100:.1f}%)")
    
    return features_lda, lda, scaler


def analyze_component_separation(
    features_lda: np.ndarray,
    labels: np.ndarray,
    unique_labels: np.ndarray,
    lda: LinearDiscriminantAnalysis
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Analyze class separation for each LDA component.
    
    Args:
        features_lda: LDA-transformed features
        labels: Class labels
        unique_labels: Array of unique class labels
        lda: Fitted LDA model
        
    Returns:
        Tuple of (class_means_dict, class_std_dict)
    """
    print("\n" + "="*70)
    print("ANALYZING COMPONENT SEPARATION")
    print("="*70)
    
    # Calculate class statistics in LDA space
    class_means_lda = {}
    class_std_lda = {}
    
    for label in unique_labels:
        mask = labels == label
        class_means_lda[label] = np.mean(features_lda[mask], axis=0)
        class_std_lda[label] = np.std(features_lda[mask], axis=0)
    
    # Analyze each component
    for comp_idx in range(features_lda.shape[1]):
        print(f"\n{'─'*50}")
        print(f"LDA Component {comp_idx+1}")
        print(f"{'─'*50}")
        print(f"Variance explained: {lda.explained_variance_ratio_[comp_idx]*100:.1f}%")
        
        # Sort classes by their mean position on this component
        comp_means = {label: class_means_lda[label][comp_idx] for label in unique_labels}
        sorted_classes = sorted(comp_means.items(), key=lambda x: x[1])
        
        print("\nClass positions (low to high):")
        for label, mean_val in sorted_classes:
            std_val = class_std_lda[label][comp_idx]
            print(f"  {label:15s}: {mean_val:8.3f} (±{std_val:.3f})")
        
        # Calculate separation metrics
        mean_values = list(comp_means.values())
        separation_range = max(mean_values) - min(mean_values)
        avg_std = np.mean([class_std_lda[label][comp_idx] for label in unique_labels])
        separation_ratio = separation_range / avg_std if avg_std > 0 else 0
        
        print(f"\nSeparation metrics:")
        print(f"  Range: {separation_range:.3f}")
        print(f"  Average std: {avg_std:.3f}")
        print(f"  Separation power: {separation_ratio:.2f} (range/avg_std)")
    
    return class_means_lda, class_std_lda


def create_comprehensive_visualization(
    features_lda: np.ndarray,
    labels: np.ndarray,
    unique_labels: np.ndarray,
    lda: LinearDiscriminantAnalysis,
    class_means_lda: Dict[str, np.ndarray],
    output_file: str,
    dpi: int = DEFAULT_DPI
) -> None:
    """
    Create comprehensive LDA visualization with multiple plots.
    
    Args:
        features_lda: LDA-transformed features
        labels: Class labels
        unique_labels: Array of unique class labels
        lda: Fitted LDA model
        class_means_lda: Dictionary of class means in LDA space
        output_file: Path to save the figure
        dpi: Resolution for saved figure
    """
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    # Plot LDA component pairs
    plot_idx = 0
    n_components = features_lda.shape[1]
    
    for i in range(n_components):
        for j in range(i + 1, n_components):
            if plot_idx < 6:
                ax = axes[plot_idx // 3, plot_idx % 3]
                
                for label_idx, label in enumerate(unique_labels):
                    mask = labels == label
                    n_samples = np.sum(mask)
                    ax.scatter(
                        features_lda[mask, i],
                        features_lda[mask, j],
                        label=f'{label} (n={n_samples})',
                        alpha=0.7,
                        s=30,
                        color=colors[label_idx],
                        edgecolors='white',
                        linewidth=0.5
                    )
                
                ax.set_xlabel(f'LDA Component {i+1}', fontsize=11)
                ax.set_ylabel(f'LDA Component {j+1}', fontsize=11)
                ax.set_title(f'Components {i+1} vs {j+1}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                if plot_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                
                plot_idx += 1
    
    # Fill remaining subplots with additional analyses
    while plot_idx < 6:
        ax = axes[plot_idx // 3, plot_idx % 3]
        
        if plot_idx == 4:  # Variance explained
            components = range(1, len(lda.explained_variance_ratio_) + 1)
            variance_pct = lda.explained_variance_ratio_ * 100
            
            bars = ax.bar(components, variance_pct, color='steelblue', alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bar, val in zip(bars, variance_pct):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('LDA Component', fontsize=11)
            ax.set_ylabel('Variance Explained (%)', fontsize=11)
            ax.set_title('Variance Explained by Each Component', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_xticks(components)
        
        elif plot_idx == 5:  # Inter-class distances heatmap
            # Calculate pairwise distances between class means
            n_classes = len(unique_labels)
            separation_matrix = np.zeros((n_classes, n_classes))
            
            for i, label1 in enumerate(unique_labels):
                for j, label2 in enumerate(unique_labels):
                    if i != j:
                        mean1 = class_means_lda[label1]
                        mean2 = class_means_lda[label2]
                        separation_matrix[i, j] = np.linalg.norm(mean1 - mean2)
            
            sns.heatmap(
                separation_matrix,
                xticklabels=unique_labels,
                yticklabels=unique_labels,
                annot=True,
                fmt='.2f',
                cmap='viridis',
                ax=ax,
                cbar_kws={'label': 'Euclidean Distance'}
            )
            ax.set_title('LDA Inter-class Distances', fontsize=12, fontweight='bold')
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_file}")
    plt.show()


def print_interpretation_guide(unique_labels: np.ndarray, n_features: int) -> None:
    """
    Print detailed interpretation guide for LDA results.
    
    Args:
        unique_labels: Array of unique class labels
        n_features: Number of original features
    """
    n_components = len(unique_labels) - 1
    
    print("\n" + "="*70)
    print("MATHEMATICAL INTERPRETATION")
    print("="*70)
    
    print("\nLDA finds linear combinations of original features:")
    for comp_idx in range(min(n_components, 3)):  # Show first 3 components
        weight_terms = " + ".join([f"w{comp_idx+1}₍{i+1}₎×feature₍{i+1}₎" 
                                   for i in range(min(3, n_features))])
        print(f"  LDA_component_{comp_idx+1} = {weight_terms} + ... + "
              f"w{comp_idx+1}₍{n_features}₎×feature₍{n_features}₎")
    
    if n_components > 3:
        print(f"  ...")
        print(f"  LDA_component_{n_components} = [similar linear combination]")
    
    print(f"\nThe weights (w) are optimized to maximize class separation.")
    print(f"Each component captures a different discriminative pattern.")
    
    print("\n" + "="*70)
    print("BIOLOGICAL INTERPRETATION")
    print("="*70)
    
    print("\nFor UNI pathology features, LDA components likely capture:")
    component_interpretations = [
        "Primary tissue architecture differences",
        "Secondary morphological patterns",
        "Cellular organization features",
        "Fine-grained histological details"
    ]
    
    for i, interp in enumerate(component_interpretations[:n_components]):
        print(f"  Component {i+1}: {interp}")
    
    print("\n" + "="*70)
    print("WHY LDA WORKS WELL FOR WSI CLASSIFICATION")
    print("="*70)
    
    compression_ratio = n_features / n_components
    print(f"\nAdvantages:")
    print(f"  • Dimensionality reduction: {n_features}D → {n_components}D "
          f"({compression_ratio:.0f}x compression)")
    print(f"  • Each dimension optimized for class separation")
    print(f"  • Removes noise and irrelevant features")
    print(f"  • Creates linearly separable representation")
    print(f"  • Supervised learning using class labels")
    
    print("\n" + "="*70)
    print("COMPARISON WITH OTHER METHODS")
    print("="*70)
    
    print("\nPCA (Principal Component Analysis):")
    print("  • Unsupervised: finds directions of maximum variance")
    print("  • May capture noise or class-irrelevant variation")
    print("  • Not optimized for classification tasks")
    
    print("\nLDA (Linear Discriminant Analysis):")
    print("  • Supervised: finds directions of maximum class separation")
    print("  • Uses class labels to guide transformation")
    print("  • Specifically optimized for classification")
    print("  • Maximizes between-class / within-class variance ratio")
    
    print("\nResult: LDA transforms overlapping classes into well-separated clusters!")


def main(
    data_file: str = None,
    output_file: str = DEFAULT_OUTPUT_FILENAME,
    label_mapping: Dict[str, str] = None,
    dpi: int = DEFAULT_DPI
) -> None:
    """
    Main function to perform comprehensive LDA analysis.
    
    Args:
        data_file: Path to pickle file containing WSI features
        output_file: Path to save visualization
        label_mapping: Custom label mapping dictionary
        dpi: Resolution for saved figures
    """
    print("="*70)
    print("LINEAR DISCRIMINANT ANALYSIS FOR WSI FEATURES")
    print("="*70)
    
    # Default data file if not provided
    if data_file is None:
        # UPDATE THIS PATH TO YOUR DATA FILE
        data_file = "/path/to/your/mean_pooled_features.pkl"
        print(f"\n⚠️  Using default data file path: {data_file}")
        print(f"⚠️  Update the 'data_file' parameter in main() with your actual path")
    
    # Load and clean data
    features, labels, names, unique_labels = load_and_clean_data(
        data_file, label_mapping
    )
    
    # Perform LDA
    features_lda, lda, scaler = perform_lda_analysis(
        features, labels, unique_labels
    )
    
    # Analyze component separation
    class_means_lda, class_std_lda = analyze_component_separation(
        features_lda, labels, unique_labels, lda
    )
    
    # Create visualizations
    create_comprehensive_visualization(
        features_lda, labels, unique_labels, lda, class_means_lda,
        output_file, dpi
    )
    
    # Print interpretation guide
    print_interpretation_guide(unique_labels, features.shape[1])
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n✅ LDA analysis complete!")
    print(f"📊 Visualization saved: {output_file}")
    print(f"📐 {features.shape[1]}D → {features_lda.shape[1]}D transformation")
    print(f"🎯 {len(unique_labels)} classes analyzed")


if __name__ == "__main__":
    # Configuration - Update these paths for your setup
    DATA_FILE = "/path/to/your/TCGAclean_Warwick_CPTAC_Mergedmean_features.pkl"
    OUTPUT_FILE = "lda_detailed_analysis.png"
    
    main(
        data_file=DATA_FILE,
        output_file=OUTPUT_FILE,
        label_mapping=DEFAULT_LABEL_MAPPING,
        dpi=DEFAULT_DPI
    )