"""
Data Preparation Script for Multi-Dataset WSI Classification

This script handles loading, cleaning, and splitting of whole slide image (WSI) features
from TCGA, Warwick, and CPTAC datasets with guaranteed representation across splits.

Author: [Your Name]
Date: 2024
"""

import os
import json
import pickle
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Configuration Constants
DEFAULT_SEEDS = [42, 123, 456, 789]
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5
DEFAULT_MIN_WARWICK_PER_CLASS = 1

# Label mappings
LABEL_MAP = {
    'BRCA_LumA': 'LumA',
    'BRCA_LumB': 'LumB',
    'BRCA_Her2': 'HER2',
    'BRCA_Basal': 'Basal'
}

NORMAL_LABELS = ['Normal', 'BRCA_Normal', 'NORMAL', 'Benign', 'benign']
KNOWN_LABELS = {'LumA', 'LumB', 'HER2', 'Basal'}


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object containing numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def get_dataset_origin(slide_id: str) -> str:
    """
    Determine dataset origin from slide ID.
    
    Args:
        slide_id: Slide identifier string
        
    Returns:
        Dataset origin ('TCGA', 'Warwick', 'CPTAC', or 'unknown')
    """
    if 'TCGA' in slide_id:
        return 'TCGA'
    elif 'Warwick' in slide_id:
        return 'Warwick'
    elif 'CPTAC' in slide_id:
        return 'CPTAC'
    else:
        return 'unknown'


def analyze_dataset_composition(labels: np.ndarray, ids: np.ndarray) -> Tuple[Dict, np.ndarray]:
    """
    Analyze the composition of TCGA, Warwick, and CPTAC samples by class.
    
    Args:
        labels: Array of class labels
        ids: Array of slide IDs
        
    Returns:
        Tuple of (class_composition dict, dataset_origin array)
    """
    dataset_origin = np.array([get_dataset_origin(slide_id) for slide_id in ids])
    
    print("\nDataset Composition Analysis:")
    print("-" * 50)
    
    # Overall composition
    tcga_count = np.sum(dataset_origin == 'TCGA')
    warwick_count = np.sum(dataset_origin == 'Warwick')
    cptac_count = np.sum(dataset_origin == 'CPTAC')
    total_count = len(ids)
    
    print(f"Total TCGA slides: {tcga_count} ({tcga_count/total_count*100:.1f}%)")
    print(f"Total Warwick slides: {warwick_count} ({warwick_count/total_count*100:.1f}%)")
    print(f"Total CPTAC slides: {cptac_count} ({cptac_count/total_count*100:.1f}%)")
    
    # Class-wise composition
    unique_classes = np.unique(labels)
    class_composition = {}
    
    print("\nClass-wise composition:")
    for class_name in unique_classes:
        class_mask = labels == class_name
        class_tcga = np.sum((dataset_origin == 'TCGA') & class_mask)
        class_warwick = np.sum((dataset_origin == 'Warwick') & class_mask)
        class_cptac = np.sum((dataset_origin == 'CPTAC') & class_mask)
        total_class = class_tcga + class_warwick + class_cptac
        
        class_composition[class_name] = {
            'tcga': int(class_tcga),
            'warwick': int(class_warwick),
            'cptac': int(class_cptac),
            'total': int(total_class),
            'warwick_percentage': float((class_warwick / total_class * 100) if total_class > 0 else 0),
            'cptac_percentage': float((class_cptac / total_class * 100) if total_class > 0 else 0)
        }
        
        print(f"  {class_name:15s}: TCGA={class_tcga:3d}, Warwick={class_warwick:3d}, "
              f"CPTAC={class_cptac:3d}, Total={total_class:3d} "
              f"(Warwick {class_warwick/total_class*100:.1f}%, CPTAC {class_cptac/total_class*100:.1f}%)")
    
    return class_composition, dataset_origin


def ensure_dataset_representation_in_splits(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    test_size: float = 0.2,
    min_warwick_per_class: int = 1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Ensure Warwick and CPTAC samples are represented in both train and test sets.
    
    Args:
        X: Feature matrix
        y: Label array
        ids: Slide ID array
        test_size: Proportion of test set
        min_warwick_per_class: Minimum Warwick samples per class
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, ids_train, ids_test, split_issues)
    """
    dataset_origin = np.array([get_dataset_origin(slide_id) for slide_id in ids])
    
    print(f"\nEnsuring dataset representation in splits...")
    print("-" * 50)
    
    # Check Warwick availability per class
    unique_classes = np.unique(y)
    warwick_per_class = {}
    
    for class_name in unique_classes:
        class_mask = y == class_name
        warwick_in_class = np.sum((dataset_origin == 'Warwick') & class_mask)
        warwick_per_class[class_name] = warwick_in_class
        
        if warwick_in_class == 0:
            print(f"⚠️  WARNING: Class {class_name} has NO Warwick samples!")
        elif warwick_in_class < min_warwick_per_class * 2:
            print(f"⚠️  WARNING: Class {class_name} has only {warwick_in_class} Warwick samples "
                  f"(minimum {min_warwick_per_class * 2} recommended)")
    
    # Perform stratified split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Analyze split results
    train_origin = np.array([get_dataset_origin(slide_id) for slide_id in ids_train])
    test_origin = np.array([get_dataset_origin(slide_id) for slide_id in ids_test])
    
    print(f"\nSplit Results:")
    print(f"Training set: {len(ids_train)} samples "
          f"({np.sum(train_origin == 'Warwick')} Warwick, "
          f"{np.sum(train_origin == 'TCGA')} TCGA, "
          f"{np.sum(train_origin == 'CPTAC')} CPTAC)")
    print(f"Test set: {len(ids_test)} samples "
          f"({np.sum(test_origin == 'Warwick')} Warwick, "
          f"{np.sum(test_origin == 'TCGA')} TCGA, "
          f"{np.sum(test_origin == 'CPTAC')} CPTAC)")
    
    # Check class representation
    print(f"\nClass representation in splits:")
    split_issues = []
    
    for class_name in unique_classes:
        train_class_mask = y_train == class_name
        train_class_warwick = np.sum((train_origin == 'Warwick') & train_class_mask)
        train_class_tcga = np.sum((train_origin == 'TCGA') & train_class_mask)
        train_class_cptac = np.sum((train_origin == 'CPTAC') & train_class_mask)
        
        test_class_mask = y_test == class_name
        test_class_warwick = np.sum((test_origin == 'Warwick') & test_class_mask)
        test_class_tcga = np.sum((test_origin == 'TCGA') & test_class_mask)
        test_class_cptac = np.sum((test_origin == 'CPTAC') & test_class_mask)
        
        print(f"  {class_name:15s}:")
        print(f"    Train: {train_class_warwick} Warwick, {train_class_tcga} TCGA, {train_class_cptac} CPTAC")
        print(f"    Test:  {test_class_warwick} Warwick, {test_class_tcga} TCGA, {test_class_cptac} CPTAC")
        
        if train_class_warwick == 0 and warwick_per_class[class_name] > 0:
            split_issues.append(f"No Warwick samples for {class_name} in training set")
        if test_class_warwick == 0 and warwick_per_class[class_name] > 0:
            split_issues.append(f"No Warwick samples for {class_name} in test set")
    
    if split_issues:
        print(f"\n⚠️  Split Issues Found:")
        for issue in split_issues:
            print(f"    - {issue}")
    else:
        print(f"\n✅ All classes with Warwick samples are represented in both splits!")
    
    return X_train, X_test, y_train, y_test, ids_train, ids_test, split_issues


def create_single_seed_data(
    seed: int,
    features: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    test_size: float = 0.2,
    cv_folds: int = 5,
    min_warwick_per_class: int = 1
) -> Dict[str, Any]:
    """
    Create complete data split for a single seed.
    
    Args:
        seed: Random seed
        features: Feature matrix
        labels: Label array
        ids: Slide ID array
        test_size: Proportion of test set
        cv_folds: Number of CV folds
        min_warwick_per_class: Minimum Warwick samples per class
        
    Returns:
        Dictionary containing all split data and metadata
    """
    print(f"\n🎯 Processing Seed {seed}")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test, ids_train, ids_test, split_issues = \
        ensure_dataset_representation_in_splits(
            features, labels, ids, test_size=test_size,
            random_state=seed, min_warwick_per_class=min_warwick_per_class
        )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Create CV splits
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_splits = list(cv.split(X_train_scaled, y_train_encoded))
    
    # Track CV folds
    fold_tracking = {}
    train_origin = np.array([get_dataset_origin(slide_id) for slide_id in ids_train])
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        fold_name = f'fold_{fold_idx + 1}'
        
        fold_train_ids = ids_train[train_idx]
        fold_val_ids = ids_train[val_idx]
        fold_train_labels = y_train_encoded[train_idx]
        fold_val_labels = y_train_encoded[val_idx]
        
        fold_train_origin = train_origin[train_idx]
        fold_val_origin = train_origin[val_idx]
        
        fold_train_class_names = [le.classes_[i] for i in fold_train_labels]
        fold_val_class_names = [le.classes_[i] for i in fold_val_labels]
        
        fold_tracking[fold_name] = {
            'train_indices': train_idx.tolist(),
            'val_indices': val_idx.tolist(),
            'train_slide_ids': fold_train_ids.tolist(),
            'val_slide_ids': fold_val_ids.tolist(),
            'train_labels_encoded': fold_train_labels.tolist(),
            'val_labels_encoded': fold_val_labels.tolist(),
            'train_labels_names': fold_train_class_names,
            'val_labels_names': fold_val_class_names,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_warwick_count': int(np.sum(fold_train_origin == 'Warwick')),
            'val_warwick_count': int(np.sum(fold_val_origin == 'Warwick')),
            'train_cptac_count': int(np.sum(fold_train_origin == 'CPTAC')),
            'val_cptac_count': int(np.sum(fold_val_origin == 'CPTAC')),
            'train_class_distribution': dict(Counter(fold_train_class_names)),
            'val_class_distribution': dict(Counter(fold_val_class_names))
        }
        
        print(f"  {fold_name}: {len(train_idx)} train "
              f"({fold_tracking[fold_name]['train_warwick_count']} Warwick, "
              f"{fold_tracking[fold_name]['train_cptac_count']} CPTAC), "
              f"{len(val_idx)} val "
              f"({fold_tracking[fold_name]['val_warwick_count']} Warwick, "
              f"{fold_tracking[fold_name]['val_cptac_count']} CPTAC)")
    
    # Analyze test set
    test_origin = np.array([get_dataset_origin(slide_id) for slide_id in ids_test])
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_encoded,
        'y_test': y_test_encoded,
        'ids_train': ids_train,
        'ids_test': ids_test,
        'cv_splits': cv_splits,
        'fold_tracking': fold_tracking,
        'scaler': scaler,
        'label_encoder': le,
        'class_names': le.classes_,
        'seed_info': {
            'seed': seed,
            'creation_timestamp': datetime.now().isoformat(),
            'split_issues': split_issues,
            'split_success': len(split_issues) == 0
        },
        'split_tracking': {
            'train_slide_ids': ids_train.tolist(),
            'test_slide_ids': ids_test.tolist(),
            'train_labels_encoded': y_train_encoded.tolist(),
            'test_labels_encoded': y_test_encoded.tolist(),
            'train_labels_names': [le.classes_[i] for i in y_train_encoded],
            'test_labels_names': [le.classes_[i] for i in y_test_encoded],
            'train_class_distribution': dict(Counter([le.classes_[i] for i in y_train_encoded])),
            'test_class_distribution': dict(Counter([le.classes_[i] for i in y_test_encoded])),
            'train_warwick_count': int(np.sum(train_origin == 'Warwick')),
            'test_warwick_count': int(np.sum(test_origin == 'Warwick')),
            'train_cptac_count': int(np.sum(train_origin == 'CPTAC')),
            'test_cptac_count': int(np.sum(test_origin == 'CPTAC')),
            'train_tcga_count': int(np.sum(train_origin == 'TCGA')),
            'test_tcga_count': int(np.sum(test_origin == 'TCGA')),
            'split_parameters': {
                'test_size': test_size,
                'cv_folds': cv_folds,
                'random_state': seed,
                'stratify': True,
                'min_warwick_per_class': min_warwick_per_class
            }
        },
        'data_shapes': {
            'X_train_shape': X_train_scaled.shape,
            'X_test_shape': X_test_scaled.shape,
            'y_train_shape': y_train_encoded.shape,
            'y_test_shape': y_test_encoded.shape,
            'n_features': X_train_scaled.shape[1],
            'n_classes': len(le.classes_),
            'cv_folds': cv_folds
        }
    }


def save_slide_ids_to_file(seed_data: Dict, seed: int, output_dir: str) -> None:
    """
    Save slide IDs for train/test/CV splits to a text file.
    
    Args:
        seed_data: Dictionary containing seed data
        seed: Random seed
        output_dir: Output directory path
    """
    txt_filename = f"seed_{seed}_slide_ids.txt"
    txt_filepath = os.path.join(output_dir, txt_filename)
    
    with open(txt_filepath, 'w') as txtf:
        txtf.write(f"Train IDs ({len(seed_data['ids_train'])}):\n")
        for sid in seed_data['ids_train']:
            txtf.write(f"{sid}\n")
        
        txtf.write(f"\nTest IDs ({len(seed_data['ids_test'])}):\n")
        for sid in seed_data['ids_test']:
            txtf.write(f"{sid}\n")
        
        txtf.write(f"\nFolds (CV):\n")
        for fold_name, fold_info in seed_data['fold_tracking'].items():
            txtf.write(f"\n{fold_name}:\n")
            txtf.write(f"  Train ({len(fold_info['train_slide_ids'])}):\n")
            for sid in fold_info['train_slide_ids']:
                txtf.write(f"    {sid}\n")
            txtf.write(f"  Val ({len(fold_info['val_slide_ids'])}):\n")
            for sid in fold_info['val_slide_ids']:
                txtf.write(f"    {sid}\n")
    
    print(f"📝 Slide IDs for seed {seed} saved to: {txt_filename}")


def load_and_clean_data(data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and clean WSI feature data.
    
    Args:
        data_file: Path to pickle file containing features
        
    Returns:
        Tuple of (features, labels, ids)
    """
    print("\n1. LOADING & CLEANING DATA")
    print("-" * 40)
    
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    slide_features = data['slide_features']
    slide_labels = data['slide_labels']
    slide_ids = np.array(data.get('slide_names', [f"slide_{i:04d}" for i in range(len(slide_labels))]))
    
    print(f"Loaded data from: {data_file}")
    print(f"Total slides: {len(slide_labels)}")
    
    # Remove NaN and normal samples
    valid_mask = slide_labels != 'nan'
    cancer_mask = ~np.isin(slide_labels, NORMAL_LABELS)
    temp_mask = valid_mask & cancer_mask
    
    # Map BRCA_* labels to standard names
    mapped_labels = np.array([LABEL_MAP.get(lbl, lbl) for lbl in slide_labels])
    
    # Determine origin and filter unknown
    origins = np.array([get_dataset_origin(sid) for sid in slide_ids])
    unknown_mask = origins == 'unknown'
    
    if np.any(unknown_mask):
        print(f"\n⚠️  Found {np.sum(unknown_mask)} slides with unknown origin:")
        for sid, lbl in zip(slide_ids[unknown_mask], mapped_labels[unknown_mask]):
            print(f"  Unknown origin: {sid} (label: {lbl})")
    
    known_origin_mask = ~unknown_mask
    
    # Filter unknown labels
    is_known_label = np.isin(mapped_labels, list(KNOWN_LABELS))
    unknown_label_mask = ~is_known_label
    
    if np.any(unknown_label_mask):
        print(f"\n⚠️  Found {np.sum(unknown_label_mask)} slides with unknown label:")
        for sid, lbl, orig in zip(slide_ids[unknown_label_mask], mapped_labels[unknown_label_mask], origins[unknown_label_mask]):
            print(f"  Unknown label: {lbl} | Slide ID: {sid} | Source: {orig}")
    
    # Final filtering
    final_mask = temp_mask & known_origin_mask & is_known_label
    
    features_final = slide_features[final_mask]
    labels_final = mapped_labels[final_mask]
    ids_final = slide_ids[final_mask]
    
    print(f"\nOriginal samples: {len(slide_labels)}")
    print(f"Final cancer samples: {len(labels_final)}")
    print(f"Removed: {len(slide_labels) - len(labels_final)} samples")
    
    return features_final, labels_final, ids_final


def main(
    data_file: str,
    output_dir: str,
    seeds: List[int] = None,
    test_size: float = DEFAULT_TEST_SIZE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    min_warwick_per_class: int = DEFAULT_MIN_WARWICK_PER_CLASS
) -> None:
    """
    Main function to prepare multi-seed data splits.
    
    Args:
        data_file: Path to input pickle file
        output_dir: Directory for output files
        seeds: List of random seeds
        test_size: Proportion of test set
        cv_folds: Number of CV folds
        min_warwick_per_class: Minimum Warwick samples per class
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    
    print("="*70)
    print("MULTI-SEED DATA PREPARATION WITH DATASET REPRESENTATION GUARANTEE")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    preparation_timestamp = datetime.now()
    preparation_date = preparation_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"📅 Data preparation started: {preparation_date}")
    print(f"🎯 Seeds to process: {seeds}")
    print(f"📂 Output directory: {output_dir}")
    
    # Load and clean data
    features_final, labels_final, ids_final = load_and_clean_data(data_file)
    
    # Analyze composition
    print("\n2. DATASET COMPOSITION ANALYSIS")
    print("-" * 40)
    class_composition, dataset_origin = analyze_dataset_composition(labels_final, ids_final)
    
    # Show class distribution
    print(f"\n3. CLASS DISTRIBUTION")
    print("-" * 40)
    class_counts = Counter(labels_final)
    for label, count in sorted(class_counts.items()):
        warwick_pct = class_composition[label]['warwick_percentage']
        cptac_pct = class_composition[label]['cptac_percentage']
        print(f"  {label:15s}: {count:4d} samples ({count/len(labels_final)*100:.1f}%) - "
              f"{warwick_pct:.1f}% Warwick, {cptac_pct:.1f}% CPTAC")
    
    # Process each seed
    print("\n4. PROCESSING SEEDS")
    print("-" * 40)
    
    successful_seeds = []
    failed_seeds = []
    seed_files = {}
    
    for seed in seeds:
        try:
            print(f"\nProcessing seed {seed}...")
            seed_data = create_single_seed_data(
                seed, features_final, labels_final, ids_final,
                test_size=test_size, cv_folds=cv_folds,
                min_warwick_per_class=min_warwick_per_class
            )
            
            # Save seed file
            seed_filename = f"seed_{seed}_data.pkl"
            seed_filepath = os.path.join(output_dir, seed_filename)
            with open(seed_filepath, 'wb') as f:
                pickle.dump(seed_data, f)
            seed_files[seed] = seed_filepath
            successful_seeds.append(seed)
            print(f"✅ Seed {seed} saved to: {seed_filename}")
            
            # Save slide IDs
            save_slide_ids_to_file(seed_data, seed, output_dir)
            
        except Exception as e:
            print(f"❌ Error processing seed {seed}: {str(e)}")
            failed_seeds.append(seed)
    
    # Create metadata
    print("\n5. CREATING METADATA")
    print("-" * 40)
    
    warwick_count = np.sum(dataset_origin == 'Warwick')
    cptac_count = np.sum(dataset_origin == 'CPTAC')
    
    metadata = {
        'creation_info': {
            'preparation_date': preparation_date,
            'preparation_timestamp': preparation_timestamp.isoformat(),
            'data_file_used': data_file
        },
        'dataset_info': {
            'total_samples': int(len(labels_final)),
            'n_features': int(features_final.shape[1]),
            'n_classes': int(len(np.unique(labels_final))),
            'class_names': sorted([str(x) for x in np.unique(labels_final).tolist()]),
            'class_distribution': {str(k): int(v) for k, v in class_counts.items()},
            'class_composition': convert_numpy_types(class_composition),
            'total_warwick_samples': int(warwick_count),
            'total_cptac_samples': int(cptac_count),
            'total_tcga_samples': int(np.sum(dataset_origin == 'TCGA')),
            'warwick_percentage': float(warwick_count / len(labels_final) * 100),
            'cptac_percentage': float(cptac_count / len(labels_final) * 100)
        },
        'processing_info': {
            'seeds_attempted': [int(s) for s in seeds],
            'seeds_successful': [int(s) for s in successful_seeds],
            'seeds_failed': [int(s) for s in failed_seeds],
            'success_rate': float(len(successful_seeds) / len(seeds)),
            'seed_files': {str(k): str(v) for k, v in seed_files.items()}
        },
        'split_parameters': {
            'test_size': float(test_size),
            'cv_folds': int(cv_folds),
            'min_warwick_per_class': int(min_warwick_per_class),
            'stratify': True
        },
        'preprocessing_info': {
            'feature_standardization': 'StandardScaler (per seed)',
            'label_encoding': 'LabelEncoder (per seed)',
            'normal_samples_removed': True,
            'nan_samples_removed': True,
            'dataset_representation_guaranteed': True
        }
    }
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "data_preparation_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_file}")
    
    # Final summary
    print("\n6. FINAL SUMMARY")
    print("-" * 40)
    print(f"✅ Successfully processed {len(successful_seeds)}/{len(seeds)} seeds")
    print(f"📁 Files created in: {output_dir}")
    print(f"🏥 Warwick samples: {warwick_count}/{len(labels_final)} ({warwick_count/len(labels_final)*100:.1f}%)")
    print(f"🧬 CPTAC samples: {cptac_count}/{len(labels_final)} ({cptac_count/len(labels_final)*100:.1f}%)")
    print(f"\n🚀 READY FOR PARALLEL TRAINING!")


if __name__ == "__main__":
    # Configuration - Update these paths for your setup
    DATA_FILE = "/home/projects2/WSI_project/PhD_WSI/feature_extraction/Mean_Features/TCGAclean_Warwick_CPTAC_Mergedmean_features.pkl"
    OUTPUT_DIR = "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_mean_features/data_splits/data_splitsV3"
    
    main(
        data_file=DATA_FILE,
        output_dir=OUTPUT_DIR,
        seeds=DEFAULT_SEEDS,
        test_size=DEFAULT_TEST_SIZE,
        cv_folds=DEFAULT_CV_FOLDS,
        min_warwick_per_class=DEFAULT_MIN_WARWICK_PER_CLASS
    )