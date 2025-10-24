"""
Mean Pooling Feature Vector Creation for WSI Analysis

This script processes tile-level features from whole slide images (WSI) and creates
slide-level representations using mean pooling. It supports:
- Examining pickle file structures
- Creating mean-pooled features from tile dictionaries
- Concatenating multiple datasets

The output format is compatible with downstream ML pipelines.

Author: Konstantinos Papagoras
Date: 2025-06
"""

import os
import glob
import pickle
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


# Configuration Constants
DEFAULT_PROGRESS_INTERVAL = 10  # Print progress every N slides
EXPECTED_FEATURE_DIM = 1536  # Expected feature dimension (UNI model)


def examine_pickle_structure(file_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Examine the structure of a pickle file to understand the data format.
    
    Args:
        file_path: Path to the pickle file
        verbose: If True, print detailed information
        
    Returns:
        Dictionary containing the loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pickle.UnpicklingError: If the file is corrupted
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    if verbose:
        print(f"File: {os.path.basename(file_path)}")
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            keys = list(data.keys())
            print(f"Total keys: {len(keys)}")
            print(f"Keys: {keys}")
            
            # Show details for first key
            if keys:
                first_key = keys[0]
                first_value = data[first_key]
                print(f"\nFirst key '{first_key}': {type(first_value)}")
                
                if isinstance(first_value, dict):
                    print(f"  Dictionary keys: {list(first_value.keys())}")
                    for sub_key, sub_value in first_value.items():
                        if isinstance(sub_value, np.ndarray):
                            print(f"    {sub_key}: array shape {sub_value.shape}, dtype {sub_value.dtype}")
                        else:
                            print(f"    {sub_key}: {type(sub_value)}")
                            
                elif isinstance(first_value, np.ndarray):
                    print(f"  Shape: {first_value.shape}")
                    print(f"  Data type: {first_value.dtype}")
                    print(f"  Min: {first_value.min():.4f}, Max: {first_value.max():.4f}")
                    
                elif isinstance(first_value, list):
                    print(f"  Length: {len(first_value)}")
                    if len(first_value) > 0:
                        print(f"  First element type: {type(first_value[0])}")
                        if isinstance(first_value[0], np.ndarray):
                            print(f"  First element shape: {first_value[0].shape}")
    
    return data


def create_mean_pooled_features(
    input_dir: str,
    output_file: str,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
    validate_labels: bool = True
) -> Dict[str, np.ndarray]:
    """
    Process tile dictionaries and create mean-pooled features at slide level.
    
    This function loads tile-level features from individual pickle files,
    computes mean pooling across tiles for each slide, and saves the result
    in a standardized format.
    
    Args:
        input_dir: Directory containing pickle files with tile features
        output_file: Path to save the processed data
        progress_interval: Print progress every N slides
        validate_labels: If True, verify all tiles in a slide have the same label
        
    Returns:
        Dictionary containing:
            - slide_features: np.ndarray of shape (n_slides, feature_dim)
            - slide_labels: np.ndarray of shape (n_slides,)
            - slide_names: np.ndarray of shape (n_slides,)
            
    Raises:
        FileNotFoundError: If input_dir doesn't exist
        ValueError: If no pickle files found or labels are inconsistent
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    pickle_files = glob.glob(os.path.join(input_dir, "*.pkl"))
    
    if not pickle_files:
        raise ValueError(f"No pickle files found in {input_dir}")
    
    print(f"Found {len(pickle_files)} slides to process")
    print(f"Input directory: {input_dir}")
    
    slide_features = []
    slide_labels = []
    slide_names = []
    
    errors = []
    
    for i, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                tile_data = pickle.load(f)
            
            # Extract slide ID from filename
            slide_id = os.path.splitext(os.path.basename(pickle_file))[0]
            
            # Collect all features and labels
            features = []
            labels = []
            
            for tile_key, tile_info in tile_data.items():
                if 'feature' not in tile_info or 'label' not in tile_info:
                    raise ValueError(f"Missing 'feature' or 'label' in tile {tile_key}")
                
                features.append(tile_info['feature'])
                labels.append(tile_info['label'])
            
            # Validate labels consistency
            if validate_labels and len(set(labels)) > 1:
                raise ValueError(f"Inconsistent labels in slide {slide_id}: {set(labels)}")
            
            # Convert to numpy array and compute mean pooling
            features_array = np.array(features)  # Shape: (n_tiles, feature_dim)
            mean_pooled_feature = np.mean(features_array, axis=0)  # Shape: (feature_dim,)
            
            # Validate feature dimension
            if mean_pooled_feature.shape[0] != EXPECTED_FEATURE_DIM:
                print(f"Warning: Unexpected feature dimension {mean_pooled_feature.shape[0]} "
                      f"(expected {EXPECTED_FEATURE_DIM}) for slide {slide_id}")
            
            # Get the slide label (all tiles should have the same label)
            slide_label = labels[0]
            
            # Append to lists
            slide_features.append(mean_pooled_feature)
            slide_labels.append(slide_label)
            slide_names.append(slide_id)
            
            if (i + 1) % progress_interval == 0:
                print(f"✓ Processed {i + 1}/{len(pickle_files)} slides "
                      f"({(i + 1) / len(pickle_files) * 100:.1f}%)")
                
        except Exception as e:
            error_msg = f"Error processing {os.path.basename(pickle_file)}: {str(e)}"
            errors.append(error_msg)
            print(f"✗ {error_msg}")
    
    if errors:
        print(f"\n⚠️  Encountered {len(errors)} errors during processing")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    # Convert to numpy arrays
    slide_features = np.array(slide_features)  # Shape: (n_slides, feature_dim)
    slide_labels = np.array(slide_labels)      # Shape: (n_slides,)
    slide_names = np.array(slide_names)        # Shape: (n_slides,)
    
    # Create standardized output dictionary
    processed_data = {
        'slide_features': slide_features,
        'slide_labels': slide_labels,
        'slide_names': slide_names
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the processed data
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\n✅ Successfully processed {len(slide_features)} slides")
    print(f"Features shape: {slide_features.shape}")
    print(f"Labels shape: {slide_labels.shape}")
    print(f"Names shape: {slide_names.shape}")
    print(f"Output saved to: {output_file}")
    
    # Print label distribution
    unique_labels, counts = np.unique(slide_labels, return_counts=True)
    print(f"\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} slides ({count / len(slide_labels) * 100:.1f}%)")
    
    return processed_data


def concatenate_datasets(
    existing_file: str,
    new_file: str,
    output_file: str,
    check_duplicates: bool = True
) -> Dict[str, np.ndarray]:
    """
    Concatenate two datasets with the same structure.
    
    Args:
        existing_file: Path to existing mean features file
        new_file: Path to new mean features file
        output_file: Path to save concatenated data
        check_duplicates: If True, check for duplicate slide names
        
    Returns:
        Dictionary containing concatenated data with keys:
            - slide_features: Combined feature arrays
            - slide_labels: Combined label arrays
            - slide_names: Combined name arrays
            
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If data structures are incompatible
    """
    # Validate input files
    if not os.path.exists(existing_file):
        raise FileNotFoundError(f"Existing file not found: {existing_file}")
    if not os.path.exists(new_file):
        raise FileNotFoundError(f"New file not found: {new_file}")
    
    # Load both datasets
    print(f"Loading existing dataset from: {existing_file}")
    with open(existing_file, 'rb') as f:
        existing_data = pickle.load(f)
    
    print(f"Loading new dataset from: {new_file}")
    with open(new_file, 'rb') as f:
        new_data = pickle.load(f)
    
    # Validate data structure
    required_keys = ['slide_features', 'slide_labels', 'slide_names']
    for key in required_keys:
        if key not in existing_data or key not in new_data:
            raise ValueError(f"Missing required key '{key}' in one of the datasets")
    
    # Validate feature dimensions match
    existing_dim = existing_data['slide_features'].shape[1]
    new_dim = new_data['slide_features'].shape[1]
    if existing_dim != new_dim:
        raise ValueError(f"Feature dimensions don't match: {existing_dim} vs {new_dim}")
    
    print(f"\nExisting dataset: {len(existing_data['slide_features'])} slides")
    print(f"New dataset: {len(new_data['slide_features'])} slides")
    
    # Check for duplicates
    if check_duplicates:
        existing_names = set(existing_data['slide_names'])
        new_names = set(new_data['slide_names'])
        duplicates = existing_names.intersection(new_names)
        
        if duplicates:
            print(f"\n⚠️  Warning: Found {len(duplicates)} duplicate slide names:")
            for dup in list(duplicates)[:5]:
                print(f"  - {dup}")
            if len(duplicates) > 5:
                print(f"  ... and {len(duplicates) - 5} more")
            
            response = input("\nContinue with duplicates? (y/n): ")
            if response.lower() != 'y':
                raise ValueError("Concatenation cancelled due to duplicates")
    
    # Concatenate arrays
    combined_features = np.concatenate([
        existing_data['slide_features'],
        new_data['slide_features']
    ], axis=0)
    
    combined_labels = np.concatenate([
        existing_data['slide_labels'],
        new_data['slide_labels']
    ], axis=0)
    
    combined_names = np.concatenate([
        existing_data['slide_names'],
        new_data['slide_names']
    ], axis=0)
    
    # Create combined dataset
    combined_data = {
        'slide_features': combined_features,
        'slide_labels': combined_labels,
        'slide_names': combined_names
    }
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save combined data
    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f)
    
    print(f"\n✅ Combined dataset saved to: {output_file}")
    print(f"Total slides: {len(combined_features)}")
    print(f"Features shape: {combined_features.shape}")
    
    # Print combined label distribution
    unique_labels, counts = np.unique(combined_labels, return_counts=True)
    print(f"\nCombined label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} slides ({count / len(combined_labels) * 100:.1f}%)")
    
    return combined_data


def main():
    """
    Main function to process and concatenate WSI mean-pooled features.
    
    This function orchestrates the entire pipeline:
    1. Examine existing data structure
    2. Process new tile-level features
    3. Concatenate with existing features
    """
    print("="*70)
    print("WSI MEAN-POOLED FEATURE CREATION AND CONCATENATION")
    print("="*70)
    
    # Configuration - Update these paths for your setup
    existing_mean_features_path = (
        "/home/projects2/WSI_project/PhD_WSI/feature_extraction/"
        "Mean_Features/TCGAclean_Warwick_Mergedmean_features.pkl"
    )
    
    input_directory = (
        "/home/projects2/WSI_project/PhD_WSI/download_v1/CPTAC_BRCA/"
        "CPTAC_BRCA_20x_UNI2_dicts/"
    )
    
    new_features_file = (
        "/home/projects2/WSI_project/PhD_WSI/download_v1/CPTAC_BRCA/"
        "CPTAC_BRCA_UNI2_mean_pooled_features.pkl"
    )
    
    combined_output_file = (
        "/home/projects2/WSI_project/PhD_WSI/feature_extraction/"
        "Mean_Features/TCGAclean_Warwick_CPTAC_Mergedmean_features.pkl"
    )
    
    # Step 1: Examine existing structure
    print("\n" + "="*70)
    print("STEP 1: EXAMINING EXISTING DATA STRUCTURE")
    print("="*70)
    
    if os.path.exists(existing_mean_features_path):
        existing_data = examine_pickle_structure(existing_mean_features_path, verbose=True)
    else:
        print(f"Warning: Existing file not found at {existing_mean_features_path}")
        existing_data = None
    
    # Step 2: Process new data
    print("\n" + "="*70)
    print("STEP 2: PROCESSING NEW TILE-LEVEL FEATURES")
    print("="*70)
    
    new_data = create_mean_pooled_features(
        input_dir=input_directory,
        output_file=new_features_file,
        progress_interval=DEFAULT_PROGRESS_INTERVAL
    )
    
    # Step 3: Concatenate datasets (if existing data exists)
    if existing_data is not None:
        print("\n" + "="*70)
        print("STEP 3: CONCATENATING DATASETS")
        print("="*70)
        
        combined_data = concatenate_datasets(
            existing_file=existing_mean_features_path,
            new_file=new_features_file,
            output_file=combined_output_file,
            check_duplicates=True
        )
    else:
        print("\n⚠️  Skipping concatenation: existing data file not found")
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()