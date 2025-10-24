"""
UNI2 Feature Extraction for Whole Slide Images

This script extracts tile-level features from whole slide images (WSI) using the
UNI2 (Universal Network for Medical Imaging) foundation model from MahmoodLab.

Features:
- Batch processing of tiles for GPU efficiency
- Memory monitoring (GPU and RAM)
- Automatic verification of extracted features
- Resume capability (skip already processed slides)
- Logging of incomplete extractions

The UNI2 model generates 1536-dimensional feature vectors for each 224×224 tile.

Requirements:
- PyTorch with CUDA support
- timm (PyTorch Image Models)
- huggingface_hub
- Access to MahmoodLab/UNI2-h on Hugging Face

Usage:
    python UNI2_fextraction.py

Configuration:
    Set the following variables before running:
    - HF_TOKEN: Your Hugging Face access token
    - INPUT_DIR: Directory containing slide tiles
    - OUTPUT_DIR: Directory to save extracted features
    - BATCH_SIZE: Number of tiles to process simultaneously

Author: [Your Name]
Date: 2025
"""

import os
import logging
from pathlib import Path
from typing import Optional, Set, Tuple
import warnings

import numpy as np
import psutil
import torch
import timm
from PIL import Image
from tqdm import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('uni2_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================
# TODO: Replace with your actual Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Directory paths - UPDATE THESE FOR YOUR SETUP
INPUT_DIR = "/home/projects2/WSI_project/PhD_WSI/download_v1/HER2_Warwick/HER2_Warwick_Tiles/HER2_organised_tiles"
OUTPUT_DIR = "/home/projects2/WSI_project/PhD_WSI/UNI2/featuresHER2Warwick"

# Processing parameters
BATCH_SIZE = 32  # Adjust based on GPU memory (reduce if OOM errors occur)
TILE_LEVEL = "20.0"  # Magnification level directory name
TILE_EXTENSION = ".jpeg"  # Tile image format
EXPECTED_FEATURE_DIM = 1536  # UNI2 output dimension
IMAGE_SIZE = 224  # UNI2 input size

# UNI2 Model configuration
UNI2_CONFIG = {
    'img_size': IMAGE_SIZE,
    'patch_size': 14,
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5,
    'embed_dim': EXPECTED_FEATURE_DIM,
    'mlp_ratio': 2.66667 * 2,
    'num_classes': 0,
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked,
    'act_layer': torch.nn.SiLU,
    'reg_tokens': 8,
    'dynamic_img_size': True
}


# ==================== Model Setup ====================
def setup_model(device: torch.device) -> Tuple[torch.nn.Module, callable]:
    """
    Initialize the UNI2 model and preprocessing transform.
    
    Args:
        device: PyTorch device (cuda or cpu)
        
    Returns:
        Tuple of (model, transform)
        
    Raises:
        RuntimeError: If model loading fails
    """
    logger.info("Setting up UNI2 model...")
    
    try:
        # Authenticate with Hugging Face
        login(token=HF_TOKEN)
        
        # Load model
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            **UNI2_CONFIG
        )
        model.eval().to(device)
        
        # Create preprocessing transform
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        
        logger.info(f"✅ Model loaded successfully on {device}")
        logger.info(f"Model output dimension: {EXPECTED_FEATURE_DIM}")
        
        return model, transform
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def log_memory_usage(stage: str = "") -> None:
    """
    Log current GPU and RAM memory usage.
    
    Args:
        stage: Optional description of current processing stage
    """
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / 1e6
        gpu_max = torch.cuda.max_memory_allocated() / 1e6
        logger.debug(f"{stage} GPU: {gpu_alloc:.2f} MB (max: {gpu_max:.2f} MB)")
    
    ram_used = psutil.virtual_memory().used / 1e9
    logger.debug(f"{stage} RAM: {ram_used:.2f} GB")


# ==================== Feature Extraction ====================
def extract_slide_features(
    slide_folder: str,
    out_path: str,
    model: torch.nn.Module,
    transform: callable,
    device: torch.device,
    batch_size: int = BATCH_SIZE
) -> bool:
    """
    Extract features from all tiles in a slide folder.
    
    Args:
        slide_folder: Path to slide's tile directory
        out_path: Path to save extracted features (.npy)
        model: UNI2 model
        transform: Preprocessing transform
        device: PyTorch device
        batch_size: Number of tiles per batch
        
    Returns:
        True if extraction successful, False otherwise
    """
    tile_dir = Path(slide_folder) / TILE_LEVEL
    
    if not tile_dir.is_dir():
        logger.warning(f"Tile directory not found: {tile_dir}")
        return False
    
    # Get all tile files
    tile_files = sorted([
        f for f in tile_dir.iterdir()
        if f.suffix == TILE_EXTENSION
    ])
    
    if not tile_files:
        logger.warning(f"No tiles found in {tile_dir}")
        return False
    
    logger.info(f"Processing {len(tile_files)} tiles from {slide_folder}")
    
    features = []
    batch_imgs = []
    failed_tiles = 0
    
    for idx, tile_path in enumerate(tile_files):
        try:
            # Load and preprocess image
            img = Image.open(tile_path).convert("RGB")
            img_tensor = transform(img)
            batch_imgs.append(img_tensor)
            
        except Exception as e:
            logger.warning(f"Failed to load {tile_path}: {e}")
            failed_tiles += 1
            continue
        
        # Process batch when full or at end
        if len(batch_imgs) == batch_size or idx == len(tile_files) - 1:
            if batch_imgs:  # Only process if we have images
                batch_tensor = torch.stack(batch_imgs).to(device)
                
                with torch.inference_mode():
                    feat = model(batch_tensor)
                
                features.append(feat.cpu().numpy())
                batch_imgs = []
                
                # Optional: log memory every 10 batches
                if (idx + 1) % (batch_size * 10) == 0:
                    log_memory_usage(f"Batch {idx // batch_size + 1}")
                
                # Clear GPU cache
                torch.cuda.empty_cache()
    
    if not features:
        logger.error(f"No features extracted from {slide_folder}")
        return False
    
    # Concatenate all features
    features_np = np.concatenate(features, axis=0)
    
    # Save features
    np.save(out_path, features_np)
    logger.info(f"✅ Saved features: {out_path}, shape: {features_np.shape}")
    
    # Verify tile count matches feature count
    n_tiles = len(tile_files)
    n_features = features_np.shape[0]
    
    if failed_tiles > 0:
        logger.warning(f"Failed to process {failed_tiles}/{n_tiles} tiles")
    
    if n_tiles != n_features + failed_tiles:
        logger.warning(
            f"Tile/feature mismatch: {n_tiles} tiles, "
            f"{n_features} features, {failed_tiles} failed"
        )
        log_incomplete_slide(out_path, n_tiles, n_features, failed_tiles)
        return False
    
    logger.info(f"Verification passed: {n_tiles} tiles → {n_features} features")
    return True


def log_incomplete_slide(
    slide_path: str,
    n_tiles: int,
    n_features: int,
    n_failed: int
) -> None:
    """
    Log information about incomplete feature extraction.
    
    Args:
        slide_path: Path to the slide's feature file
        n_tiles: Total number of tiles
        n_features: Number of features extracted
        n_failed: Number of failed tiles
    """
    log_file = Path(slide_path).parent / "incomplete_slides.log"
    
    with open(log_file, "a") as f:
        f.write(
            f"{slide_path}: tiles={n_tiles}, features={n_features}, "
            f"failed={n_failed}\n"
        )


# ==================== Verification ====================
def verify_existing_features(input_dir: str, output_dir: str) -> Set[str]:
    """
    Verify all previously extracted features and return verified slide names.
    
    Args:
        input_dir: Directory containing slide tiles
        output_dir: Directory containing extracted features
        
    Returns:
        Set of slide names that are verified and complete
    """
    logger.info("Verifying existing features...")
    
    verified_slides = set()
    output_path = Path(output_dir)
    
    for npy_file in output_path.glob("*.npy"):
        slide_name = npy_file.stem
        slide_folder = Path(input_dir) / f"{slide_name}_files"
        tile_dir = slide_folder / TILE_LEVEL
        
        if not tile_dir.is_dir():
            logger.warning(f"Tile directory missing for {slide_name}")
            continue
        
        # Count tiles
        tile_files = list(tile_dir.glob(f"*{TILE_EXTENSION}"))
        n_tiles = len(tile_files)
        
        try:
            # Load features
            features_np = np.load(npy_file)
            n_features = features_np.shape[0]
            
            # Verify match
            if n_tiles == n_features:
                verified_slides.add(slide_name)
                logger.debug(f"✅ {slide_name}: {n_tiles} tiles verified")
            else:
                logger.warning(
                    f"⚠️  {slide_name}: tiles={n_tiles}, features={n_features}"
                )
                log_incomplete_slide(str(npy_file), n_tiles, n_features, 0)
                
        except Exception as e:
            logger.warning(f"Could not load {npy_file}: {e}")
    
    logger.info(f"Verified {len(verified_slides)} slides")
    return verified_slides


# ==================== Main Processing ====================
def main():
    """
    Main function to orchestrate feature extraction process.
    """
    logger.info("="*70)
    logger.info("UNI2 FEATURE EXTRACTION FOR WHOLE SLIDE IMAGES")
    logger.info("="*70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Using CPU (will be very slow)")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    
    # Load model
    try:
        model, transform = setup_model(device)
    except Exception as e:
        logger.error(f"Cannot proceed without model: {e}")
        return
    
    # Verify existing features
    verified_slides = verify_existing_features(INPUT_DIR, OUTPUT_DIR)
    
    if verified_slides:
        logger.info(f"\nAlready verified slides ({len(verified_slides)}):")
        for slide in sorted(verified_slides):
            logger.info(f"  ✓ {slide}")
    
    # Find slides to process
    input_path = Path(INPUT_DIR)
    all_slide_folders = list(input_path.glob("*_files"))
    
    to_process = [
        folder for folder in all_slide_folders
        if folder.stem.replace("_files", "") not in verified_slides
    ]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Total slides: {len(all_slide_folders)}")
    logger.info(f"Already verified: {len(verified_slides)}")
    logger.info(f"To process: {len(to_process)}")
    logger.info(f"{'='*70}\n")
    
    if not to_process:
        logger.info("✅ All slides already processed and verified!")
        return
    
    # Process slides
    successful = 0
    failed = 0
    
    for slide_folder in tqdm(to_process, desc="Processing slides"):
        slide_name = slide_folder.stem.replace("_files", "")
        out_file = Path(OUTPUT_DIR) / f"{slide_name}.npy"
        
        logger.info(f"\nProcessing: {slide_name}")
        
        try:
            success = extract_slide_features(
                str(slide_folder),
                str(out_file),
                model,
                transform,
                device,
                BATCH_SIZE
            )
            
            if success:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"Error processing {slide_name}: {e}")
            failed += 1
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total processed: {successful + failed}")
    
    # Final verification
    logger.info("\nRunning final verification...")
    final_verified = verify_existing_features(INPUT_DIR, OUTPUT_DIR)
    logger.info(f"Total verified slides: {len(final_verified)}")
    
    logger.info("\n✅ Feature extraction pipeline complete!")


if __name__ == "__main__":
    # Suppress unnecessary warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
