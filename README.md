# Breast Cancer Molecular Subtyping from Whole Slide Images using Computational Pathology Foundation Model

This repository contains the code and experimental pipeline for our study on **annotation-free and weakly supervised classification of breast cancer molecular subtypes (LumA, LumB, Basal, HER2) from histopathology whole slide images (WSI)**. 

The workflow integrates **Preprocessing and Tiling**, **Feature Extraction using the UNI-2 Foundation Model**, **Dataset Splitting and Mean Pooling**, and **One vs Rest Model training and evaluation** for reproducible benchmarking across multiple data sources (TCGA-BRCA, CPTAC-BRCA, and Warwick HER2).

---

## 🧩 Overview

Our end-to-end pipeline for weakly supervised breast cancer molecular subtyping consists of four main stages:

1. **Preprocessing and Tiling**

   Whole Slide Images (WSIs) are preprocessed to detect tissue regions and tiled at 20× magnification into 224×224 patches without overlap.
    
   Background, white, and blurry tiles are filtered out using simple image-statistics thresholds.
   
   Tiles are renamed and organized to preserve Slide_ID, coordinates, data source, and molecular subtype labels.

3. **Feature Extraction using the UNI-2 Foundation Model**
     
   Each tile is passed through the **UNI-2** ViT-H/14 encoder (trained via DINOv2 self-supervision), producing a **1536-dimensional embedding** per tile.
   
   Per-slide dictionaries are created, where each Slide_ID maps to its corresponding set of tile embeddings.

5. **Dataset Splitting and Mean Pooling**

   Patient-level splits were created with an **80/20 train-test** ratio, followed by an **8/1/1** subdivision of the training set into train, validation, and      calibration subsets for cross-validation and probability calibration where applicable.
   
   To prevent data leakage, all tiles from a single patient were assigned to the same split, with stratification by molecular subtype and data source.  
   Mean-pooling of tile embeddings produced a single 1536-dimensional slide vector for classical ML models.
    
   Linear Discriminant Analysis (LDA) was applied to visualize subtype separability in feature space.

   
6. **One vs Rest Model training and evaluation**
   
   Each molecular subtype was modeled independently using an OvR setup, where one binary classifier distinguishes each subtype from all others.

   Cosine similarity baseline: non-parametric k-NN on mean-pooled slide vectors.

   Logistic Regression (LR) and Linear Discriminant Analysis (LDA) on pooled slide features.

   Attention-based Deep MIL using tile-level embeddings and attention pooling.

   Tile-level Logistic Regression MIL for interpretable, per-tile predictions.
  
   Optional components include class balancing (oversampling, downsampling, SMOTE) and probability calibration (isotonic or temperature scaling).
   Evaluation metrics include accuracy, balanced accuracy, macro F1, weighted F1, ROC-AUC, precision and recall computed per class, with macro-averaged scores    reported as overall performance summaries across all subtypes.

---

## ⚙️ Repository Structure

```
src/
├── preprocessing/                         # WSI preprocessing and tiling
│   ├── tile_filtering.py                  # Removes background, white, and blurry tiles based on pixel intensity and variance
│   └── tile_rename.py                     # Renames and organizes tiles while retaining coordinates and magnification, Slide_ID, data source, and molecular subtype label

├── feature_extraction/                    # Tile-level feature extraction with UNI-2
│   ├── UNI2_fextraction.py                # Extracts 1536-D embeddings from the UNI-2 encoder (ViT-H/14, DINOv2)
│   └── Create_Tile_Features_dicts.py      # Builds per-slide dictionaries of extracted features;
│                                          # each Slide_ID is a key containing tiles as subkeys and their 1536-D feature vectors as values

├── mean_pooling_features/                 # Slide-level (mean-pooled) feature ML training and analysis
│   ├── Create_MeanPooling_vectors.py      # Generates mean-pooled 1536-D slide representations from tile embeddings
│   ├── LDA_analysis_mean_vectors.py       # Performs Linear Discriminant Analysis (LDA) for feature-space visualization
│   └── ML_mean_features/                  # Classical ML models on pooled slide features
│       ├── Logistic_models.py             # One-vs-Rest (OvR) logistic regression with/without calibration and class balancing
│       ├── data_preperation/              # Train/validation/calibration split creation
│       │   └── data_preperation.py        # Generates 80/20 train-test split and 8/1/1 train-eval-calib subsets and CV splits
│                                          # Tracks all Slide_IDs to prevent data leakage
│       └── Cosine_Similarity.py           # k-NN (cosine similarity) baseline on mean-pooled vectors

├── MIL/                                   # Multiple Instance Learning (MIL) approaches
│   ├── MIL_training/                      # Attention-based deep MIL and tile-level MIL training
│   │   ├── AttentionMIL_Callibration_Balance_training.py  # Attention-based MIL with class balancing and temperature calibration
│   │   └── LR_MIL.py                      # Tile-level logistic regression MIL
│   ├── MIL_evaluate/                      # MIL model evaluation and calibration
│   │   └── AttentionMIL_evaluate.py       # Evaluates MIL models and applies temperature scaling
│   └── Heatmaps/                          # Visualization of spatial attention and probability maps
│       └── HeatmapV2.py                   # Generates subtype-specific heatmaps highlighting informative tissue regions
```




## 🧠 Notes

Access to TCGA-BRCA, CPTAC-BRCA, and Warwick HER2 datasets requires appropriate data use permissions

The scripts are designed to run on a local filesystem; paths and hyperparameters may need editing for other environments.

Users can consult docstrings in each script for detailed parameter explanations.”

Due to dataset size, logistic regression at tile level was not retrained for calibration because of limited performance and higher computational cost compared to GPU-trained MIL models.

All core code for feature extraction, model training, and evaluation is fully included for transparency and reuse.

Researchers are encouraged to adapt the pipeline to their own datasets using the provided modules.

