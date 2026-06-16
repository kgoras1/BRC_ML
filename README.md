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
   
   To prevent data leakage, all tiles and slides from a single patient were assigned to the same split, with stratification by molecular subtype and data source.  
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
├── preprocessing/                              # WSI preprocessing and tiling
│   ├── tile_filtering.py                       # Removes background, white, and blurry tiles based on pixel intensity and variance
│   └── tile_rename.py                          # Renames and organizes tiles retaining coordinates, Slide_ID, data source, and molecular subtype label
│
├── feature_extraction/                         # Tile-level feature extraction with UNI-2
│   ├── UNI2_fextraction.py                     # Extracts 1536-D embeddings per tile from the UNI-2 encoder (ViT-H/14, DINOv2)
│   └── Create_Tile_Features_dicts.py           # Builds per-slide dictionaries mapping each Slide_ID to its tile embeddings
│
├── mean_pooling_features/                      # Slide-level aggregation and visualization
│   ├── Create_MeanPooling_vectors.py           # Generates mean-pooled 1536-D slide representations from tile embeddings
│   └── LDA_analysis_mean_vectors.py            # Linear Discriminant Analysis for feature-space visualization of subtype separability
│
└── MIL/                                        # Multiple Instance Learning approaches
    ├── MIL_training/                           # Attention-based deep MIL and tile-level LR training
    │   ├── AttentionMIL_Callibration_Balance_training.py   # Attention MIL with class balancing and temperature calibration
    │   └── LR_MIL.py                           # Tile-level logistic regression MIL (weakly supervised)
    ├── MIL_evaluate/                           # MIL model evaluation
    │   └── AttentionMIL_evaluate.py            # Evaluates trained MIL models and applies temperature scaling
    └── Heatmaps/                               # Spatial attention and probability map visualizations
        ├── HeatmapV2.py                        # Generates subtype-specific heatmaps highlighting informative tissue regions
        └── heatmap_Local_optimized.py          # Optimized local version of the heatmap generation pipeline

src_revision/
├── data_split/                                 # Reproducible patient-level stratified data splitting
│   ├── Datasplit_train_test.py                 # Creates 80/20 patient-level train/test split for MIL PKL data;
│   │                                           # stratifies on label × cohort composite key to prevent patient-level leakage
│   └── mean_features_datasplit.py              # Replicates the same patient split for mean-pooled feature matrices
│                                               # using the split report JSON (avoids reloading the full MIL PKL)
│
├── Slide_Level_LR_kNN_train_eval/              # Slide-level (mean-pooled) model training and evaluation
│   ├── LR_MeanPooling_OvR.py                  # OvR Logistic Regression on mean-pooled slide features; patient-level 80/10/10
│   │                                           # train/val/cal split; optional temperature scaling and class-imbalance strategies
│   ├── LR_evaluate_case_level.py              # Evaluates LR slide-level models at slide and patient/case level with 95% CIs
│   └── CosineKNN_MeanPooling.py               # k-NN classifier with cosine similarity on mean-pooled embeddings;
│                                               # K selected by 5-fold stratified CV (K ∈ {1,3,5,7,9,15,21}) on the training split
│
├── MIL_train_eval/                             # Attention-based MIL training and evaluation (revised)
│   ├── Attention_based_MIL.py                 # Trains OvR AttentionMIL classifiers with balancing and temperature calibration;
│   │                                           # evaluates per epoch with binary metrics; saves best model by validation F1
│   └── Attention_MIL_evaluate_per_case.py     # Evaluates MIL models at slide and patient/case level; exports attention weights,
│                                               # PR/ROC curves, confusion matrices, and JSON metrics
│
└── LR_tile_train_eval/                         # Tile-level Logistic Regression training and evaluation (revised)
    ├── LR_tile_level.py                        # Weakly supervised OvR LR on tile features; patient-level 5-fold CV splits;
    │                                           # oversampling via imblearn Pipeline; optional slide-level probability aggregation
    └── Evaluate_per_case_tile_level_LR.py      # Evaluates tile-level LR at tile, slide, and patient/case level with 95% CIs;
                                                # source-stratified metrics (TCGA / CPTAC / Warwick); exports PR/ROC curves
```




## 🧠 Notes

Access to TCGA-BRCA, CPTAC-BRCA, and Warwick HER2 datasets requires appropriate data use permissions

The scripts are designed to run on a local filesystem; paths and hyperparameters may need editing for other environments.

Users can consult docstrings in each script for detailed parameter explanations.

Due to dataset size, logistic regression at tile level was not retrained for calibration because of limited performance and higher computational cost compared to GPU-trained MIL models.

All core code for feature extraction, model training, and evaluation is fully included for transparency and reuse.

Researchers are encouraged to adapt the pipeline to their own datasets using the provided modules.

