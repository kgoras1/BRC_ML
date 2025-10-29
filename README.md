# Breast Cancer Molecular Subtyping from Whole Slide Images using UNI-2 Features

This repository contains the code and experimental pipeline for our study on **annotation-free and weakly supervised classification of breast cancer molecular subtypes from histopathology whole slide images (WSI)**.  
The workflow integrates ****data_preprocessing** **foundation model feature extraction**, **multiple instance learning (MIL)**, and **classical machine learning baselines** for reproducible benchmarking across multiple data sources (TCGA-BRCA, CPTAC-BRCA, and Warwick HER2).

---

## 🧩 Overview

Our end-to-end pipeline consists of four main stages:

1. **Dataset assembly and patient-level splitting**  
   Stratified 80/20 train-test partitioning by subtype and source (TCGA, CPTAC, Warwick), with fixed seeds for reproducibility.

2. **Preprocessing and tiling**  
   Tissue detection and tiling at 20× magnification into 224×224 patches with no overlap, excluding background and blurry tiles, using simple image-statistics filters.

3. **Feature extraction using the UNI-2 foundation model**  
   Tile embeddings (1536-D) were extracted using the UNI-2 ViT-H/14 encoder trained via the DINOv2 self-supervised framework.
   Slide-level representations were obtained by mean pooling tile embeddings or aggregating through attention-based MIL.

4. **One vs Rest Model training and evaluation**  
   Each molecular subtype was modeled independently using an OvR setup, where one binary classifier distinguishes each subtype from all others.

   Cosine similarity baseline: non-parametric k-NN on mean-pooled slide vectors.

   Logistic Regression (LR) and Linear Discriminant Analysis (LDA) on pooled slide features.

   Attention-based Deep MIL using tile-level embeddings and attention pooling.

  Tile-level Logistic Regression MIL for interpretable, per-tile predictions.
  
  Optional components include class balancing (oversampling, downsampling, SMOTE) and probability calibration (isotonic or temperature scaling).
  Evaluation metrics include accuracy, balanced accuracy, macro F1, weighted F1, ROC-AUC, precision and recall computed per class, with macro-averaged scores reported as overall performance summaries across all subtypes.

---

## ⚙️ Repository Structure

src/
├── preprocessing/ # WSI preprocessing and tiling
│ ├── tile_filtering.py # Removes background, white, and blurry tiles based on pixel intensity and variance
│ └── tile_rename.py # Renames and organizes tiles while retaining coordinates and magnification, Slide_ID, data source, and molecular subtype label

├── feature_extraction/ # Tile-level feature extraction with UNI-2
│ ├── UNI2_fextraction.py # Extracts 1536-D embeddings from the UNI-2 encoder (ViT-H/14, DINOv2)
│ └── Create_Tile_Features_dicts.py # Builds per-slide dictionaries of extracted features;
│ # each Slide_ID is a key containing tiles as subkeys and their 1536-D feature vectors as values

├── mean_pooling_features/ # Slide-level (mean-pooled) feature ML training and analysis
│ ├── Create_MeanPooling_vectors.py # Generates mean-pooled 1536-D slide representations from tile embeddings
│ ├── LDA_analysis_mean_vectors.py # Performs Linear Discriminant Analysis (LDA) for feature-space visualization
│ └── ML_mean_features/ # Classical ML models on pooled slide features
│ ├── Logistic_models.py # One-vs-Rest (OvR) logistic regression with/without calibration and class balancing
│ ├── data_preperation/ # Train/validation/calibration split creation
│ │ └── data_preperation.py # Generates 80/20 train-test split and 8/1/1 train-eval-calib subsets and CV splits
│ # tracks all Slide_IDs to prevent data leakage
│ └── Cosine_Similarity.py # k-NN (cosine similarity) baseline on mean-pooled vectors

├── MIL/ # Multiple Instance Learning (MIL) approaches
│ ├── MIL_training/ # Attention-based deep MIL and tile-level MIL training
│ │ ├── AttentionMIL_Callibration_Balance_training.py # Attention-based MIL with class balancing and temperature calibration
│ │ └── LR_MIL.py # Tile-level logistic regression MIL
│ ├── MIL_evaluate/ # MIL model evaluation and calibration
│ │ └── AttentionMIL_evaluate.py # Evaluates MIL models and applies temperature scaling
│ └── Heatmaps/ # Visualization of spatial attention and probability maps
│ └── HeatmapV2.py # Generates subtype-specific heatmaps highlighting informative tissue regions




## 🧠 Notes

Access to TCGA-BRCA, CPTAC-BRCA, and Warwick HER2 datasets requires appropriate data use permissions

Some file paths, slide identifiers, and hyperparameters are dataset-specific and will need modification.

Each script includes docstrings describing its purpose, parameters, and expected inputs/outputs — users are encouraged to review these for guidance on customization.

Due to dataset size, logistic regression at tile level was not retrained for calibration because of limited performance and higher computational cost compared to GPU-trained MIL models.

All core code for feature extraction, model training, and evaluation is fully included for transparency and reuse.

Researchers are encouraged to adapt the pipeline to their own datasets using the provided modules.

