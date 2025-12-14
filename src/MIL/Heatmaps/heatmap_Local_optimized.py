import os
import math
import numpy as np
from PIL import Image
import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

# UNI2 imports
import timm
import sys
import importlib
import importlib.util
import json
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
from torchvision import transforms
# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
    _TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    _TQDM_AVAILABLE = False


# ---------------- CONFIG ----------------
SLIDE_PATH = "/home/projects2/WSI_project/PhD_WSI/download_v1/data_TCGA_BRCA/TCGA_WSI_svs/TCGA-A2-A04T-01Z-00-DX1.71444266-BD56-4183-9603-C7AC20C9DA1E.svs"
TRUE_LABEL = "LumA"  # <-- Hardcoded ground truth label
CHECKPOINT_PATH = "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/results/Mil_OvR_training_calib_nooversample/class_2_LumA/model_2_LumA.pt"
OUT_DIR = "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/Heatmap/heatmap_webapp_results"
os.makedirs(OUT_DIR, exist_ok=True)

TILE_PX = 224
STRIDE_PX = 224
FEATURE_DIM = 1536     # UNI2-h outputs 1536
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THUMB_MAX_SIZE = 2048
GAUSSIAN_SIGMA_PX = 8
SAVE_FEATURES = True  # set False to avoid writing features to disk
# Progress/monitoring options
USE_TQDM = True            # show tqdm bars if available
LOG_EVERY_TILES = 200      # fallback: print status every N tiles if no tqdm
# ----------------------------------------

CHECKPOINT_PATHS = {
    "Basal": "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/results/Mil_OvR_training_calib_nooversample/class_0_Basal/model_0_Basal.pt",
    "Her2": "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/results/Mil_OvR_training_calib_nooversample/class_1_HER2/model_1_HER2.pt",
    "LumA": "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/results/Mil_OvR_training_calib_nooversample/class_2_LumA/model_2_LumA.pt",
    "LumB": "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/results/Mil_OvR_training_calib_nooversample/class_3_LumB/model_3_LumB.pt"
}

def load_all_ovr_models():
    """Load all 4 OvR classifiers and return as dictionary"""
    models = {}
    for subtype, ckpt_path in CHECKPOINT_PATHS.items():
        print(f"Loading {subtype} classifier...")
        model = AttentionMIL(input_dim=FEATURE_DIM, hidden_dim=256, n_classes=2, dropout_rate=0.0)
        try:
            load_checkpoint_to_model(model, ckpt_path)
            models[subtype] = model
            print(f"✓ Successfully loaded {subtype} model")
        except Exception as e:
            print(f"✗ Failed to load {subtype} model: {e}")
            models[subtype] = None
    return models

def compute_all_ovr_predictions(models, features_np):
    """Compute predictions for all OvR classifiers"""
    results = {}
    ovr_scores = []
    
    for subtype, model in models.items():
        if model is None:
            print(f"Skipping {subtype} - model not loaded")
            continue
            
        print(f"Computing predictions for {subtype}...")
        logits, att_vec, p_tile, att_weighted = compute_tile_attention_and_scores_nochunk(
            model, features_np, pos_class_idx=1
        )
        
        # Store results for this subtype
        results[subtype] = {
            'logits': logits,
            'attention': att_vec,
            'p_tile': p_tile,
            'att_weighted': att_weighted,
            'slide_score': np.mean(p_tile)  # Average probability across all tiles
        }
        ovr_scores.append(results[subtype]['slide_score'])
        
        print(f"{subtype} - Slide score: {results[subtype]['slide_score']:.4f}")
        print(f"{subtype} - High prob tiles (>0.5): {(p_tile > 0.5).sum()}/{len(p_tile)}")
    
    # Determine final prediction (highest OvR score)
    if ovr_scores:
        subtypes = list(results.keys())
        final_prediction = subtypes[np.argmax(ovr_scores)]
        max_score = max(ovr_scores)
        
        print(f"\n=== FINAL PREDICTION ===")
        print(f"Predicted subtype: {final_prediction}")
        print(f"Confidence score: {max_score:.4f}")
        print(f"All scores: {dict(zip(subtypes, ovr_scores))}")
        
        results['final_prediction'] = {
            'subtype': final_prediction,
            'confidence': max_score,
            'all_scores': dict(zip(subtypes, ovr_scores))
        }
    
    return results

def render_all_heatmaps(slide_path, centers, results, slide_name, out_dir):
    """Generate heatmaps for all subtypes"""
    heatmap_paths = {}
    
    # Open slide once and generate thumbnail
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.dimensions
    thumb = slide.get_thumbnail((THUMB_MAX_SIZE, THUMB_MAX_SIZE)).convert("RGB")
    slide.close()
    
    for subtype, result_data in results.items():
        if subtype == 'final_prediction':
            continue
            
        print(f"Rendering heatmaps for {subtype}...")
        
        # Attention heatmap
        out_png_att = os.path.join(out_dir, f"{slide_name}_{subtype}_heatmap_attention.png")
        render_heatmap_on_thumbnail(thumb, (W, H), centers, result_data['attention'], 
                                  out_png_att, classifier_name=subtype, value_type="attention")
        
        # Probability heatmap
        out_png_prob = os.path.join(out_dir, f"{slide_name}_{subtype}_heatmap_probability.png")
        render_heatmap_on_thumbnail(thumb, (W, H), centers, result_data['p_tile'], 
                                  out_png_prob, classifier_name=subtype, value_type="probability")
        
        # High-contrast probability heatmap
        threshold = np.percentile(result_data['p_tile'], 80)
        p_tile_threshold = np.where(result_data['p_tile'] >= threshold, result_data['p_tile'], 0)
        out_png_contrast = os.path.join(out_dir, f"{slide_name}_{subtype}_heatmap_prob_highcontrast.png")
        render_heatmap_on_thumbnail(thumb, (W, H), centers, p_tile_threshold, 
                                  out_png_contrast, classifier_name=subtype, value_type="probability (top 20%)")
        
        heatmap_paths[subtype] = {
            'attention': out_png_att,
            'probability': out_png_prob,
            'high_contrast': out_png_contrast
        }
    return heatmap_paths

def save_all_results(out_dir, slide_name, centers, results):
    """Save comprehensive results including all OvR predictions"""
    # Create comprehensive dataframe
    df_data = {
        "loc_x": centers[:,0],
        "loc_y": centers[:,1],
    }
    
    # Add columns for each subtype
    for subtype, result_data in results.items():
        if subtype == 'final_prediction':
            continue
        df_data[f"{subtype}_attention"] = result_data['attention']
        df_data[f"{subtype}_probability"] = result_data['p_tile']
        df_data[f"{subtype}_att_weighted"] = result_data['att_weighted']
        
        # Add logits if available
        if result_data['logits'].ndim == 2:
            for i in range(result_data['logits'].shape[1]):
                df_data[f"{subtype}_logit_{i}"] = result_data['logits'][:, i]
    
    # Save as parquet
    df = pd.DataFrame(df_data)
    parquet_path = os.path.join(out_dir, f"{slide_name}_all_subtypes.parquet")
    df.to_parquet(parquet_path)
    
    # Save summary results as JSON
    def to_py(obj):
        # Recursively convert numpy types to Python types
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_py(v) for v in obj]
        return obj

    summary = {
        'slide_name': slide_name,
        'final_prediction': to_py(results.get('final_prediction', {})),
        'subtype_scores': {k: to_py(v.get('slide_score', 0)) for k, v in results.items() if k != 'final_prediction'},
        'tile_counts': {k: to_py(len(v.get('p_tile', []))) for k, v in results.items() if k != 'final_prediction'}
    }

    summary_path = os.path.join(out_dir, f"{slide_name}_prediction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved comprehensive results to {parquet_path}")
    print(f"Saved prediction summary to {summary_path}")

    return parquet_path, summary_path

# Build UNI2 extractor
def build_uni2_extractor(device=DEVICE, target_dim=FEATURE_DIM):
    local_dir = "/home/projects2/WSI_project/PhD_WSI/feature_extraction/UNI2_model/"
    weights_path = os.path.join(local_dir, "pytorch_model.bin")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"[UNI2] Missing weights file: {weights_path}")

    print(f"[UNI2] Loading UNI2 locally from {weights_path}")
    print(f"[UNI2] Device: {device}")
    if torch.cuda.is_available():
        print(f"[UNI2] GPU: {torch.cuda.get_device_name(0)} (count={torch.cuda.device_count()})")
    else:
        print("[UNI2] CUDA not available, using CPU")

    # IMPORTANT: mlp_ratio must match checkpoint (4096 hidden => 4096 / 1536 ≈ 2.66667)
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2 ,     
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked if hasattr(timm, "layers") else None,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }

    model = timm.create_model(timm_kwargs.pop('model_name'), pretrained=False, **timm_kwargs)

    sd = torch.load(weights_path, map_location='cpu')
    if isinstance(sd, dict):
        if 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']
        elif 'model' in sd and isinstance(sd['model'], dict):
            sd = sd['model']
    # strip "module." prefixes if any
    if isinstance(sd, dict):
        sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

    # Load strictly now that shapes should match
    model.load_state_dict(sd, strict=True)
    print("[UNI2] Weights loaded (strict=True)")

    model = model.to(device).eval()
    print(f"[UNI2] Model moved to {device}. Param dtype: {next(model.parameters()).dtype}")

    # Use fixed ImageNet-like transform (lab doc) — stable & matches training
    from torchvision import transforms as tv_transforms
    transform = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225)),
    ])

    def extract_batch(pil_images):
        # Always run inference in float32 (drop amp to avoid dtype mismatch)
        tensors = [transform(im) for im in pil_images]
        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            feats = model(batch)  # (N, 1536)
        feats = feats.cpu().numpy()
        if feats.shape[1] != target_dim:
            print(f"[UNI2] Adjusting feature dim {feats.shape[1]} -> {target_dim}")
            if feats.shape[1] > target_dim:
                feats = feats[:, :target_dim]
            else:
                pad = np.zeros((feats.shape[0], target_dim - feats.shape[1]), dtype=feats.dtype)
                feats = np.concatenate([feats, pad], axis=1)
        return feats.astype(np.float32)

    print("[UNI2] Extractor ready.")
    return extract_batch, transform, model


# ----- Model import / checkpoint helpers -----
def _import_attention_mil():
    """Try to import AttentionMIL from the project's evaluation script, else define locally."""
    try:
        spec = importlib.util.find_spec("scripts.Evaluate_Mil")
        if spec is not None:
            mod = importlib.import_module("scripts.Evaluate_Mil")
            if hasattr(mod, "AttentionMIL"):
                return mod.AttentionMIL
    except Exception:
        pass
    # Fallback: local minimal AttentionMIL matching Evaluate_Mil.py
    class AttentionMILFallback(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_classes=2, dropout_rate=0.0):
            super().__init__()
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            self.feature_extractor = nn.Sequential(*layers)
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
            self.classifier = nn.Linear(hidden_dim, n_classes)
        def forward(self, bag):
            H = self.feature_extractor(bag)
            A = self.attention(H)
            A = torch.softmax(A, dim=0)
            M = torch.sum(A * H, dim=0)
            logits = self.classifier(M)
            return logits, A.squeeze(-1)
    return AttentionMILFallback


AttentionMIL = _import_attention_mil()


def load_checkpoint_to_model(model: nn.Module, ckpt_path: str, map_location=None):
    """Simple loader: load state_dict (or dict), strip 'module.' prefixes and try load_state_dict.
    Falls back to strict=False if exact keys don't match.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(ckpt_path, map_location=map_location)
    # extract state_dict-like mapping
    if isinstance(data, dict) and "state_dict" in data:
        sd = data["state_dict"]
    elif isinstance(data, dict):
        sd = data
    else:
        try:
            sd = data.state_dict()
        except Exception:
            raise RuntimeError("Unrecognized checkpoint format")
    # strip module. prefix
    sd2 = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items() }
    try:
        model.load_state_dict(sd2)
        print(f"Loaded checkpoint into model from {ckpt_path} (strict=True)")
    except Exception as e:
        # try non-strict load
        model.load_state_dict(sd2, strict=False)
        print(f"Loaded checkpoint into model from {ckpt_path} with strict=False (partial load). Error: {e}")
    model.to(DEVICE).eval()
    return True


# Tile slide -> centers and top-lefts
def get_tile_centers(slide_path, tile_px=TILE_PX, stride_px=STRIDE_PX):
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.dimensions
    centers = []
    top_lefts = []
    for x in range(0, W - tile_px + 1, stride_px):
        for y in range(0, H - tile_px + 1, stride_px):
            centers.append((x + tile_px//2, y + tile_px//2))
            top_lefts.append((x, y))
    slide.close()
    return np.array(centers, dtype=int), top_lefts

def extract_tile_image(slide_path, top_left, size=TILE_PX):
    slide = openslide.OpenSlide(slide_path)
    img = slide.read_region(top_left, 0, (size, size)).convert("RGB")
    slide.close()
    return img

# Replace compute_attention_and_logits_torch with a no-chunk tile-scoring helper
def compute_tile_attention_and_scores_nochunk(model, features_np, pos_class_idx=1):
    """
    Compute per-tile logits, attention (softmax across tiles), per-tile class prob and attention-weighted score.
    Assumes features_np shape (N, input_dim). No chunking; suitable for single-slide tests.
    Returns: logits_np (N,C), att_vec (N,), p_tile (N,), att_weighted (N,)
    """
    model = model.to(DEVICE).eval()
    X = torch.from_numpy(np.asarray(features_np, dtype=np.float32)).to(DEVICE)
    with torch.no_grad():
        H = model.feature_extractor(X)               # (N, hidden_dim)
        logits_tile = model.classifier(H)            # (N, C)
        att_raw = model.attention(H).squeeze(-1)     # (N,)
        # attention softmax across tiles (stable)
        att_exp = torch.exp(att_raw - torch.max(att_raw))
        att = att_exp / (att_exp.sum() + 1e-12)
        probs = F.softmax(logits_tile, dim=1)        # (N, C)
    logits_np = logits_tile.cpu().numpy()
    att_vec = att.cpu().numpy().astype(np.float32)
    p_tile = probs[:, pos_class_idx].cpu().numpy().astype(np.float32)
    att_weighted = (att_vec * p_tile).astype(np.float32)

    def norm01(x):
        mn = np.nanmin(x); mx = np.nanmax(x)
        if np.isnan(mn) or np.isnan(mx) or mx <= mn:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)
    return logits_np, norm01(att_vec), norm01(p_tile), norm01(att_weighted)

def render_heatmap_on_thumbnail(thumb, slide_dims, centers, att_vec, out_png, classifier_name=None, thumb_max=THUMB_MAX_SIZE, sigma=GAUSSIAN_SIGMA_PX, cmap="inferno", value_type="attention"):
    """
    Args:
        classifier_name (str): The name of the classifier (e.g., "Basal", "LumA")
    """
    W, H = slide_dims
    
    # Validate input coordinates
    validate_coordinates(centers, (W, H), TILE_PX)
    
    # Use the same thumbnail approach as debug visualization for consistency
    tw, th = thumb.size
    
    # Map coordinates exactly like in debug visualization
    coords = np.array([(int(cx * tw/W), int(cy * th/H)) for cx, cy in centers], dtype=int)
    check_thumbnail_mapping((W, H), (tw, th), centers, coords)
    
    # Enhance signal strength: Apply power transformation to boost high attention values
    att_enhanced = np.power(att_vec, 0.5)  # Square root to boost mid-range values
    # Alternative: att_enhanced = np.power(att_vec, 0.3) for even stronger enhancement
    
    # Create heatmap using the same coordinate system as debug visualization
    heat = np.zeros((th, tw), dtype=np.float32)
    for (x, y), v in zip(coords, att_enhanced):
        if 0 <= x < tw and 0 <= y < th:
            heat[y, x] += float(v)
            
    try:
        from scipy.ndimage import gaussian_filter
        # Reduce blur to keep signals more localized
        heat = gaussian_filter(heat, sigma=max(1, sigma//2))
        hm = heat - heat.min() if heat.max() != heat.min() else heat
        if hm.max() != 0:
            hm = hm / hm.max()
        
        # Boost contrast further with histogram stretching
        hm_stretched = np.power(hm, 0.7)  # Power transformation for better contrast
        
        # Create side-by-side visualization: thumbnail + heatmap + colorbar
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Original thumbnail
        ax1.imshow(thumb)
        ax1.set_title("Original Tissue", fontsize=14, fontweight='bold')
        ax1.axis("off")
        ax1.invert_yaxis()
        
        # Right: Heatmap overlay
        ax2.imshow(thumb)
        im = ax2.imshow(hm_stretched, cmap=cmap, alpha=0.8, vmin=0.1, vmax=1.0)
        map_type = "Attention" if "attention" in value_type.lower() else "Probability"
        ax2.set_title(f"{classifier_name} Classifier {map_type} Map", fontsize=14, fontweight='bold')
        ax2.axis("off")
        ax2.invert_yaxis()
        
        # Add colorbar with proper labels
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        if value_type == "probability":
            cbar.set_label('Basal Probability\n(0=Low, 1=High)', rotation=270, labelpad=20, fontsize=12)
        elif value_type == "attention":
            cbar.set_label('Attention Weight\n(0=Ignored, 1=Important)', rotation=270, labelpad=20, fontsize=12)
        else:
            cbar.set_label(f'{value_type.title()}\n(0=Low, 1=High)', rotation=270, labelpad=20, fontsize=12)
        
        # Add statistics text box
        stats_text = f"Stats: Min={att_vec.min():.3f}, Max={att_vec.max():.3f}, Mean={att_vec.mean():.3f}\n"
        stats_text += f"High values (>0.5): {(att_vec > 0.5).sum()}/{len(att_vec)} tiles"
        fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1, dpi=150)
        plt.close(fig)
    except Exception:
        # Fallback to scatter plot with enhanced visibility
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Original thumbnail
        ax1.imshow(thumb)
        ax1.set_title("Original Tissue", fontsize=14, fontweight='bold')
        ax1.axis("off")
        ax1.invert_yaxis()
        
        # Right: Scatter plot overlay
        ax2.imshow(thumb)
        xs = coords[:,0]; ys = coords[:,1]
        # Enhance attention values for scatter plot
        att_enhanced = np.power(att_vec, 0.5)
        # Larger points and higher alpha for better visibility
        sc = ax2.scatter(xs, ys, c=att_enhanced, cmap=cmap, s=(TILE_PX*tw/W)**2/8, alpha=0.8, 
                       vmin=0.1, vmax=1.0, edgecolors='white', linewidths=0.5)
        map_type = "Attention" if "attention" in value_type.lower() else "Probability"
        ax2.set_title(f"{classifier_name} Classifier {map_type} Map", fontsize=14, fontweight='bold')
        ax2.axis("off")
        ax2.invert_yaxis()
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
        if value_type == "probability":
            cbar.set_label('Basal Probability\n(0=Low, 1=High)', rotation=270, labelpad=20, fontsize=12)
        elif value_type == "attention":
            cbar.set_label('Attention Weight\n(0=Ignored, 1=Important)', rotation=270, labelpad=20, fontsize=12)
        else:
            cbar.set_label(f'{value_type.title()}\n(0=Low, 1=High)', rotation=270, labelpad=20, fontsize=12)
        
        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1, dpi=150)
        plt.close(fig)

# ---- NEW: helpers to save maps/images/npz/parquet for both random test and real slide ----
def _save_npz_and_parquet(outdir, name, centers, att_vec, p_tile, att_weighted, logits=None, meta=None):
    os.makedirs(outdir, exist_ok=True)
    # save npz
    npz_path = os.path.join(outdir, f"{name}_maps.npz")
    save_dict = {
        "loc_x": np.asarray(centers)[:, 0].astype(np.int32),
        "loc_y": np.asarray(centers)[:, 1].astype(np.int32),
        "attention": np.asarray(att_vec).astype(np.float32),
        "p_tile": np.asarray(p_tile).astype(np.float32),
        "att_weighted": np.asarray(att_weighted).astype(np.float32),
    }
    if logits is not None:
        save_dict["logits"] = np.asarray(logits)
    if meta is not None:
        import json
        save_dict["meta"] = np.array([json.dumps(meta)], dtype=object)
    np.savez_compressed(npz_path, **save_dict)

    # save parquet/csv
    try:
        df = pd.DataFrame({
            "loc_x": np.asarray(centers)[:, 0],
            "loc_y": np.asarray(centers)[:, 1],
            "attention": np.asarray(att_vec).astype(np.float32),
            "p_tile": np.asarray(p_tile).astype(np.float32),
            "att_weighted": np.asarray(att_weighted).astype(np.float32),
        })
        if logits is not None:
            lar = np.asarray(logits)
            if lar.ndim == 1:
                df["logit"] = lar
            elif lar.ndim == 2:
                for i in range(lar.shape[1]):
                    df[f"logit_{i}"] = lar[:, i]
        parquet_path = os.path.join(outdir, f"{name}_tiles.parquet")
        df.to_parquet(parquet_path)
    except Exception:
        csv_path = os.path.join(outdir, f"{name}_tiles.csv")
        df.to_csv(csv_path, index=False)

    return npz_path

def _save_map_image(values, centers, out_png, grid_w=None, cmap="inferno"):
    vals = np.asarray(values).astype(np.float32)
    n = vals.shape[0]
    # Attempt grid reshape if possible
    if grid_w is None:
        root = int(np.sqrt(n))
        if root * root == n:
            grid_h = grid_w = root
        else:
            grid_w = int(np.ceil(np.sqrt(n)))
            grid_h = int(np.ceil(n / grid_w))
    else:
        grid_w = int(grid_w)
        grid_h = int(np.ceil(n / grid_w))
    # pad to full grid
    pad = grid_w * grid_h - n
    grid = np.pad(vals, (0, pad), constant_values=0.0).reshape((grid_h, grid_w))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_all_maps(outdir, name, centers, att_vec, p_tile, att_weighted, logits=None, meta=None):
    npz_path = _save_npz_and_parquet(outdir, name, centers, att_vec, p_tile, att_weighted, logits=logits, meta=meta)
    # save images (grid representation)
    _save_map_image(att_vec, centers, os.path.join(outdir, f"{name}_attention.png"), cmap="magma")
    _save_map_image(p_tile, centers, os.path.join(outdir, f"{name}_p_tile.png"), cmap="viridis")
    _save_map_image(att_weighted, centers, os.path.join(outdir, f"{name}_att_weighted.png"), cmap="inferno")
    return npz_path

# ---- NEW: random test runner (quick sanity on one slide worth of tiles) ----
def random_test_and_generate_maps(n_tiles=256, seed=0, outdir=OUT_DIR):
    """
    Run a quick random-features test through your AttentionMIL to validate numeric stability
    and generate attention / p_tile / att_weighted maps saved to outdir.
    """
    rng = np.random.RandomState(seed)
    features = rng.randn(n_tiles, FEATURE_DIM).astype(np.float32)
    # synthetic centers laid out on a simple grid (for visualization)
    grid_w = int(math.ceil(math.sqrt(n_tiles)))
    centers = []
    for i in range(n_tiles):
        gx = i % grid_w
        gy = i // grid_w
        centers.append((gx, gy))
    centers = np.array(centers, dtype=int)

    # instantiate model and load weights (safe to run even if weights missing)
    mil = AttentionMIL(input_dim=FEATURE_DIM, hidden_dim=256, n_classes=2, dropout_rate=0.0)
    try:
        load_checkpoint_to_model(mil, CHECKPOINT_PATH)
    except Exception:
        # proceed with random init model if checkpoint fails
        pass

    # compute maps (no-chunk path)
    logits, att_vec, p_tile, att_weighted = compute_tile_attention_and_scores_nochunk(mil, features, pos_class_idx=1)
    name = "random_test"
    meta = {"test": "random", "n_tiles": n_tiles}
    npz_path = save_all_maps(outdir, name, centers, att_vec, p_tile, att_weighted, logits=logits, meta=meta)
    print(f"Random test saved to {npz_path}")
    return True

class TileDataset(Dataset):
    def __init__(self, slide_path, top_lefts, tile_px, transform):
        self.slide_path = slide_path
        self.top_lefts = top_lefts
        self.tile_px = tile_px
        self.transform = transform
        self._slide = None

    def _ensure_open(self):
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.slide_path)

    def __len__(self):
        return len(self.top_lefts)

    def __getitem__(self, idx):
        self._ensure_open()
        tl = self.top_lefts[idx]
        img = self._slide.read_region(tl, 0, (self.tile_px, self.tile_px)).convert("RGB")
        return self.transform(img)

    def __del__(self):
        try:
            if self._slide is not None:
                self._slide.close()
        except Exception:
            pass

def filter_tissue_tiles(slide, top_lefts, centers, tile_px=224, threshold=220, tissue_percent=0.05):
    """
    Fast tissue filtering using downsampled reads.
    Args:
        slide: Already-open OpenSlide handle (reuse across calls)
    """
    keep_idxs = []
    
    # Use 8× downsampling for fast preview
    try:
        level = slide.get_best_level_for_downsample(8)
        downsample = slide.level_downsamples[level]
    except:
        level = 0
        downsample = 1.0
    
    size_at_level = (
        max(1, int(tile_px / downsample)),
        max(1, int(tile_px / downsample))
    )
    
    for i, tl in enumerate(top_lefts):
        img = slide.read_region(tl, level, size_at_level).convert("RGB")
        arr = np.array(img)
        tissue_mask = (arr[..., :3] < threshold).any(axis=-1)
        tissue_ratio = tissue_mask.mean()
        if tissue_ratio > tissue_percent:
            keep_idxs.append(i)
    
    centers_filt = np.array(centers)[keep_idxs]
    top_lefts_filt = [top_lefts[i] for i in keep_idxs]
    return centers_filt, top_lefts_filt, keep_idxs

def validate_coordinates(centers, slide_dims, tile_px):
    """Validate that tile centers are within slide bounds"""
    W, H = slide_dims
    centers = np.asarray(centers)
    
    # Check bounds
    x_valid = (centers[:, 0] >= tile_px//2) & (centers[:, 0] < W - tile_px//2)
    y_valid = (centers[:, 1] >= tile_px//2) & (centers[:, 1] < H - tile_px//2)
    
    if not (x_valid & y_valid).all():
        bad_coords = centers[~(x_valid & y_valid)]
        raise ValueError(f"Found {len(bad_coords)} tile centers outside slide bounds: {bad_coords[:5]}...")
    
    # Note: Skip stride validation for filtered tissue tiles as they won't have regular spacing
    
    return True

def check_thumbnail_mapping(slide_dims, thumb_dims, centers, mapped_coords):
    """Verify thumbnail coordinate mapping maintains relative positions"""
    W, H = slide_dims
    tw, th = thumb_dims
    
    # Check scale factors
    scale_x = tw / W
    scale_y = th / H
    
    # For filtered tissue tiles, just verify that coordinates are within bounds
    # rather than checking exact mapping since spacing is irregular
    if not ((mapped_coords[:, 0] >= 0) & (mapped_coords[:, 0] < tw)).all():
        raise ValueError(f"Some X coordinates are outside thumbnail bounds [0, {tw})")
    if not ((mapped_coords[:, 1] >= 0) & (mapped_coords[:, 1] < th)).all():
        raise ValueError(f"Some Y coordinates are outside thumbnail bounds [0, {th})")
    
    return True

def debug_coordinate_mapping(slide_path, centers, mapped_coords, out_debug_png):
    """Generate debug visualization of coordinate mapping"""
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.dimensions
    thumb = slide.get_thumbnail((1024, 1024)).convert("RGB")
    tw, th = thumb.size
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original coordinates
    ax1.imshow(thumb)
    ax1.scatter(centers[:, 0] * tw/W, centers[:, 1] * th/H, c='r', alpha=0.5, s=1)
    ax1.set_title("Original Centers")
    ax1.invert_yaxis()

    # Mapped coordinates
    ax2.imshow(thumb)
    ax2.scatter(mapped_coords[:, 0], mapped_coords[:, 1], c='b', alpha=0.5, s=1)
    ax2.set_title("Mapped Centers")
    ax2.invert_yaxis()

    plt.savefig(out_debug_png)
    plt.close()

def print_attention_stats(att_vec):
    """
    Print summary statistics for attention weights.
    """
    print(f"Attention stats: min={att_vec.min():.4f}, max={att_vec.max():.4f}, mean={att_vec.mean():.4f}, std={att_vec.std():.4f}")
    print(f"Tiles with attention > 0.5: {(att_vec > 0.5).sum()} / {len(att_vec)}")

def print_top_attention_tiles(att_vec, p_tile, centers, N=10):
    """
    Print the top-N tiles by attention weight for debugging.
    """
    top_idxs = np.argsort(att_vec)[-N:][::-1]
    print("\nTop attention tiles:")
    for i in top_idxs:
        print(f"Tile {i}: Center={centers[i]}, Attention={att_vec[i]:.4f}, Probability={p_tile[i]:.4f}")

def plot_attention_histogram(att_vec, slide_name, out_dir):
    """
    Plot and save a histogram of attention weights for debugging.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(att_vec, bins=50, color='purple', alpha=0.7)
    plt.title(f"Attention Weight Distribution: {slide_name}")
    plt.xlabel("Attention Weight")
    plt.ylabel("Number of Tiles")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{slide_name}_attention_histogram.png")
    plt.savefig(out_png)
    plt.close()
    print(f"Saved attention histogram: {out_png}")

def plot_attention_vs_probability(att_vec, p_tile, slide_name, out_dir):
    """
    Plot and save a scatter plot of attention weights vs tile probabilities.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(att_vec, p_tile, alpha=0.5, color='teal')
    plt.xlabel("Attention Weight")
    plt.ylabel("Tile Probability")
    plt.title(f"Attention vs Probability: {slide_name}")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{slide_name}_attention_vs_probability.png")
    plt.savefig(out_png)
    plt.close()
    print(f"Saved attention vs probability scatter: {out_png}")

def main():
    # Simple global progress file until we know slide_name
    global_progress_path = os.path.join(OUT_DIR, "progress.json")
    _write_progress(global_progress_path, stage="start", current=0, total=1, extra={"msg": "starting run"})

    # --- prepare slide & feature storage paths ---
    slide_name = os.path.splitext(os.path.basename(SLIDE_PATH))[0]
    features_npy = os.path.join(OUT_DIR, f"{slide_name}_features.npy")
    features_pt = os.path.join(OUT_DIR, f"{slide_name}_features.pt")
    bag_pt = os.path.join(OUT_DIR, f"{slide_name}.pt")
    idx_npz = os.path.join(OUT_DIR, f"{slide_name}.index.npz")

    # Progress file now uses slide-specific name
    slide_progress_path = os.path.join(OUT_DIR, f"{slide_name}.progress.json")
    _write_progress(slide_progress_path, stage="init", current=0, total=1, extra={"slide": slide_name})

    # Get all possible tile locations and filter for tissue tiles
    # This needs to be done regardless of whether features are cached or not
    print("Getting tile locations and filtering for tissue...")
    all_centers, all_top_lefts = get_tile_centers(SLIDE_PATH, TILE_PX, STRIDE_PX)
    
    # Open slide once and pass handle to filter (avoids reopening for each tile)
    slide_for_filter = openslide.OpenSlide(SLIDE_PATH)
    centers, top_lefts, keep_idxs = filter_tissue_tiles(
        slide_for_filter, all_top_lefts, all_centers,
        tile_px=TILE_PX,
        threshold=220,
        tissue_percent=0.05
    )
    slide_for_filter.close()
    
    n_tiles = len(centers)
    print(f"Kept {n_tiles} tissue tiles after filtering")

    # If pre-extracted features exist, load them (fast). Otherwise extract and optionally save.
    if os.path.exists(features_npy):
        print(f"Loading pre-extracted features from {features_npy}")
        _write_progress(slide_progress_path, stage="load_cached_features", current=0, total=1)
        features = np.load(features_npy)
        # ensure shape and dtype
        features = np.asarray(features, dtype=np.float32)
        _write_progress(slide_progress_path, stage="load_cached_features_done", current=1, total=1, extra={"n_tiles": int(features.shape[0])})
    elif os.path.exists(features_pt):
        print(f"Loading pre-extracted features from {features_pt}")
        _write_progress(slide_progress_path, stage="load_cached_features", current=0, total=1)
        tmp = torch.load(features_pt, map_location='cpu')
        # handle tensor or numpy saved inside
        if isinstance(tmp, torch.Tensor):
            features = tmp.cpu().numpy().astype(np.float32)
        else:
            features = np.asarray(tmp, dtype=np.float32)
        _write_progress(slide_progress_path, stage="load_cached_features_done", current=1, total=1, extra={"n_tiles": int(features.shape[0])})
    else:
        # GPU-optimized feature extraction
        extractor, transform, model = build_uni2_extractor(device=DEVICE, target_dim=FEATURE_DIM)
        
        # Extract features only for tissue tiles
        _write_progress(slide_progress_path, stage="extract_features", current=0, total=int(n_tiles))

        dataset = TileDataset(SLIDE_PATH, top_lefts, TILE_PX, transform)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

        features_parts = []
        processed = 0
        start_t = time.time()
        pbar = tqdm(total=n_tiles, desc="Extracting features", unit="tile") if USE_TQDM and _TQDM_AVAILABLE else None

        torch.backends.cudnn.benchmark = True
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE, non_blocking=True)
                # with torch.cuda.amp.autocast():
                feats = model(batch)
                feats = feats.float().cpu().numpy()
                features_parts.append(feats)
                processed += feats.shape[0]
                if pbar is not None:
                    pbar.update(feats.shape[0])
                else:
                    if processed % max(1, LOG_EVERY_TILES) == 0 or processed == n_tiles:
                        elapsed = time.time() - start_t
                        rate = processed / max(1e-9, elapsed)
                        remaining = n_tiles - processed
                        eta_sec = remaining / max(1e-9, rate)
                        print(f"Processed {processed}/{n_tiles} tiles | {rate:.2f} tiles/s | ETA ~{int(eta_sec)}s")
                _write_progress(
                    slide_progress_path,
                    stage="extract_features",
                    current=int(processed),
                    total=int(n_tiles),
                    extra={"batch": int(feats.shape[0]), "elapsed_sec": round(time.time()-start_t, 2)}
                )
        if pbar is not None:
            pbar.close()
        features = np.concatenate(features_parts, axis=0)
        print("Features shape:", features.shape)
        _write_progress(slide_progress_path, stage="extract_features_done", current=int(processed), total=int(n_tiles), extra={"n_tiles": int(features.shape[0])})

        # Save extracted features for reuse
        if SAVE_FEATURES:
            try:
                np.save(features_npy, features)
                torch.save(torch.from_numpy(features), features_pt)
                print(f"Saved features to {features_npy} and {features_pt}")
                _write_progress(slide_progress_path, stage="features_saved", current=1, total=1)
            except Exception as e:
                print("Failed saving features:", e)
                _write_progress(slide_progress_path, stage="features_save_failed", current=1, total=1, extra={"error": str(e)})

    # Save bag & index for compatibility (this mirrors what the script did previously)
    # Use filtered tissue tile centers for all outputs
    torch.save(torch.from_numpy(features.astype(np.float32)), bag_pt)
    np.savez(idx_npz, centers.astype(np.int32))
    print("Saved bag and index:", bag_pt, idx_npz)
    _write_progress(slide_progress_path, stage="bag_and_index_saved", current=1, total=1)

    # Generate thumbnail once for all heatmaps
    slide = openslide.OpenSlide(SLIDE_PATH)
    W, H = slide.dimensions
    thumb = slide.get_thumbnail((THUMB_MAX_SIZE, THUMB_MAX_SIZE)).convert("RGB")
    slide.close()
    print(f"Generated thumbnail: {thumb.size}")

    # Load MIL model and weights (same as before) and compute maps
    mil = AttentionMIL(input_dim=FEATURE_DIM, hidden_dim=256, n_classes=2, dropout_rate=0.0)
    load_checkpoint_to_model(mil, CHECKPOINT_PATH)
    _write_progress(slide_progress_path, stage="scoring", current=0, total=int(features.shape[0]))
    logits, att_vec, p_tile, att_weighted = compute_tile_attention_and_scores_nochunk(mil, features, pos_class_idx=1)
    print("Attention vector shape:", att_vec.shape)
    print(f"Attention values - min: {att_vec.min():.4f}, max: {att_vec.max():.4f}, mean: {att_vec.mean():.4f}")
    print(f"Tile probabilities - min: {p_tile.min():.4f}, max: {p_tile.max():.4f}, mean: {p_tile.mean():.4f}")
    print(f"Attention-weighted - min: {att_weighted.min():.4f}, max: {att_weighted.max():.4f}, mean: {att_weighted.mean():.4f}")
    print(f"Number of tiles with attention > 0.1: {(att_vec > 0.1).sum()}")
    print(f"Number of tiles with probability > 0.5: {(p_tile > 0.5).sum()}")
    print(f"Number of tiles with probability > 0.7: {(p_tile > 0.7).sum()}")

    # --- NEW: Debugging and visualization for attention ---
    _write_progress(slide_progress_path, stage="scoring_done", current=int(features.shape[0]), total=int(features.shape[0]))

    # Create heatmap using attention weights
    out_png_attention = os.path.join(OUT_DIR, f"{slide_name}_heatmap_attention.png")
    render_heatmap_on_thumbnail(thumb, (W, H), centers, att_vec, out_png_attention, value_type="attention")
    print("Saved attention heatmap:", out_png_attention)
    
    # Create heatmap using tile probabilities (this might be more informative!)
    out_png_prob = os.path.join(OUT_DIR, f"{slide_name}_heatmap_probability.png")
    render_heatmap_on_thumbnail(thumb, (W, H), centers, p_tile, out_png_prob, value_type="probability")
    print("Saved probability heatmap:", out_png_prob)
    
    # Create heatmap using attention-weighted probabilities
    out_png_weighted = os.path.join(OUT_DIR, f"{slide_name}_heatmap_weighted.png")
    render_heatmap_on_thumbnail(thumb, (W, H), centers, att_weighted, out_png_weighted, value_type="weighted")
    print("Saved attention-weighted heatmap:", out_png_weighted)
    
    # Also create a high-contrast version using top percentile thresholding of probabilities
    out_png_highcontrast = os.path.join(OUT_DIR, f"{slide_name}_heatmap_prob_highcontrast.png")
    # Keep only top 20% of probability values for high contrast visualization
    threshold = np.percentile(p_tile, 80)
    p_tile_threshold = np.where(p_tile >= threshold, p_tile, 0)
    render_heatmap_on_thumbnail(thumb, (W, H), centers, p_tile_threshold, out_png_highcontrast, value_type="probability (top 20%)")
    print("Saved high-contrast probability heatmap:", out_png_highcontrast)
    
    _write_progress(slide_progress_path, stage="render_done", current=1, total=1, extra={"heatmaps": [out_png_attention, out_png_prob, out_png_weighted, out_png_highcontrast]})

    # Save dataframe (parquet)
    try:
        import pandas as pd
        df = pd.DataFrame({
            "loc_x": centers[:,0],
            "loc_y": centers[:,1],
            "attention": att_vec,
            "p_tile": p_tile,
            "att_weighted": att_weighted
        })
        larr = np.asarray(logits)
        if larr.ndim == 2:
            for i in range(larr.shape[1]):
                df[f"logit_{i}"] = larr[:, i]
        df.to_parquet(os.path.join(OUT_DIR, f"{slide_name}_tiles.parquet"))
        print("Saved tile dataframe.")
        _write_progress(slide_progress_path, stage="parquet_saved", current=1, total=1)
    except Exception:
        print("pandas not available; skipping parquet export.")
        _write_progress(slide_progress_path, stage="parquet_skip", current=1, total=1)

    _write_progress(slide_progress_path, stage="done", current=1, total=1)

    # --- NEW: debug output for coordinate mapping ---
    debug_path = os.path.join(OUT_DIR, f"{slide_name}_coordinate_debug.png")
    # Calculate scale_x and scale_y for thumbnail mapping
    slide = openslide.OpenSlide(SLIDE_PATH)
    W, H = slide.dimensions
    scale_x = min(THUMB_MAX_SIZE / W, 1.0)
    scale_y = min(THUMB_MAX_SIZE / H, 1.0)
    slide.close()
    mapped_coords = np.array([(int(cx * scale_x), int(cy * scale_y)) for cx, cy in centers])
    
    # Load all OvR models instead of just one
    models = load_all_ovr_models()
    
    # Compute predictions for all subtypes
    _write_progress(slide_progress_path, stage="scoring_all_subtypes", current=0, total=len(models))
    all_results = compute_all_ovr_predictions(models, features)
    _write_progress(slide_progress_path, stage="scoring_done", current=len(models), total=len(models))
    
    # Generate heatmaps for all subtypes
    _write_progress(slide_progress_path, stage="render_all_heatmaps", current=0, total=len(models))
    heatmap_paths = render_all_heatmaps(SLIDE_PATH, centers, all_results, slide_name, OUT_DIR)
    _write_progress(slide_progress_path, stage="render_done", current=len(models), total=len(models))
    
    # Save comprehensive results
    parquet_path, summary_path = save_all_results(OUT_DIR, slide_name, centers, all_results)
    
    # Update progress with final prediction info
    final_pred = all_results.get('final_prediction', {})
    _write_progress(slide_progress_path, stage="done", current=1, total=1, 
                   extra={
                       "final_prediction": final_pred,
                       "heatmaps_generated": len(heatmap_paths),
                       "results_saved": [parquet_path, summary_path]
                   })
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Final prediction: {final_pred.get('subtype', 'Unknown')}")
    print(f"Confidence: {final_pred.get('confidence', 0):.4f}")
    print(f"Results saved to: {OUT_DIR}")

def _write_progress(path, stage, current, total, extra=None):
    """Write progress info to a JSON file."""
    info = {
        "stage": stage,
        "current": current,
        "total": total,
        "timestamp": datetime.now().isoformat()
    }
    if extra is not None:
        info["extra"] = extra
    try:
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
    except Exception as e:
        print(f"Failed to write progress file {path}: {e}")

if __name__  == "__main__":
    main()

