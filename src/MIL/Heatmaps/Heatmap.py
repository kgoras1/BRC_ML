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
import argparse
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
# UNI2 imports
import timm
import sys
import importlib
import importlib.util
import json
from torchvision import transforms
# Optional progress bar
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    _TQDM_AVAILABLE = False


# ---------------- DEFAULT CONFIG ----------------
OUT_DIR = "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/Heatmap/heatmap_webapp_results"
DEFAULT_CHECKPOINT_PATHS = {
    "Basal": "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/results/Mil_OvR_training_calib_nooversample/class_0_Basal/model_0_Basal.pt",
    "Her2": "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/results/Mil_OvR_training_calib_nooversample/class_1_HER2/model_1_HER2.pt",
    "LumA": "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/results/Mil_OvR_training_calib_nooversample/class_2_LumA/model_2_LumA.pt",
    "LumB": "/home/projects2/WSI_project/PhD_WSI/feature_extraction/ml_attention_pooling/results/Mil_OvR_training_calib_nooversample/class_3_LumB/model_3_LumB.pt"
}
DEFAULT_UNI2_WEIGHTS = "/home/projects2/WSI_project/PhD_WSI/feature_extraction/UNI2_model/pytorch_model.bin"

TILE_PX = 224          # 224×224 tiles
STRIDE_PX = 224        # No overlap
FEATURE_DIM = 1536
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THUMB_MAX_SIZE = 2048
GAUSSIAN_SIGMA_PX = 8
USE_TQDM = True
LOG_EVERY_TILES = 200
# ----------------------------------------

CHECKPOINT_PATHS = DEFAULT_CHECKPOINT_PATHS.copy()


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
    """Compute predictions for all OvR classifiers with GPU batching"""
    results = {}
    ovr_scores = []
    
    X = torch.from_numpy(features_np.astype(np.float32)).to(DEVICE)
    
    for subtype, model in models.items():
        if model is None:
            print(f"Skipping {subtype} - model not loaded")
            continue
            
        print(f"Computing predictions for {subtype}...")
        model = model.to(DEVICE).eval()
        
        with torch.no_grad():
            H = model.feature_extractor(X)
            logits_tile = model.classifier(H)
            att_raw = model.attention(H).squeeze(-1)
            att = F.softmax(att_raw, dim=0)
            probs = F.softmax(logits_tile, dim=1)
        
        logits_np = logits_tile.cpu().numpy()
        att_vec = att.cpu().numpy()
        p_tile = probs[:, 1].cpu().numpy()
        att_weighted = att_vec * p_tile
        
        def norm01(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(x)
        
        results[subtype] = {
            'logits': logits_np,
            'attention': norm01(att_vec),
            'p_tile': norm01(p_tile),
            'att_weighted': norm01(att_weighted),
            'slide_score': np.mean(p_tile)
        }
        ovr_scores.append(results[subtype]['slide_score'])
        
        print(f"{subtype} - Slide score: {results[subtype]['slide_score']:.4f}")
    
    if ovr_scores:
        subtypes = list(results.keys())
        final_prediction = subtypes[np.argmax(ovr_scores)]
        max_score = max(ovr_scores)
        
        print(f"\n=== FINAL PREDICTION ===")
        print(f"Predicted subtype: {final_prediction}")
        print(f"Confidence score: {max_score:.4f}")
        
        results['final_prediction'] = {
            'subtype': final_prediction,
            'confidence': float(max_score),
            'all_scores': {k: float(v) for k, v in zip(subtypes, ovr_scores)}
        }
    
    return results


def render_all_heatmaps(slide_path, centers, results, slide_name, out_dir):
    """Generate attention, probability, attention-weighted, and high-contrast heatmaps for all subtypes."""
    heatmap_paths = {}
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
        render_heatmap_on_thumbnail(
            thumb, (W, H), centers, result_data['attention'],
            out_png_att, classifier_name=subtype, value_type="attention"
        )

        # Probability heatmap
        out_png_prob = os.path.join(out_dir, f"{slide_name}_{subtype}_heatmap_probability.png")
        render_heatmap_on_thumbnail(
            thumb, (W, H), centers, result_data['p_tile'],
            out_png_prob, classifier_name=subtype, value_type="probability"
        )

        # Attention-weighted probability heatmap
        out_png_weighted = os.path.join(out_dir, f"{slide_name}_{subtype}_heatmap_att_weighted_prob.png")
        render_heatmap_on_thumbnail(
            thumb, (W, H), centers, result_data['att_weighted'],
            out_png_weighted, classifier_name=subtype, value_type="attention-weighted probability"
        )

        # High-contrast (top 20% probabilities)
        threshold = np.percentile(result_data['p_tile'], 80)
        p_tile_threshold = np.where(result_data['p_tile'] >= threshold, result_data['p_tile'], 0)
        out_png_contrast = os.path.join(out_dir, f"{slide_name}_{subtype}_heatmap_prob_highcontrast.png")
        render_heatmap_on_thumbnail(
            thumb, (W, H), centers, p_tile_threshold,
            out_png_contrast, classifier_name=subtype, value_type="probability (top 20%)"
        )

        heatmap_paths[subtype] = {
            "attention": out_png_att,
            "probability": out_png_prob,
            "att_weighted_probability": out_png_weighted,
            "high_contrast_probability": out_png_contrast
        }
    return heatmap_paths


def save_all_results(out_dir, slide_name, centers, results):
    """Save comprehensive results including all OvR predictions"""
    df_data = {
        "loc_x": centers[:,0],
        "loc_y": centers[:,1],
    }
    
    # Get expected length from centers
    n_tiles = len(centers)
    
    for subtype, result_data in results.items():
        if subtype == 'final_prediction':
            continue
        
        # Validate array lengths match
        if len(result_data['attention']) != n_tiles:
            print(f"Warning: {subtype} attention length mismatch. Expected {n_tiles}, got {len(result_data['attention'])}")
            continue
            
        df_data[f"{subtype}_attention"] = result_data['attention']
        df_data[f"{subtype}_probability"] = result_data['p_tile']
        df_data[f"{subtype}_att_weighted"] = result_data['att_weighted']
        
        if result_data['logits'].ndim == 2 and result_data['logits'].shape[0] == n_tiles:
            for i in range(result_data['logits'].shape[1]):
                df_data[f"{subtype}_logit_{i}"] = result_data['logits'][:, i]
    
    df = pd.DataFrame(df_data)
    parquet_path = os.path.join(out_dir, f"{slide_name}_all_subtypes.parquet")
    df.to_parquet(parquet_path)
    
    def to_py(obj):
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


def build_uni2_extractor(device=DEVICE, target_dim=FEATURE_DIM):
    weights_path = DEFAULT_UNI2_WEIGHTS

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"[UNI2] Missing weights file: {weights_path}")

    print(f"[UNI2] Loading UNI2 from {weights_path} on {device}")

    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,  # match checkpoint
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }

    model = timm.create_model(timm_kwargs.pop('model_name'), pretrained=False, **timm_kwargs)

    sd = torch.load(weights_path, map_location='cpu')
    if isinstance(sd, dict):
        if 'state_dict' in sd:
            sd = sd['state_dict']
        elif 'model' in sd:
            sd = sd['model']
    if isinstance(sd, dict):
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    print("[UNI2] Weights loaded (strict=True)")

    model = model.to(device).eval()
    print(f"[UNI2] Model on {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    def extract_batch(pil_images):
        tensors = [transform(im) for im in pil_images]
        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            feats = model(batch)
        feats = feats.cpu().numpy()
        if feats.shape[1] != target_dim:
            if feats.shape[1] > target_dim:
                feats = feats[:, :target_dim]
            else:
                pad = np.zeros((feats.shape[0], target_dim - feats.shape[1]), dtype=feats.dtype)
                feats = np.concatenate([feats, pad], axis=1)
        return feats.astype(np.float32)

    print("[UNI2] Extractor ready")
    return extract_batch, transform, model


def _import_attention_mil():
    try:
        spec = importlib.util.find_spec("scripts.Evaluate_Mil")
        if spec is not None:
            mod = importlib.import_module("scripts.Evaluate_Mil")
            if hasattr(mod, "AttentionMIL"):
                return mod.AttentionMIL
    except Exception:
        pass
    
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
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(ckpt_path, map_location=map_location)
    
    if isinstance(data, dict) and "state_dict" in data:
        sd = data["state_dict"]
    elif isinstance(data, dict):
        sd = data
    else:
        try:
            sd = data.state_dict()
        except Exception:
            raise RuntimeError("Unrecognized checkpoint format")
    
    sd2 = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    
    try:
        model.load_state_dict(sd2)
        print(f"Loaded checkpoint from {ckpt_path} (strict=True)")
    except Exception as e:
        model.load_state_dict(sd2, strict=False)
        print(f"Loaded checkpoint from {ckpt_path} (strict=False). Error: {e}")
    
    model.to(DEVICE).eval()
    return True


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


def render_heatmap_on_thumbnail(thumb, slide_dims, centers, values, out_png,
                                classifier_name=None, value_type="attention",
                                cmap="inferno"):
    W, H = slide_dims
    tw, th = thumb.size
    coords = np.array([(int(cx * tw / W), int(cy * th / H)) for cx, cy in centers], dtype=int)

    # Different preprocessing for attention vs probability
    v = values.astype(np.float32)
    
    if "attention" in value_type.lower():
        # FOR ATTENTION: Moderate boosting (more balanced)
        # 1. Clip at 95th percentile (middle ground)
        p_hi = np.percentile(v, 95.0)
        if p_hi > 0:
            v = np.clip(v, 0, p_hi) / p_hi
        
        # 2. Moderate power boost BEFORE gridding
        v = np.power(v, 0.4)  # More moderate (between 0.25 and 0.5)
        
    else:
        # FOR PROBABILITY: Keep original gentle processing
        p_hi = np.percentile(v, 99.0)
        if p_hi > 0:
            v = np.clip(v, 0, p_hi) / p_hi

    # Build heat grid
    heat = np.zeros((th, tw), dtype=np.float32)
    for (x, y), val in zip(coords, v):
        if 0 <= x < tw and 0 <= y < th:
            heat[y, x] += val

    try:
        from scipy.ndimage import gaussian_filter
        if "attention" in value_type.lower():
            heat = gaussian_filter(heat, sigma=2.5)  # Moderate blur
        else:
            heat = gaussian_filter(heat, sigma=2)
    except Exception:
        pass

    # Normalize to [0, 1]
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    
    # Second power boost AFTER normalization (only for attention)
    if "attention" in value_type.lower():
        heat = np.power(heat, 0.5)  # Gentler boost (sqrt)
        # Re-normalize
        if heat.max() > heat.min():
            heat = (heat - heat.min()) / (heat.max() - heat.min())
    else:
        # For probability: gentle boost only
        heat = np.power(heat, 0.5)

    # Create RGBA heatmap
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    
    base_cmap = cm.get_cmap(cmap)
    heat_rgba = base_cmap(heat)  # (H, W, 4)
    
    # Set alpha channel
    if "attention" in value_type.lower():
        # For attention: moderate alpha
        alpha = np.power(heat, 0.6)  # More moderate alpha curve
        heat_rgba[..., 3] = alpha * 0.65  # Moderate max opacity
        heat_rgba[heat < 0.08, 3] = 0     # Moderate threshold
    else:
        # For probability: keep original
        alpha = np.power(heat, 0.7)
        heat_rgba[..., 3] = alpha * 0.7
        heat_rgba[heat < 0.05, 3] = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: original tissue
    ax1.imshow(thumb)
    ax1.set_title("Original Tissue", fontsize=14)
    ax1.axis("off")
    
    # Right: tissue + transparent heatmap overlay
    ax2.imshow(thumb)
    ax2.imshow(heat_rgba, interpolation='bilinear')
    ax2.set_title(f"{classifier_name} {value_type.title()} Map", fontsize=14)
    ax2.axis("off")
    
    # Add colorbar
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=base_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(f'{value_type} (0=Low, 1=High)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor='white')
    plt.close(fig)


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


def filter_tissue_tiles(slide, top_lefts, centers, tile_px=224, threshold=220, tissue_percent=0.25):
    """Fast tissue filtering using downsampled reads"""
    keep_idxs = []
    
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate attention heatmaps for whole-slide images using UNI2 + MIL models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--slide_path", type=str, required=True, help="Path to input whole-slide image")
    parser.add_argument("--true_label", type=str, default=None, help="Ground truth label (optional)")
    parser.add_argument("--checkpoint_basal", type=str, default=DEFAULT_CHECKPOINT_PATHS["Basal"])
    parser.add_argument("--checkpoint_her2", type=str, default=DEFAULT_CHECKPOINT_PATHS["Her2"])
    parser.add_argument("--checkpoint_luma", type=str, default=DEFAULT_CHECKPOINT_PATHS["LumA"])
    parser.add_argument("--checkpoint_lumb", type=str, default=DEFAULT_CHECKPOINT_PATHS["LumB"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no_cache_features", action="store_true")
    parser.add_argument("--tissue_threshold", type=int, default=220)
    parser.add_argument("--tissue_percent", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--no_tqdm", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.isfile(args.slide_path):
        raise FileNotFoundError(f"Slide not found: {args.slide_path}")
    
    global DEVICE, BATCH_SIZE, USE_TQDM, CHECKPOINT_PATHS
    SLIDE_PATH = args.slide_path
    TRUE_LABEL = args.true_label
    BATCH_SIZE = args.batch_size
    SAVE_FEATURES = not args.no_cache_features
    USE_TQDM = not args.no_tqdm
    
    if args.device:
        DEVICE = args.device
    
    CHECKPOINT_PATHS = {
        "Basal": args.checkpoint_basal,
        "Her2": args.checkpoint_her2,
        "LumA": args.checkpoint_luma,
        "LumB": args.checkpoint_lumb
    }
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("="*80)
    print("HEATMAP GENERATION CONFIGURATION")
    print("="*80)
    print(f"Slide path:        {SLIDE_PATH}")
    print(f"Output directory:  {OUT_DIR}")
    print(f"True label:        {TRUE_LABEL or 'Not provided'}")
    print(f"Device:            {DEVICE}")
    print(f"Tile size:         {TILE_PX}px (no overlap)")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Cache features:    {SAVE_FEATURES}")
    print("="*80)
    
    global_progress_path = os.path.join(OUT_DIR, "progress.json")
    _write_progress(global_progress_path, stage="start", current=0, total=1, extra={"msg": "starting run"})

    slide_name = os.path.splitext(os.path.basename(SLIDE_PATH))[0]
    features_npy = os.path.join(OUT_DIR, f"{slide_name}_features.npy")
    features_pt = os.path.join(OUT_DIR, f"{slide_name}_features.pt")
    bag_pt = os.path.join(OUT_DIR, f"{slide_name}.pt")
    idx_npz = os.path.join(OUT_DIR, f"{slide_name}.index.npz")

    slide_progress_path = os.path.join(OUT_DIR, f"{slide_name}.progress.json")
    _write_progress(slide_progress_path, stage="init", current=0, total=1, extra={"slide": slide_name})

    print("Getting tile locations and filtering for tissue...")
    all_centers, all_top_lefts = get_tile_centers(SLIDE_PATH, TILE_PX, STRIDE_PX)
    
    slide_for_filter = openslide.OpenSlide(SLIDE_PATH)
    centers, top_lefts, keep_idxs = filter_tissue_tiles(
        slide_for_filter, all_top_lefts, all_centers,
        tile_px=TILE_PX,
        threshold=args.tissue_threshold,
        tissue_percent=args.tissue_percent
    )
    slide_for_filter.close()
    
    n_tiles = len(centers)
    print(f"Kept {n_tiles} tissue tiles after filtering")

    if SAVE_FEATURES and os.path.exists(features_npy):
        print(f"Loading pre-extracted features from {features_npy}")
        features = np.load(features_npy)
        features = np.asarray(features, dtype=np.float32)
    elif SAVE_FEATURES and os.path.exists(features_pt):
        print(f"Loading pre-extracted features from {features_pt}")
        tmp = torch.load(features_pt, map_location='cpu')
        if isinstance(tmp, torch.Tensor):
            features = tmp.cpu().numpy().astype(np.float32)
        else:
            features = np.asarray(tmp, dtype=np.float32)
    else:
        extractor, transform, model = build_uni2_extractor(device=DEVICE, target_dim=FEATURE_DIM)
        
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
        if pbar is not None:
            pbar.close()
        features = np.concatenate(features_parts, axis=0)
        print("Features shape:", features.shape)

        if SAVE_FEATURES:
            try:
                np.save(features_npy, features)
                torch.save(torch.from_numpy(features), features_pt)
                print(f"Saved features to {features_npy} and {features_pt}")
            except Exception as e:
                print("Failed saving features:", e)

    torch.save(torch.from_numpy(features.astype(np.float32)), bag_pt)
    np.savez(idx_npz, centers.astype(np.int32))
    print("Saved bag and index:", bag_pt, idx_npz)

    slide = openslide.OpenSlide(SLIDE_PATH)
    W, H = slide.dimensions
    thumb = slide.get_thumbnail((THUMB_MAX_SIZE, THUMB_MAX_SIZE)).convert("RGB")
    slide.close()
    print(f"Generated thumbnail: {thumb.size}")

    models = load_all_ovr_models()
    
    all_results = compute_all_ovr_predictions(models, features)
    
    heatmap_paths = render_all_heatmaps(SLIDE_PATH, centers, all_results, slide_name, OUT_DIR)
    
    parquet_path, summary_path = save_all_results(OUT_DIR, slide_name, centers, all_results)
    
    # Write metrics summary txt
    write_metrics_txt(
        OUT_DIR,
        slide_name,
        TILE_PX,
        STRIDE_PX,
        total_tiles=len(all_centers),
        kept_tiles=len(centers),
        discarded_tiles=len(all_centers) - len(centers),
        tissue_threshold=args.tissue_threshold,
        tissue_percent=args.tissue_percent,
        results=all_results
    )
    
    final_pred = all_results.get('final_prediction', {})
    _write_progress(slide_progress_path, stage="done", current=1, total=1, 
                   extra={
                       "final_prediction": final_pred,
                       "true_label": TRUE_LABEL,
                       "heatmaps_generated": len(heatmap_paths),
                       "results_saved": [parquet_path, summary_path]
                   })
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Final prediction: {final_pred.get('subtype', 'Unknown')}")
    print(f"Confidence: {final_pred.get('confidence', 0):.4f}")
    if TRUE_LABEL:
        print(f"Ground truth: {TRUE_LABEL}")
        print(f"Correct: {final_pred.get('subtype') == TRUE_LABEL}")
    print(f"Results saved to: {OUT_DIR}")


def write_metrics_txt(
    out_dir,
    slide_name,
    tile_px,
    stride_px,
    total_tiles,
    kept_tiles,
    discarded_tiles,
    tissue_threshold,
    tissue_percent,
    results
):
    """Write run metrics to a plain text file."""
    metrics_path = os.path.join(out_dir, f"{slide_name}_metrics.txt")
    lines = []
    lines.append(f"Slide: {slide_name}")
    lines.append(f"Tile size: {tile_px}x{tile_px}")
    lines.append(f"Stride: {stride_px} (overlap: {max(0, tile_px - stride_px)})")
    lines.append(f"Total candidate tiles: {total_tiles}")
    lines.append(f"Kept tissue tiles: {kept_tiles}")
    lines.append(f"Discarded tiles: {discarded_tiles}")
    kept_ratio = kept_tiles / total_tiles if total_tiles > 0 else 0
    lines.append(f"Kept ratio: {kept_ratio:.4f}")
    lines.append(f"Tissue threshold (pixel intensity): {tissue_threshold}")
    lines.append(f"Tissue percent min: {tissue_percent}")
    lines.append("")
    lines.append("Per-class slide scores:")
    for subtype, data in results.items():
        if subtype == "final_prediction":
            continue
        score = data.get("slide_score", 0.0)
        lines.append(f"  {subtype}: {score:.6f}")
    lines.append("")
    final_pred = results.get("final_prediction", {})
    lines.append("Final prediction:")
    lines.append(f"  Subtype: {final_pred.get('subtype', 'NA')}")
    lines.append(f"  Confidence: {final_pred.get('confidence', 0):.6f}")
    all_scores = final_pred.get('all_scores', {})
    if all_scores:
        lines.append("  All scores:")
        for k, v in all_scores.items():
            lines.append(f"    {k}: {v:.6f}")
    lines.append("")
    lines.append("Tile counts recorded per subtype (length of probability vector):")
    for subtype, data in results.items():
        if subtype == "final_prediction":
            continue
        lines.append(f"  {subtype}: {len(data.get('p_tile', []))}")
    try:
        with open(metrics_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved metrics txt: {metrics_path}")
    except Exception as e:
        print(f"Failed writing metrics file: {e}")


def _write_progress(path, stage, current, total, extra=None):
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


if __name__ == "__main__":
    main()

