#!/usr/bin/env python3
"""
HeatmapV2: OvR MIL heatmaps over whole-slide images (WSI) with UNI2 features.

Overview:
- Extracts tile-level features from a WSI using the UNI2-h model (timm checkpoint).
- Scores tiles with one-vs-rest (OvR) AttentionMIL binary classifiers per subtype.
- Produces attention, probability, and attention-weighted heatmaps over a slide thumbnail.
- Exports per-tile tables (parquet), compressed maps (npz), debug figures, and a JSON summary.
- Optionally runs a quick numerical sanity check on random features.

Key features:
- Flexible CLI: no hard-coded paths required (defaults remain available).
- Robust I/O with progress JSONs to monitor long-running jobs.
- Optional tissue filtering before feature extraction.
- Loads OvR models either from a JSON mapping or from default in-script mapping.
- Headless-safe plotting (matplotlib Agg backend).
- Reproducible runs (seeded RNG).

Example:
  python HeatmapV2.py \
    --slide_path /path/to/slide.svs \
    --checkpoint_paths_json checkpoints.json \
    --output_dir out/slide_X \
    --tile_px 224 --stride_px 224 --batch_size 256 --device cuda \
    --save_features

Where checkpoints.json contains:
{
  "Basal": "/path/to/model_0_Basal.pt",
  "Her2":  "/path/to/model_1_HER2.pt",
  "LumA":  "/path/to/model_2_LumA.pt",
  "LumB":  "/path/to/model_3_LumB.pt"
}

Notes:
- AttentionMIL is imported from scripts.Evaluate_Mil if present; otherwise a compatible fallback is used.
- AUROC/AP are not computed here; this script focuses on per-tile visualization and slide-level OvR scoring.

Author: Konstantinos Papagoras
Date: 2025-10
"""

import os
import math
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import pandas as pd
import openslide
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# UNI2 imports
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
    _TQDM_AVAILABLE = True
except Exception:
    tqdm = None
    _TQDM_AVAILABLE = False

# ------------- Defaults (can be overridden via CLI) -------------
# Reasonable defaults; prefer passing values via CLI for reproducible experiments.
DEFAULT_SLIDE_PATH = "/path/to/slide.svs"
DEFAULT_TRUE_LABEL = "Basal"
DEFAULT_OUT_DIR = "./heatmap_v2_output"
DEFAULT_FEATURE_DIM = 1536      # UNI2-h outputs 1536
DEFAULT_TILE_PX = 224
DEFAULT_STRIDE_PX = 224
DEFAULT_BATCH_SIZE = 256
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_THUMB_MAX_SIZE = 2048
DEFAULT_GAUSSIAN_SIGMA_PX = 8
DEFAULT_SAVE_FEATURES = True
DEFAULT_USE_TQDM = True
DEFAULT_LOG_EVERY_TILES = 200  # if no tqdm
# Optional: default OvR checkpoints (override with --checkpoint_paths_json)
DEFAULT_CHECKPOINT_PATHS = {
    "Basal": "/path/to/model_0_Basal.pt",
    "Her2": "/path/to/model_1_HER2.pt",
    "LumA": "/path/to/model_2_LumA.pt",
    "LumB": "/path/to/model_3_LumB.pt",
}

LOGGER = logging.getLogger("heatmap_v2")


# ---------------------- Utilities ----------------------
def setup_logging(out_dir: str, level: int = logging.INFO) -> None:
    """Configure console and file logging."""
    os.makedirs(out_dir, exist_ok=True)
    LOGGER.setLevel(level)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    LOGGER.addHandler(ch)

    fh = logging.FileHandler(os.path.join(out_dir, "heatmap_v2.log"))
    fh.setLevel(level)
    fh.setFormatter(fmt)
    LOGGER.addHandler(fh)


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility (numpy and torch)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_progress(path: str, stage: str, current: int, total: int, extra: Optional[Dict] = None) -> None:
    """Write progress information to a JSON file for monitoring."""
    info = {
        "stage": stage,
        "current": current,
        "total": total,
        "timestamp": datetime.now().isoformat(),
    }
    if extra is not None:
        info["extra"] = extra
    try:
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
    except Exception as e:
        LOGGER.warning(f"Failed to write progress file {path}: {e}")


# ------------------- AttentionMIL import -------------------
def _import_attention_mil():
    """
    Import AttentionMIL from scripts.Evaluate_Mil if available.
    Fallback to a minimal compatible local implementation otherwise.
    """
    import importlib
    import importlib.util

    try:
        spec = importlib.util.find_spec("scripts.Evaluate_Mil")
        if spec is not None:
            mod = importlib.import_module("scripts.Evaluate_Mil")
            if hasattr(mod, "AttentionMIL"):
                return mod.AttentionMIL
    except Exception:
        pass

    class AttentionMILFallback(nn.Module):
        """Minimal Attention-based MIL head compatible with Evaluate_Mil.AttentionMIL."""
        def __init__(self, input_dim: int, hidden_dim: int, n_classes: int = 2, dropout_rate: float = 0.0):
            super().__init__()
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            self.feature_extractor = nn.Sequential(*layers)
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
            self.classifier = nn.Linear(hidden_dim, n_classes)

        def forward(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                bag: [N_tiles, input_dim]
            Returns:
                logits: [2]
                attn: [N_tiles]
            """
            H = self.feature_extractor(bag)
            A = self.attention(H)
            A = torch.softmax(A, dim=0)
            M = torch.sum(A * H, dim=0)
            logits = self.classifier(M)
            return logits, A.squeeze(-1)

    return AttentionMILFallback


AttentionMIL = _import_attention_mil()


# ---------------------- Checkpoint I/O ----------------------
def load_checkpoint_to_model(model: nn.Module, ckpt_path: str, map_location: Optional[str] = None) -> bool:
    """
    Load model weights from a checkpoint. Strips 'module.' prefixes if present.
    Falls back to strict=False if exact key match fails.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    map_location = map_location or (DEFAULT_DEVICE)
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

    sd = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items() }
    try:
        model.load_state_dict(sd)
        LOGGER.info(f"Loaded checkpoint (strict=True): {ckpt_path}")
    except Exception as e:
        model.load_state_dict(sd, strict=False)
        LOGGER.warning(f"Loaded checkpoint with strict=False (partial): {ckpt_path} | reason: {e}")
    model.to(DEFAULT_DEVICE).eval()
    return True


def load_all_ovr_models(checkpoint_paths: Dict[str, str], feature_dim: int, hidden_dim: int = 256) -> Dict[str, Optional[nn.Module]]:
    """
    Load all OvR classifiers defined in checkpoint_paths.
    Returns a dict subtype -> model (or None if loading failed).
    """
    models: Dict[str, Optional[nn.Module]] = {}
    for subtype, ckpt_path in checkpoint_paths.items():
        LOGGER.info(f"Loading classifier: {subtype}")
        model = AttentionMIL(input_dim=feature_dim, hidden_dim=hidden_dim, n_classes=2, dropout_rate=0.0)
        try:
            load_checkpoint_to_model(model, ckpt_path)
            models[subtype] = model
            LOGGER.info(f"Loaded {subtype} model")
        except Exception as e:
            LOGGER.error(f"Failed to load {subtype} model: {e}")
            models[subtype] = None
    return models


# ------------------------ UNI2 extractor ------------------------
def build_uni2_extractor(device: str = DEFAULT_DEVICE, target_dim: int = DEFAULT_FEATURE_DIM):
    """
    Build UNI2-h model from timm hub and return a callable that extracts features for batches of PIL images.
    Returns:
        extract_batch(pil_images) -> np.ndarray [N, target_dim]
        transform (torchvision-like)
        model (torch.nn.Module)
    """
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': torch.nn.SiLU,  # fallback if Swish-GLU is not available
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True,
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model = model.to(device).eval()
    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**cfg, use_prefetcher=False)

    def extract_batch(pil_images: List[Image.Image]) -> np.ndarray:
        tensors = [transform(im) for im in pil_images]
        xs = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            feats = model(xs)  # [N, 1536]
        feats = feats.detach().cpu().numpy()
        if feats.shape[1] != target_dim:
            if feats.shape[1] > target_dim:
                feats = feats[:, :target_dim]
            else:
                pad = np.zeros((feats.shape[0], target_dim - feats.shape[1]), dtype=feats.dtype)
                feats = np.concatenate([feats, pad], axis=1)
        return feats.astype(np.float32)

    return extract_batch, transform, model


# -------------------- Slide tiling and I/O --------------------
def get_tile_centers(slide_path: str, tile_px: int, stride_px: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Compute tile centers and top-left coordinates across the full WSI at level 0."""
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.dimensions
    centers, top_lefts = [], []
    for x in range(0, W - tile_px + 1, stride_px):
        for y in range(0, H - tile_px + 1, stride_px):
            centers.append((x + tile_px // 2, y + tile_px // 2))
            top_lefts.append((x, y))
    slide.close()
    return np.array(centers, dtype=int), top_lefts


class TileDataset(Dataset):
    """Dataset for reading slide tiles and applying the given transform."""
    def __init__(self, slide_path: str, top_lefts: List[Tuple[int, int]], tile_px: int, transform):
        self.slide_path = slide_path
        self.top_lefts = top_lefts
        self.tile_px = tile_px
        self.transform = transform
        self._slide: Optional[openslide.OpenSlide] = None

    def _ensure_open(self):
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.slide_path)

    def __len__(self):
        return len(self.top_lefts)

    def __getitem__(self, idx: int):
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


def filter_tissue_tiles(
    slide_path: str,
    top_lefts: List[Tuple[int, int]],
    centers: np.ndarray,
    tile_px: int = 224,
    threshold: int = 220,
    tissue_percent: float = 0.05,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[int]]:
    """
    Simple tissue filtering by brightness threshold.
    Keeps tiles having at least a fraction of non-white pixels (per channel).
    """
    slide = openslide.OpenSlide(slide_path)
    keep_idxs: List[int] = []
    for i, tl in enumerate(top_lefts):
        img = slide.read_region(tl, 0, (tile_px, tile_px)).convert("RGB")
        arr = np.array(img)
        tissue_mask = (arr[..., :3] < threshold).any(axis=-1)
        if tissue_mask.mean() > tissue_percent:
            keep_idxs.append(i)
    slide.close()
    centers_filt = np.array(centers)[keep_idxs]
    top_lefts_filt = [top_lefts[i] for i in keep_idxs]
    return centers_filt, top_lefts_filt, keep_idxs


def validate_coordinates(centers: np.ndarray, slide_dims: Tuple[int, int], tile_px: int) -> bool:
    """Validate tile centers lie within slide bounds (with tile margins)."""
    W, H = slide_dims
    centers = np.asarray(centers)
    x_valid = (centers[:, 0] >= tile_px // 2) & (centers[:, 0] < W - tile_px // 2)
    y_valid = (centers[:, 1] >= tile_px // 2) & (centers[:, 1] < H - tile_px // 2)
    if not (x_valid & y_valid).all():
        bad_coords = centers[~(x_valid & y_valid)]
        raise ValueError(f"Found {len(bad_coords)} tile centers outside slide bounds. First examples: {bad_coords[:5]}")
    return True


def check_thumbnail_mapping(
    slide_dims: Tuple[int, int],
    thumb_dims: Tuple[int, int],
    centers: np.ndarray,
    mapped_coords: np.ndarray,
) -> bool:
    """Basic sanity checks that mapped coordinates fit within thumbnail bounds."""
    tw, th = thumb_dims
    if not ((mapped_coords[:, 0] >= 0) & (mapped_coords[:, 0] < tw)).all():
        raise ValueError(f"Some X coords are outside thumbnail bounds [0, {tw})")
    if not ((mapped_coords[:, 1] >= 0) & (mapped_coords[:, 1] < th)).all():
        raise ValueError(f"Some Y coords are outside thumbnail bounds [0, {th})")
    return True


def debug_coordinate_mapping(slide_path: str, centers: np.ndarray, mapped_coords: np.ndarray, out_debug_png: str) -> None:
    """Save a diagnostic visualization comparing original and mapped coordinates on a slide thumbnail."""
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.dimensions
    thumb = slide.get_thumbnail((1024, 1024)).convert("RGB")
    tw, th = thumb.size

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(thumb)
    ax1.scatter(centers[:, 0] * tw / W, centers[:, 1] * th / H, c="r", alpha=0.5, s=1)
    ax1.set_title("Original Centers")
    ax1.invert_yaxis()

    ax2.imshow(thumb)
    ax2.scatter(mapped_coords[:, 0], mapped_coords[:, 1], c="b", alpha=0.5, s=1)
    ax2.set_title("Mapped Centers")
    ax2.invert_yaxis()

    plt.savefig(out_debug_png, dpi=160, bbox_inches="tight")
    plt.close()
    slide.close()


# -------------------- Scoring and heatmaps --------------------
def compute_tile_attention_and_scores_nochunk(
    model: nn.Module, features_np: np.ndarray, pos_class_idx: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Score all tiles at once (no chunking) using a binary AttentionMIL classifier.
    Returns:
        logits_np: [N, C]
        att_vec: [N]  (softmax-normalized over tiles)
        p_tile: [N]   (probability of positive class)
        att_weighted: [N] (attention * p_tile), all normalized to [0,1]
    """
    device = DEFAULT_DEVICE
    model = model.to(device).eval()
    X = torch.from_numpy(np.asarray(features_np, dtype=np.float32)).to(device)
    with torch.no_grad():
        H = model.feature_extractor(X)               # (N, hidden_dim)
        logits_tile = model.classifier(H)            # (N, C)
        att_raw = model.attention(H).squeeze(-1)     # (N,)
        att_exp = torch.exp(att_raw - torch.max(att_raw))  # stable softmax across tiles
        att = att_exp / (att_exp.sum() + 1e-12)
        probs = F.softmax(logits_tile, dim=1)        # (N, C)

    logits_np = logits_tile.cpu().numpy()
    att_vec = att.cpu().numpy().astype(np.float32)
    p_tile = probs[:, pos_class_idx].cpu().numpy().astype(np.float32)
    att_weighted = (att_vec * p_tile).astype(np.float32)

    def norm01(x: np.ndarray) -> np.ndarray:
        mn, mx = np.nanmin(x), np.nanmax(x)
        if np.isnan(mn) or np.isnan(mx) or mx <= mn:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - mn) / (mx - mn)).astype(np.float32)

    return logits_np, norm01(att_vec), norm01(p_tile), norm01(att_weighted)


def render_heatmap_on_thumbnail(
    slide_path: str,
    centers: np.ndarray,
    values: np.ndarray,
    out_png: str,
    classifier_name: Optional[str] = None,
    thumb_max: int = DEFAULT_THUMB_MAX_SIZE,
    sigma: int = DEFAULT_GAUSSIAN_SIGMA_PX,
    cmap: str = "inferno",
    value_type: str = "attention",
) -> None:
    """
    Render a heatmap (attention/probability/weighted) over a slide thumbnail.

    Args:
        values: Per-tile scalar values in [0,1] (will be lightly contrast-enhanced).
        value_type: 'attention', 'probability', or a custom label to annotate the colorbar.
    """
    slide = openslide.OpenSlide(slide_path)
    W, H = slide.dimensions
    validate_coordinates(centers, (W, H), DEFAULT_TILE_PX)

    thumb = slide.get_thumbnail((1024, 1024)).convert("RGB")
    tw, th = thumb.size
    coords = np.array([(int(cx * tw / W), int(cy * th / H)) for cx, cy in centers], dtype=int)
    check_thumbnail_mapping((W, H), (tw, th), centers, coords)

    # Gentle contrast enhancement (square-root)
    val_enh = np.power(values, 0.5)

    heat = np.zeros((th, tw), dtype=np.float32)
    for (x, y), v in zip(coords, val_enh):
        if 0 <= x < tw and 0 <= y < th:
            heat[y, x] += float(v)

    try:
        from scipy.ndimage import gaussian_filter
        heat = gaussian_filter(heat, sigma=max(1, sigma // 2))
        hm = heat - heat.min() if heat.max() != heat.min() else heat
        if hm.max() != 0:
            hm = hm / hm.max()
        hm_stretched = np.power(hm, 0.7)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(thumb)
        ax1.set_title("Original Tissue", fontsize=14, fontweight="bold")
        ax1.axis("off")
        ax1.invert_yaxis()

        ax2.imshow(thumb)
        im = ax2.imshow(hm_stretched, cmap=cmap, alpha=0.8, vmin=0.1, vmax=1.0)
        map_type = "Attention" if "attention" in value_type.lower() else "Probability"
        title = f"{classifier_name} Classifier {map_type} Map" if classifier_name else f"{map_type} Map"
        ax2.set_title(title, fontsize=14, fontweight="bold")
        ax2.axis("off")
        ax2.invert_yaxis()

        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        if value_type == "probability":
            cbar.set_label("Probability (0=Low, 1=High)", rotation=270, labelpad=20, fontsize=12)
        elif value_type == "attention":
            cbar.set_label("Attention (0=Ignored, 1=Important)", rotation=270, labelpad=20, fontsize=12)
        else:
            cbar.set_label(f"{value_type.title()} (0=Low, 1=High)", rotation=270, labelpad=20, fontsize=12)

        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1, dpi=150)
        plt.close(fig)
    except Exception:
        # Fallback to scatter overlay if scipy not available
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(thumb)
        ax1.set_title("Original Tissue", fontsize=14, fontweight="bold")
        ax1.axis("off")
        ax1.invert_yaxis()

        ax2.imshow(thumb)
        xs, ys = coords[:, 0], coords[:, 1]
        val_enh = np.power(values, 0.5)
        sc = ax2.scatter(
            xs, ys, c=val_enh, cmap=cmap,
            s=(DEFAULT_TILE_PX * tw / W) ** 2 / 8, alpha=0.8,
            vmin=0.1, vmax=1.0, edgecolors="white", linewidths=0.5,
        )
        map_type = "Attention" if "attention" in value_type.lower() else "Probability"
        title = f"{classifier_name} Classifier {map_type} Map" if classifier_name else f"{map_type} Map"
        ax2.set_title(title, fontsize=14, fontweight="bold")
        ax2.axis("off")
        ax2.invert_yaxis()

        cbar = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label(f"{map_type} (0=Low, 1=High)", rotation=270, labelpad=20, fontsize=12)

        plt.tight_layout()
        fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1, dpi=150)
        plt.close(fig)
    finally:
        slide.close()


# --------------------- Maps persistence ---------------------
def _save_npz_and_parquet(
    outdir: str,
    name: str,
    centers: np.ndarray,
    att_vec: np.ndarray,
    p_tile: np.ndarray,
    att_weighted: np.ndarray,
    logits: Optional[np.ndarray] = None,
    meta: Optional[Dict] = None,
) -> str:
    """
    Save maps to a compressed NPZ and a parquet file with per-tile rows.
    Returns path to NPZ file.
    """
    os.makedirs(outdir, exist_ok=True)
    npz_path = os.path.join(outdir, f"{name}_maps.npz")
    save_dict: Dict[str, np.ndarray] = {
        "loc_x": np.asarray(centers)[:, 0].astype(np.int32),
        "loc_y": np.asarray(centers)[:, 1].astype(np.int32),
        "attention": np.asarray(att_vec).astype(np.float32),
        "p_tile": np.asarray(p_tile).astype(np.float32),
        "att_weighted": np.asarray(att_weighted).astype(np.float32),
    }
    if logits is not None:
        save_dict["logits"] = np.asarray(logits)

    if meta is not None:
        # store JSON string to preserve metadata
        save_dict["meta"] = np.array([json.dumps(meta)], dtype=object)

    np.savez_compressed(npz_path, **save_dict)

    # Per-tile table
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
    try:
        df.to_parquet(parquet_path)
    except Exception:
        csv_path = os.path.join(outdir, f"{name}_tiles.csv")
        df.to_csv(csv_path, index=False)
        LOGGER.warning(f"Parquet not available, wrote CSV: {csv_path}")

    return npz_path


def _save_map_image(values: np.ndarray, centers: np.ndarray, out_png: str, grid_w: Optional[int] = None, cmap: str = "inferno") -> None:
    """Save a simple grid visualization (not spatially accurate; for quick inspection)."""
    vals = np.asarray(values).astype(np.float32)
    n = vals.shape[0]
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
    pad = grid_w * grid_h - n
    grid = np.pad(vals, (0, pad), constant_values=0.0).reshape((grid_h, grid_w))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, interpolation="nearest")
    ax.axis("off")
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_all_maps(
    outdir: str,
    name: str,
    centers: np.ndarray,
    att_vec: np.ndarray,
    p_tile: np.ndarray,
    att_weighted: np.ndarray,
    logits: Optional[np.ndarray] = None,
    meta: Optional[Dict] = None,
) -> str:
    """Convenience wrapper to persist NPZ/parquet and small preview images."""
    npz_path = _save_npz_and_parquet(outdir, name, centers, att_vec, p_tile, att_weighted, logits=logits, meta=meta)
    _save_map_image(att_vec, centers, os.path.join(outdir, f"{name}_attention.png"), cmap="magma")
    _save_map_image(p_tile, centers, os.path.join(outdir, f"{name}_p_tile.png"), cmap="viridis")
    _save_map_image(att_weighted, centers, os.path.join(outdir, f"{name}_att_weighted.png"), cmap="inferno")
    return npz_path


# --------------------- OvR inference helpers ---------------------
def compute_all_ovr_predictions(models: Dict[str, Optional[nn.Module]], features_np: np.ndarray) -> Dict[str, Dict]:
    """
    Compute per-tile scores for all OvR classifiers and return:
      - 'logits', 'attention', 'p_tile', 'att_weighted', 'slide_score' per subtype
      - 'final_prediction' entry with best subtype and scores
    """
    results: Dict[str, Dict] = {}
    ovr_scores: List[float] = []

    for subtype, model in models.items():
        if model is None:
            LOGGER.warning(f"Skipping {subtype} (model not loaded)")
            continue
        LOGGER.info(f"Scoring tiles for subtype: {subtype}")
        logits, att_vec, p_tile, att_weighted = compute_tile_attention_and_scores_nochunk(model, features_np, pos_class_idx=1)
        slide_score = float(np.mean(p_tile)) if len(p_tile) else 0.0
        results[subtype] = {
            "logits": logits,
            "attention": att_vec,
            "p_tile": p_tile,
            "att_weighted": att_weighted,
            "slide_score": slide_score,
        }
        ovr_scores.append(slide_score)
        LOGGER.info(f"{subtype} — slide score (mean P): {slide_score:.4f}; high prob (>0.5): {(p_tile > 0.5).sum()}/{len(p_tile)}")

    if ovr_scores:
        subtypes = list(results.keys())
        best_idx = int(np.argmax([results[s]["slide_score"] for s in subtypes]))
        final_prediction = subtypes[best_idx]
        results["final_prediction"] = {
            "subtype": final_prediction,
            "confidence": results[final_prediction]["slide_score"],
            "all_scores": {k: float(v["slide_score"]) for k, v in results.items() if k != "final_prediction"},
        }
        LOGGER.info(f"FINAL PREDICTION: {final_prediction} (score={results['final_prediction']['confidence']:.4f})")
    return results


def render_all_heatmaps(
    slide_path: str,
    centers: np.ndarray,
    results: Dict[str, Dict],
    slide_name: str,
    out_dir: str,
) -> Dict[str, Dict[str, str]]:
    """
    Render attention, probability, and high-contrast probability heatmaps for each subtype.
    Returns dict subtype -> paths of created images.
    """
    heatmap_paths: Dict[str, Dict[str, str]] = {}
    for subtype, result_data in results.items():
        if subtype == "final_prediction":
            continue
        LOGGER.info(f"Rendering heatmaps for {subtype}")
        out_png_att = os.path.join(out_dir, f"{slide_name}_{subtype}_heatmap_attention.png")
        render_heatmap_on_thumbnail(slide_path, centers, result_data["attention"], out_png_att, classifier_name=subtype, value_type="attention")

        out_png_prob = os.path.join(out_dir, f"{slide_name}_{subtype}_heatmap_probability.png")
        render_heatmap_on_thumbnail(slide_path, centers, result_data["p_tile"], out_png_prob, classifier_name=subtype, value_type="probability")

        threshold = float(np.percentile(result_data["p_tile"], 80)) if len(result_data["p_tile"]) else 1.0
        p_tile_threshold = np.where(result_data["p_tile"] >= threshold, result_data["p_tile"], 0)
        out_png_contrast = os.path.join(out_dir, f"{slide_name}_{subtype}_heatmap_prob_highcontrast.png")
        render_heatmap_on_thumbnail(slide_path, centers, p_tile_threshold, out_png_contrast, classifier_name=subtype, value_type="probability (top 20%)")

        heatmap_paths[subtype] = {
            "attention": out_png_att,
            "probability": out_png_prob,
            "high_contrast": out_png_contrast,
        }
    return heatmap_paths


def save_all_results(out_dir: str, slide_name: str, centers: np.ndarray, results: Dict[str, Dict]) -> Tuple[str, str]:
    """
    Save a parquet with per-tile metrics across subtypes and a JSON summary with slide-level scores.
    """
    # Per-tile table across all subtypes
    df_data: Dict[str, np.ndarray] = {
        "loc_x": centers[:, 0],
        "loc_y": centers[:, 1],
    }
    for subtype, result_data in results.items():
        if subtype == "final_prediction":
            continue
        df_data[f"{subtype}_attention"] = result_data["attention"]
        df_data[f"{subtype}_probability"] = result_data["p_tile"]
        df_data[f"{subtype}_att_weighted"] = result_data["att_weighted"]
        if np.ndim(result_data["logits"]) == 2:
            for i in range(result_data["logits"].shape[1]):
                df_data[f"{subtype}_logit_{i}"] = result_data["logits"][:, i]

    df = pd.DataFrame(df_data)
    parquet_path = os.path.join(out_dir, f"{slide_name}_all_subtypes.parquet")
    df.to_parquet(parquet_path)

    # Slide-level summary
    summary = {
        "slide_name": slide_name,
        "final_prediction": results.get("final_prediction", {}),
        "subtype_scores": {k: float(v.get("slide_score", 0.0)) for k, v in results.items() if k != "final_prediction"},
        "tile_counts": {k: int(len(v.get("p_tile", []))) for k, v in results.items() if k != "final_prediction"},
    }
    summary_path = os.path.join(out_dir, f"{slide_name}_prediction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info(f"Saved tile table: {parquet_path}")
    LOGGER.info(f"Saved slide summary: {summary_path}")
    return parquet_path, summary_path


# ------------------------ Diagnostics ------------------------
def print_attention_stats(att_vec: np.ndarray) -> None:
    """Print distribution stats for attention weights."""
    LOGGER.info(f"Attention stats: min={att_vec.min():.4f}, max={att_vec.max():.4f}, mean={att_vec.mean():.4f}, std={att_vec.std():.4f}")
    LOGGER.info(f"Tiles with attention > 0.5: {(att_vec > 0.5).sum()} / {len(att_vec)}")


def print_top_attention_tiles(att_vec: np.ndarray, p_tile: np.ndarray, centers: np.ndarray, N: int = 10) -> None:
    """Log top-N tiles by attention weight."""
    top_idxs = np.argsort(att_vec)[-N:][::-1]
    LOGGER.info("Top attention tiles:")
    for i in top_idxs:
        LOGGER.info(f"Tile {i}: Center={centers[i]}, Attention={att_vec[i]:.4f}, Probability={p_tile[i]:.4f}")


def plot_attention_histogram(att_vec: np.ndarray, slide_name: str, out_dir: str) -> None:
    """Save histogram of attention weights."""
    plt.figure(figsize=(8, 4))
    plt.hist(att_vec, bins=50, color="purple", alpha=0.7)
    plt.title(f"Attention Weight Distribution: {slide_name}")
    plt.xlabel("Attention Weight")
    plt.ylabel("Tiles")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{slide_name}_attention_histogram.png")
    plt.savefig(out_png, dpi=160)
    plt.close()
    LOGGER.info(f"Saved attention histogram: {out_png}")


def plot_attention_vs_probability(att_vec: np.ndarray, p_tile: np.ndarray, slide_name: str, out_dir: str) -> None:
    """Save scatter of attention vs. probability."""
    plt.figure(figsize=(6, 6))
    plt.scatter(att_vec, p_tile, alpha=0.5, color="teal")
    plt.xlabel("Attention Weight")
    plt.ylabel("Tile Probability")
    plt.title(f"Attention vs Probability: {slide_name}")
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"{slide_name}_attention_vs_probability.png")
    plt.savefig(out_png, dpi=160)
    plt.close()
    LOGGER.info(f"Saved attention vs probability scatter: {out_png}")


# ------------------------ Random test ------------------------
def random_test_and_generate_maps(
    n_tiles: int,
    feature_dim: int,
    checkpoint_path: Optional[str],
    outdir: str,
    seed: int = 0,
) -> bool:
    """
    Quick sanity test: random features -> AttentionMIL -> save maps (npz + previews).
    Intended to validate numeric stability and outputs without loading a slide.
    """
    rng = np.random.RandomState(seed)
    features = rng.randn(n_tiles, feature_dim).astype(np.float32)
    grid_w = int(math.ceil(math.sqrt(n_tiles)))
    centers = np.array([(i % grid_w, i // grid_w) for i in range(n_tiles)], dtype=int)

    mil = AttentionMIL(input_dim=feature_dim, hidden_dim=256, n_classes=2, dropout_rate=0.0)
    if checkpoint_path:
        try:
            load_checkpoint_to_model(mil, checkpoint_path)
        except Exception as e:
            LOGGER.warning(f"Random test: failed to load checkpoint, proceeding with random weights: {e}")

    logits, att_vec, p_tile, att_weighted = compute_tile_attention_and_scores_nochunk(mil, features, pos_class_idx=1)
    meta = {"test": "random", "n_tiles": n_tiles}
    npz_path = save_all_maps(outdir, "random_test", centers, att_vec, p_tile, att_weighted, logits=logits, meta=meta)
    LOGGER.info(f"Random test saved to: {npz_path}")
    return True


# ---------------------------- CLI/Main ----------------------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="OvR MIL heatmaps over WSI with UNI2 features")
    ap.add_argument("--slide_path", type=str, default=DEFAULT_SLIDE_PATH, help="Path to WSI (e.g., .svs)")
    ap.add_argument("--true_label", type=str, default=DEFAULT_TRUE_LABEL, help="Optional ground-truth label (for logging only)")
    ap.add_argument("--output_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for artifacts")
    ap.add_argument("--checkpoint_paths_json", type=str, default=None, help="JSON mapping {subtype: ckpt_path}")
    ap.add_argument("--single_checkpoint_path", type=str, default=None, help="Single checkpoint for quick test run")
    ap.add_argument("--feature_dim", type=int, default=DEFAULT_FEATURE_DIM, help="Feature dimension (UNI2-h=1536)")
    ap.add_argument("--tile_px", type=int, default=DEFAULT_TILE_PX, help="Tile size (px)")
    ap.add_argument("--stride_px", type=int, default=DEFAULT_STRIDE_PX, help="Stride (px)")
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for feature extraction")
    ap.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="'cuda' or 'cpu'")
    ap.add_argument("--thumb_max", type=int, default=DEFAULT_THUMB_MAX_SIZE, help="Max thumbnail size (px)")
    ap.add_argument("--gaussian_sigma", type=int, default=DEFAULT_GAUSSIAN_SIGMA_PX, help="Gaussian sigma for heatmap blur")
    ap.add_argument("--save_features", action="store_true", default=DEFAULT_SAVE_FEATURES, help="Persist extracted features")
    ap.add_argument("--skip_random_test", action="store_true", help="Skip the initial random-features sanity test")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--tissue_threshold", type=int, default=220, help="Background threshold (0-255)")
    ap.add_argument("--tissue_min_percent", type=float, default=0.05, help="Minimum tissue fraction to keep tile")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    set_seed(args.seed)

    global DEFAULT_DEVICE
    DEFAULT_DEVICE = args.device

    # Resolve checkpoints dict
    if args.checkpoint_paths_json and os.path.isfile(args.checkpoint_paths_json):
        with open(args.checkpoint_paths_json, "r") as f:
            checkpoint_paths = json.load(f)
        assert isinstance(checkpoint_paths, dict), "checkpoint_paths_json must map subtype -> path"
    else:
        checkpoint_paths = DEFAULT_CHECKPOINT_PATHS

    # Progress file (global then slide-specific)
    global_progress_path = os.path.join(args.output_dir, "progress.json")
    _write_progress(global_progress_path, stage="start", current=0, total=1, extra={"msg": "starting run"})

    # Optional random test
    if not args.skip_random_test:
        _write_progress(global_progress_path, stage="random_test", current=0, total=1)
        try:
            random_test_and_generate_maps(
                n_tiles=256,
                feature_dim=args.feature_dim,
                checkpoint_path=args.single_checkpoint_path or None,
                outdir=args.output_dir,
                seed=args.seed,
            )
            _write_progress(global_progress_path, stage="random_test_done", current=1, total=1)
        except Exception as e:
            LOGGER.warning(f"Random test failed: {e}")
            _write_progress(global_progress_path, stage="random_test_failed", current=1, total=1, extra={"error": str(e)})

    # Slide setup
    if not os.path.isfile(args.slide_path):
        raise FileNotFoundError(f"Slide not found: {args.slide_path}")

    slide_name = os.path.splitext(os.path.basename(args.slide_path))[0]
    slide_progress_path = os.path.join(args.output_dir, f"{slide_name}.progress.json")
    _write_progress(slide_progress_path, stage="init", current=0, total=1, extra={"slide": slide_name})

    features_npy = os.path.join(args.output_dir, f"{slide_name}_features.npy")
    features_pt = os.path.join(args.output_dir, f"{slide_name}_features.pt")
    bag_pt = os.path.join(args.output_dir, f"{slide_name}.pt")
    idx_npz = os.path.join(args.output_dir, f"{slide_name}.index.npz")

    # Compute tiling and filter tissue
    LOGGER.info("Computing tile grid and filtering tissue tiles...")
    all_centers, all_top_lefts = get_tile_centers(args.slide_path, args.tile_px, args.stride_px)
    centers, top_lefts, keep_idxs = filter_tissue_tiles(
        args.slide_path,
        all_top_lefts,
        all_centers,
        tile_px=args.tile_px,
        threshold=args.tissue_threshold,
        tissue_percent=args.tissue_min_percent,
    )
    n_tiles = len(centers)
    LOGGER.info(f"Kept {n_tiles} tissue tiles after filtering")

    # Features: load cached if present, else extract
    if os.path.exists(features_npy):
        LOGGER.info(f"Loading cached features: {features_npy}")
        _write_progress(slide_progress_path, stage="load_cached_features", current=0, total=1)
        features = np.asarray(np.load(features_npy), dtype=np.float32)
        _write_progress(slide_progress_path, stage="load_cached_features_done", current=1, total=1, extra={"n_tiles": int(features.shape[0])})
    elif os.path.exists(features_pt):
        LOGGER.info(f"Loading cached features: {features_pt}")
        _write_progress(slide_progress_path, stage="load_cached_features", current=0, total=1)
        tmp = torch.load(features_pt, map_location="cpu")
        features = tmp.cpu().numpy().astype(np.float32) if isinstance(tmp, torch.Tensor) else np.asarray(tmp, dtype=np.float32)
        _write_progress(slide_progress_path, stage="load_cached_features_done", current=1, total=1, extra={"n_tiles": int(features.shape[0])})
    else:
        # Extract with UNI2
        extractor, transform, uni2_model = build_uni2_extractor(device=args.device, target_dim=args.feature_dim)
        _write_progress(slide_progress_path, stage="extract_features", current=0, total=int(n_tiles))

        dataset = TileDataset(args.slide_path, top_lefts, args.tile_px, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=(args.device == "cuda"),
            prefetch_factor=2 if args.device == "cuda" else 2,
        )

        features_parts: List[np.ndarray] = []
        processed = 0
        start_t = time.time()
        pbar = tqdm(total=n_tiles, desc="Extracting features", unit="tile") if DEFAULT_USE_TQDM and _TQDM_AVAILABLE else None

        torch.backends.cudnn.benchmark = True
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(args.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(args.device == "cuda")):
                    feats = uni2_model(batch)
                feats = feats.float().cpu().numpy()
                features_parts.append(feats)
                processed += feats.shape[0]
                if pbar is not None:
                    pbar.update(feats.shape[0])
                else:
                    if processed % max(1, DEFAULT_LOG_EVERY_TILES) == 0 or processed == n_tiles:
                        elapsed = time.time() - start_t
                        rate = processed / max(1e-9, elapsed)
                        remaining = n_tiles - processed
                        eta_sec = remaining / max(1e-9, rate)
                        LOGGER.info(f"Processed {processed}/{n_tiles} tiles | {rate:.2f} tiles/s | ETA ~{int(eta_sec)} s")
                _write_progress(
                    slide_progress_path,
                    stage="extract_features",
                    current=int(processed),
                    total=int(n_tiles),
                    extra={"batch": int(feats.shape[0]), "elapsed_sec": round(time.time() - start_t, 2)},
                )
        if pbar is not None:
            pbar.close()

        features = np.concatenate(features_parts, axis=0).astype(np.float32)
        LOGGER.info(f"Features shape: {features.shape}")
        _write_progress(slide_progress_path, stage="extract_features_done", current=int(processed), total=int(n_tiles), extra={"n_tiles": int(features.shape[0])})

        if args.save_features:
            try:
                np.save(features_npy, features)
                torch.save(torch.from_numpy(features), features_pt)
                LOGGER.info(f"Saved features to {features_npy} and {features_pt}")
                _write_progress(slide_progress_path, stage="features_saved", current=1, total=1)
            except Exception as e:
                LOGGER.warning(f"Failed saving features: {e}")
                _write_progress(slide_progress_path, stage="features_save_failed", current=1, total=1, extra={"error": str(e)})

    # Save bag & index for compatibility
    torch.save(torch.from_numpy(features.astype(np.float32)), bag_pt)
    np.savez(idx_npz, centers.astype(np.int32))
    LOGGER.info(f"Saved bag and index: {bag_pt} | {idx_npz}")
    _write_progress(slide_progress_path, stage="bag_and_index_saved", current=1, total=1)

    # Score with a single classifier (optional quick look)
    if args.single_checkpoint_path:
        mil = AttentionMIL(input_dim=args.feature_dim, hidden_dim=256, n_classes=2, dropout_rate=0.0)
        load_checkpoint_to_model(mil, args.single_checkpoint_path)
        _write_progress(slide_progress_path, stage="scoring_single", current=0, total=int(features.shape[0]))
        logits, att_vec, p_tile, att_weighted = compute_tile_attention_and_scores_nochunk(mil, features, pos_class_idx=1)
        LOGGER.info(
            f"Attention range: [{att_vec.min():.4f}, {att_vec.max():.4f}]  "
            f"P(range): [{p_tile.min():.4f}, {p_tile.max():.4f}]  "
            f"Weighted(range): [{att_weighted.min():.4f}, {att_weighted.max():.4f}]"
        )
        print_attention_stats(att_vec)
        print_top_attention_tiles(att_vec, p_tile, centers, N=10)
        plot_attention_histogram(att_vec, slide_name, args.output_dir)
        plot_attention_vs_probability(att_vec, p_tile, slide_name, args.output_dir)
        _write_progress(slide_progress_path, stage="scoring_single_done", current=int(features.shape[0]), total=int(features.shape[0]))

        # Heatmaps
        out_png_attention = os.path.join(args.output_dir, f"{slide_name}_heatmap_attention.png")
        render_heatmap_on_thumbnail(args.slide_path, centers, att_vec, out_png_attention, value_type="attention")
        out_png_prob = os.path.join(args.output_dir, f"{slide_name}_heatmap_probability.png")
        render_heatmap_on_thumbnail(args.slide_path, centers, p_tile, out_png_prob, value_type="probability")
        out_png_weighted = os.path.join(args.output_dir, f"{slide_name}_heatmap_weighted.png")
        render_heatmap_on_thumbnail(args.slide_path, centers, att_weighted, out_png_weighted, value_type="weighted")
        # High-contrast prob
        out_png_highcontrast = os.path.join(args.output_dir, f"{slide_name}_heatmap_prob_highcontrast.png")
        threshold = float(np.percentile(p_tile, 80)) if len(p_tile) else 1.0
        p_tile_threshold = np.where(p_tile >= threshold, p_tile, 0)
        render_heatmap_on_thumbnail(args.slide_path, centers, p_tile_threshold, out_png_highcontrast, value_type="probability (top 20%)")
        _write_progress(
            slide_progress_path,
            stage="render_single_done",
            current=1,
            total=1,
            extra={"heatmaps": [out_png_attention, out_png_prob, out_png_weighted, out_png_highcontrast]},
        )

        # Parquet dump
        try:
            df = pd.DataFrame({
                "loc_x": centers[:, 0],
                "loc_y": centers[:, 1],
                "attention": att_vec,
                "p_tile": p_tile,
                "att_weighted": att_weighted,
            })
            larr = np.asarray(logits)
            if larr.ndim == 2:
                for i in range(larr.shape[1]):
                    df[f"logit_{i}"] = larr[:, i]
            df.to_parquet(os.path.join(args.output_dir, f"{slide_name}_tiles.parquet"))
            LOGGER.info("Saved single-class tile dataframe.")
            _write_progress(slide_progress_path, stage="parquet_saved", current=1, total=1)
        except Exception:
            LOGGER.warning("pandas/pyarrow not fully available; skipping parquet export.")
            _write_progress(slide_progress_path, stage="parquet_skip", current=1, total=1)

    # Debug visualization for coordinate mapping
    debug_path = os.path.join(args.output_dir, f"{slide_name}_coordinate_debug.png")
    slide = openslide.OpenSlide(args.slide_path)
    W, H = slide.dimensions
    scale_x = min(DEFAULT_THUMB_MAX_SIZE / W, 1.0)
    scale_y = min(DEFAULT_THUMB_MAX_SIZE / H, 1.0)
    slide.close()
    mapped_coords = np.array([(int(cx * scale_x), int(cy * scale_y)) for cx, cy in centers])
    debug_coordinate_mapping(args.slide_path, centers, mapped_coords, debug_path)
    LOGGER.info(f"Saved coordinate debug visualization: {debug_path}")

    # OvR: load all models and compute subtype predictions
    models = load_all_ovr_models(checkpoint_paths=checkpoint_paths, feature_dim=args.feature_dim)
    _write_progress(slide_progress_path, stage="scoring_all_subtypes", current=0, total=len(models))
    all_results = compute_all_ovr_predictions(models, features)
    _write_progress(slide_progress_path, stage="scoring_done", current=len(models), total=len(models))

    # Render heatmaps for all subtypes
    _write_progress(slide_progress_path, stage="render_all_heatmaps", current=0, total=len(models))
    heatmap_paths = render_all_heatmaps(args.slide_path, centers, all_results, slide_name, args.output_dir)
    _write_progress(slide_progress_path, stage="render_done", current=len(models), total=len(models))

    # Save comprehensive results
    parquet_path, summary_path = save_all_results(args.output_dir, slide_name, centers, all_results)

    # Final progress
    final_pred = all_results.get("final_prediction", {})
    _write_progress(
        slide_progress_path,
        stage="done",
        current=1,
        total=1,
        extra={
            "final_prediction": final_pred,
            "heatmaps_generated": len(heatmap_paths),
            "results_saved": [parquet_path, summary_path],
        },
    )
    LOGGER.info("Analysis complete.")
    LOGGER.info(f"Final prediction: {final_pred.get('subtype', 'Unknown')} | confidence: {final_pred.get('confidence', 0.0):.4f}")
    LOGGER.info(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

