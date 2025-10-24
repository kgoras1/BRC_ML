import os
import pandas as pd
import shutil
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
slides_dir = Path("/home/projects2/WSI_project/PhD_WSI/download_v1/data_TCGA_BRCA/common_svs")
tiles_root_dir = Path("/home/projects2/WSI_project/PhD_WSI/download_v1/tiling/svs_tiles")
output_dir = Path("/home/projects2/WSI_project/PhD_WSI/download_v1/tiling/tiles_organised")     # Where renamed tiles will go
metadata_csv = Path("/home/projects2/WSI_project/PhD_WSI/download_v1/data_TCGA_BRCA/metadata.xlsx")
output_metadata_csv = Path("/home/projects2/WSI_project/PhD_WSI/download_v1/tiling/tiles_organised/tile_metadata.csv")
target_magnification = 20.0  # Target magnification level for tiles
max_workers = 8  # Adjust based on system

# === LOAD LABEL METADATA ===
df_meta = pd.read_excel(metadata_csv)
sample_to_label = dict(zip(df_meta['Patient ID'].str.upper(), df_meta['Subtype']))

# === FUNCTION TO PROCESS A SINGLE SLIDE ===
def process_slide(slide_file):
    tile_records = []

    slide_id = slide_file.stem
    parts = slide_id.split("-")
    if len(parts) < 3:
        return tile_records

    sample_id = "-".join(parts[:3]).upper()
    if sample_id not in sample_to_label:
        return tile_records

    label = sample_to_label[sample_id]
    slide_tile_dir = tiles_root_dir / f"{slide_id}_files" / f"{target_magnification:.1f}"
    if not slide_tile_dir.exists():
        return tile_records

    for tile_file in slide_tile_dir.glob("*.jpeg"):
        match = re.match(r"(\d+)_(\d+)\.jpeg", tile_file.name)
        if not match:
            continue
        x, y = match.groups()
        new_filename = f"{slide_id}__x_{x}__y_{y}__mag_{target_magnification:.1f}__label_{label}.jpeg"
        new_tile_path = tile_file.parent / new_filename
        if not new_tile_path.exists():
            tile_file.rename(new_tile_path)
        else:
            new_tile_path = tile_file  # If already renamed, just use the existing path

        tile_records.append({
            "tile_id": new_filename,
            "slide_id": slide_id,
            "sample_id": sample_id,
            "x": int(x),
            "y": int(y),
            "mag": target_magnification,
            "disease": label,
            "tile_path": str(new_tile_path),
            "bag_id": sample_id,
        })

    return tile_records

# === MULTITHREADING EXECUTION ===
all_slide_files = list(slides_dir.glob("*.svs"))
tile_records = []

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_slide, slide): slide for slide in all_slide_files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing slides in parallel"):
        tile_records.extend(future.result())

# === SAVE TILE METADATA ===
df_tiles = pd.DataFrame(tile_records)
df_tiles.to_csv(output_metadata_csv, index=False)
print(f"Saved tile metadata to {output_metadata_csv}")