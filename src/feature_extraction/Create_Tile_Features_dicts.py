import os
import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool

def build_tile_dict(tile_dir, feature_file, output_pickle):
    # Load features
    features = np.load(feature_file)

    # Get tile filenames in sorted order
    tile_fnames = sorted([
        f for f in os.listdir(tile_dir) if f.endswith(".jpeg")
    ])

    if len(tile_fnames) != features.shape[0]:
        print(f"[WARN] Feature count mismatch in {tile_dir}")
        return

    tile_dict = {}

    for fname, feat in zip(tile_fnames, features):
        full_path = os.path.join(tile_dir, fname)

        # Extract label from filename (e.g., ends with __label_BRCA_LumA.jpeg)
        if "__label_" in fname:
            label = fname.split("__label_")[-1].replace(".jpeg", "")
        else:
            label = "unknown"

        tile_dict[fname] = {
            "path": full_path,
            "feature": feat,
            "label": label
        }

    # Save as pickle
    with open(output_pickle, "wb") as f:
        pickle.dump(tile_dict, f)

    print(f"[✓] Saved tile dict to {output_pickle}")

input_root = "/home/projects2/WSI_project/PhD_WSI/download_v1/tiling/svs_tiles"
features_root = "/home/projects2/WSI_project/PhD_WSI/UNI2/features"
dict_output_root = "/home/projects2/WSI_project/PhD_WSI/UNI2/tile_dicts"

os.makedirs(dict_output_root, exist_ok=True)

def process_slide(slide_folder):
    if not slide_folder.endswith("_files"): 
        return
    slide_name = slide_folder.replace("_files", "")
    tile_dir = os.path.join(input_root, slide_folder, "20.0")
    feature_file = os.path.join(features_root, slide_name + ".npy")
    output_pickle = os.path.join(dict_output_root, slide_name + ".pkl")
    if not os.path.exists(tile_dir) or not os.path.exists(feature_file):
        print(f"[SKIP] Missing tile or feature for {slide_name}")
        return
    build_tile_dict(tile_dir, feature_file, output_pickle)

if __name__ == "__main__":
    slide_folders = os.listdir(input_root)
    with Pool() as pool:
        list(tqdm(pool.imap_unordered(process_slide, slide_folders), total=len(slide_folders)))
