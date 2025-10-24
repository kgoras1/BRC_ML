import os
import numpy as np
import cupy as cp
from multiprocessing import Process, Queue
from glob import glob
from PIL import Image
from tqdm import tqdm
import pandas as pd

def extract_slide_id(tile_path):
    # Extract slide ID as the first three '-' separated parts
    basename = os.path.basename(tile_path)
    parts = basename.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    return "unknown"

def load_tile(path):
    try:
        img = Image.open(path).convert("RGB").resize((224, 224))
        return np.array(img)
    except:
        return None

def tile_filter_worker(tile_paths, gpu_id, result_queue):
    cp.cuda.Device(gpu_id).use()
    batch_size = 128
    for i in range(0, len(tile_paths), batch_size):
        batch_paths = tile_paths[i:i + batch_size]
        batch_imgs, valid_paths = [], []

        for p in batch_paths:
            img = load_tile(p)
            if img is not None:
                batch_imgs.append(img)
                valid_paths.append(p)

        if not batch_imgs:
            continue

        imgs_gpu = cp.asarray(np.stack(batch_imgs))
        gray_gpu = 0.2989 * imgs_gpu[:, :, :, 0] + 0.5870 * imgs_gpu[:, :, :, 1] + 0.1140 * imgs_gpu[:, :, :, 2]
        bright_pixels = (gray_gpu > 230).sum(axis=(1, 2)) / (224 * 224)
        background_mask = bright_pixels > 0.8

        std_per_image = cp.std(imgs_gpu.reshape(len(imgs_gpu), -1), axis=1)
        low_variance_mask = std_per_image < 15

        from cupyx.scipy.ndimage import laplace
        blur_variance = cp.var(laplace(gray_gpu), axis=(1, 2))
        blurry_mask = blur_variance < 100

        # Log statistics for every tile
        for idx, path in enumerate(valid_paths):
            slide_id = extract_slide_id(path)
            stats = {
                "slide_id": slide_id,
                "tile_path": path,
                "bright_pixel_ratio": float(bright_pixels[idx].get()),
                "is_background": bool(background_mask[idx].get()),
                "std": float(std_per_image[idx].get()),
                "is_low_variance": bool(low_variance_mask[idx].get()),
                "blur_variance": float(blur_variance[idx].get()),
                "is_blurry": bool(blurry_mask[idx].get())
            }
            result_queue.put(stats)

    result_queue.put("DONE")

def main():
    input_dir = "/home/projects2/WSI_project/PhD_WSI/download_v1/tiling/svs_tiles"
    tile_paths = glob(os.path.join(input_dir, "**/*.jpeg"), recursive=True)

    gpu_0_paths = tile_paths[::2]
    gpu_1_paths = tile_paths[1::2]

    result_queue = Queue()

    p0 = Process(target=tile_filter_worker, args=(gpu_0_paths, 0, result_queue))
    p1 = Process(target=tile_filter_worker, args=(gpu_1_paths, 1, result_queue))

    p0.start()
    p1.start()

    stats_list = []
    done_count = 0
    with tqdm(total=len(tile_paths)) as pbar:
        while done_count < 2:
            item = result_queue.get()
            if item == "DONE":
                done_count += 1
            else:
                stats_list.append(item)
            pbar.update(1)

    p0.join()
    p1.join()

    df = pd.DataFrame(stats_list)
    df.to_csv("tile_statistics_by_slide.csv", index=False)
    print(f"Saved statistics for {len(df)} tiles to tile_statistics_by_slide.csv")

if __name__ == "__main__":
    main()
