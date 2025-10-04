"""
validate_floodnet.py

Scans the FloodNet dataset, verifies image <-> mask pairs, check sizes and
mask encodings

Usage:
    pip install pillow numpy
    python -m scripts.validate_floodnet.py --data-root "data"
"""

import argparse
import csv
import os
import numpy as np
from PIL import Image

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def is_image(fname):
    """
    Check if a file is an image based on its extension.
    Args:
        fname (str): Filename to check.
    Returns:
        bool: True if the file is an image, False otherwise.
    """
    return os.path.splitext(fname)[1].lower() in IMG_EXTS


def list_dir_images(path):
    """
    List all image files in a directory.
    Args:
        path (str): Directory path.
    Returns:
        files (list): Sorted list of image filenames.
    """
    files = []
    if not os.path.isdir(path):
        return files
    files = sorted([f for f in os.listdir(path) if is_image(f)])
    return files


def safe_open_image(path):
    """
    Safely open an image file.
    Args:
        path (str): Path to the image file.
    Returns:
        Image or None: PIL Image object if successful, None otherwise.
    """
    try:
        im = Image.open(path)
        im.load()
        return im
    except Exception as e:
        print(f"    ERROR opening {path}: {e}", flush=True)
        return None


def analyze_mask_array(mask_array):
    """
    Analyze a mask array to determine its type and unique values/colors.
    Args:
        mask_array (np.ndarray): Numpy array of the mask image.
    Returns:
        (str, list): Tuple of mask type and list of unique values/colors.
    """
    if mask_array.ndim == 2:
        uniques = np.unique(mask_array)
        return 'indexed', uniques.tolist()
    elif mask_array.ndim == 3 and mask_array.shape[2] == 3:
        if np.all(mask_array[:, :, 0] == mask_array[:, :, 1]) and \
           np.all(mask_array[:, :, 1] == mask_array[:, :, 2]):
            uniques = np.unique(mask_array[:, :, 0])
            return 'indexed_3ch', uniques.tolist()
        else:
            # color mask
            flat = mask_array.reshape(-1, 3)
            # find unique colors but cap to resonable number
            uniq_colors = np.unique(flat, axis=0)
            colors = [tuple(map(int, c)) for c in uniq_colors]
            return 'color', colors
    else:
        return 'unknown', []


def try_match_by_stem(imgs, masks):
    """
    Attempt to match by filename stem. Returns list of (img_path, mask_path) plus unmatched lists.
    Args:
        imgs (list): List of image file paths.
        masks (list): List of mask file paths.
    Returns:
        (list, list, list): Tuple of matched pairs, unmatched images, unmatched masks.
    """
    img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in imgs}
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}

    matches = []
    unmatched_imgs = []
    for stem, img_path in img_map.items():
        if stem in mask_map:
            matches.append((img_path, mask_map[stem]))
        else:
            # try partial matches (one contains the other)
            found = None
            for mstem, mpath in mask_map.items():
                if stem == mstem or stem in mstem or mstem in stem:
                    found = mpath
                    break
            if found:
                matches.append((img_path, found))
            else:
                unmatched_imgs.append(img_path)

    matched_masks = set(m for _, m in matches)
    unmatched_masks = [mp for mp in masks if mp not in matched_masks]
    return matches, unmatched_imgs, unmatched_masks


def scan_split(img_dir, mask_dir, out_rows, report):
    print(f"Scanning split: {img_dir} vs {mask_dir}", flush=True)
    imgs = [os.path.join(img_dir, f) for f in list_dir_images(img_dir)]
    masks = [os.path.join(mask_dir, f) for f in list_dir_images(mask_dir)]
    report.append(f"  Images found: {len(imgs)}    Masks found: {len(masks)}")

    matches, unmatched_imgs, unmatched_masks = try_match_by_stem(imgs, masks)
    report.append(
        f"  Matched pairs: {len(matches)}    Unmatched images: {len(unmatched_imgs)}    Unmatched masks: {len(unmatched_masks)}")

    for i, (img_path, mask_path) in enumerate(matches):
        print(
            f"  [{i+1}/{len(matches)}] Checking {os.path.basename(img_path)}", flush=True)
        img = safe_open_image(img_path)
        mask = safe_open_image(mask_path)
        if img is None or mask is None:
            continue
        width, height = img.size
        if mask.size != (width, height):
            report.append(
                f"    SIZE MISMATCH: {os.path.basename(img_path)}  img={img.size}  mask={mask.size}")
        arr = np.array(mask)
        mtype, uniques = analyze_mask_array(arr)
        out_rows.append({
            'image_path': img_path,
            'mask_path': mask_path,
            'width': width,
            'height': height,
            'mask_type': mtype,
            'mask_unique_count': len(uniques)
        })
    # report some unmatched samples
    if unmatched_imgs:
        report.append("  Sample unmatched images (first 5):")
        for p in unmatched_imgs[:5]:
            report.append(f"    {os.path.basename(p)}")
    if unmatched_masks:
        report.append("  Sample unmatched masks (first 5):")
        for p in unmatched_masks[:5]:
            report.append(f"    {os.path.basename(p)}")
    return matches, unmatched_imgs, unmatched_masks


def main(data_root):
    report_lines = []
    out_rows = []
    print(f"Starting validation on: {data_root}", flush=True)

    fn_root = os.path.join(data_root, 'FloodNet-Supervised_v1.0')
    color_root = os.path.join(data_root, 'ColorMasks-FloodNetv1.0')

    if os.path.isdir(fn_root):
        print("Found FloodNet-Supervised_v1.0 structure.", flush=True)
        for split in ['train', 'val', 'test']:
            print(f"\nProcessing split: {split}", flush=True)
            img_dir = os.path.join(fn_root, split, f"{split}-org-img")
            mask_dir = os.path.join(fn_root, split, f"{split}-label-img")
            if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
                scan_split(img_dir, mask_dir, out_rows, report_lines)
            else:
                print(
                    f"WARNING: Missing expected dirs for split {split}", flush=True)
    else:
        print("FloodNet-Supervised_v1.0 NOT found under data root.", flush=True)

    csv_path = os.path.join(data_root, 'file_mapping.csv')
    txt_path = os.path.join(data_root, 'dataset_report.txt')

    print(f"Writing CSV to {csv_path}", flush=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=[
            'image_path', 'mask_path', 'width', 'height', 'mask_type', 'mask_unique_count'])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Writing report to {txt_path}", flush=True)
    with open(txt_path, 'w', encoding='utf-8') as tf:
        tf.write("\n".join(report_lines))

    print("==== DONE ====", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data',
                        help='path to your data folder')
    args = parser.parse_args()
    main(args.data_root)
