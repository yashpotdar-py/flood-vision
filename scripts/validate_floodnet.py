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
import json
import numpy as np
from PIL import Image
from collections import defaultdict

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
        # check if actually identical channels
        # grayscale stored as 3-channel rgb
        if np.all(mask_array[:, :, 0] == mask_array[:, :, 1] and
                  np.all(mask_array[:, :, 1] == mask_array[:, :, 2])):
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
    imgs = [os.path.join(img_dir, f) for f in list_dir_images(img_dir)]
    masks = [os.path.join(mask_dir, f) for f in list_dir_images(mask_dir)]
    report.append(f"  Images found: {len(imgs)}    Masks found: {len(masks)}")
    matches, unmatched_imgs, unmatched_masks = try_match_by_stem(imgs, masks)
    report.append(
        f"  Matched pairs: {len(matches)}    Unmatched images: {len(unmatched_imgs)}    Unmatched masks: {len(unmatched_masks)}")
    # analyze matched pairs
    for img_path, mask_path in matches:
        img = safe_open_image(img_path)
        mask = safe_open_image(mask_path)
        width, height = (None, None)
        mask_info = {'type': 'error', 'unique': []}
        if img is None:
            report.append(f"    ERROR opening image: {img_path}")
            continue
        if mask is None:
            report.append(f"    ERROR opening mask: {mask_path}")
            continue
        width, height = img.size
        if mask.size != (width, height):
            report.append(
                f"    SIZE MISMATCH: {os.path.basename(img_path)}  img={img.size}  mask={mask.size}")
        arr = np.array(mask)
        mtype, uniques = analyze_mask_array(arr)
        mask_info = {'type': mtype, 'unique_count': len(uniques), 'sample_uniques': (
            uniques[:10] if isinstance(uniques, list) else uniques)}
        out_rows.append({
            'image_path': img_path,
            'mask_path': mask_path,
            'width': width,
            'height': height,
            'mask_type': mask_info['type'],
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
    report_lines.append(f"VALIDATION REPORT for data root: {data_root}")
    # try to detect FloodNet-Supervised_v1.0 structure
    fn_root = os.path.join(data_root, 'FloodNet-Supervised_v1.0')
    # also check for color masks folder
    color_root = os.path.join(data_root, 'ColorMasks-FloodNetv1.0')
    report_lines.append("")
    if os.path.isdir(fn_root):
        report_lines.append("Found FloodNet-Supervised_v1.0 structure.")
        for split in ['train', 'val', 'test']:
            report_lines.append(f"\nSplit: {split}")
            img_dir = os.path.join(fn_root, split, f"{split}-org-img")
            mask_dir = os.path.join(fn_root, split, f"{split}-label-img")
            if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
                report_lines.append(
                    f"  WARNING: Expected dirs not found: {img_dir} or {mask_dir}")
                # try fallback to any images under this split
                img_dir2 = os.path.join(fn_root, split)
                # list images directly under split if above not present
                if os.path.isdir(img_dir2):
                    # look for images and attempt label dirs
                    # try to infer mask dir by presence of '-label' directories
                    # fallback: find any subdir containing 'org' or 'label'
                    found_img_dir = None
                    found_mask_dir = None
                    for sub in sorted(os.listdir(img_dir2)):
                        subp = os.path.join(img_dir2, sub)
                        if os.path.isdir(subp):
                            if 'org' in sub.lower():
                                found_img_dir = subp
                            if 'label' in sub.lower() or 'mask' in sub.lower():
                                found_mask_dir = subp
                    if found_img_dir and found_mask_dir:
                        report_lines.append(
                            f"  Using inferred {found_img_dir} and {found_mask_dir}")
                        scan_split(found_img_dir, found_mask_dir,
                                   out_rows, report_lines)
                    else:
                        report_lines.append(
                            f"  Could not auto-find org/mask subfolders under {img_dir2}")
                continue
            else:
                scan_split(img_dir, mask_dir, out_rows, report_lines)
    else:
        report_lines.append(
            "FloodNet-Supervised_v1.0 NOT found under data root.")
    # Check color masks folder if exists
    if os.path.isdir(color_root):
        report_lines.append(
            "\nFound ColorMasks-FloodNetv1.0 folder. Scanning color mask files (train/val/test)...")
        for csplit in sorted(os.listdir(color_root)):
            cdir = os.path.join(color_root, csplit)
            if os.path.isdir(cdir):
                report_lines.append(
                    f"  Color split: {csplit}  files: {len(list_dir_images(cdir))}")
                # sample one file to inspect color palette
                imgs = [os.path.join(cdir, f) for f in list_dir_images(cdir)]
                if imgs:
                    sample = imgs[0]
                    im = safe_open_image(sample)
                    if im is not None:
                        arr = np.array(im)
                        mtype, uniques = analyze_mask_array(arr)
                        report_lines.append(
                            f"    sample {os.path.basename(sample)} -> type={mtype}, uniques_found={len(uniques)}")
    # write CSV and text report
    csv_path = os.path.join(data_root, 'file_mapping.csv')
    txt_path = os.path.join(data_root, 'dataset_report.txt')
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=[
                                'image_path', 'mask_path', 'width', 'height', 'mask_type', 'mask_unique_count'])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    with open(txt_path, 'w', encoding='utf-8') as tf:
        tf.write("\n".join(report_lines))
    print("==== DONE ====")
    print(f"Wrote CSV mapping: {csv_path}")
    print(f"Wrote text report: {txt_path}")
    print("Summary (first 40 lines):\n")
    for line in report_lines[:40]:
        print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data',
                        help='path to your data folder (the one containing FloodNet-Supervised_v1.0)')
    args = parser.parse_args()
    main(args.data_root)
