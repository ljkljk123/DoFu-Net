#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate FINAL MID visual perturbations for AV2S TEST QUERY images only.
HDD-optimized parallel version.

Design goals
------------
- Optimized for MECHANICAL HDD:
  * conservative worker count (default 4)
  * one source image is read once, then all requested conditions are generated together
  * resume-friendly: skips images whose condition files already exist
- Generates perturbations only for TEST QUERY images
- Database images stay clean
- Same perturbation folders can be reused by all methods later

Output
------
<out_root>/
  exposure/<relative_image_path>
  lowlight/<relative_image_path>
  rain/<relative_image_path>
  smoke/<relative_image_path>
  perturb_configs/perturb_manifest.csv
  perturb_configs/perturb_parameters.txt
"""
import argparse
import csv
import hashlib
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

DEFAULT_CONDITIONS = ["exposure", "lowlight", "rain", "smoke"]

PERTURB_CFG_MID = {
    'exposure': {'gain': 1.28, 'gamma': 0.965, 'v_lift': 0.035},
    'lowlight': {'brightness_scale': 0.82, 'gamma': 1.08, 'noise_sigma': 0.002, 'blur_sigma': 0.15},
    'rain': {
        'density': 0.00045, 'min_count': 220, 'angle_deg': -13.0, 'base_len': 16, 'len_jitter': 4,
        'ksize': 17, 'layer_blur': 0.55, 'val_range': (0.76, 1.00), 'thick_prob2': 0.18,
        'darkening': 0.94, 'rain_alpha': 0.16, 'veil_value': 0.79, 'veil_alpha': 0.08,
        'saturation_scale': 0.95, 'contrast_scale': 0.95, 'blur_sigma': 0.40,
    },
    'smoke': {
        'trans_hi': 0.90, 'trans_delta': 0.18, 'trans_clip': (0.68, 0.90),
        'smoke_gray': 0.80, 'contrast_scale': 0.90, 'noise_base': 28,
    },
}


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def check_path(*paths):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)


def check_dir(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def auto_discover_test_splits(cache_root: str, include_all: bool = False) -> List[Dict]:
    pat = re.compile(r'^av2_test_([a-z0-9]+)_db\.npy$')
    city_splits = []
    for fn in sorted(os.listdir(cache_root)):
        m = pat.match(fn)
        if not m:
            continue
        ctag = m.group(1)
        dbp = os.path.join(cache_root, fn)
        qp = os.path.join(cache_root, f'av2_test_{ctag}_query.npy')
        if os.path.exists(qp):
            city_splits.append({
                "name": f"AV2_TEST_{ctag.upper()}",
                "city_tag": ctag.upper(),
                "database_path": dbp,
                "query_path": qp,
            })

    all_db = os.path.join(cache_root, "av2_test_db.npy")
    all_q = os.path.join(cache_root, "av2_test_query.npy")
    all_split = []
    if os.path.exists(all_db) and os.path.exists(all_q):
        all_split.append({
            "name": "AV2_TEST_ALL",
            "city_tag": "ALL",
            "database_path": all_db,
            "query_path": all_q,
        })

    if city_splits:
        return (all_split + city_splits) if include_all else city_splits
    return all_split


def filter_splits_by_tags(splits: List[Dict], city_tags: str) -> List[Dict]:
    if not city_tags:
        return splits
    wanted = {x.strip().upper() for x in city_tags.split(",") if x.strip()}
    if not wanted:
        return splits
    return [s for s in splits if s["city_tag"].upper() in wanted or s["name"].upper() in wanted]


def collect_unique_query_indices(splits: List[Dict]) -> np.ndarray:
    ids = []
    for sp in splits:
        arr = np.load(sp["query_path"])
        if arr.size == 0:
            continue
        ids.extend(arr[:, 0].astype(np.int64).tolist())
    if not ids:
        return np.empty((0,), dtype=np.int64)
    return np.array(sorted(set(ids)), dtype=np.int64)


def load_infos(info_path: str):
    with open(info_path, "rb") as f:
        return pickle.load(f)


def stable_seed(rel_path: str, condition: str) -> int:
    key = f"MID|{condition}|{rel_path}".encode("utf-8")
    h = hashlib.sha1(key).hexdigest()[:8]
    return int(h, 16)


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def perturb_exposure(img_rgb: np.ndarray) -> np.ndarray:
    cfg = PERTURB_CFG_MID['exposure']
    x = to_float(img_rgb)
    x = clip01((x ** cfg['gamma']) * cfg['gain'])
    hsv = cv2.cvtColor(to_uint8(x), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = np.clip(hsv[..., 2] + 255.0 * cfg['v_lift'], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def perturb_lowlight(img_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cfg = PERTURB_CFG_MID['lowlight']
    x = to_float(img_rgb)
    x = clip01((x ** cfg['gamma']) * cfg['brightness_scale'])
    noise = rng.normal(0.0, cfg['noise_sigma'], size=x.shape).astype(np.float32)
    x = clip01(x + noise)
    x = cv2.GaussianBlur(x, (0, 0), cfg['blur_sigma'])
    return to_uint8(x)


def _draw_rain_layer(h: int, w: int, cfg: Dict, rng: np.random.Generator) -> np.ndarray:
    layer = np.zeros((h, w), dtype=np.float32)
    n = max(int(cfg['min_count']), int(h * w * cfg['density']))
    angle = np.deg2rad(cfg['angle_deg'])
    base_len = int(cfg['base_len'])
    jitter = int(cfg['len_jitter'])
    lo_val, hi_val = cfg['val_range']
    for _ in range(n):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        length = int(base_len + rng.integers(-jitter, jitter + 1))
        dx = int(length * np.sin(angle))
        dy = int(length * np.cos(angle))
        x1 = int(np.clip(x0 + dx, 0, w - 1))
        y1 = int(np.clip(y0 + dy, 0, h - 1))
        val = float(rng.uniform(lo_val, hi_val))
        thick = 2 if float(rng.random()) < cfg['thick_prob2'] else 1
        cv2.line(layer, (x0, y0), (x1, y1), color=val, thickness=thick)

    ksize = int(cfg['ksize'])
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    cv2.line(kernel, (ksize - 1, 0), (0, ksize - 1), 1.0, 1)
    kernel /= max(kernel.sum(), 1e-6)
    layer = cv2.filter2D(layer, -1, kernel)
    layer = cv2.GaussianBlur(layer, (3, 3), cfg['layer_blur'])
    return np.clip(layer, 0.0, 1.0)


def perturb_rain(img_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cfg = PERTURB_CFG_MID['rain']
    x = to_float(img_rgb)
    h, w = x.shape[:2]
    rain = _draw_rain_layer(h, w, cfg, rng)
    rain_rgb = np.repeat(rain[..., None], 3, axis=2)
    x = clip01(x * (1.0 - cfg['rain_alpha']) + rain_rgb * cfg['rain_alpha'])
    x = x * cfg['darkening']
    veil = np.ones_like(x, dtype=np.float32) * cfg['veil_value']
    x = clip01(x * (1.0 - cfg['veil_alpha']) + veil * cfg['veil_alpha'])
    hsv = cv2.cvtColor(to_uint8(x), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * cfg['saturation_scale'], 0, 255)
    x = to_float(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    x = clip01((x - 0.5) * cfg['contrast_scale'] + 0.5)
    x = cv2.GaussianBlur(x, (0, 0), cfg['blur_sigma'])
    return to_uint8(x)


def _upsampled_noise(h: int, w: int, rng: np.random.Generator, base: int = 28) -> np.ndarray:
    sh = max(6, h // base)
    sw = max(6, w // base)
    noise = rng.random((sh, sw), dtype=np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), 11.0)
    noise -= noise.min()
    if noise.max() > 1e-6:
        noise /= noise.max()
    return noise.astype(np.float32)


def perturb_smoke(img_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cfg = PERTURB_CFG_MID['smoke']
    x = to_float(img_rgb)
    h, w = x.shape[:2]
    n = _upsampled_noise(h, w, rng, base=int(cfg['noise_base']))
    trans = cfg['trans_hi'] - cfg['trans_delta'] * n
    trans = np.clip(trans, cfg['trans_clip'][0], cfg['trans_clip'][1])
    smoke_color = np.ones_like(x, dtype=np.float32) * cfg['smoke_gray']
    x = x * trans[..., None] + smoke_color * (1.0 - trans[..., None])
    x = clip01((x - 0.5) * cfg['contrast_scale'] + 0.5)
    return to_uint8(x)


def apply_condition(img_rgb: np.ndarray, condition: str, rel_path: str) -> np.ndarray:
    if condition == 'exposure':
        return perturb_exposure(img_rgb)
    rng = np.random.default_rng(stable_seed(rel_path, condition))
    if condition == 'lowlight':
        return perturb_lowlight(img_rgb, rng)
    if condition == 'rain':
        return perturb_rain(img_rgb, rng)
    if condition == 'smoke':
        return perturb_smoke(img_rgb, rng)
    raise ValueError(condition)


def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: str, img: np.ndarray, jpg_quality: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        ok = cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    elif ext == ".png":
        ok = cv2.imwrite(path, bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        ok = cv2.imwrite(path, bgr)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def build_jobs(query_indices: np.ndarray,
               infos,
               data_root_dir: str,
               out_root: str,
               conditions: List[str],
               overwrite: bool) -> Tuple[List[Tuple], List[Dict]]:
    jobs = []
    manifest_rows = []

    for idx in query_indices:
        idx = int(idx)
        info = infos[idx]
        for cam_name, cam_info in info["cam_infos"].items():
            rel_img = cam_info["filename"]
            src_path = os.path.join(data_root_dir, rel_img)

            needed_conditions = []
            for condition in conditions:
                dst_path = os.path.join(out_root, condition, rel_img)
                if overwrite or (not os.path.exists(dst_path)):
                    needed_conditions.append(condition)

            if needed_conditions:
                jobs.append((src_path, rel_img, out_root, tuple(needed_conditions)))

            manifest_rows.append({
                "frame_index": idx,
                "timestamp": int(info.get("timestamp", cam_info.get("timestamp", -1))),
                "log_id": info.get("log_id", info.get("scene_token", "")),
                "city": info.get("city", "UNK"),
                "camera": cam_name,
                "relative_image_path": rel_img,
                "severity": "mid",
            })
    return jobs, manifest_rows


def worker_run(args):
    src_path, rel_img, out_root, conditions, jpg_quality = args
    img = imread_rgb(src_path)
    written = 0
    for condition in conditions:
        out = apply_condition(img, condition, rel_img)
        dst_path = os.path.join(out_root, condition, rel_img)
        imwrite_rgb(dst_path, out, jpg_quality)
        written += 1
    return written


def parse_args():
    ap = argparse.ArgumentParser(description="Generate FINAL MID AV2S query-only visual perturbations (HDD optimized).")
    ap.add_argument("--config", default="config/config_av2.yaml")
    ap.add_argument("--data_root_dir", default=None)
    ap.add_argument("--cache_root", default=None)
    ap.add_argument("--info_path", default=None)
    ap.add_argument("--out_root", required=True, help="e.g. /media/ros/DataSets2/AV2S_QUERY_PERTURB_MID")
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=DEFAULT_CONDITIONS)
    ap.add_argument("--city_tags", default="", help="e.g. DTW,MIA,PIT")
    ap.add_argument("--include_all", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4, help="For HDD, 4 is a good default. Try 3~6.")
    ap.add_argument("--chunksize", type=int, default=8, help="Executor chunksize. For HDD, keep modest.")
    ap.add_argument("--jpg_quality", type=int, default=90, help="JPEG output quality.")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    data_root_dir = args.data_root_dir or cfg["data"]["data_root_dir"]
    cache_root = args.cache_root or os.path.dirname(cfg["data"]["database_path"])
    info_path = args.info_path or cfg["data"].get("test_info_path", os.path.join(cache_root, "av2_infos_test.pkl"))
    out_root = args.out_root

    check_path(data_root_dir, cache_root, info_path)
    check_dir(out_root, os.path.join(out_root, "perturb_configs"))

    splits = auto_discover_test_splits(cache_root, include_all=args.include_all)
    splits = filter_splits_by_tags(splits, args.city_tags)
    if not splits:
        raise FileNotFoundError(f"No AV2 test split query files found in {cache_root}")

    print("[splits]")
    for s in splits:
        print(f"  {s['name']}: query={s['query_path']}")

    infos = load_infos(info_path)
    query_indices = collect_unique_query_indices(splits)
    if query_indices.size == 0:
        raise RuntimeError("No query frame indices found in the selected splits.")

    jobs, manifest_rows = build_jobs(
        query_indices=query_indices,
        infos=infos,
        data_root_dir=data_root_dir,
        out_root=out_root,
        conditions=list(args.conditions),
        overwrite=bool(args.overwrite),
    )

    total_query_frames = len(query_indices)
    total_camera_images = len(manifest_rows)
    total_output_images = total_camera_images * len(args.conditions)
    remaining_outputs = sum(len(j[3]) for j in jobs)

    print(f"[query frames]      {total_query_frames}")
    print(f"[camera images]     {total_camera_images}")
    print(f"[output images all] {total_output_images}")
    print(f"[to generate now]   {remaining_outputs}")
    print(f"[workers]           {args.num_workers} (HDD recommended: 4)")
    print(f"[chunksize]         {args.chunksize}")
    print(f"[jpg_quality]       {args.jpg_quality}")

    # Ensure top-level condition dirs exist before workers start writing
    for condition in args.conditions:
        check_dir(os.path.join(out_root, condition))

    if jobs:
        job_args = [(src_path, rel_img, out_root, conds, int(args.jpg_quality)) for (src_path, rel_img, out_root, conds) in jobs]
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as ex:
            pbar = tqdm(
                ex.map(worker_run, job_args, chunksize=int(args.chunksize)),
                total=len(job_args),
                desc="gen final MID visual query perturb (HDD opt)",
                ncols=120
            )
            generated = 0
            for written in pbar:
                generated += int(written)
                pbar.set_postfix(written=generated)
    else:
        print("[skip] all requested perturbation files already exist.")

    manifest_csv = os.path.join(out_root, "perturb_configs", "perturb_manifest.csv")
    with open(manifest_csv, "w", newline="") as f:
        fieldnames = ["frame_index", "timestamp", "log_id", "city", "camera", "relative_image_path", "severity"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

    param_txt = os.path.join(out_root, "perturb_configs", "perturb_parameters.txt")
    with open(param_txt, "w", encoding="utf-8") as f:
        f.write("Visual perturbation protocol (FINAL MID severity only)\n")
        f.write("Generator version: HDD-optimized parallel version\n\n")
        for k, v in PERTURB_CFG_MID.items():
            f.write(f"{k}: {v}\n")

    print("\n[done]")
    print(f"out_root      : {out_root}")
    print(f"manifest_csv  : {manifest_csv}")
    print(f"param_txt     : {param_txt}")
    print("conditions    :", ", ".join(args.conditions))


if __name__ == "__main__":
    main()
