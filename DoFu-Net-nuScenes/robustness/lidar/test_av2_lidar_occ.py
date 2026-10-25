"""
Full AV2S LiDAR occlusion robustness evaluation for DoFu-Net models.

Protocol:
  - database always uses clean range_data_dir from config
  - query uses clean range_data_dir for the clean baseline
  - query uses RANGE_DOME3VIEW_OCC/occ_{030,060,090,120}_seedXXXX for occlusion tests
  - no retraining, no GT/index change

Default output:
  <result_dir>/lidar_occ_robustness/
      robustness_summary_lidar.csv
      <split>/clean/features_clean.pkl, pr_clean.csv
      <split>/occ_030_seed2026/features_occ_030_seed2026.pkl, pr_occ_030_seed2026.csv
      ...
"""
import argparse
import csv
import json
import os
import pickle
import re
import time
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset.AV2Dataset import DatabaseQueryDataset
from modules.DoFuNet import DoFuNet
from tools.utils import check_dir, check_path
from tools.utils_av2 import RING_CAMERA_ORDER, load_config

METHOD_NAME = 'lidar'
DEFAULT_OCC_DEGREES = [30, 60, 90, 120]


class LidarBranchWrapper(nn.Module):
    """
    Force evaluation to use the LiDAR descriptor branch, regardless of what DoFuNet.forward()
    currently returns in modules/DoFuNet.py. This avoids accidentally evaluating the visual branch
    when the file is temporarily switched during development.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, camera_feature, lidar_feature):
        py, ry, rp = self.base_model.unpack_dome_views(lidar_feature)
        lidar_mid = self.base_model._extract_lidar_mid_feature(py, ry, rp)
        return self.base_model._forward_lidar_head(lidar_mid)


def save_csv(rows: List[Dict], csv_path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def auto_discover_test_splits(cache_root: str, info_path: str, include_all: bool = False) -> List[Dict]:
    """
    Discover AV2S test splits.

    By default, if city/test split files exist, skip AV2_TEST_ALL so the common
    three-test-set setup outputs exactly three rows per condition. Use --include_all
    if you also want the global ALL result.
    """
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
                'name': f'AV2_TEST_{ctag.upper()}',
                'database_path': dbp,
                'query_path': qp,
                'info_path': info_path,
                'city_tag': ctag.upper(),
            })

    all_db = os.path.join(cache_root, 'av2_test_db.npy')
    all_q = os.path.join(cache_root, 'av2_test_query.npy')
    all_split = []
    if os.path.exists(all_db) and os.path.exists(all_q):
        all_split.append({
            'name': 'AV2_TEST_ALL',
            'database_path': all_db,
            'query_path': all_q,
            'info_path': info_path,
            'city_tag': 'ALL',
        })

    if city_splits:
        return (all_split + city_splits) if include_all else city_splits
    return all_split


def filter_splits_by_tags(splits: List[Dict], city_tags: str) -> List[Dict]:
    if not city_tags:
        return splits
    wanted = {x.strip().upper() for x in city_tags.split(',') if x.strip()}
    if not wanted:
        return splits
    return [s for s in splits if s['city_tag'].upper() in wanted or s['name'].upper() in wanted]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def extract_descriptors(model: nn.Module,
                        loader: DataLoader,
                        device: torch.device,
                        output_dim: int,
                        desc: str) -> Tuple[np.ndarray, float]:
    model.eval()
    feats = np.empty((len(loader.dataset), output_dim), dtype=np.float32)
    cursor = 0
    total_ms = 0.0
    pbar = tqdm(loader, ncols=100, desc=desc)
    for input_dict in pbar:
        camera_feature = input_dict['camera_feature'].to(device, non_blocking=True)
        lidar_feature = input_dict['lidar_feature'].to(device, non_blocking=True)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        descriptor = model(camera_feature, lidar_feature)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_ms += elapsed_ms

        desc_np = descriptor.detach().cpu().numpy().astype(np.float32)
        bs = desc_np.shape[0]
        feats[cursor:cursor + bs] = desc_np
        cursor += bs
        pbar.set_postfix(saved=cursor, ms=f'{elapsed_ms / max(bs, 1):.3f}')
    return feats, total_ms


@torch.no_grad()
def extract_full_dataset(model: nn.Module,
                         dataset: DatabaseQueryDataset,
                         batch_size: int,
                         num_workers: int,
                         device: torch.device,
                         output_dim: int,
                         desc: str) -> Tuple[np.ndarray, np.ndarray, float]:
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    feats, total_ms = extract_descriptors(model, loader, device, output_dim, desc)
    dbFeat = feats[:dataset.num_db].astype(np.float32)
    qFeat = feats[dataset.num_db:].astype(np.float32)
    return dbFeat, qFeat, total_ms


@torch.no_grad()
def extract_query_only(model: nn.Module,
                       dataset: DatabaseQueryDataset,
                       batch_size: int,
                       num_workers: int,
                       device: torch.device,
                       output_dim: int,
                       desc: str) -> Tuple[np.ndarray, float]:
    query_indices = list(range(dataset.num_db, len(dataset)))
    subset = Subset(dataset, query_indices)
    loader = DataLoader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    qFeat, total_ms = extract_descriptors(model, loader, device, output_dim, desc)
    return qFeat.astype(np.float32), total_ms


def compute_metrics(dbFeat: np.ndarray,
                    qFeat: np.ndarray,
                    gt: List[np.ndarray],
                    n_values=(1, 5, 10, 20),
                    pr_points: int = 1000,
                    query_topk: int = 5) -> Dict:
    output_dim = int(dbFeat.shape[1])
    faiss_index = faiss.IndexFlatL2(output_dim)
    faiss_index.add(dbFeat.astype(np.float32))
    search_k = min(len(dbFeat), max(max(n_values), int(query_topk), 1))
    dists_top, preds_top = faiss_index.search(qFeat.astype(np.float32), search_k)

    correct_at_n = np.zeros(len(n_values), dtype=np.float64)
    for qIx, pred in enumerate(preds_top):
        positives = gt[qIx]
        for i, n in enumerate(n_values):
            k = min(n, pred.shape[0])
            if np.any(np.in1d(pred[:k], positives)):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / max(len(qFeat), 1) * 100.0
    recalls = {f'recall@{n}': float(recall_at_n[i]) for i, n in enumerate(n_values)}

    best_dists = dists_top[:, 0] if len(dists_top) > 0 else np.empty((0,), dtype=np.float32)
    best_preds = preds_top[:, 0] if len(preds_top) > 0 else np.empty((0,), dtype=np.int64)
    best_is_pos = np.array([np.any(np.in1d(best_preds[i], gt[i])) for i in range(len(best_preds))], dtype=bool)

    if len(best_dists) == 0:
        pr_rows = []
        f1max = 0.0
        pr_auc = 0.0
        best_threshold = 0.0
    else:
        dmin = float(best_dists.min())
        dmax = float(best_dists.max())
        if dmin - 0.1 > 0:
            dmin -= 0.1
        thresholds = np.linspace(dmin, dmax + 0.1, int(pr_points))
        pr_rows = []
        f1max = 0.0
        best_threshold = float(thresholds[0])
        for th in thresholds:
            predicted_positive = best_dists < th
            tp = int(np.sum(predicted_positive & best_is_pos))
            fp = int(np.sum(predicted_positive & ~best_is_pos))
            fn = int(np.sum((~predicted_positive) & best_is_pos))
            tn = int(np.sum((~predicted_positive) & ~best_is_pos))
            precision = 0.0 if (tp + fp) == 0 else tp / float(tp + fp)
            recall = 0.0 if (tp + fn) == 0 else tp / float(tp + fn)
            f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
            if f1 > f1max:
                f1max = float(f1)
                best_threshold = float(th)
            pr_rows.append({
                'threshold': float(th),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
            })
        rr = np.array([r['recall'] for r in pr_rows], dtype=np.float64)
        pp = np.array([r['precision'] for r in pr_rows], dtype=np.float64)
        order = np.argsort(rr)
        pr_auc = float(np.trapz(pp[order], rr[order])) if len(rr) > 1 else 0.0

    out = {
        **recalls,
        'f1max': float(f1max),
        'best_threshold': float(best_threshold),
        'pr_auc': float(pr_auc),
        'pr_rows': pr_rows,
        'dists_top': dists_top,
        'preds_top': preds_top,
    }
    return out



def _json_int_list(x) -> str:
    return json.dumps([int(v) for v in x], ensure_ascii=False)


def _json_float_list(x) -> str:
    return json.dumps([float(v) for v in x], ensure_ascii=False)


def build_query_topk_rows(method_name: str,
                         split_name: str,
                         condition_name: str,
                         occ_degree: int,
                         seed: int,
                         db_frame_indices: np.ndarray,
                         query_frame_indices: np.ndarray,
                         gt: List[np.ndarray],
                         preds_top: np.ndarray,
                         dists_top: np.ndarray,
                         best_threshold: float,
                         positive_if_leq: bool = False) -> List[Dict]:
    rows = []
    k = int(preds_top.shape[1]) if preds_top.ndim == 2 else 0
    for qi in range(len(query_frame_indices)):
        pred_local = preds_top[qi].astype(np.int64).tolist() if k > 0 else []
        pred_frames = [int(db_frame_indices[idx]) for idx in pred_local]
        gt_local = np.asarray(gt[qi], dtype=np.int64).tolist()
        gt_frames = [int(db_frame_indices[idx]) for idx in gt_local]
        top1_dist = float(dists_top[qi, 0]) if k > 0 else float('nan')
        top1_local = int(pred_local[0]) if pred_local else -1
        top1_frame = int(pred_frames[0]) if pred_frames else -1
        top1_correct = bool(top1_local in set(gt_local))
        pred_positive = (top1_dist <= float(best_threshold)) if positive_if_leq else (top1_dist < float(best_threshold))
        rows.append({
            'method': method_name,
            'split': split_name,
            'condition': condition_name,
            'occ_degree': int(occ_degree),
            'seed': int(seed),
            'query_order': int(qi),
            'query_frame_idx': int(query_frame_indices[qi]),
            'pred_top1_db_local_idx': int(top1_local),
            'pred_top1_db_frame_idx': int(top1_frame),
            'top1_distance': float(top1_dist),
            'top1_is_correct': int(top1_correct),
            'pred_positive_at_best_threshold': int(pred_positive),
            'best_threshold': float(best_threshold),
            'gt_db_local_indices': _json_int_list(gt_local),
            'gt_db_frame_indices': _json_int_list(gt_frames),
            'topk_db_local_indices': _json_int_list(pred_local),
            'topk_db_frame_indices': _json_int_list(pred_frames),
            'topk_distances': _json_float_list(dists_top[qi].tolist() if k > 0 else []),
        })
    return rows


def save_feature_pickle(path: str, dbFeat: np.ndarray, qFeat: np.ndarray, gt: List[np.ndarray], meta: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'dbFeat': dbFeat, 'qFeat': qFeat, 'gt': gt, 'meta': meta}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_feature_pickle(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['dbFeat'], obj['qFeat'], obj['gt'], obj.get('meta', {})


def parse_args():
    ap = argparse.ArgumentParser(description='DoFu AV2S LiDAR occlusion robustness test')
    ap.add_argument('--config', default='config/config_av2.yaml')
    ap.add_argument('--weights', default=None, help='Override cfg[runner][eval_weights]')
    ap.add_argument('--result_dir', default=None, help='Override cfg[runner][result_dir]')
    ap.add_argument('--occ_root', default=None, help='Default: <cache_root>/RANGE_DOME3VIEW_OCC')
    ap.add_argument('--occ_degrees', type=int, nargs='+', default=DEFAULT_OCC_DEGREES)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--city_tags', default='', help='Comma separated tags, e.g. MIA,PIT. Default: all discovered city/test splits.')
    ap.add_argument('--include_all', action='store_true', help='Also evaluate AV2_TEST_ALL when city/test splits exist.')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--output_dim', type=int, default=256)
    ap.add_argument('--pr_points', type=int, default=1000)
    ap.add_argument('--query_topk', type=int, default=5, help='How many top retrievals to export per query to query_topk.csv')
    ap.add_argument('--force_recompute', action='store_true', help='Recompute feature pkl even if it already exists.')
    ap.add_argument('--branch', default='lidar', choices=['lidar', 'current_forward'],
                    help='lidar forces _forward_lidar_descriptor; current_forward uses modules/DoFuNet.py forward() as-is.')
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    data_root_dir = cfg['data']['data_root_dir']
    clean_range_data_dir = cfg['data']['range_data_dir']
    resize = cfg['runner']['resize']
    num_workers = args.num_workers if args.num_workers is not None else cfg['runner']['num_workers_test']
    num_cam = int(cfg['runner'].get('num_cam', 7))
    camera_channels = cfg['data'].get('camera_channels', RING_CAMERA_ORDER)[:num_cam]
    cache_root = os.path.dirname(cfg['data']['database_path'])
    info_path = cfg['data'].get('test_info_path', os.path.join(cache_root, 'av2_infos_test.pkl'))
    occ_root = args.occ_root or os.path.join(cache_root, 'RANGE_DOME3VIEW_OCC')

    result_dir = args.result_dir or cfg['runner']['result_dir']
    result_dir = os.path.join(result_dir, 'lidar_occ_robustness')
    check_dir(result_dir)

    weights = args.weights or cfg['runner']['eval_weights']
    check_path(data_root_dir, clean_range_data_dir, info_path, weights)

    splits = auto_discover_test_splits(cache_root, info_path, include_all=args.include_all)
    splits = filter_splits_by_tags(splits, args.city_tags)
    if not splits:
        raise FileNotFoundError(f'No AV2S test splits found in {cache_root}')

    print('[splits]')
    for sp in splits:
        print(f"  {sp['name']}: db={sp['database_path']}, query={sp['query_path']}")

    visual_backbone = cfg['runner'].get('visual_backbone', 'convnextv2_atto')
    convnextv2_ckpt_path = cfg['runner'].get('convnextv2_ckpt_path', '')
    base_model = DoFuNet.create(weights=None, num_cam=num_cam, visual_backbone=visual_backbone,
                             convnext_local_ckpt_path=convnextv2_ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(weights, map_location=device)
    state = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    base_model.load_state_dict(state)
    if args.branch == 'lidar':
        model = LidarBranchWrapper(base_model)
    else:
        model = base_model
    model = model.to(device)
    model.eval()

    total_params, trainable_params = count_parameters(base_model)
    print(f'[model] branch={args.branch}, params={total_params/1e6:.3f}M, weights={weights}')

    img_transforms = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    summary_rows = []
    for sp in splits:
        split_name = sp['name']
        split_dir = os.path.join(result_dir, split_name)
        check_dir(split_dir)
        check_path(sp['database_path'], sp['query_path'], sp['info_path'])
        db_meta = np.load(sp['database_path'])
        query_meta = np.load(sp['query_path'])
        db_frame_indices = db_meta[:, 0].astype(np.int64)
        query_frame_indices = query_meta[:, 0].astype(np.int64)

        clean_dataset = DatabaseQueryDataset(
            data_root_dir=data_root_dir,
            database_path=sp['database_path'],
            query_path=sp['query_path'],
            info_path=sp['info_path'],
            transforms=img_transforms,
            nonTrivPosDistThres=cfg['runner']['nonTrivPosDistThres'],
            range_data_dir=clean_range_data_dir,
            query_range_data_dir=None,
            camera_channels=camera_channels,
        )
        gt = clean_dataset.getPositives()

        # Clean baseline: extract both DB and query once, then reuse DB for all occlusion degrees.
        clean_cond = 'clean'
        clean_pkl = os.path.join(split_dir, clean_cond, f'features_{clean_cond}.pkl')
        if os.path.exists(clean_pkl) and not args.force_recompute:
            dbFeat_clean, qFeat_clean, gt, _ = load_feature_pickle(clean_pkl)
            clean_extract_ms = 0.0
            print(f'[{split_name}] loaded cached clean features: {clean_pkl}')
        else:
            dbFeat_clean, qFeat_clean, clean_extract_ms = extract_full_dataset(
                model=model,
                dataset=clean_dataset,
                batch_size=args.batch_size,
                num_workers=num_workers,
                device=device,
                output_dim=args.output_dim,
                desc=f'{split_name} clean full',
            )
            save_feature_pickle(clean_pkl, dbFeat_clean, qFeat_clean, gt, {
                'split': split_name,
                'condition': clean_cond,
                'range_data_dir': clean_range_data_dir,
                'query_range_data_dir': None,
                'weights': weights,
                'branch': args.branch,
            })

        metrics = compute_metrics(
            dbFeat_clean, qFeat_clean, gt, pr_points=args.pr_points, query_topk=args.query_topk
        )
        pr_csv = os.path.join(split_dir, clean_cond, f'pr_{clean_cond}.csv')
        save_csv(metrics['pr_rows'], pr_csv,
                 ['threshold', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn'])
        query_topk_csv = os.path.join(split_dir, clean_cond, 'query_topk.csv')
        query_rows = build_query_topk_rows(
            method_name=METHOD_NAME,
            split_name=split_name,
            condition_name=clean_cond,
            occ_degree=0,
            seed=int(args.seed),
            db_frame_indices=db_frame_indices,
            query_frame_indices=query_frame_indices,
            gt=gt,
            preds_top=metrics['preds_top'][:, :min(metrics['preds_top'].shape[1], int(args.query_topk))],
            dists_top=metrics['dists_top'][:, :min(metrics['dists_top'].shape[1], int(args.query_topk))],
            best_threshold=metrics['best_threshold'],
            positive_if_leq=False,
        )
        save_csv(query_rows, query_topk_csv, [
            'method', 'split', 'condition', 'occ_degree', 'seed', 'query_order', 'query_frame_idx',
            'pred_top1_db_local_idx', 'pred_top1_db_frame_idx', 'top1_distance', 'top1_is_correct',
            'pred_positive_at_best_threshold', 'best_threshold',
            'gt_db_local_indices', 'gt_db_frame_indices',
            'topk_db_local_indices', 'topk_db_frame_indices', 'topk_distances'
        ])
        clean_r1 = metrics['recall@1']
        clean_f1 = metrics['f1max']
        row = {
            'method': METHOD_NAME,
            'branch': args.branch,
            'split': split_name,
            'condition': clean_cond,
            'occ_degree': 0,
            'seed': int(args.seed),
            'num_db': int(clean_dataset.num_db),
            'num_query': int(clean_dataset.num_query),
            'descriptor_dim': int(args.output_dim),
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'recall@1': metrics['recall@1'],
            'recall@5': metrics['recall@5'],
            'recall@10': metrics['recall@10'],
            'recall@20': metrics['recall@20'],
            'f1max': metrics['f1max'],
            'pr_auc': metrics['pr_auc'],
            'best_threshold': metrics['best_threshold'],
            'retention_r1_percent': 100.0,
            'delta_r1': 0.0,
            'retention_f1_percent': 100.0,
            'delta_f1': 0.0,
            'feature_pkl': clean_pkl,
            'pr_csv': pr_csv,
            'query_topk_csv': query_topk_csv,
            'query_range_data_dir': '',
            'extract_total_ms': float(clean_extract_ms),
        }
        summary_rows.append(row)
        print(f"[{split_name}][clean] R@1={metrics['recall@1']:.2f}, F1max={metrics['f1max']:.4f}")

        # Occlusion degrees: clean DB descriptors are reused; only query descriptors are recomputed.
        for deg in args.occ_degrees:
            cond = f'occ_{int(deg):03d}_seed{int(args.seed)}'
            query_occ_dir = os.path.join(occ_root, cond)
            check_path(query_occ_dir)
            occ_dataset = DatabaseQueryDataset(
                data_root_dir=data_root_dir,
                database_path=sp['database_path'],
                query_path=sp['query_path'],
                info_path=sp['info_path'],
                transforms=img_transforms,
                nonTrivPosDistThres=cfg['runner']['nonTrivPosDistThres'],
                range_data_dir=clean_range_data_dir,
                query_range_data_dir=query_occ_dir,
                camera_channels=camera_channels,
            )
            occ_pkl = os.path.join(split_dir, cond, f'features_{cond}.pkl')
            if os.path.exists(occ_pkl) and not args.force_recompute:
                _db_cached, qFeat_occ, _gt_cached, _ = load_feature_pickle(occ_pkl)
                occ_extract_ms = 0.0
                print(f'[{split_name}] loaded cached {cond} query features: {occ_pkl}')
            else:
                qFeat_occ, occ_extract_ms = extract_query_only(
                    model=model,
                    dataset=occ_dataset,
                    batch_size=args.batch_size,
                    num_workers=num_workers,
                    device=device,
                    output_dim=args.output_dim,
                    desc=f'{split_name} {cond} query',
                )
                save_feature_pickle(occ_pkl, dbFeat_clean, qFeat_occ, gt, {
                    'split': split_name,
                    'condition': cond,
                    'range_data_dir': clean_range_data_dir,
                    'query_range_data_dir': query_occ_dir,
                    'weights': weights,
                    'branch': args.branch,
                })

            metrics = compute_metrics(
                dbFeat_clean, qFeat_occ, gt, pr_points=args.pr_points, query_topk=args.query_topk
            )
            pr_csv = os.path.join(split_dir, cond, f'pr_{cond}.csv')
            save_csv(metrics['pr_rows'], pr_csv,
                     ['threshold', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn'])
            query_topk_csv = os.path.join(split_dir, cond, 'query_topk.csv')
            query_rows = build_query_topk_rows(
                method_name=METHOD_NAME,
                split_name=split_name,
                condition_name=cond,
                occ_degree=int(deg),
                seed=int(args.seed),
                db_frame_indices=db_frame_indices,
                query_frame_indices=query_frame_indices,
                gt=gt,
                preds_top=metrics['preds_top'][:, :min(metrics['preds_top'].shape[1], int(args.query_topk))],
                dists_top=metrics['dists_top'][:, :min(metrics['dists_top'].shape[1], int(args.query_topk))],
                best_threshold=metrics['best_threshold'],
                positive_if_leq=False,
            )
            save_csv(query_rows, query_topk_csv, [
                'method', 'split', 'condition', 'occ_degree', 'seed', 'query_order', 'query_frame_idx',
                'pred_top1_db_local_idx', 'pred_top1_db_frame_idx', 'top1_distance', 'top1_is_correct',
                'pred_positive_at_best_threshold', 'best_threshold',
                'gt_db_local_indices', 'gt_db_frame_indices',
                'topk_db_local_indices', 'topk_db_frame_indices', 'topk_distances'
            ])

            retention_r1 = 0.0 if clean_r1 == 0 else metrics['recall@1'] / clean_r1 * 100.0
            retention_f1 = 0.0 if clean_f1 == 0 else metrics['f1max'] / clean_f1 * 100.0
            row = {
                'method': METHOD_NAME,
                'branch': args.branch,
                'split': split_name,
                'condition': cond,
                'occ_degree': int(deg),
                'seed': int(args.seed),
                'num_db': int(occ_dataset.num_db),
                'num_query': int(occ_dataset.num_query),
                'descriptor_dim': int(args.output_dim),
                'total_params': int(total_params),
                'trainable_params': int(trainable_params),
                'recall@1': metrics['recall@1'],
                'recall@5': metrics['recall@5'],
                'recall@10': metrics['recall@10'],
                'recall@20': metrics['recall@20'],
                'f1max': metrics['f1max'],
                'pr_auc': metrics['pr_auc'],
                'best_threshold': metrics['best_threshold'],
                'retention_r1_percent': float(retention_r1),
                'delta_r1': float(metrics['recall@1'] - clean_r1),
                'retention_f1_percent': float(retention_f1),
                'delta_f1': float(metrics['f1max'] - clean_f1),
                'feature_pkl': occ_pkl,
                'pr_csv': pr_csv,
                'query_topk_csv': query_topk_csv,
                'query_range_data_dir': query_occ_dir,
                'extract_total_ms': float(occ_extract_ms),
            }
            summary_rows.append(row)
            print(f"[{split_name}][{cond}] R@1={metrics['recall@1']:.2f}, "
                  f"F1max={metrics['f1max']:.4f}, retention={retention_r1:.2f}%")

        # Save/update after every split, so long runs are not lost.
        summary_csv_path = os.path.join(result_dir, f'robustness_summary_{METHOD_NAME}.csv')
        save_csv(summary_rows, summary_csv_path, fieldnames=[
            'method', 'branch', 'split', 'condition', 'occ_degree', 'seed',
            'num_db', 'num_query', 'descriptor_dim', 'total_params', 'trainable_params',
            'recall@1', 'recall@5', 'recall@10', 'recall@20',
            'f1max', 'pr_auc', 'best_threshold',
            'retention_r1_percent', 'delta_r1', 'retention_f1_percent', 'delta_f1',
            'feature_pkl', 'pr_csv', 'query_topk_csv', 'query_range_data_dir', 'extract_total_ms'
        ])
        print('summary saved at:', summary_csv_path)

    print('[done] Full LiDAR occlusion robustness evaluation finished.')


if __name__ == '__main__':
    main()
