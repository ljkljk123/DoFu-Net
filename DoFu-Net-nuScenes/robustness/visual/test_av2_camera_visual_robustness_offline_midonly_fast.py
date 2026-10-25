#!/usr/bin/env python3
"""
AV2S camera visual robustness evaluation
OFFLINE pre-generated query images, MID only, BUNDLED FAST version
"""
import argparse, csv, os, pickle, re, time
from typing import Dict, List, Tuple
import cv2, faiss, numpy as np, torch, torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from modules.DoFuNet import DoFuNet
from tools.utils import check_dir, check_path
from tools.utils_av2 import RING_CAMERA_ORDER, load_config

METHOD_NAME = 'camera'
DEFAULT_CONDITIONS = ['exposure', 'lowlight', 'rain', 'smoke']
MID_SEVERITY = 'mid'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class VisualBranchWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
    def forward(self, camera_feature):
        return self.base_model.visual_branch(camera_feature)

def save_csv(rows: List[Dict], csv_path: str, fieldnames: List[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def auto_discover_test_splits(cache_root: str, info_path: str, include_all: bool = False) -> List[Dict]:
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
            city_splits.append({'name': f'AV2_TEST_{ctag.upper()}', 'database_path': dbp, 'query_path': qp, 'info_path': info_path, 'city_tag': ctag.upper()})
    all_db = os.path.join(cache_root, 'av2_test_db.npy')
    all_q = os.path.join(cache_root, 'av2_test_query.npy')
    all_split = []
    if os.path.exists(all_db) and os.path.exists(all_q):
        all_split.append({'name': 'AV2_TEST_ALL', 'database_path': all_db, 'query_path': all_q, 'info_path': info_path, 'city_tag': 'ALL'})
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

def preprocess_rgb_to_tensor(img_rgb: np.ndarray, resize_hw) -> torch.Tensor:
    if isinstance(resize_hw, int):
        out_h = out_w = int(resize_hw)
    else:
        out_h, out_w = int(resize_hw[0]), int(resize_hw[1])
    img = cv2.resize(img_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x.copy())

def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

class BaseAV2CameraDataset(Dataset):
    def __init__(self, data_root_dir: str, database_path: str, query_path: str, info_path: str, resize, nonTrivPosDistThres: float, camera_channels=None):
        super().__init__()
        self.dataroot = data_root_dir
        self.resize = resize
        self.camera_channels = camera_channels or list(RING_CAMERA_ORDER)
        self.nonTrivPosDistThres = nonTrivPosDistThres
        with open(info_path, 'rb') as f:
            self.infos = pickle.load(f)
        data_base = np.load(database_path)
        query = np.load(query_path)
        self.dataset = np.concatenate((data_base, query), axis=0)
        self.num_db = len(data_base)
        self.num_query = len(query)
        self.positives = None
    def getPositives(self):
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            dataset = np.ascontiguousarray(self.dataset[:self.num_db, 1:3])
            knn.fit(dataset)
            self.positives = list(knn.radius_neighbors(self.dataset[self.num_db:, 1:3], radius=self.nonTrivPosDistThres, return_distance=False))
        return self.positives

class CleanTensorDataset(BaseAV2CameraDataset):
    def __len__(self): return len(self.dataset)
    def __getitem__(self, item):
        frame_index = int(self.dataset[item][0]); imgs = []
        for channel in self.camera_channels:
            rel_path = self.infos[frame_index]['cam_infos'][channel]['filename']
            rgb = imread_rgb(os.path.join(self.dataroot, rel_path))
            imgs.append(preprocess_rgb_to_tensor(rgb, self.resize))
        return {'camera_feature': torch.stack(imgs)}

class BundledPerturbedQueryTensorDataset(BaseAV2CameraDataset):
    def __init__(self, perturb_root: str, conditions: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturb_root = perturb_root
        self.conditions = list(conditions)
    def __len__(self): return self.num_query
    def __getitem__(self, item):
        global_idx = self.num_db + int(item)
        frame_index = int(self.dataset[global_idx][0])
        cond_tensors = []
        for cond in self.conditions:
            imgs = []
            for channel in self.camera_channels:
                rel_path = self.infos[frame_index]['cam_infos'][channel]['filename']
                rgb = imread_rgb(os.path.join(self.perturb_root, cond, rel_path))
                imgs.append(preprocess_rgb_to_tensor(rgb, self.resize))
            cond_tensors.append(torch.stack(imgs))
        return {'camera_feature': torch.stack(cond_tensors)}

def make_loader(dataset, batch_size, num_workers):
    kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers and num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return DataLoader(**kwargs)

@torch.no_grad()
def extract_descriptors(model: nn.Module, loader: DataLoader, device: torch.device, output_dim: int, desc: str) -> Tuple[np.ndarray, float]:
    model.eval()
    feats = np.empty((len(loader.dataset), output_dim), dtype=np.float32)
    cursor = 0; total_ms = 0.0
    pbar = tqdm(loader, ncols=100, desc=desc)
    use_amp = (device.type == 'cuda')
    for input_dict in pbar:
        camera_feature = input_dict['camera_feature'].to(device, non_blocking=True)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            descriptor = model(camera_feature)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        total_ms += (time.perf_counter() - t0) * 1000.0
        desc_np = descriptor.detach().float().cpu().numpy().astype(np.float32)
        bs = desc_np.shape[0]
        feats[cursor:cursor + bs] = desc_np
        cursor += bs
    return feats, total_ms

@torch.no_grad()
def extract_bundled_query_descriptors(model: nn.Module, loader: DataLoader, device: torch.device, output_dim: int, conditions: List[str], condition_chunk: int, desc: str) -> Tuple[Dict[str, np.ndarray], float]:
    model.eval()
    num_query = len(loader.dataset)
    out = {cond: np.empty((num_query, output_dim), dtype=np.float32) for cond in conditions}
    total_ms = 0.0; cursor = 0
    pbar = tqdm(loader, ncols=110, desc=desc)
    use_amp = (device.type == 'cuda')
    for input_dict in pbar:
        camera_feature = input_dict['camera_feature']  # [B,Ncond,Ncam,C,H,W]
        B, Ncond, Ncam, C, H, W = camera_feature.shape
        cond_keys = list(conditions)
        for start in range(0, Ncond, int(condition_chunk)):
            chunk_keys = cond_keys[start:start + int(condition_chunk)]
            chunk = camera_feature[:, start:start + len(chunk_keys)]
            K = chunk.shape[1]
            big = chunk.reshape(B * K, Ncam, C, H, W).to(device, non_blocking=True)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                descriptor = model(big)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            total_ms += (time.perf_counter() - t0) * 1000.0
            desc_np = descriptor.detach().float().cpu().numpy().astype(np.float32).reshape(B, K, output_dim)
            for j, cond in enumerate(chunk_keys):
                out[cond][cursor:cursor + B] = desc_np[:, j, :]
        cursor += B
        pbar.set_postfix(done=cursor)
    return out, total_ms

def compute_metrics(dbFeat: np.ndarray, qFeat: np.ndarray, gt: List[np.ndarray], n_values=(1,5,10,20), pr_points: int=1000, query_topk: int=5) -> Dict:
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
    best_dists = dists_top[:, 0]; best_preds = preds_top[:, 0]
    best_is_pos = np.array([np.any(np.in1d(best_preds[i], gt[i])) for i in range(len(best_preds))], dtype=bool)
    if len(best_dists) == 0:
        return {**recalls, 'f1max':0.0, 'best_threshold':0.0, 'pr_auc':0.0, 'pr_rows':[], 'preds_top':preds_top, 'dists_top':dists_top}
    dmin = float(best_dists.min()); dmax = float(best_dists.max())
    if dmin - 0.1 > 0: dmin -= 0.1
    thresholds = np.linspace(dmin, dmax + 0.1, int(pr_points))
    pr_rows = []; f1max = 0.0; best_threshold = float(thresholds[0])
    for th in thresholds:
        pred_pos = best_dists < th
        tp = int(np.sum(pred_pos & best_is_pos)); fp = int(np.sum(pred_pos & ~best_is_pos))
        fn = int(np.sum((~pred_pos) & best_is_pos)); tn = int(np.sum((~pred_pos) & ~best_is_pos))
        precision = 0.0 if (tp + fp) == 0 else tp / float(tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / float(tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
        if f1 > f1max: f1max = float(f1); best_threshold = float(th)
        pr_rows.append({'threshold':float(th),'precision':float(precision),'recall':float(recall),'f1':float(f1),'tp':tp,'fp':fp,'fn':fn,'tn':tn})
    rr = np.array([r['recall'] for r in pr_rows], dtype=np.float64)
    pp = np.array([r['precision'] for r in pr_rows], dtype=np.float64)
    order = np.argsort(rr)
    pr_auc = float(np.trapz(pp[order], rr[order])) if len(rr) > 1 else 0.0
    return {**recalls,'f1max':float(f1max),'best_threshold':float(best_threshold),'pr_auc':float(pr_auc),'pr_rows':pr_rows,'preds_top':preds_top,'dists_top':dists_top}

def save_feature_pickle(path: str, dbFeat: np.ndarray, qFeat: np.ndarray, gt: List[np.ndarray], meta: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'dbFeat': dbFeat, 'qFeat': qFeat, 'gt': gt, 'meta': meta}, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_feature_pickle(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['dbFeat'], obj['qFeat'], obj['gt'], obj.get('meta', {})

def build_query_topk_rows(method_name: str, split_name: str, condition_name: str, dataset_query: BaseAV2CameraDataset, gt: List[np.ndarray], preds_top: np.ndarray, dists_top: np.ndarray, best_threshold: float) -> List[Dict]:
    query_frame_indices = dataset_query.dataset[dataset_query.num_db:, 0].astype(int).tolist()
    db_frame_indices = dataset_query.dataset[:dataset_query.num_db, 0].astype(int).tolist()
    rows = []
    for q_ord, q_frame in enumerate(query_frame_indices):
        gt_local = np.asarray(gt[q_ord], dtype=np.int64)
        gt_frames = [db_frame_indices[int(x)] for x in gt_local.tolist()]
        pred_local = preds_top[q_ord].astype(int).tolist()
        pred_frames = [db_frame_indices[int(x)] for x in pred_local]
        pred_dists = dists_top[q_ord].astype(float).tolist()
        top1_local = int(pred_local[0]) if pred_local else -1
        top1_frame = int(pred_frames[0]) if pred_frames else -1
        top1_dist = float(pred_dists[0]) if pred_dists else float('inf')
        top1_correct = int(top1_local in set(gt_local.tolist())) if pred_local else 0
        pred_positive = int(top1_dist < float(best_threshold))
        rows.append({'method':method_name,'split':split_name,'severity':MID_SEVERITY,'condition':condition_name,'condition_label':condition_name if condition_name == 'clean' else f'{MID_SEVERITY}_{condition_name}','query_order':int(q_ord),'query_frame_idx':int(q_frame),'pred_top1_db_local_idx':int(top1_local),'pred_top1_db_frame_idx':int(top1_frame),'top1_distance':float(top1_dist),'top1_is_correct':int(top1_correct),'pred_positive_at_best_threshold':int(pred_positive),'best_threshold':float(best_threshold),'gt_db_local_indices':str(gt_local.tolist()),'gt_db_frame_indices':str(gt_frames),'topk_db_local_indices':str(pred_local),'topk_db_frame_indices':str(pred_frames),'topk_distances':str(pred_dists)})
    return rows

def parse_args():
    ap = argparse.ArgumentParser(description='DoFu AV2S camera visual robustness test (OFFLINE pre-generated, MID ONLY, BUNDLED FAST)')
    ap.add_argument('--config', default='config/config_av2.yaml')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--result_dir', default=None)
    ap.add_argument('--perturb_root', required=True)
    ap.add_argument('--conditions', nargs='+', default=DEFAULT_CONDITIONS, choices=DEFAULT_CONDITIONS)
    ap.add_argument('--city_tags', default='')
    ap.add_argument('--include_all', action='store_true')
    ap.add_argument('--batch_size_clean', type=int, default=16)
    ap.add_argument('--batch_size_query', type=int, default=4)
    ap.add_argument('--condition_chunk', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=None)
    ap.add_argument('--output_dim', type=int, default=256)
    ap.add_argument('--pr_points', type=int, default=1000)
    ap.add_argument('--query_topk', type=int, default=5)
    ap.add_argument('--force_recompute', action='store_true')
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    data_root_dir = cfg['data']['data_root_dir']
    resize = cfg['runner']['resize']
    num_workers = args.num_workers if args.num_workers is not None else cfg['runner']['num_workers_test']
    num_cam = int(cfg['runner'].get('num_cam', 7))
    camera_channels = cfg['data'].get('camera_channels', RING_CAMERA_ORDER)[:num_cam]
    cache_root = os.path.dirname(cfg['data']['database_path'])
    info_path = cfg['data'].get('test_info_path', os.path.join(cache_root, 'av2_infos_test.pkl'))
    result_dir = args.result_dir or cfg['runner']['result_dir']
    result_dir = os.path.join(result_dir, 'camera_visual_robustness_offline_midonly_fast')
    check_dir(result_dir)
    weights = args.weights or cfg['runner']['eval_weights']
    check_path(data_root_dir, info_path, weights, args.perturb_root)
    splits = auto_discover_test_splits(cache_root, info_path, include_all=args.include_all)
    splits = filter_splits_by_tags(splits, args.city_tags)
    if not splits:
        raise FileNotFoundError(f'No AV2S test splits found in {cache_root}')
    print('[splits]')
    for sp in splits:
        print(f"  {sp['name']}: db={sp['database_path']}, query={sp['query_path']}")
    visual_backbone = cfg['runner'].get('visual_backbone', 'convnextv2_atto')
    convnextv2_ckpt_path = cfg['runner'].get('convnextv2_ckpt_path', '')
    base_model = DoFuNet.create(weights=None, num_cam=num_cam, visual_backbone=visual_backbone, convnext_local_ckpt_path=convnextv2_ckpt_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(weights, map_location=device)
    state = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    base_model.load_state_dict(state)
    model = VisualBranchWrapper(base_model).to(device)
    model.eval()
    total_params, trainable_params = count_parameters(base_model)
    print(f'[model] branch=visual, params={total_params/1e6:.3f}M, weights={weights}')
    summary_rows = []; conditions = list(args.conditions)

    for sp in splits:
        split_name = sp['name']
        split_dir = os.path.join(result_dir, split_name)
        check_dir(split_dir)
        base_dataset = BaseAV2CameraDataset(data_root_dir=data_root_dir, database_path=sp['database_path'], query_path=sp['query_path'], info_path=sp['info_path'], resize=resize, nonTrivPosDistThres=cfg['runner']['nonTrivPosDistThres'], camera_channels=camera_channels)
        gt = base_dataset.getPositives()
        clean_dataset = CleanTensorDataset(data_root_dir=data_root_dir, database_path=sp['database_path'], query_path=sp['query_path'], info_path=sp['info_path'], resize=resize, nonTrivPosDistThres=cfg['runner']['nonTrivPosDistThres'], camera_channels=camera_channels)
        clean_dir = os.path.join(split_dir, 'clean')
        clean_pkl = os.path.join(clean_dir, 'features_clean.pkl')
        if os.path.exists(clean_pkl) and not args.force_recompute:
            dbFeat_clean, qFeat_clean, gt, _ = load_feature_pickle(clean_pkl); clean_extract_ms = 0.0
            print(f'[{split_name}] loaded cached clean features: {clean_pkl}')
        else:
            clean_loader = make_loader(clean_dataset, args.batch_size_clean, num_workers)
            feats, clean_extract_ms = extract_descriptors(model, clean_loader, device, args.output_dim, f'{split_name} clean full')
            dbFeat_clean = feats[:clean_dataset.num_db].astype(np.float32); qFeat_clean = feats[clean_dataset.num_db:].astype(np.float32)
            save_feature_pickle(clean_pkl, dbFeat_clean, qFeat_clean, gt, {'split':split_name,'severity':'clean','condition':'clean','weights':weights,'branch':'visual','offline_midonly_fast':True})
        metrics = compute_metrics(dbFeat_clean, qFeat_clean, gt, pr_points=args.pr_points, query_topk=args.query_topk)
        pr_csv = os.path.join(clean_dir, 'pr_clean.csv')
        save_csv(metrics['pr_rows'], pr_csv, ['threshold','precision','recall','f1','tp','fp','fn','tn'])
        query_topk_csv = os.path.join(clean_dir, 'query_topk.csv')
        query_rows = build_query_topk_rows(METHOD_NAME, split_name, 'clean', base_dataset, gt, metrics['preds_top'][:, :min(metrics['preds_top'].shape[1], int(args.query_topk))], metrics['dists_top'][:, :min(metrics['dists_top'].shape[1], int(args.query_topk))], metrics['best_threshold'])
        save_csv(query_rows, query_topk_csv, ['method','split','severity','condition','condition_label','query_order','query_frame_idx','pred_top1_db_local_idx','pred_top1_db_frame_idx','top1_distance','top1_is_correct','pred_positive_at_best_threshold','best_threshold','gt_db_local_indices','gt_db_frame_indices','topk_db_local_indices','topk_db_frame_indices','topk_distances'])
        clean_r1 = metrics['recall@1']; clean_f1 = metrics['f1max']
        summary_rows.append({'method':METHOD_NAME,'branch':'visual','split':split_name,'severity':'clean','condition':'clean','condition_label':'clean','num_db':int(base_dataset.num_db),'num_query':int(base_dataset.num_query),'descriptor_dim':int(args.output_dim),'total_params':int(total_params),'trainable_params':int(trainable_params),'recall@1':metrics['recall@1'],'recall@5':metrics['recall@5'],'recall@10':metrics['recall@10'],'recall@20':metrics['recall@20'],'f1max':metrics['f1max'],'pr_auc':metrics['pr_auc'],'best_threshold':metrics['best_threshold'],'retention_r1_percent':100.0,'delta_r1':0.0,'retention_f1_percent':100.0,'delta_f1':0.0,'feature_pkl':clean_pkl,'pr_csv':pr_csv,'query_topk_csv':query_topk_csv,'extract_total_ms':float(clean_extract_ms)})
        print(f"[{split_name}][clean] R@1={metrics['recall@1']:.2f}, F1max={metrics['f1max']:.4f}")

        need_extract = args.force_recompute
        for cond in conditions:
            feat_pkl = os.path.join(split_dir, cond, f'features_{MID_SEVERITY}_{cond}.pkl')
            if not os.path.exists(feat_pkl):
                need_extract = True; break
        qFeat_map = {}; bundled_extract_ms = 0.0
        if need_extract:
            q_dataset = BundledPerturbedQueryTensorDataset(perturb_root=args.perturb_root, conditions=conditions, data_root_dir=data_root_dir, database_path=sp['database_path'], query_path=sp['query_path'], info_path=sp['info_path'], resize=resize, nonTrivPosDistThres=cfg['runner']['nonTrivPosDistThres'], camera_channels=camera_channels)
            q_loader = make_loader(q_dataset, args.batch_size_query, num_workers)
            qFeat_map, bundled_extract_ms = extract_bundled_query_descriptors(model, q_loader, device, args.output_dim, conditions, args.condition_chunk, f'{split_name} bundled mid query')
            for cond in conditions:
                feat_pkl = os.path.join(split_dir, cond, f'features_{MID_SEVERITY}_{cond}.pkl')
                save_feature_pickle(feat_pkl, dbFeat_clean, qFeat_map[cond], gt, {'split':split_name,'severity':MID_SEVERITY,'condition':cond,'weights':weights,'branch':'visual','offline_midonly_fast':True})
        else:
            for cond in conditions:
                feat_pkl = os.path.join(split_dir, cond, f'features_{MID_SEVERITY}_{cond}.pkl')
                _db, qFeat_map[cond], _gt, _meta = load_feature_pickle(feat_pkl)
            print(f'[{split_name}] loaded cached bundled condition features.')

        for cond in conditions:
            cond_dir = os.path.join(split_dir, cond)
            feat_pkl = os.path.join(cond_dir, f'features_{MID_SEVERITY}_{cond}.pkl')
            qFeat_pert = qFeat_map[cond]
            metrics = compute_metrics(dbFeat_clean, qFeat_pert, gt, pr_points=args.pr_points, query_topk=args.query_topk)
            pr_csv = os.path.join(cond_dir, f'pr_{MID_SEVERITY}_{cond}.csv')
            save_csv(metrics['pr_rows'], pr_csv, ['threshold','precision','recall','f1','tp','fp','fn','tn'])
            query_topk_csv = os.path.join(cond_dir, 'query_topk.csv')
            query_rows = build_query_topk_rows(METHOD_NAME, split_name, cond, base_dataset, gt, metrics['preds_top'][:, :min(metrics['preds_top'].shape[1], int(args.query_topk))], metrics['dists_top'][:, :min(metrics['dists_top'].shape[1], int(args.query_topk))], metrics['best_threshold'])
            save_csv(query_rows, query_topk_csv, ['method','split','severity','condition','condition_label','query_order','query_frame_idx','pred_top1_db_local_idx','pred_top1_db_frame_idx','top1_distance','top1_is_correct','pred_positive_at_best_threshold','best_threshold','gt_db_local_indices','gt_db_frame_indices','topk_db_local_indices','topk_db_frame_indices','topk_distances'])
            retention_r1 = 0.0 if clean_r1 == 0 else metrics['recall@1'] / clean_r1 * 100.0
            retention_f1 = 0.0 if clean_f1 == 0 else metrics['f1max'] / clean_f1 * 100.0
            summary_rows.append({'method':METHOD_NAME,'branch':'visual','split':split_name,'severity':MID_SEVERITY,'condition':cond,'condition_label':f'{MID_SEVERITY}_{cond}','num_db':int(base_dataset.num_db),'num_query':int(base_dataset.num_query),'descriptor_dim':int(args.output_dim),'total_params':int(total_params),'trainable_params':int(trainable_params),'recall@1':metrics['recall@1'],'recall@5':metrics['recall@5'],'recall@10':metrics['recall@10'],'recall@20':metrics['recall@20'],'f1max':metrics['f1max'],'pr_auc':metrics['pr_auc'],'best_threshold':metrics['best_threshold'],'retention_r1_percent':float(retention_r1),'delta_r1':float(metrics['recall@1'] - clean_r1),'retention_f1_percent':float(retention_f1),'delta_f1':float(metrics['f1max'] - clean_f1),'feature_pkl':feat_pkl,'pr_csv':pr_csv,'query_topk_csv':query_topk_csv,'extract_total_ms':float(bundled_extract_ms / max(len(conditions), 1))})
            print(f"[{split_name}][mid/{cond}] R@1={metrics['recall@1']:.2f}, F1max={metrics['f1max']:.4f}, retention={retention_r1:.2f}%")

        summary_csv_path = os.path.join(result_dir, f'robustness_summary_{METHOD_NAME}.csv')
        save_csv(summary_rows, summary_csv_path, ['method','branch','split','severity','condition','condition_label','num_db','num_query','descriptor_dim','total_params','trainable_params','recall@1','recall@5','recall@10','recall@20','f1max','pr_auc','best_threshold','retention_r1_percent','delta_r1','retention_f1_percent','delta_f1','feature_pkl','pr_csv','query_topk_csv','extract_total_ms'])
        print('summary saved at:', summary_csv_path)

    print('[done] OFFLINE MID-ONLY camera visual robustness evaluation finished.')

if __name__ == '__main__':
    main()
