import os
import csv
import pickle
import numpy as np
import torch
import faiss
import argparse

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from dataset.NuScenesDataset import DatabaseQueryDataset
from modules.DoFuNet import DoFuNet
from tools.utils import load_config, check_dir, check_path


def save_csv(rows, csv_path, fieldnames):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class UnifiedEvaluator:
    def __init__(self, model, dataset, loader, device, output_dim=256):
        self.model = model
        self.dataset = dataset
        self.loader = loader
        self.device = device
        self.output_dim = output_dim

    @torch.no_grad()
    def extract_features(self, ckpt_path, feature_path):
        print('================= extracting features =================')
        print('load weights from:', ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['net'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print('database:', self.dataset.num_db, 'query:', self.dataset.num_query)

        feats = np.empty((len(self.dataset), self.output_dim), dtype=np.float32)

        for i, input_dict in enumerate(tqdm(self.loader, ncols=80)):
            camera_feature = input_dict['camera_feature'].to(self.device)
            lidar_feature = input_dict['lidar_feature'].to(self.device)
            descriptor = self.model(camera_feature, lidar_feature)
            feats[i, :] = descriptor.detach().cpu().numpy()

        qFeat = feats[self.dataset.num_db:].astype(np.float32)
        dbFeat = feats[:self.dataset.num_db].astype(np.float32)
        gt = self.dataset.getPositives()

        with open(feature_path, 'wb') as f:
            pickle.dump(
                {'qFeat': qFeat, 'dbFeat': dbFeat, 'gt': gt},
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        print('features saved at:', feature_path)

    def _load_feature(self, feature_path):
        with open(feature_path, 'rb') as f:
            feature = pickle.load(f)
        qFeat = feature['qFeat']
        dbFeat = feature['dbFeat']
        gt = feature['gt']
        return qFeat, dbFeat, gt

    def _search(self, feature_path):
        qFeat, dbFeat, gt = self._load_feature(feature_path)
        faiss_index = faiss.IndexFlatL2(self.output_dim)
        faiss_index.add(dbFeat)
        dists, preds = faiss_index.search(qFeat, len(dbFeat))  # ascending for L2
        return qFeat, dbFeat, gt, dists, preds

    def get_recall_at_n(self, feature_path, n_values=(1, 5, 10, 20)):
        _, _, gt, _, preds = self._search(feature_path)

        correct_at_n = np.zeros(len(n_values), dtype=np.float64)

        print('getting Recall@N ...')
        for qIx, pred in enumerate(preds):
            for i, n in enumerate(n_values):
                if np.any(np.in1d(pred[:n], gt[qIx])):
                    correct_at_n[i:] += 1
                    break

        recall_at_n = correct_at_n / self.dataset.num_query * 100.0
        recalls = {}
        for i, n in enumerate(n_values):
            recalls[n] = float(recall_at_n[i])

        print('[Recall]')
        for n in n_values:
            print(f'Recall@{n}: {recalls[n]:.2f}\t', end='')
        print()
        return recalls

    def get_pr_rows(self, feature_path, num_thresholds=1000):
        _, _, gt, dists, preds = self._search(feature_path)

        top1_dists = dists[:, 0]
        top1_preds = preds[:, 0]

        dmin = float(top1_dists.min())
        dmax = float(top1_dists.max())
        thresholds = np.linspace(dmin, dmax + 1e-6, num_thresholds)

        rows = []
        print('getting PR curve ...')
        for th in tqdm(thresholds, ncols=80):
            tp = fp = fn = tn = 0

            for qIx in range(len(top1_dists)):
                pred_positive = top1_dists[qIx] <= th
                gt_positive = np.any(np.in1d(top1_preds[qIx], gt[qIx]))

                if pred_positive and gt_positive:
                    tp += 1
                elif pred_positive and (not gt_positive):
                    fp += 1
                elif (not pred_positive) and gt_positive:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            rows.append({
                'threshold': float(th),
                'precision': float(precision),
                'recall': float(recall),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
            })

        return rows

    def get_roc_rows(self, feature_path, num_thresholds=1000):
        _, _, gt, dists, preds = self._search(feature_path)

        top1_dists = dists[:, 0]
        top1_preds = preds[:, 0]

        dmin = float(top1_dists.min())
        dmax = float(top1_dists.max())
        thresholds = np.linspace(dmin, dmax + 1e-6, num_thresholds)

        rows = []
        print('getting ROC curve ...')
        for th in tqdm(thresholds, ncols=80):
            tp = fp = fn = tn = 0

            for qIx in range(len(top1_dists)):
                pred_positive = top1_dists[qIx] <= th
                gt_positive = np.any(np.in1d(top1_preds[qIx], gt[qIx]))

                if pred_positive and gt_positive:
                    tp += 1
                elif pred_positive and (not gt_positive):
                    fp += 1
                elif (not pred_positive) and gt_positive:
                    fn += 1
                else:
                    tn += 1

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            rows.append({
                'threshold': float(th),
                'tpr': float(tpr),
                'fpr': float(fpr),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
            })

        return rows

    @staticmethod
    def get_max_f1_from_pr_rows(pr_rows):
        best_f1 = -1.0
        best_row = None

        for row in pr_rows:
            p = row['precision']
            r = row['recall']
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_row = dict(row)
                best_row['f1'] = float(f1)

        return float(best_f1), best_row

    def evaluate_all(self, ckpt_path, feature_path, pr_csv_path, roc_csv_path):
        self.extract_features(ckpt_path, feature_path)

        recalls = self.get_recall_at_n(feature_path, n_values=(1, 5, 10, 20))
        pr_rows = self.get_pr_rows(feature_path, num_thresholds=1000)
        roc_rows = self.get_roc_rows(feature_path, num_thresholds=1000)
        max_f1, best_pr_row = self.get_max_f1_from_pr_rows(pr_rows)

        save_csv(
            pr_rows,
            pr_csv_path,
            fieldnames=['threshold', 'precision', 'recall', 'tp', 'fp', 'fn', 'tn']
        )

        save_csv(
            roc_rows,
            roc_csv_path,
            fieldnames=['threshold', 'tpr', 'fpr', 'tp', 'fp', 'fn', 'tn']
        )

        print('[Max F1]')
        print(f'max F1: {max_f1:.6f}')
        print(f'best threshold: {best_pr_row["threshold"]:.6f}')
        print(f'best precision: {best_pr_row["precision"]:.6f}')
        print(f'best recall: {best_pr_row["recall"]:.6f}')

        result = {
            'recall@1': recalls[1],
            'recall@5': recalls[5],
            'recall@10': recalls[10],
            'recall@20': recalls[20],
            'max_f1': max_f1,
            'best_threshold': best_pr_row['threshold'],
            'best_precision': best_pr_row['precision'],
            'best_recall': best_pr_row['recall'],
            'num_db': int(self.dataset.num_db),
            'num_query': int(self.dataset.num_query),
        }
        return result


def build_sq_query_all(dataroot):
    sq_all_path = os.path.join(dataroot, 'sq_query_all.npy')
    if os.path.exists(sq_all_path):
        return sq_all_path

    sq_train_path = os.path.join(dataroot, 'sq_train_query.npy')
    sq_test_path = os.path.join(dataroot, 'sq_test_query.npy')
    check_path(sq_train_path, sq_test_path)

    sq_train = np.load(sq_train_path)
    sq_test = np.load(sq_test_path)
    sq_all = np.concatenate([sq_train, sq_test], axis=0)
    np.save(sq_all_path, sq_all)
    print('generated:', sq_all_path)
    return sq_all_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate DoFu-Net on nuScenes')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--checkpoint', required=True, help='Trained DoFu-Net .pth.tar checkpoint')
    parser.add_argument('--splits', nargs='+', choices=['BS', 'SON', 'SHV', 'SQ'], default=['BS', 'SON', 'SHV'])
    args = parser.parse_args()
    cfg = load_config(args.config)

    # ==================== config ====================
    data_root_dir = cfg['data']['data_root_dir']
    info_path = cfg['data']['info_path']   # 这里只作为默认值，不同地点会被覆盖
    resize = cfg['runner']['resize']
    num_workers_test = cfg['runner']['num_workers_test']

    result_dir = cfg['runner']['result_dir']
    check_dir(result_dir)

    # ============== your fixed best model ==============
    weights = os.path.abspath(os.path.expanduser(args.checkpoint))
    check_path(weights)

    # ==================== model ====================
    model = DoFuNet.create(weights=None, visual_backbone=cfg['runner'].get('visual_backbone', 'convnextv2_atto'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_transforms = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ============ prepare SQ full-query file ============
    sq_query_all_path = (build_sq_query_all(data_root_dir) if 'SQ' in args.splits
                         else os.path.join(data_root_dir, 'sq_query_all.npy'))

    # ==================== tests ====================
    tests = [
        {
            'name': 'BS',
            'database_path': os.path.join(data_root_dir, 'bs_db.npy'),
            'query_path': os.path.join(data_root_dir, 'bs_test_query.npy'),
            'info_path': os.path.join(data_root_dir, 'nuscenes_infos-bs.pkl')
        },
        {
            'name': 'SON',
            'database_path': os.path.join(data_root_dir, 'son_db.npy'),
            'query_path': os.path.join(data_root_dir, 'son_query.npy'),
            'info_path': os.path.join(data_root_dir, 'nuscenes_infos-son.pkl')
        },
        {
            'name': 'SHV',
            'database_path': os.path.join(data_root_dir, 'shv_db.npy'),
            'query_path': os.path.join(data_root_dir, 'shv_query.npy'),
            'info_path': os.path.join(data_root_dir, 'nuscenes_infos-shv.pkl')
        },
        {
            'name': 'SQ',
            'database_path': os.path.join(data_root_dir, 'sq_db.npy'),
            'query_path': sq_query_all_path,
            'info_path': os.path.join(data_root_dir, 'nuscenes_infos-sq.pkl')
        },
    ]

    summary_rows = []

    for test_cfg in tests:
        if test_cfg['name'] not in args.splits:
            continue
        name = test_cfg['name']
        database_path = test_cfg['database_path']
        query_path = test_cfg['query_path']
        info_path = test_cfg['info_path']

        print('\n' + '=' * 80)
        print(f'Now testing on {name}')
        print('=' * 80)

        check_path(data_root_dir, database_path, query_path, info_path)

        dataset = DatabaseQueryDataset(
            data_root_dir=data_root_dir,
            database_path=database_path,
            query_path=query_path,
            info_path=info_path,
            transforms=img_transforms,
            nonTrivPosDistThres=cfg['runner']['nonTrivPosDistThres'],
            lidar_view_dir=cfg['data'].get('lidar_view_dir', 'samples/RANGE_DOFU_3VIEW'),
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers_test
        )

        evaluator = UnifiedEvaluator(
            model=model,
            dataset=dataset,
            loader=loader,
            device=device,
            output_dim=256
        )

        place_dir = os.path.join(result_dir, name)
        check_dir(place_dir)

        feature_path = os.path.join(place_dir, f'{name.lower()}_features.pkl')
        pr_csv_path = os.path.join(place_dir, f'{name.lower()}_pr.csv')
        roc_csv_path = os.path.join(place_dir, f'{name.lower()}_roc.csv')

        result = evaluator.evaluate_all(
            ckpt_path=weights,
            feature_path=feature_path,
            pr_csv_path=pr_csv_path,
            roc_csv_path=roc_csv_path
        )

        row = {'split': name}
        row.update(result)
        summary_rows.append(row)

    summary_csv_path = os.path.join(result_dir, 'nuscenes_summary.csv')
    save_csv(
        summary_rows,
        summary_csv_path,
        fieldnames=[
            'split',
            'num_db',
            'num_query',
            'recall@1',
            'recall@5',
            'recall@10',
            'recall@20',
            'max_f1',
            'best_threshold',
            'best_precision',
            'best_recall',
        ]
    )

    print('\nAll done.')
    print('summary saved at:', summary_csv_path)


if __name__ == '__main__':
    main()