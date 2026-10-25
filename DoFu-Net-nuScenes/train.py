import torch
from tools.runner import Trainer
from tools.utils import load_config, check_path, check_dir
from modules.DoFuNet import DoFuNet
import os
import argparse
from torch.utils.data import DataLoader
from dataset.NuScenesDataset import TripletDataset, DatabaseQueryDataset, collate_fn
from torchvision.transforms import transforms


def main():
    parser = argparse.ArgumentParser(description='Train DoFu-Net on nuScenes')
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)

    # ====================parse config====================
    data_root_dir = cfg['data']['data_root_dir']
    database_path = cfg['data']['database_path']
    train_query_path = cfg['data']['train_query_path']
    test_query_path = cfg['data']['test_query_path']
    val_query_path = cfg['data']['val_query_path']
    info_path = cfg['data']['info_path']

    nonTrivPosDistThres = cfg['runner']['nonTrivPosDistThres']
    posDistThr = cfg['runner']['posDistThr']
    nNeg = cfg['runner']['nNeg']
    nNegSample = cfg['runner']['nNegSample']
    margin = cfg['runner']['margin']
    resize = cfg['runner']['resize']
    lr = cfg['runner']['lr']
    step_size = cfg['runner']['step_size']
    gamma = cfg['runner']['gamma']
    num_epochs = cfg['runner']['num_epochs']
    num_workers_train = cfg['runner']['num_workers_train']
    num_workers_test = cfg['runner']['num_workers_test']
    resume_path = cfg['runner']['resume_path']
    log = cfg['runner']['log']
    resume_scheduler = cfg['runner']['resume_scheduler']

    # Speed knobs: how often to rebuild cache / run validation
    cache_every = cfg['runner'].get('cache_every', 5)
    val_every = cfg['runner'].get('val_every', 1)

    ckpt_dir = cfg['runner']['ckpt_dir']
    result_dir = cfg['runner']['result_dir']
    cache_dir = cfg['runner']['cache_dir']
    log_dir = cfg['runner']['log_dir']

    # ====================check dirs and paths====================
    check_path(data_root_dir, database_path, train_query_path, test_query_path, val_query_path, info_path)
    check_dir(ckpt_dir, result_dir, cache_dir, log_dir)

    # ==========================dataset===========================

    img_transforms = transforms.Compose([transforms.Resize(resize),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                         ])

    train_set = TripletDataset(data_root_dir, database_path, train_query_path, info_path, cache_dir,
                               img_transforms, nNeg, nNegSample, nonTrivPosDistThres, posDistThr, margin,
                                lidar_view_dir=cfg['data'].get('lidar_view_dir', 'samples/RANGE_DOFU_3VIEW'))

    # DataLoader speed options:
    # - pin_memory speeds host->GPU transfer
    # - prefetch_factor only valid when num_workers > 0
    if num_workers_train > 0:
        train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, collate_fn=collate_fn,
                                  num_workers=num_workers_train, pin_memory=True,
                                  prefetch_factor=2, persistent_workers=False)
    else:
        train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, collate_fn=collate_fn,
                                  num_workers=0, pin_memory=True)

    whole_train_set = DatabaseQueryDataset(data_root_dir, database_path, train_query_path, info_path,
                                           img_transforms, nonTrivPosDistThres,
                                            lidar_view_dir=cfg['data'].get('lidar_view_dir', 'samples/RANGE_DOFU_3VIEW'))

    whole_train_loader = DataLoader(dataset=whole_train_set, batch_size=1, shuffle=False,
                                    num_workers=num_workers_test)
    whole_val_set = DatabaseQueryDataset(data_root_dir, database_path, val_query_path, info_path,
                                         img_transforms, nonTrivPosDistThres,
                                            lidar_view_dir=cfg['data'].get('lidar_view_dir', 'samples/RANGE_DOFU_3VIEW'))
    whole_val_loader = DataLoader(dataset=whole_val_set, batch_size=1, shuffle=False,
                                  num_workers=num_workers_test)

    # ConvNeXt V2 tiny + same PR head as the current ResNet18 baseline
    # Set path either in config.yaml: runner.convnextv2_ckpt_path
    # or via environment variable: CONVNEXTV2_CKPT_PATH
    visual_backbone = cfg['runner'].get('visual_backbone', 'convnextv2_atto')
    convnextv2_ckpt_path = cfg['runner'].get('convnextv2_ckpt_path', '')
    if not convnextv2_ckpt_path:
        convnextv2_ckpt_path = os.environ.get('CONVNEXTV2_CKPT_PATH', '')

    if visual_backbone.startswith('convnextv2') and (not convnextv2_ckpt_path or not os.path.isfile(convnextv2_ckpt_path)):
        raise ValueError(
            'ConvNeXt V2 local checkpoint path is required. '
            'Please set runner.convnextv2_ckpt_path in config.yaml or export CONVNEXTV2_CKPT_PATH.'
        )

    model = DoFuNet.create(
        weights=None,
        visual_backbone=visual_backbone,
        convnext_local_ckpt_path=convnextv2_ckpt_path,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, train_loader, whole_train_loader, whole_val_set, whole_val_loader, device,
                      num_epochs, resume_path, log, log_dir, ckpt_dir, cache_dir,
                      resume_scheduler, lr, step_size, gamma, margin,
                      cache_every=cache_every, val_every=val_every)
    trainer.train()


if __name__ == '__main__':
    main()
