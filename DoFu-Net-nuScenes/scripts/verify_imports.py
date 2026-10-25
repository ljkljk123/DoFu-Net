#!/usr/bin/env python3
"""Quick import/config check. Run after installing dependencies."""
from tools.utils import load_config
from modules.DoFuNet import DoFuNet
from dataset.NuScenesDataset import DatabaseQueryDataset, TripletDataset
cfg = load_config('config/config.yaml')
print('Model class:', DoFuNet.__name__)
print('Data root:', cfg['data']['data_root_dir'])
print('LiDAR view folder:', cfg['data']['lidar_view_dir'])
print('Imports OK')
