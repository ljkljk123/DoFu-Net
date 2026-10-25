"""Minimal AV2S helpers for reproducible robustness evaluations."""
import os
from pathlib import Path
import yaml
RING_CAMERA_ORDER = [
    'ring_front_center','ring_front_left','ring_front_right',
    'ring_side_left','ring_side_right','ring_rear_left','ring_rear_right',
]
def load_config(config_path):
    with open(config_path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)
def check_dir(*dirs):
    for d in dirs:
        if d: os.makedirs(d,exist_ok=True)
def check_path(*paths):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Path not found: {p}')
