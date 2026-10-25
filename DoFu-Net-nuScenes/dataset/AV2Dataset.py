"""AV2S database/query adapter for robustness tests.

Uses 7 Ring cameras and the original PY/RY/RP NPZ layout.
For occlusion tests, only queries read `query_range_data_dir`; DB always reads clean range data.
"""
import os
import pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tools.utils_av2 import RING_CAMERA_ORDER

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DatabaseQueryDataset(Dataset):
    def __init__(self, data_root_dir, database_path, query_path, info_path,
                 transforms, nonTrivPosDistThres, range_data_dir=None,
                 query_range_data_dir=None, camera_channels=None):
        self.dataroot = str(data_root_dir)
        with open(info_path,'rb') as f:
            self.infos = pickle.load(f)
        db = np.load(database_path)
        q = np.load(query_path)
        self.dataset = np.concatenate((db,q), axis=0)
        self.num_db, self.num_query = len(db), len(q)
        self.positives = None
        self.nonTrivPosDistThres = float(nonTrivPosDistThres)
        self.img_transforms = transforms
        self.range_data_dir = str(range_data_dir or os.path.join(self.dataroot, 'samples','RANGE_DOFU_3VIEW'))
        self.query_range_data_dir = str(query_range_data_dir) if query_range_data_dir else None
        self.camera_channels = list(camera_channels) if camera_channels is not None else list(RING_CAMERA_ORDER)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        idx = int(self.dataset[item][0])
        cams = []
        for channel in self.camera_channels:
            rel = self.infos[idx]['cam_infos'][channel]['filename']
            path = os.path.join(self.dataroot,rel)
            with Image.open(path) as img:
                cams.append(self.img_transforms(img.convert('RGB')))
        is_query = int(item) >= self.num_db
        lidar_dir = self.query_range_data_dir if is_query and self.query_range_data_dir else self.range_data_dir
        rel_lidar = self.infos[idx]['lidar_infos']['LIDAR_TOP']['filename']
        name = os.path.splitext(os.path.basename(rel_lidar))[0]+'.npz'
        with np.load(os.path.join(lidar_dir,name)) as data:
            py,ry,rp=[data[k] for k in ('py','ry','rp')]
        H,W=64,768
        PAD=-1000.0
        out = np.full((3,H,W),PAD,dtype=np.float32)
        for i, view in enumerate((py,ry,rp)):
            if view.ndim!=2 or view.shape[0]>H or view.shape[1]>W:
                raise ValueError(f'Unexpected {("py","ry","rp")[i]} shape: {view.shape}')
            out[i,:view.shape[0],:view.shape[1]]=view
        return {'camera_feature':torch.stack(cams), 'lidar_feature':torch.from_numpy(out)}

    def getPositives(self):
        if self.positives is None:
            db_xy = np.ascontiguousarray(self.dataset[:self.num_db,1:3])
            q_xy = np.ascontiguousarray(self.dataset[self.num_db:,1:3])
            knn=NearestNeighbors(n_jobs=-1)
            knn.fit(db_xy)
            self.positives=list(knn.radius_neighbors(q_xy, radius=self.nonTrivPosDistThres, return_distance=False))
        return self.positives
