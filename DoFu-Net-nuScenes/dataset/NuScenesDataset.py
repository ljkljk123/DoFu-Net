import os
import h5py
import torch
import pickle
import numpy as np
from PIL import Image
from PIL import ImageFile
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, default_collate

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    def __init__(self, data_root_dir, info_path, lidar_view_dir='samples/RANGE_DOFU_3VIEW'):
        super().__init__()
        self.infos = self.read_infos(info_path)
        self.dataroot = data_root_dir
        self.lidar_view_dir = lidar_view_dir
        self.img_transforms = None

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def read_infos(self, info_path):
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        return infos

    # 读取相机和激光雷达的实际图像数据
    def load_data(self, index):
        channels = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        #channels = ['CAM_FRONT']
        camera_images = self.load_camera_data(index, channels)
        lidar_images = self.load_lidar_data(index)
        return camera_images, lidar_images

    # # 读取雷达球面映射数据-----对应于Dome的输入就必须进行修改
    def load_lidar_data(self, index):
        lidar_data = self.infos[index]['lidar_infos']['LIDAR_TOP']
        fname = os.path.split(lidar_data['filename'].split('.')[0])[-1] + '.npz'
        # fname = os.path.split(lidar_data['filename'].split('.')[0])[-1] + '.npy' #获取当前雷达帧所对应的npy文件
        # depth_data_path = os.path.join(self.dataroot, 'samples', 'RANGE_DATA', fname) #寻找雷达bin深度文件
        data_path = os.path.join(self.dataroot, self.lidar_view_dir, fname)
        
        if not os.path.exists(data_path):
            raise Exception(f'FileNotFound! {data_path}')
        
        # depth_data = np.load(depth_data_path, allow_pickle=True)
        data = np.load(data_path)
        # print(list(data.keys())) data中的结构为 ['py', 'ry', 'rp', 'pitch_min', 'pitch_max']
        py = data['py']          # (H, W)
        ry = data['ry']
        rp = data['rp']
        # print("py shape:",py.shape)
        # print("ry shape:",ry.shape)
        # print("rp shape:",rp.shape)
        
        """
            torch.from_numpy：将NumPy数组转换为PyTorch张量
            unsqueeze(0)
            作用：在维度0添加一个维度
            效果：从(H, W)变为(1, H, W)
        """
        # ========= 统一目标尺寸 =========
        H_max = 64
        W_max = 768
        PAD_VAL = -1000.0

        # ---------- PY: (32, 768) -> (64, 768) ----------
        py_pad = np.full((H_max, W_max), PAD_VAL, dtype=np.float32)
        py_pad[:py.shape[0], :py.shape[1]] = py

        # ---------- RY: (64, 768) -> (64, 768) ----------
        ry_pad = np.full((H_max, W_max), PAD_VAL, dtype=np.float32)
        ry_pad[:ry.shape[0], :ry.shape[1]] = ry

        # ---------- RP: (64, 32) -> (64, 768) ----------
        rp_pad = np.full((H_max, W_max), PAD_VAL, dtype=np.float32)
        rp_pad[:rp.shape[0], :rp.shape[1]] = rp

        # 转 tensor，加 channel 维
        # py_t = torch.from_numpy(py_pad).unsqueeze(0)  # (1, 64, 768)
        # ry_t = torch.from_numpy(ry_pad).unsqueeze(0)  # (1, 64, 768)
        # rp_t = torch.from_numpy(rp_pad).unsqueeze(0)  # (1, 64, 768)
        py_t = torch.from_numpy(py_pad).float()  # (1, 64, 768)
        ry_t = torch.from_numpy(ry_pad).float()  # (1, 64, 768)
        rp_t = torch.from_numpy(rp_pad).float()  # (1, 64, 768)
        lidar = torch.stack([py_t, ry_t, rp_t], dim = 0) # 变成了(3,H,W)
        return lidar
    
    
    
    # 读取相机映射数据
    def load_camera_data(self, index, channels):
        imgs = []
        # 需要循环6个相机
        for channel in channels: 
            cam_data = self.infos[index]['cam_infos'][channel]
            filename = cam_data['filename']
            img_path = os.path.join(self.dataroot, filename) #获取图像路径
            
            if not os.path.exists(img_path):
                raise Exception(f'FileNotFound! {img_path}')
            
            img = Image.open(img_path)
            img_tensor = self.img_transforms(img)
            imgs.append(img_tensor)

        """
        堆叠前：
        torch.Size([3, 256, 704])一共三张这样的图片
        堆叠后：
         torch.Size([3, 3, 256, 704])一张4维数据
        """
        imgs = torch.stack(imgs) #torch.stack堆叠图像
        return imgs

# 三元组损失训练
class TripletDataset(BaseDataset):
    def __init__(self, data_root_dir, # 数据集根目录
                 database_path,  # 数据库文件路径
                 query_path,  # 训练查询集路径
                 info_path,  # 元信息文件路径
                 cache_dir,  # 缓存目录('runs/cache/')
                 img_transforms, # 图像预处理流水线
                 nNeg,  # 每个锚点的负样本数量(6个)
                 nNegSample,  # 负样本候选池大小(1000个)
                 nonTrivPosDistThres,  # 困难正样本距离阈值(9米)
                 posDistThr, # 正样本距离阈值(18米)
                 margin, # 三元组损失边界值(0.5)       正样本应该比负样本至少近margin个单位
                 lidar_view_dir='samples/RANGE_DOFU_3VIEW'
                 ):
        
        super().__init__(data_root_dir, info_path, lidar_view_dir)
        
        self.data_base = np.load(database_path)
        self.nNeg = nNeg  # number of negatives used for training
        self.nNegSample = nNegSample  # number of negatives to randomly sample
        self.nonTrivPosDistThres = nonTrivPosDistThres
        self.posDistThr = posDistThr
        self.margin = margin
        self.queries = np.load(query_path)
        self.img_transforms = img_transforms

        knn = NearestNeighbors()
        knn.fit(self.data_base[:, 1:]) # 取所有行的第1列到最后（x, y坐标），忽略索引列
        
        #有点像半径滤波 寻找困难正样本距离阈值(9米)内的所有数据
        self.nontrivial_positives = list(knn.radius_neighbors(self.queries[:, 1:], #查询集的训练集的所有数据的（x, y）坐标
                                                              radius=self.nonTrivPosDistThres, #9m内
                                                              return_distance=False))
        """
            用法：enumerate(iterable, start=0)
            1、iterable：要枚举的可迭代对象（如列表、元组等）
            2、start：计数器的起始值（默认为0）
            # i 是索引（从0开始）
            # posi 是 self.nontrivial_positives 列表中对应索引的元素
            
        """
        for i, posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)  #从小到大
            
            
        # 查找每个查询点半径18米内的所有正样本（包括困难和简单正样本）
        potential_positives = list(knn.radius_neighbors(self.queries[:, 1:], 
                                                        radius=self.posDistThr, # 18m内
                                                        return_distance=False))
        self.potential_negatives = []
        """
        np.setdiff1d(A, B)：在A中但不在B中的元素
        A：所有数据库点索引[0, 1, 2, ..., N-1]
        B：当前查询的正样本索引
        结果：当前查询的潜在负样本（距离>18米的所有点）
        """
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.data_base.shape[0]), pos, assume_unique=True))
        """
            np.setdiff1d(A, B)：在A中但不在B中的元素
            A：所有数据库点索引[0, 1, 2, ..., N-1]
            B：当前查询的正样本索引
            结果：当前查询的潜在负样本（距离>18米的所有点）
        """
        # filepath of HDF5 containing feature vectors for this epoch
        self.cache = os.path.join(cache_dir, 'feat_cache.hdf5')  # 特征缓存文件路径
        self.negCache = [np.empty((0,)) for _ in range(len(self.queries))] # 负样本缓存

# 核心方法：__getitem__（获取单个训练样本）
    def __getitem__(self, index):  # index是查询的索引
        # with执行完毕后会自动关闭数据流
        with h5py.File(self.cache, mode='r') as h5:  # 打开特征缓存文件
            h5feat = h5.get('features') # 获取特征数据
            qOffset = len(self.data_base) # 数据库点数量（特征文件中数据库点在前）
            
            qFeat = h5feat[index + qOffset] # 查询点的特征（在数据库点之后）
            qFeat = torch.tensor(qFeat) # 转换为PyTorch张量
            
            posFeat = h5feat[self.nontrivial_positives[index]] # 获取所有困难正样本特征
            posFeat = torch.tensor(posFeat) # 将所有困难正样本特征转换为张量形式
            
            dist = torch.norm(qFeat - posFeat, dim=1) # 计算查询与每个正样本的距离
            result = dist.topk(1, largest=False)  # 找到距离最小的正样本（最困难）
            posdist, posidx = result.values, result.indices #距离最小的正样本和样本值
            
            # 最困难正样本的索引     .item() - 转换为Python标量
            posIndex = self.nontrivial_positives[index][posidx].item() 
            
            #负样本缓存处理
            if self.negCache[index].ndim == 0:
                self.negCache[index] = self.negCache[index].reshape(1)

            # randomly select from potential negatives
            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)  # 随机选择1000个候选负样本
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]).astype(int)) # 合并缓存和新样本
            negFeat = h5feat[negSample] # 获取候选负样本特征
            negFeat = torch.tensor(negFeat)
            dist = torch.norm(qFeat - negFeat, dim=1) # 计算与每个负样本的距离
            result = dist.topk(self.nNeg * 10, largest=False) # 找到最接近的60个负样本，其中self.nNeg = 6
            negdist, negidx = result.values, result.indices # 返回负样本的距离值和索引
            
            #  找出违反边界条件的负样本
            vilatingNeg = negdist.numpy() < posdist.numpy() + self.margin
            if np.sum(vilatingNeg) < 1: # 如果没有违反边界的负样本
                return None  # 跳过这个样本

            negidx = negidx[vilatingNeg][:self.nNeg] # 选择前6个困难负样本
            nNeg = len(negidx) # 实际选择的负样本数量
            negIndex = negSample[negidx].astype(int) # 负样本的数据库索引
            self.negCache[index] = negIndex  # 更新负样本缓存
            # with执行完毕，数据流自动关闭
        
        # 加载图像数据
        query_idx = int(self.queries[index][0]) # 查询点在查询训练集中的索引
        pos_idx = int(self.data_base[posIndex][0])  # 正样本点在原始数据中的索引

        #分别加在查询点和正样本点的激光和相机数据
        q_camera_images, q_lidar_image = self.load_data(query_idx)
        pos_camera_images, pos_lidar_image = self.load_data(pos_idx)

        if self.data_base[negIndex].ndim == 1:
            negIndex = np.array([negIndex]) # 确保negIndex是数组
            
        neg_idx = self.data_base[negIndex][:, 0].astype(int) # 负样本的原始索引
        camera_images, lidar_images = [], []
        
        for i in range(len(neg_idx)):
            neg_camera_images, neg_lidar_image = self.load_data(neg_idx[i]) #读取负样本对应的激光和图像数据
            camera_images.append(neg_camera_images)
            lidar_images.append(neg_lidar_image)

        # 组织返回数据
        camera_images.extend([pos_camera_images, q_camera_images])
        lidar_images.extend([pos_lidar_image, q_lidar_image])

        res_dict = dict({'camera_feature': torch.stack(camera_images),
                         'lidar_feature': torch.stack(lidar_images)})
        return res_dict, nNeg   # 返回数据字典和负样本数量

    def __len__(self):
        return len(self.queries)

    def read_infos(self, infos_path):
        with open(infos_path, 'rb') as f:
            infos = pickle.load(f)
        return infos


# 自定义的批次组装函数
def collate_fn(batch):
    # filter(function, iterable) 过滤出满足条件的元素   等价于：[x for x in batch if x is not None]
    batch = list(filter(lambda x: x is not None, batch)) #移除批次中的None值
    # 边界情况处理：如果整个批次都是None，返回None
    if len(batch) == 0:
        return None, None
    # 分离输入数据和负样本数量
    input_dict, nNeg = zip(*batch)
    #default_collate：PyTorch内置的默认批次组装函数
    #作用：将多个样本的数据字典组装成批次张量
    return default_collate(input_dict), list(nNeg)



class DatabaseQueryDataset(BaseDataset):
    def __init__(self, data_root_dir, #数据集根目录
                 database_path, # 数据库路径
                 query_path, #查询集路径
                 info_path, # 元数据路径
                 transforms,  #图像预处理流水线
                 nonTrivPosDistThres, #困难正样本距离阈值(9米)
                 lidar_view_dir='samples/RANGE_DOFU_3VIEW'
                 ):
        
        super().__init__(data_root_dir, info_path, lidar_view_dir)
        data_base = np.load(database_path) # 获取数据库数据
        query = np.load(query_path) #获取查询集数据
        # 合并数据库和查询  self.dataset = [数据库点1, 数据库点2, ..., 数据库点N, 查询点1, 查询点2, ..., 查询点M]
        self.dataset = np.concatenate((data_base, query), axis=0)
        self.num_db = len(data_base) # 数据库点数量
        
        self.num_query = len(query) # 查询点数量
        #  正样本缓存（延迟计算）、距离缓存、特征缓存初始化
        self.positives = None
        self.distances = None
        self.cache = None
        
        self.nonTrivPosDistThres = nonTrivPosDistThres # 正样本阈值
        self.img_transforms = transforms # 图像预处理

    # 获取单个样本
    def __getitem__(self, item):
        index = int(self.dataset[item][0]) # 获取原始数据索引
        #获取该原始数据的相机和雷达图像
        camera_images, lidar_image = self.load_data(index) 
        # 将结果组合为字典准备输出
        res = dict({'camera_feature': camera_images,
                    'lidar_feature': lidar_image})
        return res

    def __len__(self):
        return len(self.dataset)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            dataset = np.ascontiguousarray(self.dataset[:self.num_db, 1:])
            knn.fit(dataset)
            self.positives = list(knn.radius_neighbors(self.dataset[self.num_db:, 1:], radius=self.nonTrivPosDistThres,
                                                       return_distance=False))
        return self.positives
