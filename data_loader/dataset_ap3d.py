import os
import math
import torch
import random
import numpy as np
from sklearn.externals.array_api_compat import torch

from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton
from data_loader.datareader_ap3d import DataReaderAP3D
from utils.draw import render_pictures

"""
AP3D骨骼编号
"Human3.6M":
{
    "0": ["PELVIS"],
    "1": ["RASI", "RPSI"],
    "2": ["RKNE"],
    "3": ["RANK"],
    "4": ["LASI", "LPSI"],
    "5": ["LKNE"],
    "6": ["LANK"],
    "7": ["Xiphoid", "T10"],
    "8": ["CLAV"],
    "9": ["CLAV", "Nasion"],
    "10": ["Vertex", "EARC"],
    "11": ["L_Shoulder Joint"],
    "12": ["LELB"],
    "13": ["LWRT"],
    "14": ["R_Shoulder Joint"],
    "15": ["RELB"],
    "16": ["RWRT"]
}
"""

class DatasetAP3D(Dataset):
    def __init__(self, mode, t_his=15, t_pred=60, actions='all', use_vel=False, **kwargs):
        self.use_vel = use_vel
        super().__init__(mode, t_his, t_pred, actions='all', **kwargs)

    def prepare_data(self, **kwargs):
        self.kept_joints = np.arange(17)
        self.skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
            joints_left=[4, 5, 6, 11, 12, 13],  # 左髋、左膝、左踝、左肩、左肘、左腕
            joints_right=[1, 2, 3, 14, 15, 16]  # 右髋、右膝、右踝、右肩、右肘、右腕
        )
        self.skeleton.gen_adj_mat()

        # 纯List存储所有序列
        self.data = []
        reader = DataReaderAP3D()
        raw_data = reader.get_train_sequences() if self.mode == 'train' else reader.get_test_sequences()

        for seq in raw_data:
            seq = seq[:, self.kept_joints, :].astype(np.float32)
            # seq = seq / 1000.0  # 单位转换
            # seq = seq - seq[:, 0:1, :]  #  根节点归一化

            if self.use_vel:
                vel = np.diff(seq, axis=0, prepend=seq[:1])
                seq = np.concatenate([seq, vel], axis=-1)

            if seq.shape[0] >= self.t_total:
                self.data.append(np.ascontiguousarray(seq))

        print(f"✅ AP3D-{self.mode} 序列数: {len(self.data)} | 纯List存储")

    def sample(self):
        seq = random.choice(self.data)
        max_start = seq.shape[0] - self.t_total
        start = random.randint(0, max_start)
        return seq[start:start + self.t_total][None]

    def sampling_generator(self, num_samples=50000, batch_size=256, aug=True):
        for _ in range(num_samples // batch_size):
            batch = [self.sample() for _ in range(batch_size)]
            batch = np.concatenate(batch, axis=0)
            seq_len = batch.shape[1]
            mask_indices = np.random.randint(int(seq_len * 0.08), int(seq_len * 0.92), int(seq_len * 0.2))
            mask = np.array([i not in mask_indices for i in range(seq_len)], dtype=bool)

            if aug:
                if np.random.rand() > 0.5:
                    theta = np.random.uniform(0, 2 * np.pi)
                    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    batch[..., :2] = np.matmul(batch[..., :2], rot.T)
                if np.random.uniform() > 0.5:  # x-z mirroring
                    batch[..., 0] = - batch[..., 0]
                if np.random.uniform() > 0.5:  # y-z mirroring
                    batch[..., 1] = - batch[..., 1]

            yield batch, mask

    def iter_generator(self, step=15):
        # 🔥 迭代步长统一15
        for seq in self.data:
            for i in range(0, seq.shape[0] - self.t_total + 1, step):
                yield seq[None, i:i + self.t_total]


if __name__ == '__main__':
    """全链路性能测试：Train加载时间 <0.1秒"""
    import time

    # 测试训练模式加载速度
    start = time.time()
    ap3d_train = DatasetAP3D('test', t_his=15, t_pred=60, use_vel=False)
    load_time = time.time() - start
    print(f"Train全链路加载时间: {load_time:.3f}秒")

    # 测试采样速度
    # start = time.time()
    # for _ in range(1000):
    #     ap3d_train.sample()
    # sample_time = (time.time() - start) / 1000
    # print(f"单次采样时间: {sample_time * 1000:.3f}毫秒")
    generator = ap3d_train.iter_generator(step=15)
    sg = ap3d_train.sampling_generator()
    i = 0
    all_bone_lengths = []
    max_v = -np.inf
    for data in generator:
        x = torch.from_numpy(data).reshape(-1, 17, 3)
        max_coord = max(max_v, np.max(np.abs(data)))

        # 计算所有骨骼的长度
        parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12, 8, 11, 7]
        for j in range(1, 17):
            vec = x[:, j] - x[:, parent[j]]
            length = torch.norm(vec, dim=-1)
            all_bone_lengths.append(length)

    all_bone_lengths = torch.cat(all_bone_lengths)
    mean_len = all_bone_lengths.mean().item()
    std_len = all_bone_lengths.std().item()
    print("="*70)
    print("骨骼长度方差诊断")
    print("="*70)
    print(f"平均骨骼长度: {mean_len:.4f}")
    print(f"骨骼长度标准差: {std_len:.4f}")
    print(f"变异系数: {std_len / mean_len * 100:.1f}%")
    print("="*70)
    print(max_coord)

