import os
import sys
import types
import numpy as np
from sympy.printing.pretty.pretty_symbology import root

from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton



class DatasetWP(Dataset):
    """
    WorldPose 单模态数据集
    100% 对齐 DatasetAP3D / DatasetSP 接口
    """

    def __init__(self, mode='train', t_his=25, t_pred=100, normalization=True, use_vel=False,
                 data_path=r"data/worldpose/wp_data_py3.npz",
                 max_abs=2.10):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.normalization = normalization
        self.use_vel = use_vel
        # current_path = os.path.dirname(os.path.abspath(__file__))
        # root_path = os.path.dirname(current_path)
        root_path = r"/work7/y_zhou/HumanMAC"
        # 加载之前转换好的运动数据
        self.data_path = os.path.join(root_path, data_path)
        self.max_abs = max_abs
        self._prepare_data()

    def _prepare_data(self):
        # SMPL 24关节骨骼定义
        self.skeleton = Skeleton(
            parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            joints_left=[1, 4, 7, 10, 13, 16, 18, 20, 22],
            joints_right=[2, 5, 8, 11, 14, 17, 19, 21, 23],
        )
        self.kept_joints = np.arange(24)

        # 加载运动数据
        print(f"📦 加载 WorldPose {self.mode} 数据...")
        cache = np.load(self.data_path, allow_pickle=True)
        self.train_data = list(cache['train'])
        self.test_data = list(cache['test'])
        self.data = self.train_data if self.mode == 'train' else self.test_data

        if self.normalization and self.max_abs is None:
            print("计算全局归一化参数...")
            all_root_centered = []
            for seq in self.data:
                root = seq[:, 0:1, :]
                root_centered = seq - root
                all_root_centered.append(root_centered)

            all_data = np.concatenate(all_root_centered, axis=0)
            self.max_abs = np.max(np.abs(all_data))
            if self.max_abs < 1e-8:
                self.max_abs = 1.0
            print(f"归一化参数 max_abs: {self.max_abs:.4f}")

    def _normalize(self, seq):
        """标准归一化流程（与AP3D/SportsPose完全一致）"""
        # 1. 根节点归零
        root = seq[:, 0:1, :]
        seq = seq - root
        # 2. Max归一化至[-1, 1]
        if self.normalization:
            seq = seq / self.max_abs
        return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self._normalize(self.data[idx])

        if self.use_vel:
            vel = np.zeros_like(seq)
            vel[1:] = seq[1:] - seq[:-1]
            seq = np.concatenate([seq, vel], axis=-1)

        return seq

    def sample(self):
        """单模态采样（与AP3D/SportsPose完全一致）"""
        while True:
            seq_idx = np.random.randint(len(self.data))
            seq = self.data[seq_idx]
            if len(seq) >= self.t_total:
                break
        fr_start = np.random.randint(len(seq) - self.t_total + 1)
        traj = self._normalize(seq[fr_start: fr_start + self.t_total])
        return traj[None]

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        """单模态采样生成器"""
        for _ in range(num_samples // batch_size):
            batch = []
            for _ in range(batch_size):
                seq = self.sample()
                batch.append(seq)
            batch = np.concatenate(batch, axis=0)

            if aug:
                if np.random.uniform() > 0.5:  # x-y rotating
                    theta = np.random.uniform(0, 2 * np.pi)
                    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    rotate_xy = np.matmul(batch.transpose([0, 2, 1, 3])[..., 0:2], rotate_matrix)
                    batch[..., 0:2] = rotate_xy.transpose([0, 2, 1, 3])
                    del theta, rotate_matrix, rotate_xy
                if np.random.uniform() > 0.5:  # x-z mirroring
                    batch[..., 0] = - batch[..., 0]
                if np.random.uniform() > 0.5:  # y-z mirroring
                    batch[..., 1] = - batch[..., 1]

            yield batch

    def iter_generator(self, step=25):
        """单模态迭代生成器（用于评估）"""
        for seq in self.data:
            seq_len = seq.shape[0]
            for i in range(0, seq_len - self.t_total + 1, step):
                traj = self._normalize(seq[i: i + self.t_total])
                yield traj[None]

    def get_stats(self):
        """返回归一化参数（与AP3D/SportsPose完全一致）"""
        return np.ones(3, dtype=np.float32) * self.max_abs, np.zeros(3, dtype=np.float32)

if __name__ == '__main__':
    dataset = DatasetWP(mode='test', normalization=True)
    print(f"✅ 测试集加载成功，共 {len(dataset)} 个序列")

    # 测试采样
    traj, _ = dataset.sample()
    print(f"✅ 采样成功，轨迹形状：{traj.shape}")  # 应该输出 (1, 75, 24, 3)

    # 测试迭代器
    for i, (traj, _) in enumerate(dataset.iter_generator(step=15)):
        print(f"第 {i} 个样本形状：{traj.shape}")
        if i == 3:
            break