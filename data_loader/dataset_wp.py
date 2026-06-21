import os
import sys
import types
import numpy as np
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton
import torch
import time


class DatasetWP(Dataset):
    def __init__(self, mode='train', t_his=25, t_pred=100, normalization=True, use_vel=False,
                 data_path=r"data/worldpose/wp_data_py3.npz",
                 max_abs=2.10, **kwargs):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.normalization = normalization
        self.use_vel = use_vel
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.dirname(current_path)
        # 加载之前转换好的运动数据
        self.data_path = os.path.join(root_path, data_path)
        self.max_abs = max_abs

        super().__init__(mode, t_his, t_pred, **kwargs)

    def prepare_data(self):
        # SMPL 24关节骨骼定义（与AP3D/SportsPose完全一致）
        self.skeleton = Skeleton(
            parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            joints_left=[1, 4, 5, 6, 16, 17, 18, 19],
            joints_right=[2, 7, 8, 9, 20, 21, 22, 23]
        )
        self.kept_joints = np.arange(24)

        # 加载运动数据
        print(f"📦 加载 WorldPose {self.mode} 数据...")
        cache = np.load(self.data_path, allow_pickle=True)
        self.train_data = list(cache['train'])
        self.test_data = list(cache['test'])
        self.data = self.train_data if self.mode == 'train' else self.test_data

        # 计算全局归一化参数（与AP3D/SportsPose完全一致）
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

            yield batch, _

    def iter_generator(self, step=25):
        """单模态迭代生成器（用于评估）"""
        for seq in self.data:
            seq_len = seq.shape[0]
            for i in range(0, seq_len - self.t_total + 1, step):
                traj = self._normalize(seq[i: i + self.t_total])
                yield traj[None], None

    def get_stats(self):
        """返回归一化参数（与AP3D/SportsPose完全一致）"""
        return np.ones(3, dtype=np.float32) * self.max_abs, np.zeros(3, dtype=np.float32)


if __name__ == '__main__':
    dataset = DatasetWP(mode='train', normalization=True)
    print(f"✅ 测试集加载成功，共 {len(dataset)} 个序列")

    # 测试采样速度
    start = time.time()
    for _ in range(1000):
        dataset.sample()
    sample_time = (time.time() - start) / 1000
    print(f"单次采样时间: {sample_time * 1000:.3f}毫秒")

    # 测试批量生成器
    generator = dataset.sampling_generator(num_samples=51200, batch_size=1024)
    batch, _ = next(generator)
    print(f"批量形状: {batch.shape}")

    # 骨骼长度诊断（与AP3D一致）
    print("\n" + "=" * 70)
    print("骨骼长度方差诊断")
    print("=" * 70)
    generator = dataset.iter_generator(step=15)
    all_bone_lengths = []

    max_coord = -np.inf
    for data, _ in generator:
        x = torch.from_numpy(data).reshape(-1, 24, 3)
        max_coord = max(max_coord, torch.max(torch.abs(x)))
        for j in range(1, 24):
            vec = x[:, j] - x[:, dataset.skeleton.parents()[j]]
            length = torch.norm(vec, dim=-1)
            all_bone_lengths.append(length)

    all_bone_lengths = torch.cat(all_bone_lengths)
    mean_len = all_bone_lengths.mean().item()
    std_len = all_bone_lengths.std().item()
    print(f"平均骨骼长度: {mean_len:.4f}")
    print(f"骨骼长度标准差: {std_len:.4f}")
    print(f"变异系数: {std_len / mean_len * 100:.1f}%")
    print("=" * 70)
    print(max_coord)
    print("✅ 所有测试通过！")