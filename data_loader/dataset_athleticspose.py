import os
import math
import torch
import random
import numpy as np

from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton
from utils.draw import render_pictures

"""
AthleticsPose 骨骼编号（与AP3D完全一致）
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


class DatasetAthleticsPose(Dataset):
    def __init__(self,
                 mode,
                 t_his=15,
                 t_pred=60,
                 actions='all',
                 use_vel=False,
                 data_path=r"data\AthleticsPose",
                 **kwargs):
        self.use_vel = use_vel
        cur_path = os.path.abspath(os.path.dirname(__file__))
        root_path = os.path.dirname(cur_path)
        self.data_path = os.path.join(root_path, data_path)  # 预处理npz文件所在目录
        super().__init__(mode, t_his, t_pred, actions='all', **kwargs)

    def prepare_data(self, **kwargs):
        # 骨骼拓扑与AP3D完全一致
        self.kept_joints = np.arange(17)
        self.skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
            joints_left=[4, 5, 6, 11, 12, 13],
            joints_right=[1, 2, 3, 14, 15, 16]
        )
        self.skeleton.gen_adj_mat()

        npz_file = os.path.join(self.data_path, f"{self.mode}.npz")
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"预处理文件不存在：{npz_file}\n请先运行preprocess_athletics_pose.py生成")

        data = np.load(npz_file, allow_pickle=True)

        # 读取完整序列（对象数组，每个元素是不同长度的(T,17,3)序列）
        raw_sequences = data['trajectories']
        self.subjects = data['subjects']
        self.actions = data['actions']
        self.lengths = data['lengths']
        self.global_max = data['global_max']

        # 处理序列（与AP3D逻辑完全一致）
        self.data = []
        for seq in raw_sequences:
            seq = seq.astype(np.float32)  # 统一为float32

            # 速度特征（可选，与AP3D一致）
            if self.use_vel:
                vel = np.diff(seq, axis=0, prepend=seq[:1])
                seq = np.concatenate([seq, vel], axis=-1)

            # 过滤过短序列（预处理已过滤，此处为双重保险）
            if seq.shape[0] >= self.t_total:
                self.data.append(np.ascontiguousarray(seq))

        print(f"✅ AthleticsPose-{self.mode} 加载完成")
        print(f"   序列数: {len(self.data)} | 总帧数: {sum(self.lengths)}")
        print(f"   全局归一化最大值: {self.global_max:.4f}")

    def sample(self):
        """与AP3D完全一致的随机采样逻辑"""
        seq = random.choice(self.data)
        max_start = seq.shape[0] - self.t_total
        start = random.randint(0, max_start)
        return seq[start:start + self.t_total][None]

    def sampling_generator(self, num_samples=50000, batch_size=256, aug=True):
        """与AP3D完全一致的批量采样生成器"""
        for _ in range(num_samples // batch_size):
            batch = [self.sample() for _ in range(batch_size)]
            batch = np.concatenate(batch, axis=0)
            seq_len = batch.shape[1]

            # 随机掩码（与AP3D一致）
            mask_indices = np.random.randint(int(seq_len * 0.08), int(seq_len * 0.92), int(seq_len * 0.2))
            mask = np.array([i not in mask_indices for i in range(seq_len)], dtype=bool)

            # 数据增强（与AP3D完全一致）
            if aug:
                if np.random.rand() > 0.5:
                    theta = np.random.uniform(0, 2 * np.pi)
                    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    batch[..., :2] = np.matmul(batch[..., :2], rot.T)
                if np.random.uniform() > 0.5:  # x轴镜像
                    batch[..., 0] = - batch[..., 0]
                if np.random.uniform() > 0.5:  # y轴镜像
                    batch[..., 1] = - batch[..., 1]

            yield batch, mask

    def iter_generator(self, step=15):
        """与AP3D完全一致的评估迭代器（步长15）"""
        for seq in self.data:
            for i in range(0, seq.shape[0] - self.t_total + 1, step):
                yield seq[None, i:i + self.t_total]


if __name__ == '__main__':
    """全链路测试（与AP3D测试逻辑完全一致）"""
    import time

    # 测试训练集加载
    start = time.time()
    train_dataset = DatasetAthleticsPose('test', t_his=15, t_pred=60, use_vel=False)
    load_time = time.time() - start
    print(f"\nTrain加载时间: {load_time:.3f}秒")

    # 测试采样速度
    start = time.time()
    for _ in range(1000):
        train_dataset.sample()
    sample_time = (time.time() - start) / 1000
    print(f"单次采样时间: {sample_time * 1000:.3f}毫秒")

    # 测试批量生成器
    generator = train_dataset.sampling_generator(num_samples=512000, batch_size=1025)
    batch, mask = next(generator)
    print(f"批量形状: {batch.shape} | 掩码形状: {mask.shape}")

    # 骨骼长度诊断（与AP3D一致）
    print("\n" + "=" * 70)
    print("骨骼长度方差诊断")
    print("=" * 70)
    generator = train_dataset.iter_generator(step=15)
    all_bone_lengths = []
    parent = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12, 8, 11, 7]

    max_coord = -np.inf
    for data in generator:
        x = torch.from_numpy(data).reshape(-1, 17, 3)
        max_coord = max(max_coord, torch.max(torch.abs(x)))
        for j in range(1, 17):
            vec = x[:, j] - x[:, parent[j]]
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