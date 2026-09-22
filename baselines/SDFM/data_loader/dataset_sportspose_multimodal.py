import os
import random
import numpy as np
from data_loader.dataset import Dataset
from data_loader.dataset_sportspose import DatasetSP
from data_loader.skeleton import Skeleton  # 导入你的Skeleton类


class DatasetSP_multi(Dataset):
    def __init__(self, mode, t_his=45, t_pred=180, actions='all', **kwargs):
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.dirname(current_path)
        if 'multimodal_path' in kwargs.keys() and kwargs['multimodal_path'] is not None:
            self.multimodal_path = kwargs['multimodal_path']
            self.multimodal_path = os.path.join(root_path, self.multimodal_path)
        else:
            self.multimodal_path = None

        if 'data_candi_path' in kwargs.keys() and kwargs['data_candi_path'] is not None:
            self.data_candi_path = kwargs['data_candi_path']
            self.data_candi_path = os.path.join(root_path, self.data_candi_path)
        else:
            self.data_candi_path = None
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.prepare_data()

    def prepare_data(self):

        self.subjects = ['Virtual/Test'] if self.mode == 'test' else ['Virtual/Train']

        # SportsPose 骨骼
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 0, 16, 17, 18, 19],
                                 joints_left=[5, 4, 3, 2, 11, 12, 13, 14],
                                 joints_right=[10, 9, 8, 7, 19, 18, 17, 16])
        self.kept_joints = np.arange(21)
        self.process_data()

    def process_data(self):
        from data_loader.dataset_sportspose import DatasetSP
        dataset = DatasetSP(mode=self.mode, t_his=self.t_his, t_pred=self.t_pred, normalization=True)
        seqs = [seq.astype(np.float32) for seq in dataset.data]
        self.data = {self.subjects[0]: {'all_actions': np.concatenate(seqs, axis=0).astype(np.float32)}}

        # ===================== 加载新的多模态真值 =====================
        if self.multimodal_path is None:
            current_path = os.path.dirname(os.path.abspath(__file__))
            root_path = os.path.dirname(current_path)
            multimodal_path = os.path.join(root_path,
                                           'data/sportspose/multimodal/t_his45_1_thre0.500_t_pred180_thre0.010_index_filterd.npz')
            candi_path = os.path.join(root_path,
                                      'data/sportspose/multimodal/data_candi_t_his45_t_pred180_skiprate45.npz')
            self.data_multimodal = np.load( multimodal_path, allow_pickle=True)['data_multimodal'].item()
            self.data_candi = np.load(candi_path, allow_pickle=True)['data_candidate.npy'].astype(np.float32)
        else:
            self.data_multimodal = np.load(self.multimodal_path, allow_pickle=True)['data_multimodal'].item()
            self.data_candi = np.load(self.data_candi_path, allow_pickle=True)['data_candidate.npy'].astype(np.float32)

        self.multi_dict = self.data_multimodal['all_subjects']['all_actions']
        self.max_valid_idx = max(self.multi_dict.keys()) if self.multi_dict else -1

    def iter_generator(self, step=45):
        global_idx = 0
        seq = self.data[self.subjects[0]]['all_actions']
        max_start = len(seq) - self.t_total

        for i in range(0, max_start + 1, step):
            if global_idx > self.max_valid_idx:
                print(f"\n✅ 到达最大有效索引 {self.max_valid_idx}，停止迭代")
                return

            # 主序列（完整轨迹：历史+未来）
            traj = seq[None, i:i + self.t_total].astype(np.float32)

            # 安全获取多模态索引
            key = np.int64(global_idx)
            idx = self.multi_dict.get(key, np.array([], dtype=np.int32))

            # 空索引兜底：用自身作为唯一真值
            if len(idx) == 0:
                multi_traj = traj.copy()
            else:
                # 从候选库取数据 + 强制对齐历史帧（标准流程）
                multi_traj = self.data_candi[idx]
                multi_traj[:, :self.t_his] = traj[:, :self.t_his]

            global_idx += 1
            yield traj, multi_traj

    def sample(self, n_modality=5):
        seq = self.data[self.subjects[0]]['all_actions']
        fr_start = np.random.randint(len(seq) - self.t_total)
        return seq[None, fr_start: fr_start + self.t_total].astype(np.float32), None


if __name__ == '__main__':
    np.random.seed(42)
    # ===================== 测试配置（请修改为你的文件路径） =====================
    CONFIG = {
        "mode": "valid",
        "t_his": 45,
        "t_pred": 180,
    }

    # ===================== 1. 初始化并加载数据集 =====================
    print("=" * 60)
    print("🔥 开始测试 DatasetSP_multi")
    print("=" * 60)

    # 实例化数据集
    dataset = DatasetSP_multi(mode="test")
    # 加载数据（核心初始化）
    dataset.prepare_data()

    # ===================== 2. 测试单样本 sample() 方法 =====================
    # print("\n" + "-" * 50)
    # print("✅ 测试单样本采样 sample()")
    # print("-" * 50)
    # traj, traj_multi = dataset.sample(n_modality=5)
    # print(f"主序列形状: {traj.shape}  ->  期望: (1, 75, 17, 3)")
    # print(f"多模态候选形状: {traj_multi.shape}  ->  期望: (5, 75, 17, 3)")
    # print(f"历史帧对齐验证: {np.allclose(traj_multi[:, :15], traj[:, :15])}  ->  期望: True")

    # ===================== 3. 测试批量采样 sampling_generator() =====================
    # print("\n" + "-" * 50)
    # print("✅ 测试批量生成器 sampling_generator()")
    # print("-" * 50)
    # # 生成1个batch
    # gen = dataset.sampling_generator(num_samples=256, batch_size=8, n_modality=5)
    # batch_traj, batch_multi = next(gen)
    # print(f"批量主序列形状: {batch_traj.shape}  ->  期望: (8, 75, 17, 3)")
    # print(f"批量多模态形状: {batch_multi.shape}  ->  期望: (8, 5, 75, 17, 3)")

    # ===================== 4. 测试评估迭代器 iter_generator() =====================
    print("\n" + "-" * 50)
    print("✅ 测试评估迭代器 iter_generator()")
    print("-" * 50)
    # 遍历前3个样本验证
    iter_gen = dataset.iter_generator(step=45)
    for i, (traj_iter, multi_iter) in enumerate(iter_gen):
        print(f"\n第 {i + 1} 个评估样本:")
        print(f"  主序列: {traj_iter.shape}")
        print(f"  多模态: {multi_iter.shape}")
        print(f"  历史帧对齐: {np.allclose(multi_iter[:, :45], traj_iter[:, :45])}")

    # ===================== 5. 数据集信息统计 =====================
    # print("\n" + "=" * 60)
    # print("📊 数据集最终验证")
    # print("=" * 60)
    # print(f"总测试片段数: {len(dataset.multi_indices)}")
    # print(f"候选库形状: {dataset.candidates.shape}")
    # print("🎉 所有方法测试通过！")
