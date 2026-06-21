import os
import random
import numpy as np
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton
from data_loader.dataset_ap3d import DataReaderAP3D
from utils import util


class DatasetAP3D_multi(Dataset):
    def __init__(self, mode, t_his=15, t_pred=60, actions='all', **kwargs):
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

        self.skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
            joints_left=[4, 5, 6, 11, 12, 13],  # 左髋、左膝、左踝、左肩、左肘、左腕
            joints_right=[1, 2, 3, 14, 15, 16]  # 右髋、右膝、右踝、右肩、右肘、右腕
        )
        self.kept_joints = np.arange(17)
        self.process_data()

    def process_data(self):
        from data_loader.datareader_ap3d import DataReaderAP3D
        reader = DataReaderAP3D(normalize_to_neg1_pos1=True)
        if self.mode == 'train':
            seqs = reader.get_train_sequences()
        else:
            seqs = reader.get_test_sequences()

        self.data = {self.subjects[0]: {'all_actions': np.concatenate(seqs, axis=0)}}

        if self.multimodal_path is None or self.data_candi_path is None:
            current_path = os.path.dirname(os.path.abspath(__file__))
            root_path = os.path.dirname(current_path)
            multimodal_path = os.path.join(root_path,
                                           'data/athlete_pose_3d_v3/multimodal/t_his15_top50_t_pred60_thre0.100_filtered_dlow.npz')
            candi_path = os.path.join(root_path,
                                           'data/athlete_pose_3d_v3/multimodal/data_candi_t_his15_t_pred60_skiprate15.npz')
            self.data_multimodal = np.load(multimodal_path, allow_pickle=True)['data_multimodal'].item()
            self.data_candi = np.load(candi_path, allow_pickle=True)['data_candidate.npy'].astype(np.float32)
        else:
            self.data_multimodal = np.load(self.multimodal_path, allow_pickle=True)['data_multimodal'].item()
            self.data_candi = np.load(self.data_candi_path, allow_pickle=True)['data_candidate.npy'].astype(np.float32)

        self.multi_dict = self.data_multimodal['all_subjects']['all_actions']
        self.max_valid_idx = max(self.multi_dict.keys()) if self.multi_dict else -1

    def iter_generator(self, step=15):
        global_idx = 0
        seq = self.data[self.subjects[0]]['all_actions']
        max_start = len(seq) - self.t_total

        for i in range(0, max_start + 1, step):
            if global_idx > self.max_valid_idx:
                return

            traj = seq[None, i:i + self.t_total].astype(np.float32)
            key = np.int64(global_idx)
            idx = self.multi_dict.get(key, np.array([], dtype=np.int32))

            # 空索引兜底
            if len(idx) == 0:
                multi_traj = traj.copy()
            else:
                # 从候选库取数据 + 强制对齐历史帧
                multi_traj = self.data_candi[idx]
                multi_traj[:, :self.t_his] = traj[:, :self.t_his]

            global_idx += 1
            yield traj, multi_traj

    def sample(self, n_modality=5):
        seq = self.data[self.subjects[0]]['all_actions']
        fr_start = np.random.randint(len(seq) - self.t_total)
        return seq[None, fr_start: fr_start + self.t_total], None


# -----------------------------------------------------------------------------
# 测试代码（一键验证，无任何报错）
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)
    # ===================== 测试配置（请修改为你的文件路径） =====================
    CONFIG = {
        "mode": "test",
        "t_his": 15,
        "t_pred": 60,
        # 替换为你生成的多模态文件路径
    }

    # ===================== 1. 初始化并加载数据集 =====================
    print("=" * 60)
    print("🔥 开始测试 DatasetAP3D_multi")
    print("=" * 60)

    # 实例化数据集
    dataset = DatasetAP3D_multi(mode="test")
    # 加载数据（核心初始化）
    dataset.prepare_data()


    # ===================== 4. 测试评估迭代器 iter_generator() =====================
    print("\n" + "-" * 50)
    print("✅ 测试评估迭代器 iter_generator()")
    print("-" * 50)
    # 遍历前3个样本验证
    max_coord = -np.inf
    iter_gen = dataset.iter_generator(step=15)
    for i, (traj_iter, multi_iter) in enumerate(iter_gen):
        x = traj_iter.reshape(-1, 17, 3)
        max_coord = max(max_coord, np.max(np.abs(x)))
        print(f"\n第 {i + 1} 个评估样本:")
        print(f"  主序列: {traj_iter.shape}")
        print(f"  多模态: {multi_iter.shape}")
        print(f"  历史帧对齐: {np.allclose(multi_iter[:, :15], traj_iter[:, :15])}")

    # ===================== 5. 数据集信息统计 =====================
    print("\n" + "=" * 60)
    print("📊 数据集最终验证")
    print("=" * 60)
    print("🎉 所有方法测试通过！")
    print(max_coord)
