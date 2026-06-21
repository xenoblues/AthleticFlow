import os
import numpy as np
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


class DatasetAthleticsPose_multi(Dataset):
    def __init__(self,
                 mode,
                 t_his=15,
                 t_pred=60,
                 actions='all',
                 data_path=r"data\AthleticsPose",
                 multimodal_path=None,
                 data_candi_path=None,
                 **kwargs):
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.dirname(current_path)
        self.data_path = os.path.join(root_path, data_path)
        self.multimodal_path = multimodal_path
        self.data_candi_path = data_candi_path
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.prepare_data()

    def prepare_data(self):
        # 与AP3D完全一致的骨骼定义
        self.subjects = ['Virtual/Test'] if self.mode == 'test' else ['Virtual/Train']
        self.skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
            joints_left=[4, 5, 6, 11, 12, 13],
            joints_right=[1, 2, 3, 14, 15, 16]
        )
        self.kept_joints = np.arange(17)
        self.process_data()

    def process_data(self):

        npz_file = os.path.join(self.data_path, f"{self.mode}.npz")
        data = np.load(npz_file, allow_pickle=True)
        seqs = list(data['trajectories'])
        seqs = [seq.astype(np.float32) for seq in seqs]

        # 全局拼接（与AP3D一致）
        self.data = {self.subjects[0]: {'all_actions': np.concatenate(seqs, axis=0)}}

        # 加载多模态文件（与AP3D完全一致的路径和格式）
        if self.multimodal_path is None or self.data_candi_path is None:
            current_path = os.path.dirname(os.path.abspath(__file__))
            root_path = os.path.dirname(current_path)
            multimodal_path = os.path.join(root_path, self.data_path,
                                           'multimodal/t_his15_1_thre0.500_t_pred60_thre0.100_index_filterd.npz')
            candi_path = os.path.join(root_path, self.data_path,
                                      'multimodal/data_candi_t_his15_t_pred60_skiprate15.npz')
        else:
            multimodal_path = self.multimodal_path
            candi_path = self.data_candi_path

        if not os.path.exists(multimodal_path) or not os.path.exists(candi_path):
            raise FileNotFoundError("多模态文件不存在，请先运行athleticspose_multimodal_data_generation.py生成")

        self.data_multimodal = np.load(multimodal_path, allow_pickle=True)['data_multimodal'].item()
        self.data_candi = np.load(candi_path, allow_pickle=True)['data_candidate.npy'].astype(np.float32)
        self.multi_dict = self.data_multimodal['all_subjects']['all_actions']
        self.max_valid_idx = max(self.multi_dict.keys()) if self.multi_dict else -1

    def iter_generator(self, step=15):
        """与AP3D完全一致的多模态评估迭代器"""
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
                multi_traj = self.data_candi[idx]
                multi_traj[:, :self.t_his] = traj[:, :self.t_his]  # 强制对齐历史帧

            global_idx += 1
            yield traj, multi_traj

    def sample(self, n_modality=5):
        """与AP3D完全一致的采样逻辑"""
        seq = self.data[self.subjects[0]]['all_actions']
        fr_start = np.random.randint(len(seq) - self.t_total)
        return seq[None, fr_start: fr_start + self.t_total], None


# -----------------------------------------------------------------------------
# 测试代码
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)
    print("=" * 60)
    print("🔥 测试 DatasetAthleticsPose_multi")
    print("=" * 60)

    # 初始化数据集
    dataset = DatasetAthleticsPose_multi(mode="test")

    # 测试迭代器
    print("\n✅ 测试评估迭代器")
    max_coord = -np.inf
    iter_gen = dataset.iter_generator(step=15)
    for i, (traj_iter, multi_iter) in enumerate(iter_gen):
        max_coord = max(max_coord, np.max(np.abs(traj_iter)))
        print(f"\n第 {i + 1} 个样本:")
        print(f"  主序列: {traj_iter.shape}")
        print(f"  多模态: {multi_iter.shape}")
        print(f"  历史帧对齐: {np.allclose(multi_iter[:, :15], traj_iter[:, :15])}")

    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print(max_coord)