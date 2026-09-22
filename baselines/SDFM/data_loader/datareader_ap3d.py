# Adapted from MotionBERT (https://github.com/Walter0807/MotionBERT/blob/main/lib/data/datareader_h36m.py)
import os
import numpy as np
import random
import joblib
import time


class DataReaderAP3D(object):
    def __init__(self, n_frames=125, sample_stride=1, data_stride_train=25, data_stride_test=25,
                 read_confidence=False, dt_root='data/athlete_pose_3d_v3',
                 normalize_to_neg1_pos1=True,
                 max_joint_range=2.0,
                 max_val=2.10):
        self.n_frames = n_frames  # 125
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train  # 训练集步长
        self.data_stride_test = data_stride_test  # 测试集步长
        self.read_confidence = read_confidence
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.dirname(current_path)
        dt_root_abs = os.path.join(root_path, dt_root)
        self.dt_root = dt_root_abs
        self.normalize_to_neg1_pos1 = normalize_to_neg1_pos1
        self.max_joint_range = max_joint_range

        self.max_val = max_val
        if self.normalize_to_neg1_pos1 and self.max_val is None:
            self._compute_and_save_max_val()

        # 加载：按 VIDEO 分组的 完整原始序列（核心修改）
        # st1 = time.time()
        self.dt_dataset = {
            'train': self._load_complete_sequences('train'),
            'test': self._load_complete_sequences('valid')
        }
        # st2 = time.time()
        # print(f"✅ 完整序列加载完成 | 耗时: {st2 - st1:.3f}s")

    def _is_valid_sequence(self, seq):
        """
        检查序列是否有效（根节点归一化后）
        返回：True=有效，False=异常
        """
        seq_normalized = seq - seq[:, 0:1, :]
        max_val = np.max(np.abs(seq_normalized))
        return max_val < self.max_joint_range

    def _compute_and_save_max_val(self):
        max_val_path = os.path.join(self.dt_root, 'ap3d_max_val1.npy')
        if os.path.exists(max_val_path):
            self.max_val = np.load(max_val_path)
            print(f"✅ 加载全局缩放因子: {self.max_val:.4f}")
            return

        print("🔄 计算训练集全局最大值（含异常数据清洗）...")
        meta = joblib.load(os.path.join(self.dt_root, 'train_meta_fps.joblib'), mmap_mode='r')
        pose_cam = np.load(os.path.join(self.dt_root, 'train_joint_3d_camera_fps.npy'), mmap_mode='r')

        video_ids = meta['videoid']
        unique_vids = np.unique(video_ids)
        all_poses = []
        invalid_count = 0

        for vid in unique_vids:
            seq = pose_cam[video_ids == vid].astype(np.float32)
            seq = seq / 1000.0

            # ===================== 🔥 异常数据清洗 =====================
            if not self._is_valid_sequence(seq):
                invalid_count += 1
                continue
            # ========================================================

            seq = seq - seq[:, 0:1, :]
            all_poses.append(seq.reshape(-1, 17 * 3))

        all_poses = np.concatenate(all_poses, axis=0)
        raw_max = np.max(np.abs(all_poses))
        self.max_val = np.ceil(raw_max * 10) / 10

        # np.save(max_val_path, self.max_val)
        print(f"✅ 异常视频数: {invalid_count}")
        print(f"✅ 训练集原始极值: {raw_max:.4f}")
        print(f"✅ 带安全边际最终缩放值: {self.max_val:.4f}")


        print("🔄 计算测试集全局最大值...")
        meta = joblib.load(os.path.join(self.dt_root, 'valid_meta_fps.joblib'))
        pose_cam = np.load(os.path.join(self.dt_root, 'valid_joint_3d_camera_fps.npy'), mmap_mode='r')

        video_ids = meta['videoid']
        unique_vids = np.unique(video_ids)
        all_poses = []

        for vid in unique_vids:
            seq = pose_cam[video_ids == vid].astype(np.float32)
            seq = seq / 1000.0
            seq = seq - seq[:, 0:1, :]
            all_poses.append(seq)

        # 计算训练集全局极值 + 安全边际
        all_poses = np.concatenate(all_poses, axis=0)
        raw_max = np.max(np.abs(all_poses))
        self.max_val = max(self.max_val, np.ceil(raw_max * 10) / 10)

        print(f"✅ 测试集原始极值: {raw_max:.4f}")
        print(f"✅ 带安全边际最终缩放值: {self.max_val:.4f}")


    def _load_complete_sequences(self, split):
        meta = joblib.load(os.path.join(self.dt_root, f'{split}_meta_fps.joblib'), mmap_mode='r')
        pose_cam = np.load(os.path.join(self.dt_root, f'{split}_joint_3d_camera_fps.npy'), mmap_mode='r')

        video_ids = meta['videoid']
        unique_vids = np.unique(video_ids)

        sequence_dict = {}
        invalid_count = 0

        for vid in unique_vids:
            mask = (video_ids == vid)
            seq = pose_cam[mask].astype(np.float32)

            # ===================== 标准预处理 =====================
            seq = seq / 1000.0
            # ===================== 🔥 异常数据清洗 =====================
            if not self._is_valid_sequence(seq):
                invalid_count += 1
                continue
            # ========================================================
            seq = seq - seq[:, 0:1, :]
            if self.normalize_to_neg1_pos1:
                seq = seq / self.max_val
            # ========================================================

            sequence_dict[vid] = seq

        print(f"✅ 加载 {split} 序列 | 有效视频: {len(sequence_dict)} | 异常视频: {invalid_count}")
        return sequence_dict

    # ------------------------------
    # 以下为兼容接口 + 标准调用方法
    # ------------------------------
    def get_train_sequences(self):
        """返回训练集所有完整序列列表"""
        return list(self.dt_dataset['train'].values())

    def get_test_sequences(self):
        """返回测试集所有完整序列列表"""
        return list(self.dt_dataset['test'].values())

    def inverse_normalize(self, seq):
        return seq * self.max_val if self.normalize_to_neg1_pos1 else seq


if __name__ == '__main__':
    """性能测试：DataReaderAP3D 加载时间 <0.1秒"""
    data_reader = DataReaderAP3D(normalize_to_neg1_pos1=True)
    train_data = data_reader.get_train_sequences()
    seq = train_data[0]
    # print(f"序列形状: {seq.shape}")
    # print(f"最小值: {np.min(seq):.4f}")
    # print(f"最大值: {np.max(seq):.4f}")
    # print(f"根节点坐标（第一帧）: {seq[0, 0, :]}")

    # print(f"Train labels shape: {len(train_data)}")
    # min_t = np.inf
    # max_t = -np.inf
    # for data in train_data:
    #     l = data.shape[0]
    #     # print(data.shape[0])
    #     min_t = min(min_t, l)
    #     max_t = max(max_t, l)
    # print(min_t, max_t)
    #
    # test_data = data_reader.get_test_sequences()
    # print(f"Train labels shape: {len(train_data)}")
    # min_t = np.inf
    # max_t = -np.inf
    # for data in test_data:
    #     l = data.shape[0]
    #     # print(data.shape)
    #     min_t = min(min_t, l)
    #     max_t = max(max_t, l)
    # print(min_t, max_t)
