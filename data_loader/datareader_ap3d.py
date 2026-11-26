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
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
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

        # complete sequences grouped by video
        # st1 = time.time()
        self.dt_dataset = {
            'train': self._load_complete_sequences('train'),
            'test': self._load_complete_sequences('valid')
        }
        # st2 = time.time()

    def _is_valid_sequence(self, seq):
        """
        check whether a sequence is valid (after root normalization)
        returns True = valid, False = corrupted
        """
        seq_normalized = seq - seq[:, 0:1, :]
        max_val = np.max(np.abs(seq_normalized))
        return max_val < self.max_joint_range

    def _compute_and_save_max_val(self):
        max_val_path = os.path.join(self.dt_root, 'ap3d_max_val1.npy')
        if os.path.exists(max_val_path):
            self.max_val = np.load(max_val_path)
            print(f"loaded global scale: {self.max_val:.4f}")
            return

        print("computing the global maximum of the training set (corrupted videos removed)...")
        meta = joblib.load(os.path.join(self.dt_root, 'train_meta_fps.joblib'), mmap_mode='r')
        pose_cam = np.load(os.path.join(self.dt_root, 'train_joint_3d_camera_fps.npy'), mmap_mode='r')

        video_ids = meta['videoid']
        unique_vids = np.unique(video_ids)
        all_poses = []
        invalid_count = 0

        for vid in unique_vids:
            seq = pose_cam[video_ids == vid].astype(np.float32)
            seq = seq / 1000.0

            # remove corrupted sequences
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
        print(f"corrupted videos: {invalid_count}")
        print(f"raw maximum (train): {raw_max:.4f}")
        print(f"scale with safety margin: {self.max_val:.4f}")


        print("computing the global maximum of the test set...")
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

        # global maximum + safety margin
        all_poses = np.concatenate(all_poses, axis=0)
        raw_max = np.max(np.abs(all_poses))
        self.max_val = max(self.max_val, np.ceil(raw_max * 10) / 10)

        print(f"raw maximum (test): {raw_max:.4f}")
        print(f"scale with safety margin: {self.max_val:.4f}")


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

            # millimetre -> metre
            seq = seq / 1000.0
            # remove corrupted sequences
            if not self._is_valid_sequence(seq):
                invalid_count += 1
                continue
            # ========================================================
            seq = seq - seq[:, 0:1, :]
            if self.normalize_to_neg1_pos1:
                seq = seq / self.max_val
            # ========================================================

            sequence_dict[vid] = seq

        print(f"loaded {split} sequences | valid videos: {len(sequence_dict)} | corrupted videos: {invalid_count}")
        return sequence_dict

    # ------------------------------
    # public interface
    # ------------------------------
    def get_train_sequences(self):
        """list of all complete training sequences"""
        return list(self.dt_dataset['train'].values())

    def get_test_sequences(self):
        """list of all complete test sequences"""
        return list(self.dt_dataset['test'].values())

    def inverse_normalize(self, seq):
        return seq * self.max_val if self.normalize_to_neg1_pos1 else seq


if __name__ == '__main__':
    """loading test"""
    data_reader = DataReaderAP3D(normalize_to_neg1_pos1=True)
    train_data = data_reader.get_train_sequences()
    seq = train_data[0]

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
