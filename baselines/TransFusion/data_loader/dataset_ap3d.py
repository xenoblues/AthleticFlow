"""AthletePose3D (17 joints, H36M ordering, 60 fps).

Joint order: 0 pelvis, 1 RHip, 2 RKnee, 3 RAnkle, 4 LHip, 5 LKnee, 6 LAnkle, 7 spine,
8 thorax, 9 neck, 10 head, 11 LShoulder, 12 LElbow, 13 LWrist, 14 RShoulder, 15 RElbow, 16 RWrist.

Sequences are loaded per video from {split}_joint_3d_camera_fps.npy / {split}_meta_fps.joblib
(train = 'train', test = 'valid'), converted from mm to m, root-centred and divided by
MAX_VAL = 2.1. Videos whose root-relative coordinates exceed 2 m are discarded.
The sequence reader follows MotionBERT's datareader_h36m.py.
"""
import os
import random

import joblib
import numpy as np

from data_loader.dataset import Dataset, MultimodalTestSet

MAX_VAL = 2.1
MAX_JOINT_RANGE = 2.0


def load_ap3d_sequences(data_root, split):
    root = os.path.join(data_root, 'athlete_pose_3d_v3')
    file_split = 'train' if split == 'train' else 'valid'
    meta = joblib.load(os.path.join(root, f'{file_split}_meta_fps.joblib'), mmap_mode='r')
    pose_cam = np.load(os.path.join(root, f'{file_split}_joint_3d_camera_fps.npy'), mmap_mode='r')

    video_ids = meta['videoid']
    sequences, invalid = [], 0
    for vid in np.unique(video_ids):
        seq = pose_cam[video_ids == vid].astype(np.float32) / 1000.0
        seq = seq - seq[:, 0:1, :]
        if not np.max(np.abs(seq)) < MAX_JOINT_RANGE:
            invalid += 1
            continue
        sequences.append(seq / MAX_VAL)
    print(f'AthletePose3D {file_split}: {len(sequences)} sequences ({invalid} discarded)')
    return sequences


class DatasetAP3D(Dataset):
    def __init__(self, mode, t_his=15, t_pred=60, data_root='data'):
        self.data_root = data_root
        super().__init__(mode, t_his, t_pred)

    def prepare_data(self):
        self.data = [np.ascontiguousarray(seq) for seq in load_ap3d_sequences(self.data_root, self.mode)
                     if seq.shape[0] >= self.t_total]

    def sample(self):
        seq = random.choice(self.data)
        start = random.randint(0, seq.shape[0] - self.t_total)
        return seq[start:start + self.t_total][None]

    def sampling_generator(self, num_samples=50000, batch_size=256, aug=True):
        for _ in range(num_samples // batch_size):
            batch = np.concatenate([self.sample() for _ in range(batch_size)], axis=0)
            if aug:
                if np.random.rand() > 0.5:  # random rotation about the vertical axis
                    theta = np.random.uniform(0, 2 * np.pi)
                    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    batch[..., :2] = np.matmul(batch[..., :2], rot.T)
            yield batch


def DatasetAP3D_multi(t_his, t_pred, data_root, multimodal_path, data_candi_path):
    """Test windows of every AthletePose3D test sequence (stride t_his) + multimodal GT."""
    return MultimodalTestSet(load_ap3d_sequences(data_root, 'test'), t_his, t_pred,
                             os.path.join(data_root, multimodal_path), os.path.join(data_root, data_candi_path),
                             name='ap3d')
