"""AthleticsPose (17 joints, same ordering as AthletePose3D, 60 fps).

Reads the preprocessed AthleticsPose/{train,test}.npz ('trajectories': object array of
root-centred, normalised [T, 17, 3] sequences) produced by the main repository's
data_loader/preprocess_athletics_pose.py.
"""
import os
import random

import numpy as np

from data_loader.dataset import Dataset, MultimodalTestSet


def load_ap_sequences(data_root, split):
    data = np.load(os.path.join(data_root, 'AthleticsPose', f'{split}.npz'), allow_pickle=True)
    sequences = [seq.astype(np.float32) for seq in data['trajectories']]
    print(f'AthleticsPose {split}: {len(sequences)} sequences, {sum(len(s) for s in sequences)} frames')
    return sequences


class DatasetAthleticsPose(Dataset):
    def __init__(self, mode, t_his=15, t_pred=60, data_root='data'):
        self.data_root = data_root
        super().__init__(mode, t_his, t_pred)

    def prepare_data(self):
        self.data = [np.ascontiguousarray(seq) for seq in load_ap_sequences(self.data_root, self.mode)
                     if seq.shape[0] >= self.t_total]

    def sample(self):
        seq = random.choice(self.data)
        start = random.randint(0, seq.shape[0] - self.t_total)
        return seq[start:start + self.t_total][None]

    def sampling_generator(self, num_samples=50000, batch_size=256, aug=True):
        for _ in range(num_samples // batch_size):
            batch = np.concatenate([self.sample() for _ in range(batch_size)], axis=0)
            # unused draw kept from the original training code so that the random stream is unchanged
            np.random.randint(int(batch.shape[1] * 0.08), int(batch.shape[1] * 0.92), int(batch.shape[1] * 0.2))
            if aug:
                if np.random.rand() > 0.5:  # random rotation about the vertical axis
                    theta = np.random.uniform(0, 2 * np.pi)
                    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    batch[..., :2] = np.matmul(batch[..., :2], rot.T)
                if np.random.uniform() > 0.5:  # x mirroring
                    batch[..., 0] = - batch[..., 0]
                if np.random.uniform() > 0.5:  # y mirroring
                    batch[..., 1] = - batch[..., 1]
            yield batch


def DatasetAthleticsPose_multi(t_his, t_pred, data_root, multimodal_path, data_candi_path):
    """Test windows of every AthleticsPose test sequence (stride t_his) + multimodal GT."""
    return MultimodalTestSet(load_ap_sequences(data_root, 'test'), t_his, t_pred,
                             os.path.join(data_root, multimodal_path), os.path.join(data_root, data_candi_path),
                             name='ap')
