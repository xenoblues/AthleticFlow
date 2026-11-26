"""WorldPose (SMPL, 24 joints incl. pelvis, 50 fps).

Reads worldpose/wp_data_py3.npz ('train' / 'test': object arrays of [T, 24, 3] sequences).
Every window is root-centred and divided by MAX_ABS = 2.10.
"""
import os

import numpy as np

from data_loader.dataset import Dataset, MultimodalTestSet

MAX_ABS = 2.10


def load_wp_raw(data_root, split):
    raw = np.load(os.path.join(data_root, 'worldpose', 'wp_data_py3.npz'), allow_pickle=True)
    return list(raw[split])


class DatasetWP(Dataset):
    def __init__(self, mode, t_his=25, t_pred=100, data_root='data'):
        self.data_root = data_root
        super().__init__(mode, t_his, t_pred)

    def prepare_data(self):
        self.data = load_wp_raw(self.data_root, 'train' if self.mode == 'train' else 'test')

    @staticmethod
    def _normalize(seq):
        return (seq - seq[:, 0:1, :]) / MAX_ABS

    def sample(self):
        while True:
            seq = self.data[np.random.randint(len(self.data))]
            if len(seq) >= self.t_total:
                break
        fr_start = np.random.randint(len(seq) - self.t_total + 1)
        return self._normalize(seq[fr_start: fr_start + self.t_total])[None]

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        for _ in range(num_samples // batch_size):
            batch = np.concatenate([self.sample() for _ in range(batch_size)], axis=0)
            if aug:
                if np.random.uniform() > 0.5:  # x-y rotating
                    theta = np.random.uniform(0, 2 * np.pi)
                    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    rotate_xy = np.matmul(batch.transpose([0, 2, 1, 3])[..., 0:2], rotate_matrix)
                    batch[..., 0:2] = rotate_xy.transpose([0, 2, 1, 3])
                if np.random.uniform() > 0.5:  # x-z mirroring
                    batch[..., 0] = - batch[..., 0]
                if np.random.uniform() > 0.5:  # y-z mirroring
                    batch[..., 1] = - batch[..., 1]
            yield batch


def DatasetWP_multi(t_his, t_pred, data_root, multimodal_path, data_candi_path):
    """Test windows of every WorldPose test sequence (stride t_his) + multimodal GT."""
    seqs = []
    for seq in load_wp_raw(data_root, 'test'):
        seq = seq.astype(np.float32)
        seq = (seq - seq[:, 0:1, :]) / float(MAX_ABS)
        seqs.append(np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
    return MultimodalTestSet(seqs, t_his, t_pred,
                             os.path.join(data_root, multimodal_path), os.path.join(data_root, data_candi_path),
                             name='wp')
