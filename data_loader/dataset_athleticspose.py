import os
import random
import numpy as np

from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton

"""
AthleticsPose joint indices (same H36M-17 layout as AthletePose3D)
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
                 data_path=os.path.join("data", "AthleticsPose"),
                 **kwargs):
        self.use_vel = use_vel
        cur_path = os.path.abspath(os.path.dirname(__file__))
        root_path = os.path.dirname(cur_path)
        self.data_path = os.path.join(root_path, data_path)  # directory of the preprocessed train.npz / test.npz
        super().__init__(mode, t_his, t_pred, actions='all', **kwargs)

    def prepare_data(self, **kwargs):
        # same skeleton as AthletePose3D
        self.kept_joints = np.arange(17)
        self.skeleton = Skeleton(
            parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
            joints_left=[4, 5, 6, 11, 12, 13],
            joints_right=[1, 2, 3, 14, 15, 16]
        )

        npz_file = os.path.join(self.data_path, f"{self.mode}.npz")
        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"preprocessed file not found: {npz_file}\nrun data_loader/preprocess_athletics_pose.py first")

        data = np.load(npz_file, allow_pickle=True)

        # complete sequences (object array of (T_i, 17, 3) arrays)
        raw_sequences = data['trajectories']
        self.subjects = data['subjects']
        self.actions = data['actions']
        self.lengths = data['lengths']
        self.global_max = data['global_max']

        self.data = []
        for seq in raw_sequences:
            seq = seq.astype(np.float32)

            # optional velocity features
            if self.use_vel:
                vel = np.diff(seq, axis=0, prepend=seq[:1])
                seq = np.concatenate([seq, vel], axis=-1)

            # drop sequences shorter than one window
            if seq.shape[0] >= self.t_total:
                self.data.append(np.ascontiguousarray(seq))

        print(f"AthleticsPose-{self.mode} loaded")
        print(f"   sequences: {len(self.data)} | frames: {sum(self.lengths)}")
        print(f"   global normalization scale: {self.global_max:.4f}")

    def sample(self):
        """random training window"""
        seq = random.choice(self.data)
        max_start = seq.shape[0] - self.t_total
        start = random.randint(0, max_start)
        return seq[start:start + self.t_total][None]

    def sampling_generator(self, num_samples=50000, batch_size=256, aug=True):
        """batches of random windows with random rotation / mirroring"""
        for _ in range(num_samples // batch_size):
            batch = [self.sample() for _ in range(batch_size)]
            batch = np.concatenate(batch, axis=0)
            seq_len = batch.shape[1]

            # random frame mask (unused, kept for the random-number stream)
            mask_indices = np.random.randint(int(seq_len * 0.08), int(seq_len * 0.92), int(seq_len * 0.2))
            mask = np.array([i not in mask_indices for i in range(seq_len)], dtype=bool)

            # data augmentation
            if aug:
                if np.random.rand() > 0.5:
                    theta = np.random.uniform(0, 2 * np.pi)
                    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    batch[..., :2] = np.matmul(batch[..., :2], rot.T)
                if np.random.uniform() > 0.5:  # x-z mirroring
                    batch[..., 0] = - batch[..., 0]
                if np.random.uniform() > 0.5:  # y-z mirroring
                    batch[..., 1] = - batch[..., 1]

            yield batch, mask

    def iter_generator(self, step=15):
        """all windows of every sequence with the given stride"""
        for seq in self.data:
            for i in range(0, seq.shape[0] - self.t_total + 1, step):
                yield seq[None, i:i + self.t_total]
