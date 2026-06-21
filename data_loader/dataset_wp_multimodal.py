import os
import sys
import types
import numpy as np
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton
import torch


class DatasetWP_multi(Dataset):
    """
    WorldPose multimodal dataset.

    This version is aligned with the latest WorldPose multimodal generation protocol:
    1. candidate pool is generated from train_seqs by per-sequence sliding windows;
    2. test clips are generated from test_seqs by the same per-sequence sliding windows;
    3. multi_dict key is the global test clip index;
    4. data_candi indices point to the train candidate pool;
    5. no cross-sequence concatenation is used for evaluation.
    """

    def __init__(self, mode='train', t_his=25, t_pred=100, normalization=True, use_vel=False, n_modality=5, max_abs=2.10, **kwargs):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.normalization = normalization
        self.use_vel = use_vel
        self.n_modality = n_modality
        self.max_abs = max_abs
        self.step = t_his

        current_path = os.path.dirname(os.path.abspath(__file__))
        self.root_path = os.path.dirname(current_path)

        self.multimodal_path = kwargs.get('multimodal_path', None)
        self.data_candi_path = kwargs.get('data_candi_path', None)
        self.raw_data_path = kwargs.get('raw_data_path', kwargs.get('raw_data_path ', None))

        if self.multimodal_path is not None:
            self.multimodal_path = os.path.join(self.root_path, self.multimodal_path)

        if self.data_candi_path is not None:
            self.data_candi_path = os.path.join(self.root_path, self.data_candi_path)

        if self.raw_data_path is not None:
            self.raw_data_path = os.path.join(self.root_path, self.raw_data_path)

        self._load_multimodal_data()
        self._init_skeleton()

    def _load_npz_array(self, path, preferred_key):
        data = np.load(path, allow_pickle=True)
        if preferred_key in data.files:
            return data[preferred_key]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise KeyError(f"{preferred_key} not found in {path}. Available keys: {data.files}")

    def _load_multimodal_data(self):
        if self.multimodal_path is None:
            multimodal_path = os.path.join(self.root_path, f'data/worldpose/multimodal/t_his{self.t_his}_1_thre0.500_t_pred{self.t_pred}_thre0.100_index_filterd.npz')
            candi_path = os.path.join(self.root_path, f'data/worldpose/multimodal/data_candi_t_his{self.t_his}_t_pred{self.t_pred}_skiprate{self.step}.npz')
        else:
            multimodal_path = self.multimodal_path
            candi_path = self.data_candi_path

        if self.raw_data_path is None:
            raw_data_path = os.path.join(self.root_path, 'data/worldpose/wp_data_py3.npz')
        else:
            raw_data_path = self.raw_data_path

        self.data_multimodal = np.load(multimodal_path, allow_pickle=True)['data_multimodal'].item()
        self.data_candi = self._load_npz_array(candi_path, 'data_candidate.npy').astype(np.float32)

        raw_data = np.load(raw_data_path, allow_pickle=True)
        train_raw = list(raw_data['train'])
        test_raw = list(raw_data['test'])

        if self.normalization and self.max_abs is None:
            self.max_abs = self._estimate_max_abs(train_raw + test_raw)

        self.train_seqs = [self._normalize_sequence(seq.astype(np.float32)) for seq in train_raw]
        self.test_seqs = [self._normalize_sequence(seq.astype(np.float32)) for seq in test_raw]

        raw_multi_dict = self.data_multimodal["all_subjects"]["all_actions"]
        self.multi_dict = {int(k): np.asarray(v, dtype=np.int64) for k, v in raw_multi_dict.items()}

        self.max_valid_idx = max(self.multi_dict.keys()) if len(self.multi_dict) > 0 else -1
        self.total_clips = self.max_valid_idx + 1

        if self.data_candi.ndim != 4:
            raise ValueError(f"data_candi should be [N,T,J,3], got {self.data_candi.shape}")

        if self.data_candi.shape[1] != self.t_total:
            raise ValueError(f"data_candi t_total mismatch: expected {self.t_total}, got {self.data_candi.shape[1]}")

        self.num_joints = self.data_candi.shape[2]

        self.test_clips = self._build_test_clips(self.test_seqs)

        if self.total_clips > 0 and len(self.test_clips) != self.total_clips:
            raise RuntimeError(f"Test clip count mismatch. Built test_clips={len(self.test_clips)}, but multimodal dict has {self.total_clips} keys. This means DatasetWP_multi and multimodal generation are not aligned.")

        self._validate_multimodal_indices()

        print(f"WorldPose multimodal dataset loaded.")
        print(f"mode: {self.mode}")
        print(f"data_candi: {self.data_candi.shape}")
        print(f"test_clips: {self.test_clips.shape}")
        print(f"multi_dict keys: {len(self.multi_dict)}")
        print(f"max_abs: {self.max_abs}")

    def _estimate_max_abs(self, seqs):
        max_abs = 0.0
        for seq in seqs:
            seq = seq.astype(np.float32)
            centered = seq - seq[:, 0:1, :]
            value = np.nanmax(np.abs(centered))
            if value > max_abs:
                max_abs = value
        if max_abs < 1e-8 or np.isnan(max_abs):
            max_abs = 1.0
        return float(max_abs)

    def _normalize_sequence(self, seq):
        if not self.normalization:
            return seq.astype(np.float32)
        root = seq[:, 0:1, :]
        seq = (seq - root) / float(self.max_abs)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return seq.astype(np.float32)

    def _build_test_clips(self, test_seqs):
        clips = []
        meta = []

        for seq_id, seq in enumerate(test_seqs):
            if len(seq) < self.t_total:
                continue

            max_start = len(seq) - self.t_total

            for start in range(0, max_start + 1, self.step):
                clips.append(seq[start:start + self.t_total])
                meta.append((seq_id, start))

        clips = np.asarray(clips, dtype=np.float32)
        self.test_clip_meta = meta

        return clips

    def _validate_multimodal_indices(self):
        max_idx = -1

        for v in self.multi_dict.values():
            if len(v) > 0:
                cur = int(np.max(v))
                if cur > max_idx:
                    max_idx = cur

        if max_idx >= len(self.data_candi):
            raise RuntimeError(f"multimodal index out of range: max idx={max_idx}, candidate size={len(self.data_candi)}. The index file and candidate pool are not from the same generation run.")

    def _init_skeleton(self):
        self.skeleton = Skeleton(
            parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            joints_left=[1, 4, 5, 6, 16, 17, 18, 19],
            joints_right=[2, 7, 8, 9, 20, 21, 22, 23]
        )
        self.kept_joints = np.arange(24)

    def _add_velocity(self, x):
        if not self.use_vel:
            return x

        if x.ndim == 3:
            vel = np.zeros_like(x)
            vel[1:] = x[1:] - x[:-1]
            return np.concatenate([x, vel], axis=-1)

        if x.ndim == 4:
            vel = np.zeros_like(x)
            vel[:, 1:] = x[:, 1:] - x[:, :-1]
            return np.concatenate([x, vel], axis=-1)

        raise ValueError(f"Unexpected shape for velocity feature: {x.shape}")

    def _valid_candidate_indices(self, idx_list):
        idx_list = np.asarray(idx_list, dtype=np.int64)

        if len(idx_list) == 0:
            return idx_list

        valid = (idx_list >= 0) & (idx_list < len(self.data_candi))
        idx_list = idx_list[valid]

        return idx_list

    def _fix_n_modality(self, multi_traj):
        if self.n_modality is None:
            return multi_traj

        if len(multi_traj) == self.n_modality:
            return multi_traj

        if len(multi_traj) > self.n_modality:
            return multi_traj[:self.n_modality]

        repeat_idx = np.tile(np.arange(len(multi_traj)), self.n_modality // len(multi_traj) + 1)[:self.n_modality]
        return multi_traj[repeat_idx]

    def _get_multimodal_candidates(self, global_idx, traj, fix_n_modality=False):
        key = int(global_idx)
        idx_list = self.multi_dict.get(key, np.array([], dtype=np.int32))
        idx_list = self._valid_candidate_indices(idx_list)

        if len(idx_list) == 0:
            multi_traj = traj[None].copy()
        else:
            multi_traj = self.data_candi[idx_list].astype(np.float32).copy()
            multi_traj[:, :self.t_his] = traj[None, :self.t_his]

        if fix_n_modality:
            multi_traj = self._fix_n_modality(multi_traj)

        return multi_traj.astype(np.float32)

    def __len__(self):
        if self.mode == 'train':
            return len(self.data_candi)
        return len(self.test_clips)

    def __getitem__(self, idx):
        if self.mode == 'train':
            candi_idx = np.random.randint(len(self.data_candi))
            traj = self.data_candi[candi_idx].astype(np.float32).copy()
            multi_traj = np.repeat(traj[None], self.n_modality, axis=0).astype(np.float32)
        else:
            global_idx = int(idx) % len(self.test_clips)
            traj = self.test_clips[global_idx].astype(np.float32).copy()
            multi_traj = self._get_multimodal_candidates(global_idx, traj, fix_n_modality=True)

        traj = self._add_velocity(traj)
        multi_traj = self._add_velocity(multi_traj)

        return traj.astype(np.float32), multi_traj.astype(np.float32)

    def sample(self, n_modality=None):
        if n_modality is None:
            n_modality = self.n_modality

        idx = np.random.randint(len(self))
        traj, multi_traj = self.__getitem__(idx)

        return traj[None].astype(np.float32), multi_traj[:n_modality].astype(np.float32)

    def sampling_generator(self, num_samples=1000, batch_size=8, n_modality=None):
        if n_modality is None:
            n_modality = self.n_modality

        for _ in range(num_samples // batch_size):
            batch_traj = []
            batch_multi = []

            for _ in range(batch_size):
                traj, multi_traj = self.sample(n_modality=n_modality)
                batch_traj.append(traj[0])
                batch_multi.append(multi_traj[:n_modality])

            batch_traj = np.asarray(batch_traj, dtype=np.float32)
            batch_multi = np.asarray(batch_multi, dtype=np.float32)

            yield batch_traj, batch_multi

    def iter_generator(self, step=None):
        """
        Evaluation generator.

        Important:
        This follows the exact global test clip order used during multimodal generation.
        Do not use test_global concatenation here.
        """

        for global_idx in range(len(self.test_clips)):
            traj = self.test_clips[global_idx].astype(np.float32).copy()
            multi_traj = self._get_multimodal_candidates(global_idx, traj, fix_n_modality=False)

            traj = traj[None]
            traj = self._add_velocity(traj)
            multi_traj = self._add_velocity(multi_traj)

            yield traj.astype(np.float32), multi_traj.astype(np.float32)

    def get_stats(self):
        return np.ones(3, dtype=np.float32) * float(self.max_abs), np.zeros(3, dtype=np.float32)


if __name__ == "__main__":
    print("=" * 60)
    print("测试多模态 DatasetWP_multi")
    print("=" * 60)
    ds_multi = DatasetWP_multi(mode='test', use_vel=False)
    print(f"多模态测试集大小: {len(ds_multi)}")
    itergen = ds_multi.sampling_generator(num_samples=51200, batch_size=1024, n_modality=5)
    max_coord = -np.inf
    for input_traj, future_trajs in itergen:
        print(input_traj.shape, future_trajs.shape)
        max_coord = max(max_coord, np.max(np.abs(input_traj)))
    print(max_coord)
