"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset.py
"""
import os

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Dataset:
    def __init__(self, mode, t_his, t_pred, actions='all', **kwargs):
        self.data = None
        self.dataset_name = None
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.prepare_data(**kwargs)
        self.std, self.mean = None, None
        self.data_len = sum([seq.shape[0] for data_s in self.data.values() for seq in data_s.values()]) if type(self.data).__name__ == 'dict' else sum([seq.shape[0] for seq in self.data])
        self.traj_dim = (self.kept_joints.shape[0] - 1) * 3
        self.normalized = False
        # iterator specific
        self.sample_ind = None

    def prepare_data(self, **kwargs):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        raise NotImplementedError

    def iter_generator(self, step=25):
        raise NotImplementedError


class PerSequenceMultimodalDataset:
    """
    Test windows with multimodal ground truth for AthletePose3D / AthleticsPose.

    Evaluation protocol: windows of t_his + t_pred frames are extracted WITHIN every test sequence with stride t_his
    (sequences shorter than t_his + t_pred are skipped); windows never straddle two sequences. The window order must
    equal the 'test_clip_meta' array (sequence id, start frame) of the multimodal file, whose i-th entry of
    'data_multimodal' lists the indices of the candidate futures (training windows) of the i-th test window.

    Subclasses set self.skeleton, self.kept_joints and implement load_test_sequences() and the default file names.
    """
    default_multimodal_path = None
    default_candi_path = None

    def __init__(self, mode, t_his=15, t_pred=60, multimodal_path=None, data_candi_path=None, **kwargs):
        assert mode == 'test', 'the multimodal dataset is only used for evaluation'
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.multimodal_path = os.path.join(ROOT_DIR, multimodal_path or self.default_multimodal_path)
        self.data_candi_path = os.path.join(ROOT_DIR, data_candi_path or self.default_candi_path)
        self.prepare_data()

    def load_test_sequences(self):
        """list of test sequences, each (T_i, 17, 3)"""
        raise NotImplementedError

    def prepare_data(self):
        for path in (self.multimodal_path, self.data_candi_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"multimodal file not found: {path} (see README for the download / "
                                        f"the *_multimodal_data_generation.py scripts)")

        self.seqs = [seq.astype(np.float32) for seq in self.load_test_sequences()]

        z = np.load(self.multimodal_path, allow_pickle=True)
        self.multi_dict = z['data_multimodal'].item()['all_subjects']['all_actions']
        self.data_candi = np.load(self.data_candi_path, allow_pickle=True)['data_candidate.npy'].astype(np.float32)

        # per-sequence windows, stride t_his
        self.windows = [(sid, s) for sid, seq in enumerate(self.seqs) if len(seq) >= self.t_total
                        for s in range(0, len(seq) - self.t_total + 1, self.t_his)]
        meta = z['test_clip_meta']
        assert len(meta) == len(self.windows) and np.array_equal(np.asarray(self.windows), meta), \
            f'test window order does not match test_clip_meta of {self.multimodal_path}'
        print(f'{self.__class__.__name__}: {len(self.windows)} test windows from '
              f'{sum(len(seq) >= self.t_total for seq in self.seqs)} sequences')

    def iter_generator(self, step=None):
        assert step is None or step == self.t_his, 'the evaluation protocol uses stride t_his'
        for gi, (sid, st) in enumerate(self.windows):
            traj = self.seqs[sid][None, st:st + self.t_total]
            idx = self.multi_dict.get(np.int64(gi), np.array([], dtype=np.int32))
            if len(idx) == 0:
                multi_traj = traj.copy()
            else:
                # candidate futures, history replaced by the observed history
                multi_traj = self.data_candi[idx]
                multi_traj[:, :self.t_his] = traj[:, :self.t_his]
            yield traj, multi_traj
