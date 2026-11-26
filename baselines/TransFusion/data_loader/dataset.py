import numpy as np


class Dataset:
    """Training / validation dataset: random windows of t_his + t_pred frames."""

    def __init__(self, mode, t_his, t_pred):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.prepare_data()

    def prepare_data(self):
        raise NotImplementedError


def per_sequence_windows(lengths, t_total, step):
    """(sequence id, start frame) of every evaluation window.

    Windows never cross a sequence boundary: each test sequence that is at least
    t_total frames long contributes the windows starting at 0, step, 2*step, ...
    """
    return [(sid, s) for sid, L in enumerate(lengths) if L >= t_total
            for s in range(0, L - t_total + 1, step)]


class MultimodalTestSet:
    """Per-sequence evaluation windows with their multimodal ground-truth futures.

    sequences:        list of [T_i, J, 3] float32 test sequences (already root-centred/normalised)
    multimodal_path:  npz with 'data_multimodal' (window index -> candidate indices) and, for
                      AthletePose3D / AthleticsPose, 'test_clip_meta' = [(sequence id, start), ...]
    data_candi_path:  npz with the candidate pool 'data_candidate.npy' [N, t_total, J, 3]

    The window order is the order of the multimodal file; it is asserted against
    'test_clip_meta' when the file provides it (otherwise only the count is checked).
    """

    def __init__(self, sequences, t_his, t_pred, multimodal_path, data_candi_path, name=''):
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.sequences = sequences

        self.windows = per_sequence_windows([len(s) for s in sequences], self.t_total, step=t_his)

        z = np.load(multimodal_path, allow_pickle=True)
        multi = z['data_multimodal'].item()['all_subjects']['all_actions']
        self.multi_dict = {int(k): np.asarray(v) for k, v in multi.items()}
        if 'test_clip_meta' in z.files:
            meta = np.asarray(z['test_clip_meta'])
            assert len(meta) == len(self.windows) and np.array_equal(np.asarray(self.windows), meta), \
                f'{name}: evaluation window order != test_clip_meta of {multimodal_path}'
        else:
            n_keys = max(self.multi_dict.keys()) + 1 if self.multi_dict else 0
            assert n_keys == len(self.windows), \
                f'{name}: {len(self.windows)} windows but {n_keys} entries in {multimodal_path}'

        c = np.load(data_candi_path, allow_pickle=True)
        key = 'data_candidate.npy' if 'data_candidate.npy' in c.files else c.files[0]
        self.data_candi = c[key].astype(np.float32)
        assert self.data_candi.shape[1] == self.t_total, f'{name}: candidate length != t_his + t_pred'
        max_idx = max((int(np.max(v)) for v in self.multi_dict.values() if len(v)), default=-1)
        assert max_idx < len(self.data_candi), f'{name}: candidate index out of range'

        n_seq = sum(len(s) >= self.t_total for s in sequences)
        print(f'[{name}] per-sequence evaluation windows: {len(self.windows)} from {n_seq} sequences')

    def __len__(self):
        return len(self.windows)

    def iter_generator(self):
        for gi, (sid, st) in enumerate(self.windows):
            traj = self.sequences[sid][None, st:st + self.t_total].astype(np.float32)
            idx = self.multi_dict.get(gi, np.array([], dtype=np.int64))
            if len(idx) == 0:
                multi_traj = traj.copy()
            else:
                multi_traj = self.data_candi[idx].copy()
                multi_traj[:, :self.t_his] = traj[:, :self.t_his]
            yield traj, multi_traj
