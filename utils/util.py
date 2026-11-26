import os
import random

import torch as tr
import numpy as np


def seed_set(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)


def generate_pad(padding, t_his, t_pred):
    zero_index = None
    if padding == 'Zero':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        zero_index = max(idx_pad)
    elif padding == 'Repeat':
        idx_pad = list(range(t_his)) * int(((t_pred + t_his) / t_his))
        # [0, 1, 2,....,24, 0, 1, 2,....,24, 0, 1, 2,...., 24...]
    elif padding == 'LastFrame':
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        # [0, 1, 2,....,24, 24, 24,.....]
    else:
        raise NotImplementedError(f"unknown padding method: {padding}")
    return idx_pad, zero_index


def padding_traj(traj, padding, idx_pad, zero_index):
    if padding == 'Zero':
        traj_tmp = traj
        traj_tmp[..., zero_index, :] = 0
        traj_pad = traj_tmp[..., idx_pad, :]
    else:
        traj_pad = traj[..., idx_pad, :]

    return traj_pad


def post_process(pred, cfg):
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
    pred = np.concatenate((np.tile(np.zeros((1, cfg.t_his + cfg.t_pred, 1, 3)), (pred.shape[0], 1, 1, 1)), pred),
                          axis=2)
    pred[..., :1, :] = 0
    return pred


def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    # idct_m = np.linalg.inv(dct_m)
    idct_m = dct_m.T
    if is_torch:
        dct_m = tr.from_numpy(dct_m).float()
        idct_m = tr.from_numpy(idct_m).float()
    return dct_m, idct_m


def get_keyframe_matrices(N, K, is_torch=True):
    """Time-domain control for the DCT ablation (temporal_basis: 'keyframe').

    Returns (enc, dec), both N x N, used exactly like (dct_m, idct_m):
    enc[:K] selects K uniformly spaced frames (first and last frame included),
    dec[:, :K] reconstructs all N frames by piecewise-linear interpolation
    between the selected frames, so enc[:K] @ dec[:, :K] = I_K. Rows/columns
    beyond K are zero and are never used (callers slice [:K]).
    """
    idx = np.round(np.linspace(0, N - 1, K)).astype(int)
    assert len(np.unique(idx)) == K, "keyframes must be distinct"
    enc = np.zeros((N, N))
    dec = np.zeros((N, N))
    for k, t in enumerate(idx):
        enc[k, t] = 1.0
    for t in range(N):
        j = np.searchsorted(idx, t, side='right') - 1
        j = min(max(j, 0), K - 2)
        t0, t1 = idx[j], idx[j + 1]
        w = (t - t0) / (t1 - t0)
        dec[t, j] = 1.0 - w
        dec[t, j + 1] = w
    if is_torch:
        enc = tr.from_numpy(enc).float()
        dec = tr.from_numpy(dec).float()
    return enc, dec


def get_time_discretization(nfes: int, rho=7):
    step_indices = tr.arange(nfes, dtype=tr.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = tr.cat([sigma_vec, tr.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - tr.clip(time_vec, min=0.0, max=1.0)
    return t_samples
