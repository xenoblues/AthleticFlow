import os
import random

# import adan
import torch
import torch as tr
import torch.nn.functional as F
import numpy as np
import pywt

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


def padding_vel(vel, padding, idx_pad, zero_index):
    if padding == 'Zero':
        vel_tmp = vel
        vel_tmp[..., zero_index, :, :] = 0
        vel_pad = vel_tmp[..., idx_pad, :, :]
    else:
        vel_pad = vel[..., idx_pad, :, :]

    return vel_pad


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


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = tr.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tr.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 - mask) * tr.sqrt(distances)

    return distances


def _pairwise_distances_l1(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    distances = tr.abs(embeddings[None, :, :] - embeddings[:, None, :])
    return distances


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
    return R


def absolute2relative(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / np.linalg.norm(xt, axis=-1, keepdims=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = np.linalg.norm(x0[..., 1:, :] - x0[..., parents[1:], :], axis=-1, keepdims=True)
        xt = x * limb_l
        xt0 = np.zeros_like(xt[..., :1, :])
        xt = np.concatenate([xt0, xt], axis=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt


def absolute2relative_torch(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / tr.norm(xt, dim=-1, keepdim=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = tr.norm(x0[..., 1:, :] - x0[..., parents[1:], :], dim=-1, keepdim=True)
        xt = x * limb_l
        xt0 = tr.zeros_like(xt[..., :1, :])
        xt = tr.cat([xt0, xt], dim=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def multiscale_filters(A, num):
    # L 图的拉普拉斯矩阵
    I = np.identity(A.shape[0])
    # for i in range(L.shape):
    # I[i, i] = 1
    # A_hat = 1/2 * (I + L)
    # T = I - A_hat
    filters = [A]
    assert num > 0
    for i in range(1, num):
        filters.append(filters[i - 1] ** (2 ** (i - 1)) - filters[i - 1] ** (2 ** i))
    return np.asarray(filters)


def get_temporal_graph(num_node):
    A = np.eye(num_node, dtype=float)
    for i in range(num_node):
        if i - 1 >= 0:
            A[i, i - 1] = 1
        if i + 1 < num_node:
            A[i, i + 1] = 1
    A = normalize_digraph(A)
    return A


def cal_vel_acc(traj):
    traj_tmp = traj.clone().reshape([traj.shape[0], traj.shape[1], -1, 3])[:, :-1, :, :]
    traj_tmp2 = traj.clone().reshape([traj.shape[0], traj.shape[1], -1, 3])[:, 1:, :, :]
    vel = tr.linalg.norm(traj_tmp2 - traj_tmp, dim=-1).unsqueeze(-1)
    acc = vel[:, 1:, :, :] - vel[:, :-1, :, :]
    vel = tr.cat((vel, vel[:, -1:, :, :]), dim=1)
    acc = tr.cat((acc, acc[:, -1:, :, :], acc[:, -1:, :, :]), dim=1)
    vel_acc = tr.cat((vel, acc), dim=-1)
    return vel_acc


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


def get_causal_mask(n, m):
    causal_mask = torch.zeros(n, m)
    d = abs(n - m)
    for i in range(n):
        for j in range(m):
            if j > i + d:
                causal_mask[i, j] = 0
            else:
                causal_mask[i, j] = 1
    return causal_mask

class AccelerationProjection(torch.nn.Module):
    def __init__(self, dct_m_all, n_pre, pred_len):
        super().__init__()
        K = n_pre
        T = pred_len

        # -------------------------
        # Truncated IDCT basis
        # -------------------------
        Ck = dct_m_all[:K]      # (K,T)
        Ck_inv = Ck.T           # (T,K)

        # -------------------------
        # Second-order difference
        # -------------------------
        D2 = torch.zeros(T - 2, T).to(dct_m_all.device)
        for i in range(T - 2):

            D2[i, i] = 1.
            D2[i, i + 1] = -2.
            D2[i, i + 2] = 1.

        # -------------------------
        # Acceleration projector
        # -------------------------
        acc_proj = D2 @ Ck_inv
        acc_proj = acc_proj / (acc_proj.norm(dim=1, keepdim=True) + 1e-6)

        self.register_buffer("acc_proj", acc_proj.float())

    def forward(self, dct_coeff):
        """
        dct_coeff
        shape:
        B × K × J*C

        return:
        B × (T-2) × J*C
        """

        return torch.einsum("bkj,tk->btj", dct_coeff, self.acc_proj)


import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalDCTTruncation(nn.Module):
    def __init__(self, dct_m_all, idct_m_all=None, k_max=15, k_torso=12, k_mid=14, k_end=15, num_joints=16):
        super().__init__()

        self.k_max = k_max
        self.num_joints = num_joints
        self.data_dim = num_joints * 3

        if not torch.is_tensor(dct_m_all):
            dct_m_all = torch.tensor(dct_m_all, dtype=torch.float32)
        else:
            dct_m_all = dct_m_all.float()

        self.t_total = dct_m_all.shape[1]
        dct_basis = dct_m_all[:k_max].contiguous()

        if idct_m_all is None:
            idct_basis = dct_basis.T.contiguous()
        else:
            if not torch.is_tensor(idct_m_all):
                idct_m_all = torch.tensor(idct_m_all, dtype=torch.float32)
            else:
                idct_m_all = idct_m_all.float()
            idct_basis = idct_m_all[:, :k_max].contiguous()

        self.register_buffer("dct_basis", dct_basis)
        self.register_buffer("idct_basis", idct_basis)

        torso_joints = [0, 3, 6, 7, 8, 9]
        mid_joints = [1, 4, 10, 11, 13, 14]
        end_joints = [2, 5, 12, 15]

        freq_mask = torch.zeros(k_max, num_joints, 3)
        freq_mask[:k_torso, torso_joints, :] = 1.0
        freq_mask[:k_mid, mid_joints, :] = 1.0
        freq_mask[:k_end, end_joints, :] = 1.0
        freq_mask = freq_mask.reshape(k_max, num_joints * 3)

        self.register_buffer("freq_mask", freq_mask)

    def encode(self, traj):
        B, T, D = traj.shape
        z = torch.einsum("kt,btd->bkd", self.dct_basis, traj)
        z = z * self.freq_mask[None]
        return z

    def decode(self, z):

        B, K, D = z.shape
        z = z * self.freq_mask[None]
        traj = torch.einsum("tk,bkd->btd", self.idct_basis, z)
        return traj

    def mask(self, z):
        return z * self.freq_mask[None]

    def masked_mse(self, pred, target):
        mask = self.freq_mask[None]
        loss = (pred - target).pow(2) * mask
        return loss.sum() / (mask.sum() * pred.shape[0] + 1e-8)

    def masked_smooth_l1(self, pred, target):
        mask = self.freq_mask[None]
        loss = F.smooth_l1_loss(pred, target, reduction="none") * mask
        return loss.sum() / (mask.sum() * pred.shape[0] + 1e-8)


def _to_xyz(x, num_joints=16):
    if x.dim() == 4:
        return x
    B, T, D = x.shape
    assert D == num_joints * 3, f"Expected D={num_joints * 3}, got {D}"
    return x.view(B, T, num_joints, 3)


def _safe_unit(v, eps=1e-6):
    return v / torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def _unit_dot(a, b):
    return (_safe_unit(a) * _safe_unit(b)).sum(dim=-1, keepdim=True)


def _angle_sincos(xyz, a, b, c):
    ba = _safe_unit(xyz[:, :, a] - xyz[:, :, b])
    bc = _safe_unit(xyz[:, :, c] - xyz[:, :, b])
    cos = (ba * bc).sum(dim=-1).clamp(-1.0, 1.0)
    sin = torch.norm(torch.cross(ba, bc, dim=-1), dim=-1).clamp(0.0, 1.0)
    return cos, sin


def _dist_last(xyz, i, j, eps=1e-6):
    return torch.norm(xyz[:, -1, i] - xyz[:, -1, j], dim=-1).clamp_min(eps)


def compute_contact_free_athletic_state(traj_his, num_joints=16):
    xyz = _to_xyz(traj_his, num_joints)
    B, T, J, _ = xyz.shape

    vel = xyz[:, 1:] - xyz[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1] if T > 2 else torch.zeros(B, 1, J, 3, device=xyz.device, dtype=xyz.dtype)

    torso_idx = [0, 3, 6, 7]
    end_idx = [2, 5, 12, 15]

    body_center = xyz[:, :, torso_idx].mean(dim=2)
    body_vel = body_center[:, 1:] - body_center[:, :-1]
    body_acc = body_vel[:, 1:] - body_vel[:, :-1] if T > 2 else torch.zeros(B, 1, 3, device=xyz.device, dtype=xyz.dtype)

    body_v_last = body_vel[:, -1]
    body_a_last = body_acc[:, -1]
    body_speed = torch.norm(body_vel, dim=-1)
    body_speed_feat = torch.stack([body_speed[:, -1], body_speed.mean(dim=1), body_speed.std(dim=1, unbiased=False)], dim=-1)

    end_speed = torch.norm(vel[:, :, end_idx], dim=-1)
    end_speed_last = end_speed[:, -1]
    end_speed_mean = end_speed.mean(dim=1)
    end_speed_std = end_speed.std(dim=1, unbiased=False)

    end_acc = torch.norm(acc[:, :, end_idx], dim=-1)
    end_acc_last = end_acc[:, -1]
    end_acc_mean = end_acc.mean(dim=1)

    rk_cos, rk_sin = _angle_sincos(xyz, 0, 1, 2)
    lk_cos, lk_sin = _angle_sincos(xyz, 3, 4, 5)
    le_cos, le_sin = _angle_sincos(xyz, 10, 11, 12)
    re_cos, re_sin = _angle_sincos(xyz, 13, 14, 15)

    angle_seq = torch.stack([rk_cos, rk_sin, lk_cos, lk_sin, le_cos, le_sin, re_cos, re_sin], dim=-1)

    angle_last = angle_seq[:, -1]
    angle_mean = angle_seq.mean(dim=1)
    angle_std = angle_seq.std(dim=1, unbiased=False)

    if T > 1:
        angle_delta = angle_seq[:, -1] - angle_seq[:, -2]
    else:
        angle_delta = torch.zeros_like(angle_last)

    v_last = vel[:, -1]

    rfoot_v = v_last[:, 2]
    lfoot_v = v_last[:, 5]
    lwrist_v = v_last[:, 12]
    rwrist_v = v_last[:, 15]

    coordination = torch.cat([_unit_dot(rfoot_v, lwrist_v), _unit_dot(lfoot_v, rwrist_v), _unit_dot(rfoot_v, lfoot_v), _unit_dot(lwrist_v, rwrist_v)], dim=-1)

    rleg_ext = _dist_last(xyz, 0, 2) / (_dist_last(xyz, 0, 1) + _dist_last(xyz, 1, 2))
    lleg_ext = _dist_last(xyz, 3, 5) / (_dist_last(xyz, 3, 4) + _dist_last(xyz, 4, 5))
    larm_ext = _dist_last(xyz, 10, 12) / (_dist_last(xyz, 10, 11) + _dist_last(xyz, 11, 12))
    rarm_ext = _dist_last(xyz, 13, 15) / (_dist_last(xyz, 13, 14) + _dist_last(xyz, 14, 15))

    extension = torch.stack([rleg_ext, lleg_ext, larm_ext, rarm_ext], dim=-1)

    if T > 1:
        distal_edges = [(1, 2), (4, 5), (11, 12), (14, 15)]
        dir_change = []

        for p, c in distal_edges:
            d_last = _safe_unit(xyz[:, -1, c] - xyz[:, -1, p])
            d_prev = _safe_unit(xyz[:, -2, c] - xyz[:, -2, p])
            dir_change.append(1.0 - (d_last * d_prev).sum(dim=-1))

        dir_change = torch.stack(dir_change, dim=-1)
    else:
        dir_change = torch.zeros(B, 4, device=xyz.device, dtype=xyz.dtype)

    state = torch.cat([body_v_last, body_a_last, body_speed_feat, end_speed_last, end_speed_mean, end_speed_std, end_acc_last, end_acc_mean, angle_last, angle_delta, angle_mean, angle_std, coordination, extension, dir_change], dim=-1)

    return state

if __name__ == '__main__':
    # 示例数据：T=100帧, J=17个关节, 每个3D坐标
    T, J, C = 100, 17, 3
        # coeffs, coeff_shapes = dwt(data, wavelet='db1', level=3)
        # print(coeff_shapes[0])

    # 重建动作序列
    # data_rec = idwt(coeffs, coeff_shapes, T, J, C, wavelet='db1')

    # 验证误差
    # print("最大重建误差：", np.abs(data - data_rec).max())
    mat = get_causal_mask(4, 6)
    print(mat)
