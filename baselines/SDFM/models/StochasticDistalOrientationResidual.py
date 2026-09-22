import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DISTAL_EDGES_AP3D = [(1, 2), (4, 5), (11, 12), (14, 15)]
LIMB_CHAIN_EDGES_AP3D = [(6, 0), (0, 1), (1, 2), (6, 3), (3, 4), (4, 5), (7, 10), (10, 11), (11, 12), (7, 13), (13, 14), (14, 15)]
LIMB_CHAIN_ORDER_AP3D = [[6, 0, 1, 2], [6, 3, 4, 5], [7, 10, 11, 12], [7, 13, 14, 15]]


def build_dct_basis(T, K, device=None, dtype=torch.float32):
    n = torch.arange(T, device=device, dtype=dtype)
    k = torch.arange(K, device=device, dtype=dtype).unsqueeze(1)
    basis = torch.cos(math.pi / T * (n + 0.5) * k)
    basis[0] = basis[0] * math.sqrt(1.0 / T)
    if K > 1:
        basis[1:] = basis[1:] * math.sqrt(2.0 / T)
    return basis


def safe_unit(v, eps=1e-6):
    return v / torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def to_xyz(x, num_joints=16):
    if x.dim() == 4:
        return x
    B, T, D = x.shape
    return x.reshape(B, T, num_joints, 3)


class StochasticDistalOrientationResidual(nn.Module):
    def __init__(self, dim=512, t_pred=60, k_dir=8, num_joints=16,
                 edges=DISTAL_EDGES_AP3D, dropout=0.1, init_gate=-4.0):
        super().__init__()

        self.dim = dim
        self.t_pred = t_pred
        self.k_dir = k_dir
        self.num_joints = num_joints
        self.edges = edges
        self.num_edges = len(edges)

        dct = build_dct_basis(t_pred, k_dir)
        self.register_buffer("idct", dct.T.contiguous())

        parent_ids = torch.tensor([p for p, c in edges], dtype=torch.long)
        child_ids = torch.tensor([c for p, c in edges], dtype=torch.long)

        self.register_buffer("parent_ids", parent_ids)
        self.register_buffer("child_ids", child_ids)

        self.feat_proj = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, k_dir * self.num_edges * 6)
        )

        self.gate = nn.Parameter(torch.tensor(init_gate))

        nn.init.zeros_(self.feat_proj[-1].weight)
        nn.init.zeros_(self.feat_proj[-1].bias)

    def forward(self, pred_traj, latent_x, sample=False,
                temperature=1.0, detach_base_geometry=True):
        pred_xyz = to_xyz(pred_traj, self.num_joints)
        future = pred_xyz[:, -self.t_pred:].clone()

        B = future.shape[0]

        parents = self.parent_ids
        children = self.child_ids

        global_feat = latent_x.mean(dim=1)
        parent_feat = latent_x[:, parents].mean(dim=1)
        child_feat = latent_x[:, children].mean(dim=1)

        feat = torch.cat([global_feat, parent_feat, child_feat], dim=-1)

        params = self.feat_proj(feat).view(B, self.k_dir, self.num_edges, 6)
        params = torch.einsum("tk,bked->bted", self.idct.to(params.device), params)

        mu = params[..., :3]
        logvar = params[..., 3:].clamp(-6.0, 2.0)

        parent = future[:, :, parents]
        child = future[:, :, children]

        bone = child - parent
        length = torch.norm(bone, dim=-1, keepdim=True).clamp_min(1e-6)
        u = bone / length

        if detach_base_geometry:
            parent = parent.detach()
            length = length.detach()
            u = u.detach()

        mu = mu - (mu * u).sum(dim=-1, keepdim=True) * u

        if sample:
            eps = torch.randn_like(mu)
            r = mu + temperature * torch.exp(0.5 * logvar) * eps
        else:
            r = mu

        r = r - (r * u).sum(dim=-1, keepdim=True) * u

        alpha = torch.sigmoid(self.gate)
        u_new = safe_unit(u + alpha * r)

        refined_child = parent + length * u_new

        future[:, :, children] = refined_child

        refined = pred_xyz.clone()
        refined[:, -self.t_pred:] = future

        return refined.reshape(pred_traj.shape[0], pred_traj.shape[1], -1), mu, logvar


def distal_orientation_nll_loss(pred_traj, gt_traj, mu, logvar,
                                t_pred=60, num_joints=16, edges=DISTAL_EDGES_AP3D):
    pred_xyz = to_xyz(pred_traj, num_joints)[:, -t_pred:]
    gt_xyz = to_xyz(gt_traj, num_joints)[:, -t_pred:]

    parents = torch.tensor([p for p, c in edges], device=pred_xyz.device, dtype=torch.long)
    children = torch.tensor([c for p, c in edges], device=pred_xyz.device, dtype=torch.long)

    p_parent = pred_xyz[:, :, parents]
    p_child = pred_xyz[:, :, children]

    g_parent = gt_xyz[:, :, parents]
    g_child = gt_xyz[:, :, children]

    u = safe_unit(p_child - p_parent)
    u_gt = safe_unit(g_child - g_parent)

    target = u_gt - (u_gt * u).sum(dim=-1, keepdim=True) * u

    inv_var = torch.exp(-logvar)

    loss = 0.5 * ((target - mu).pow(2) * inv_var + logvar)

    return loss.mean()


def distal_endpoint_loss(refined_traj, gt_traj, t_pred=60,
                         num_joints=16, end_joints=[2, 5, 12, 15]):
    pred = to_xyz(refined_traj, num_joints)[:, -t_pred:, end_joints]
    gt = to_xyz(gt_traj, num_joints)[:, -t_pred:, end_joints]
    return F.smooth_l1_loss(pred, gt)


def apply_distal_direction_oracle(pred_traj, gt_traj, num_joints=16, edges=DISTAL_EDGES_AP3D):
    """
    Diagnostic oracle only.

    pred_traj:
        [K, N, T, D] or [K, T, D] or [B, T, D]

    gt_traj:
        [N, T, D] or [1, T, D] or [T, D]

    return:
        same shape as pred_traj

    This function replaces only distal bone directions:
        RKnee -> RFoot
        LKnee -> LFoot
        LElbow -> LWrist
        RElbow -> RWrist

    Parent joint position and predicted bone length are preserved.
    GT distal direction is used.
    """

    pred_is_numpy = isinstance(pred_traj, np.ndarray)

    if torch.is_tensor(pred_traj):
        pred = pred_traj
        device = pred.device
        dtype = pred.dtype
    else:
        pred = torch.from_numpy(pred_traj).float()
        device = pred.device
        dtype = pred.dtype

    if torch.is_tensor(gt_traj):
        gt = gt_traj.to(device=device, dtype=dtype)
    else:
        gt = torch.from_numpy(gt_traj).to(device=device, dtype=dtype)

    pred_shape = pred.shape

    if pred.shape[-1] != num_joints * 3:
        raise ValueError(f"pred last dim should be {num_joints * 3}, got {pred.shape[-1]}")

    if gt.shape[-1] != num_joints * 3:
        raise ValueError(f"gt last dim should be {num_joints * 3}, got {gt.shape[-1]}")

    pred_xyz = pred.reshape(*pred.shape[:-1], num_joints, 3).clone()
    gt_xyz = gt.reshape(*gt.shape[:-1], num_joints, 3)

    for p, c in edges:
        parent = pred_xyz[..., p, :]
        child = pred_xyz[..., c, :]

        pred_bone = child - parent
        pred_len = torch.norm(pred_bone, dim=-1, keepdim=True).clamp_min(1e-6)

        gt_dir = gt_xyz[..., c, :] - gt_xyz[..., p, :]
        gt_dir = gt_dir / torch.norm(gt_dir, dim=-1, keepdim=True).clamp_min(1e-6)

        pred_xyz[..., c, :] = parent + pred_len * gt_dir

    out = pred_xyz.reshape(*pred_shape)

    if pred_is_numpy:
        return out.cpu().numpy()

    return out

LIMB_CHAINS_AP3D = [[0, 1, 2], [3, 4, 5], [10, 11, 12], [13, 14, 15]]
def apply_limb_chain_direction_oracle(pred_traj, gt_traj, num_joints=16, chains=LIMB_CHAINS_AP3D):
    pred_is_numpy = isinstance(pred_traj, np.ndarray)

    if torch.is_tensor(pred_traj):
        pred = pred_traj
        device = pred.device
        dtype = pred.dtype
    else:
        pred = torch.from_numpy(pred_traj).float()
        device = pred.device
        dtype = pred.dtype

    if torch.is_tensor(gt_traj):
        gt = gt_traj.to(device=device, dtype=dtype)
    else:
        gt = torch.from_numpy(gt_traj).to(device=device, dtype=dtype)

    pred_shape = pred.shape

    pred_xyz = pred.reshape(*pred.shape[:-1], num_joints, 3).clone()
    gt_xyz = gt.reshape(*gt.shape[:-1], num_joints, 3)

    for chain in chains:
        for i in range(len(chain) - 1):
            p = chain[i]
            c = chain[i + 1]

            parent = pred_xyz[..., p, :]
            child = pred_xyz[..., c, :]

            pred_bone = child - parent
            pred_len = torch.norm(pred_bone, dim=-1, keepdim=True).clamp_min(1e-6)

            gt_dir = gt_xyz[..., c, :] - gt_xyz[..., p, :]
            gt_dir = gt_dir / torch.norm(gt_dir, dim=-1, keepdim=True).clamp_min(1e-6)

            pred_xyz[..., c, :] = parent + pred_len * gt_dir

    out = pred_xyz.reshape(*pred_shape)

    if pred_is_numpy:
        return out.cpu().numpy()

    return out


class WholeLimbChainOrientationRefiner(nn.Module):
    def __init__(self, dim=512, t_pred=60, k_dir=10, num_joints=16, edges=LIMB_CHAIN_EDGES_AP3D,
                 chains=LIMB_CHAIN_ORDER_AP3D, dropout=0.2, init_gate=-2.0):
        super().__init__()
        self.dim = dim
        self.t_pred = t_pred
        self.k_dir = k_dir
        self.num_joints = num_joints
        self.edges = edges
        self.chains = chains
        self.num_edges = len(edges)

        parent_ids = torch.tensor([p for p, c in edges], dtype=torch.long)
        child_ids = torch.tensor([c for p, c in edges], dtype=torch.long)

        self.register_buffer("parent_ids", parent_ids)
        self.register_buffer("child_ids", child_ids)

        dct = build_dct_basis(t_pred, k_dir)
        self.register_buffer("idct", dct.T.contiguous())

        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(dim * 3 + 12),
            nn.Linear(dim * 3 + 12, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, k_dir * 3)
        )

        self.gate = nn.Parameter(torch.tensor(init_gate))

        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)

    def base_direction_summary(self, base_traj):
        xyz = to_xyz(base_traj, self.num_joints)[:, -self.t_pred:]
        p = xyz[:, :, self.parent_ids]
        c = xyz[:, :, self.child_ids]

        u = safe_unit(c - p)

        if u.shape[1] > 1:
            du = u[:, 1:] - u[:, :-1]
            du_last = du[:, -1]
        else:
            du_last = torch.zeros_like(u[:, -1])

        u_last = u[:, -1]
        u_mean = u.mean(dim=1)
        u_std = u.std(dim=1, unbiased=False)

        return torch.cat([u_last, u_mean, u_std, du_last], dim=-1)

    def forward(self, base_traj, latent_x, strength=None):
        base_xyz = to_xyz(base_traj, self.num_joints)
        future = base_xyz[:, -self.t_pred:].clone()
        future_orig = future.clone()

        B = future.shape[0]
        E = self.num_edges

        global_feat = latent_x.mean(dim=1, keepdim=True).expand(B, E, self.dim)
        parent_feat = latent_x[:, self.parent_ids]
        child_feat = latent_x[:, self.child_ids]
        dir_feat = self.base_direction_summary(base_traj)

        edge_feat = torch.cat([global_feat, parent_feat, child_feat, dir_feat], dim=-1)

        residual_dct = self.edge_mlp(edge_feat).view(B, E, self.k_dir, 3).permute(0, 2, 1, 3)
        residual = torch.einsum("tk,bked->bted", self.idct.to(residual_dct.device), residual_dct)

        base_parent = future_orig[:, :, self.parent_ids]
        base_child = future_orig[:, :, self.child_ids]
        base_u = safe_unit(base_child - base_parent)

        residual = residual - (residual * base_u).sum(dim=-1, keepdim=True) * base_u

        if strength is None:
            alpha = torch.sigmoid(self.gate)
        else:
            alpha = strength

        u_new_all = safe_unit(base_u + alpha * residual)

        edge_to_id = {edge: i for i, edge in enumerate(self.edges)}

        for chain in self.chains:
            for i in range(len(chain) - 1):
                p = chain[i]
                c = chain[i + 1]
                e = edge_to_id[(p, c)]

                parent_new = future[:, :, p]
                length_orig = torch.norm(future_orig[:, :, c] - future_orig[:, :, p], dim=-1, keepdim=True).clamp_min(1e-6)

                future[:, :, c] = parent_new + length_orig * u_new_all[:, :, e]

        refined = base_xyz.clone()
        refined[:, -self.t_pred:] = future

        return refined.reshape(base_traj.shape[0], base_traj.shape[1], -1), u_new_all


def limb_chain_direction_distill_loss(base_traj, refined_traj, gt_traj,
                                      t_pred=60, num_joints=16, edges=LIMB_CHAIN_EDGES_AP3D,
                                      chains=LIMB_CHAIN_ORDER_AP3D):
    base = to_xyz(base_traj, num_joints)[:, -t_pred:]
    refined = to_xyz(refined_traj, num_joints)[:, -t_pred:]
    gt = to_xyz(gt_traj, num_joints)[:, -t_pred:]

    parents = torch.tensor([p for p, c in edges], device=base.device, dtype=torch.long)
    children = torch.tensor([c for p, c in edges], device=base.device, dtype=torch.long)

    u_ref = safe_unit(refined[:, :, children] - refined[:, :, parents])
    u_gt = safe_unit(gt[:, :, children] - gt[:, :, parents])

    loss_dir = (1.0 - (u_ref * u_gt).sum(dim=-1)).mean()

    oracle = apply_limb_chain_direction_oracle_preserve_length(base_traj, gt_traj, num_joints=num_joints, chains=chains)
    oracle = oracle.to(base_traj.device) if torch.is_tensor(base_traj) else torch.from_numpy(oracle).to(base.device)

    refined_xyz = to_xyz(refined_traj, num_joints)[:, -t_pred:]
    oracle_xyz = to_xyz(oracle, num_joints)[:, -t_pred:]

    target_joints = sorted(list(set([j for chain in chains for j in chain[1:]])))
    loss_pos = F.smooth_l1_loss(refined_xyz[:, :, target_joints], oracle_xyz[:, :, target_joints])

    return loss_dir + 0.2 * loss_pos, loss_dir.detach(), loss_pos.detach()


LIMB_CHAINS_AP3D_FULL = [[6, 0, 1, 2], [6, 3, 4, 5], [7, 10, 11, 12], [7, 13, 14, 15]]

def apply_limb_chain_direction_oracle_preserve_length(pred_traj, gt_traj, num_joints=16, chains=LIMB_CHAINS_AP3D_FULL):
    pred_is_numpy = isinstance(pred_traj, np.ndarray)

    if torch.is_tensor(pred_traj):
        pred = pred_traj
        device = pred.device
        dtype = pred.dtype
    else:
        pred = torch.from_numpy(pred_traj).float()
        device = pred.device
        dtype = pred.dtype

    if torch.is_tensor(gt_traj):
        gt = gt_traj.to(device=device, dtype=dtype)
    else:
        gt = torch.from_numpy(gt_traj).to(device=device, dtype=dtype)

    pred_shape = pred.shape
    pred_xyz = pred.reshape(*pred.shape[:-1], num_joints, 3).clone()
    pred_orig = pred_xyz.clone()
    gt_xyz = gt.reshape(*gt.shape[:-1], num_joints, 3)

    for chain in chains:
        for i in range(len(chain) - 1):
            p = chain[i]
            c = chain[i + 1]

            parent_new = pred_xyz[..., p, :]
            length_orig = torch.norm(pred_orig[..., c, :] - pred_orig[..., p, :], dim=-1, keepdim=True).clamp_min(1e-6)

            gt_dir = gt_xyz[..., c, :] - gt_xyz[..., p, :]
            gt_dir = gt_dir / torch.norm(gt_dir, dim=-1, keepdim=True).clamp_min(1e-6)

            pred_xyz[..., c, :] = parent_new + length_orig * gt_dir

    out = pred_xyz.reshape(*pred_shape)

    if pred_is_numpy:
        return out.cpu().numpy()

    return out