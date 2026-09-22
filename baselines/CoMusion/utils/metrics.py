import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch
import math
from .fid import FID
from einops import rearrange


def time_slice(array, t0, t, axis):
    if t == -1:
        return torch.index_select(array, axis,
                                  torch.arange(t0, array.shape[axis], device=array.device, dtype=torch.int32))
    else:
        return torch.index_select(array, axis, torch.arange(t0, t, device=array.device, dtype=torch.int32))


def APD(pred, target, *args, t0=0, t=-1):
    """
    pred: [b, n_sample, t_pred, NC]; target: [b, t_pred, NC];
    """
    pred = time_slice(pred, t0, t, 2)
    batch_size, n_samples = pred.shape[:2]
    if n_samples == 1:  # only one sample => no APD possible
        return torch.tensor([0] * batch_size, device=pred.device)

    all_apd = torch.tensor([0.0] * batch_size, dtype=torch.float, device=pred.device)
    for b in range(batch_size):
        dist_diverse = torch.pdist(pred[b].reshape(n_samples, -1))
        all_apd += dist_diverse.mean(-1)

    return all_apd / batch_size


def ADE(pred, target, *args, t0=0, t=-1):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    target = target.reshape((batch_size, 1, seq_length, -1))

    diff = pred - target
    dist = torch.linalg.norm(diff, axis=-1).mean(axis=-1)
    return dist.min(axis=-1).values


def FDE(pred, target, *args, t0=0, t=-1):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    batch_size, n_samples, seq_length = pred.shape[:3]
    pred = pred.reshape((batch_size, n_samples, seq_length, -1))
    target = target.reshape((batch_size, 1, seq_length, -1))

    diff = pred - target
    dist = torch.linalg.norm(diff, axis=-1)[..., -1]
    return dist.min(axis=-1).values


def MMADE(pred, target, gt_multi, *args, t0=0, t=-1):  # memory efficient version
    """
    pred: [b, sample_num, t_pred, NC]
    target[b, t_pred, NC]
    """
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    gt_num = len(gt_multi)
    if isinstance(gt_multi, list):
        gt_multi = torch.cat(gt_multi, dim=0).reshape(gt_num, pred.shape[2], -1)
        if len(gt_multi.shape) == 3:
            gt_multi = gt_multi.unsqueeze(0)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size,))
    for i in range(batch_size):  # gt_multi[i] [num_similar, t_pred, nc];  pred[i]: [num_sample, t_pred, nc]
        n_gts = gt_multi[i].shape[0]
        if n_gts == 1:
            results[i] = float('nan')
            continue
        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)  # [1, num_sample, t_pred, nc]
        gt = time_slice(gt_multi[i], t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(
            1)  # [num_similar, 1, t_pred, nc]

        diff = p - gt
        dist = torch.linalg.norm(diff, axis=-1).mean(axis=-1)
        results[i] = dist.min(axis=-1).values.mean()

    return results


def MMFDE(pred, target, gt_multi, *args, t0=0, t=-1):
    pred, target = time_slice(pred, t0, t, 2), time_slice(target, t0, t, 1)
    gt_num = len(gt_multi)
    if isinstance(gt_multi, list):
        gt_multi = torch.cat(gt_multi, dim=0).reshape(gt_num, pred.shape[2], -1)
        if len(gt_multi.shape) == 3:
            gt_multi = gt_multi.unsqueeze(0)
    batch_size, n_samples, seq_length = pred.shape[:3]
    results = torch.zeros((batch_size,))
    for i in range(batch_size):
        n_gts = gt_multi[i].shape[0]
        if n_gts == 1:
            results[i] = float('nan')
            continue
        p = pred[i].reshape((n_samples, seq_length, -1)).unsqueeze(0)
        gt = time_slice(gt_multi[i], t0, t, 1).reshape((n_gts, seq_length, -1)).unsqueeze(1)

        diff = p - gt
        dist = torch.linalg.norm(diff, axis=-1)[..., -1]
        results[i] = dist.min(axis=-1).values.mean()

    return results


def APDE(curr_apds, gt_apds):
    """
    input: current batch apds [b, ]
    input: gt apds [b, ]: if zero, ignore
    return: [b, ], none indicating gt apd is 0
    """
    nonzero_idxs = torch.nonzero(gt_apds)
    zero_idxs = torch.nonzero(gt_apds.eq(0))
    ret = abs(curr_apds - gt_apds)
    ret[zero_idxs] = float('nan')
    return ret


def CMD(val_per_frame, val_ref):
    T = len(val_per_frame) + 1
    return np.sum([(T - t) * np.abs(val_per_frame[t - 1] - val_ref) for t in range(1, T)])


def CMD_helper(pred, extra, histogram_data, all_obs_classes):
    """
    pred: [b, num_s, t_pred, NC] -> [batch, num_s, t_pred, joint, 3]
    """
    pred_flat = rearrange(pred, '... (n c) -> ... n c', c=3)
    motion = (torch.linalg.norm(pred_flat[:, :, 1:] - pred_flat[:, :, :-1], axis=-1)).mean(axis=1).mean(axis=-1)

    histogram_data.append(motion.cpu().detach().numpy())
    classes = extra['act'].numpy()
    all_obs_classes.append(classes)

    return


def CMD_pose(dataset, histogram_data, all_obs_classes):
    """
    TODO: validate this function
    """
    ret = 0
    obs_classes = np.concatenate(all_obs_classes, axis=0)
    motion_data = np.concatenate(histogram_data, axis=0)
    motion_data_mean = motion_data.mean(axis=0)

    motion_per_class = np.zeros((dataset.num_actions, motion_data.shape[1]))
    # CMD weighted by class
    for i, (name, class_val_ref) in enumerate(zip(dataset.idx_to_class, dataset.mean_motion_per_class)):
        mask = obs_classes == i
        if mask.sum() == 0:
            continue
        motion_data_mean = motion_data[mask].mean(axis=0)
        motion_per_class[i] = motion_data_mean
        ret += CMD(motion_data_mean, class_val_ref) * (mask.sum() / obs_classes.shape[0])
    return ret


def FID_helper(pred, gt, classifier_for_fid, all_pred_activations, all_gt_activations, all_pred_classes,
               all_gt_classes):
    """
    pred: [b, sample_num, t_pred, NC]
    gt: [b, t_pred, NC]
    """
    b, s = pred.shape[0], pred.shape[1]

    pred_ = rearrange(pred, 'b s t d -> (b s) d t')  # [bs, nc, t_pred])
    gt_ = rearrange(gt, 'b t d -> b d t')  # [b, nc, t_pred]

    pred_activations = classifier_for_fid.get_fid_features(motion_sequence=pred_).cpu().data.numpy()
    gt_activations = classifier_for_fid.get_fid_features(motion_sequence=gt_).cpu().data.numpy()

    all_pred_activations.append(pred_activations)
    all_gt_activations.append(gt_activations)

    pred_classes = classifier_for_fid(motion_sequence=pred_.float()).cpu().data.numpy().argmax(axis=1)
    # recover the batch size and samples dimension
    pred_classes = pred_classes.reshape([b, s])
    gt_classes = classifier_for_fid(motion_sequence=gt_.float()).cpu().data.numpy().argmax(axis=1)
    # append to the list
    all_pred_classes.append(pred_classes)
    all_gt_classes.append(gt_classes)

    return


def FID_pose(all_gt_activations, all_pred_activations):
    ret = 0
    ret = FID(np.concatenate(all_gt_activations, axis=0), np.concatenate(all_pred_activations, axis=0))
    return ret


def compute_bone_length_error(pred_xyz, gt_xyz, edges):

    T, J, _ = pred_xyz.shape

    bone_errors = []

    for p, c in edges:

        pred_len = torch.norm(
            pred_xyz[:, p] - pred_xyz[:, c],
            dim=-1
        )

        gt_len = torch.norm(
            gt_xyz[:, p] - gt_xyz[:, c],
            dim=-1
        )

        bone_errors.append(
            (pred_len - gt_len).abs()
        )

    bone_errors = torch.stack(
        bone_errors,
        dim=1
    )

    return bone_errors


def compute_joint_angle(a, b, c):
    """
    a-b-c

    返回点b处夹角

    a,b,c:
        [...,3]
    """

    ba = a - b
    bc = c - b

    ba = ba / (torch.norm(ba, dim=-1, keepdim=True) + 1e-8)
    bc = bc / (torch.norm(bc, dim=-1, keepdim=True) + 1e-8)

    cos_theta = (ba * bc).sum(dim=-1)

    cos_theta = torch.clamp( cos_theta,  -1.0,  1.0)

    angle = torch.rad2deg( torch.acos(cos_theta) )

    return angle


# Joint-name triplets (a, b, c) used by compute_joint_angle_error: the angle
# is measured at `b`. Keyed by the joint-name list they were derived from, so
# the correct indices are looked up dynamically per dataset instead of being
# hardcoded to a single (H36M) joint ordering.
_ANGLE_TRIPLET_NAMES = {
    'RKnee': ('RHip', 'RKnee', 'RAnkle'),
    'LKnee': ('LHip', 'LKnee', 'LAnkle'),
    'LElbow': ('LShoulder', 'LElbow', 'LWrist'),
    'RElbow': ('RShoulder', 'RElbow', 'RWrist'),
}
# H36M's reduced 16-joint skeleton has no separate ankle joint: its 'Foot'
# joint IS the ankle location (see dataset_ap3d.py: joints_right=[...,3,...]
# is commented "right ankle" but named 'RFoot' in H36M_JOINT_NAMES). SMPL,
# on the other hand, has both a distinct RAnkle and a distinct RFoot (toe)
# joint, so for SMPL we must NOT fall back to its 'RFoot' -- only alias to
# 'Foot' when 'Ankle' truly isn't in the joint-name list (i.e. H36M).
_ANGLE_TRIPLET_NAME_ALIASES = {'RAnkle': 'RFoot', 'LAnkle': 'LFoot'}


def compute_joint_angle_error(pred_xyz, gt_xyz, dataset_name=None, num_joints=None):
    """
    pred_xyz, gt_xyz:
        [T, J, 3]

    dataset_name, num_joints:
        used to resolve the correct joint indices for the current skeleton
        ordering via get_joint_names(). Previously this function hardcoded
        H36M joint indices (0,1,2 / 3,4,5 / 10,11,12 / 13,14,15) regardless
        of dataset, which silently computed angles between the WRONG joints
        for the 'wp' (SMPL-23) skeleton, whose joint ordering is different.
        If dataset_name/num_joints are not given, falls back to the legacy
        H36M indices for backward compatibility.
    """
    J = pred_xyz.shape[-2] if num_joints is None else num_joints

    if dataset_name is not None:
        joint_names = get_joint_names(dataset_name, J)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
    else:
        joint_names = None
        name_to_idx = None

    angle_error = {}
    for key, (a_name, b_name, c_name) in _ANGLE_TRIPLET_NAMES.items():
        if name_to_idx is not None and all(
            n in name_to_idx or _ANGLE_TRIPLET_NAME_ALIASES.get(n) in name_to_idx
            for n in (a_name, b_name, c_name)
        ):
            def _idx(n):
                return name_to_idx[n] if n in name_to_idx else name_to_idx[_ANGLE_TRIPLET_NAME_ALIASES[n]]
            ia, ib, ic = _idx(a_name), _idx(b_name), _idx(c_name)
        else:
            # legacy H36M fallback (only valid for the 16-joint H36M ordering)
            legacy = {
                'RKnee': (0, 1, 2), 'LKnee': (3, 4, 5),
                'LElbow': (10, 11, 12), 'RElbow': (13, 14, 15),
            }
            ia, ib, ic = legacy[key]

        pred_angle = compute_joint_angle(pred_xyz[:, ia], pred_xyz[:, ib], pred_xyz[:, ic])
        gt_angle = compute_joint_angle(gt_xyz[:, ia], gt_xyz[:, ib], gt_xyz[:, ic])
        angle_error[key] = (pred_angle - gt_angle).abs()

    return angle_error



# =========================================================
# Dataset meta: fps / skeleton naming
# =========================================================

DATASET_FPS = {
    'ap': 60,
    'ap3d': 60,
    'wp': 50,
    'h36m': 50,
}


def get_fps(dataset_name, cfg=None):
    """cfg.fps 优先；否则按数据集默认值"""
    if cfg is not None:
        fps = getattr(cfg, 'fps', None)
        if fps:
            return float(fps)
    if dataset_name not in DATASET_FPS:
        raise ValueError(
            f'no default fps for dataset: {dataset_name}, '
            f'please set cfg.fps explicitly')
    return float(DATASET_FPS[dataset_name])


H36M_JOINT_NAMES = [
    'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist',
]

SMPL_23_JOINT_NAMES = [
    'LHip', 'RHip', 'Spine1', 'LKnee', 'RKnee', 'Spine2',
    'LAnkle', 'RAnkle', 'Spine3', 'LFoot', 'RFoot', 'Neck',
    'LCollar', 'RCollar', 'Head', 'LShoulder', 'RShoulder',
    'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHand', 'RHand',
]

# Bone edges (parent_idx, child_idx), indexed into H36M_JOINT_NAMES /
# SMPL_23_JOINT_NAMES above (i.e. AFTER the root/pelvis joint has already
# been dropped from the data, matching the `[..., 1:, :]` slicing used
# throughout data loading). These were previously referenced by
# get_skeleton_edges() below but never defined anywhere in the codebase,
# so any call to get_skeleton_edges()/get_bone_names()/compute_detailed_metrics
# raised a NameError.
H36M_SKELETON_EDGES = [
    (0, 1), (1, 2),        # RHip-RKnee-RFoot
    (3, 4), (4, 5),        # LHip-LKnee-LFoot
    (6, 7), (7, 8), (8, 9),  # Spine-Thorax-Neck-Head
    (7, 10), (10, 11), (11, 12),  # Thorax-LShoulder-LElbow-LWrist
    (7, 13), (13, 14), (14, 15),  # Thorax-RShoulder-RElbow-RWrist
]

SMPL_23_EDGES = [
    (0, 3), (3, 6), (6, 9),                          # LHip-LKnee-LAnkle-LFoot
    (1, 4), (4, 7), (7, 10),                         # RHip-RKnee-RAnkle-RFoot
    (2, 5), (5, 8), (8, 11),                         # Spine1-Spine2-Spine3-Neck
    (11, 14),                                        # Neck-Head
    (8, 12), (12, 15), (15, 17), (17, 19), (19, 21),  # Spine3-LCollar-LShoulder-LElbow-LWrist-LHand
    (8, 13), (13, 16), (16, 18), (18, 20), (20, 22),  # Spine3-RCollar-RShoulder-RElbow-RWrist-RHand
]


def get_skeleton_edges(dataset_name):
    if dataset_name in ('h36m', 'ap3d', 'ap'):
        return H36M_SKELETON_EDGES
    elif dataset_name == 'wp':
        return SMPL_23_EDGES
    raise ValueError(f'no skeleton edges defined for dataset: {dataset_name}')


def get_joint_names(dataset_name, num_joints):
    if dataset_name in ('h36m', 'ap3d', 'ap') and num_joints == len(H36M_JOINT_NAMES):
        return list(H36M_JOINT_NAMES)
    if dataset_name == 'wp' and num_joints == len(SMPL_23_JOINT_NAMES):
        return list(SMPL_23_JOINT_NAMES)
    return [f'J{i}' for i in range(num_joints)]


def get_bone_names(dataset_name, num_joints):
    edges = get_skeleton_edges(dataset_name)
    jn = get_joint_names(dataset_name, num_joints)
    names = []
    for e in edges:
        i, j = int(e[0]), int(e[1])
        a = jn[i] if 0 <= i < len(jn) else f'J{i}'
        b = jn[j] if 0 <= j < len(jn) else f'J{j}'
        names.append(f'{a}-{b}')
    return names


# =========================================================
# Acceleration / Jitter
# =========================================================

def _finite_diff(x, order, fps):
    """x: [..., T, J, 3] -> [..., T-order, J, 3]，已乘 fps**order"""
    for _ in range(order):
        x = x[..., 1:, :, :] - x[..., :-1, :, :]
    return x * (float(fps) ** order)


def compute_accel_jitter(xyz, fps, hist_xyz=None, gt_xyz=None):
    """
    xyz:      [..., T, J, 3]
    hist_xyz: [..., H, J, 3] 可选，拼在前面以覆盖 his/pred 接缝
    gt_xyz:   [..., T, J, 3] 可选，提供则额外返回 accel_error_joint
    返回的每一项形状为 [..., J]
    """
    if hist_xyz is not None:
        xyz = torch.cat([hist_xyz, xyz], dim=-3)
        if gt_xyz is not None:
            gt_xyz = torch.cat([hist_xyz, gt_xyz], dim=-3)

    T = xyz.shape[-3]
    out = {}

    if T >= 3:
        accel = _finite_diff(xyz, 2, fps)
        out['accel_joint'] = torch.linalg.norm(accel, dim=-1).mean(dim=-2)
        if gt_xyz is not None:
            gt_accel = _finite_diff(gt_xyz, 2, fps)
            out['accel_error_joint'] = torch.linalg.norm(accel - gt_accel, dim=-1).mean(dim=-2)

    if T >= 4:
        jerk = _finite_diff(xyz, 3, fps)
        out['jitter_joint'] = torch.linalg.norm(jerk, dim=-1).mean(dim=-2)

    return out


def _select_best_sample(pred, gt):
    """pred: [b, s, T, D], gt: [b, T, D] -> [b, T, D]（ADE 口径）"""
    dist = torch.linalg.norm(pred - gt[:, None], dim=-1)
    idx = dist.mean(dim=2).argmin(dim=1)
    return pred[torch.arange(pred.shape[0], device=pred.device), idx]


def compute_detailed_metrics(pred, gt, dataset_name, num_joints=None, fps=None, hist=None):
    """
    TrainerCustom 格式：pred [b, s, T, D]，gt [b, T, D]，hist [b, H, D] 可选
    返回 dict：标量为 0-dim tensor，向量已在 batch 维平均
    """
    b, s, T, D = pred.shape
    if num_joints is None:
        num_joints = D // 3
    assert D == num_joints * 3, f'D={D} 与 num_joints={num_joints} 不匹配'

    if fps is None:
        fps = get_fps(dataset_name)

    edges = get_skeleton_edges(dataset_name)

    best = _select_best_sample(pred, gt)
    pred_xyz = best.reshape(b, T, num_joints, 3)
    gt_xyz = gt.reshape(b, T, num_joints, 3)

    joint_err = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)      # [b, T, J]
    joint_ade = joint_err.mean(dim=1)                             # [b, J]
    joint_fde = joint_err[:, -1]                                  # [b, J]
    time_ade = joint_err.mean(dim=2)                              # [b, T]

    bone_list, angle_list = [], []
    for k in range(b):
        bone_list.append(compute_bone_length_error(pred_xyz[k], gt_xyz[k], edges=edges))
        angle_list.append(compute_joint_angle_error(pred_xyz[k], gt_xyz[k],
                                                     dataset_name=dataset_name, num_joints=num_joints))
    bone_err = torch.stack(bone_list, dim=0)                      # [b, T, n_bone]
    angle_keys = list(angle_list[0].keys())
    angle_err = {k: torch.stack([a[k] for a in angle_list], dim=0) for k in angle_keys}

    hist_xyz = None
    if hist is not None:
        hist_xyz = hist.reshape(b, -1, num_joints, 3)

    best_stat = compute_accel_jitter(pred_xyz, fps, hist_xyz=hist_xyz, gt_xyz=gt_xyz)
    all_stat = compute_accel_jitter(
        pred.reshape(b, s, T, num_joints, 3), fps,
        hist_xyz=None if hist_xyz is None else hist_xyz[:, None].expand(b, s, -1, -1, -1))
    gt_stat = compute_accel_jitter(gt_xyz, fps, hist_xyz=hist_xyz)

    zero_bj = torch.zeros(b, num_joints, device=pred.device, dtype=pred.dtype)
    jit_j = best_stat.get('jitter_joint', zero_bj)
    ae_j = best_stat.get('accel_error_joint', zero_bj)

    out = {
        'joint_ade':         joint_ade.mean(dim=0),
        'joint_fde':         joint_fde.mean(dim=0),
        'time_ade':          time_ade.mean(dim=0),
        'joint_time_error':  joint_err.mean(dim=0),
        'bone_ade':          bone_err.mean(dim=1).mean(dim=0),
        'bone_fde':          bone_err[:, -1].mean(dim=0),
        'bone_time_error':   bone_err.mean(dim=2).mean(dim=0),
        'whole_body_ade':    torch.sqrt((joint_ade ** 2).sum(dim=1)).mean(),
        'whole_body_fde':    torch.sqrt((joint_fde ** 2).sum(dim=1)).mean(),
        'whole_body_ble':    bone_err.mean(dim=(1, 2)).mean(),
        'whole_body_blf':    bone_err[:, -1].mean(dim=1).mean(),
        'jitter':            jit_j.mean(),
        'jitter_all':        all_stat.get('jitter_joint', zero_bj).mean(),
        'jitter_gt':         gt_stat.get('jitter_joint', zero_bj).mean(),
        'accel':             best_stat.get('accel_joint', zero_bj).mean(),
        'accel_gt':          gt_stat.get('accel_joint', zero_bj).mean(),
        'accel_error':       ae_j.mean(),
        'joint_jitter':      jit_j.mean(dim=0),
        'joint_accel_error': ae_j.mean(dim=0),
    }
    for k in angle_keys:
        out[f'angle_ade_{k}'] = angle_err[k].mean(dim=1).mean()
        out[f'angle_fde_{k}'] = angle_err[k][:, -1].mean()
    return out

def compute_all_metrics_detailed(pred, gt, gt_multi, num_joints, dataset_name,
                                 fps=None, hist=None):
    """
    pred:
        [K, T, D]

    gt:
        [1, T, D]

    gt_multi:
        [M, T, D]

    hist:
        [H, D] 可选，历史段最后若干帧，用于覆盖 his/pred 接缝处的加速度与抖动
    """

    if pred.shape[0] == 1:
        diversity = torch.tensor(0.0, device=pred.device)
    else:
        diversity = torch.pdist(pred.reshape(pred.shape[0], -1)).mean()

    if not torch.is_tensor(gt_multi):
        gt_multi = torch.from_numpy(gt_multi).to(pred.device)

    pred_expand = pred[:, None]
    gt_multi_gt = torch.cat([gt_multi, gt], dim=0)[None]

    diff_multi = pred_expand - gt_multi_gt

    # =====================================================
    # HumanMAC Original Metrics
    # =====================================================

    dist_full = torch.linalg.norm(diff_multi, dim=3)

    mmfde, _ = dist_full[:, :-1, -1].min(dim=0)
    mmfde = mmfde.mean()

    mmade, _ = dist_full[:, :-1].mean(dim=2).min(dim=0)
    mmade = mmade.mean()

    ade, best_ade_idx = dist_full[:, -1].mean(dim=1).min(dim=0)
    ade = ade.mean()

    fde, best_fde_idx = dist_full[:, -1, -1].min(dim=0)
    fde = fde.mean()

    # =====================================================
    # Detailed Analysis Metrics
    # =====================================================

    best_pred = pred[best_ade_idx]
    gt_single = gt.squeeze(0)

    diff = (best_pred - gt_single).reshape(pred.shape[1], num_joints, 3)

    joint_error = torch.linalg.norm(diff, dim=-1)

    joint_ade = joint_error.mean(dim=0)

    joint_fde = joint_error[-1]

    time_ade = joint_error.mean(dim=1)

    joint_time_error = joint_error

    edges = get_skeleton_edges(dataset_name)

    bone_error = compute_bone_length_error(
        best_pred.reshape(pred.shape[1], num_joints, 3),
        gt_single.reshape(pred.shape[1], num_joints, 3),
        edges=edges
    )

    bone_ade = bone_error.mean(dim=0)

    bone_fde = bone_error[-1]

    bone_time_error = bone_error.mean(dim=1)

    whole_body_ble = bone_error.mean()

    whole_body_blf = bone_fde.mean()

    whole_body_fde = torch.sqrt((joint_fde ** 2).sum())

    whole_body_ade = torch.sqrt((joint_ade ** 2).sum())

    pred_xyz = best_pred.reshape(pred.shape[1], num_joints, 3)

    gt_xyz = gt_single.reshape(pred.shape[1], num_joints, 3)

    angle_error = compute_joint_angle_error(pred_xyz, gt_xyz, dataset_name=dataset_name, num_joints=num_joints)

    angle_ade = {}

    angle_fde = {}

    for k, v in angle_error.items():
        angle_ade[k] = v.mean()

        angle_fde[k] = v[-1]

    # =====================================================
    # Acceleration / Jitter
    # =====================================================

    if fps is None:
        fps = get_fps(dataset_name)

    K = pred.shape[0]
    pred_all_xyz = pred.reshape(K, pred.shape[1], num_joints, 3)

    hist_xyz = None
    if hist is not None:
        if not torch.is_tensor(hist):
            hist = torch.from_numpy(hist).to(pred.device).to(pred.dtype)
        hist_xyz = hist.reshape(-1, num_joints, 3)

    best_stat = compute_accel_jitter(pred_xyz, fps, hist_xyz=hist_xyz, gt_xyz=gt_xyz)
    all_stat = compute_accel_jitter(
        pred_all_xyz, fps,
        hist_xyz=None if hist_xyz is None else hist_xyz[None].expand(K, -1, -1, -1))
    gt_stat = compute_accel_jitter(gt_xyz, fps, hist_xyz=hist_xyz)

    zero_j = torch.zeros(num_joints, device=pred.device, dtype=pred.dtype)
    joint_jitter = best_stat.get('jitter_joint', zero_j)
    joint_accel_error = best_stat.get('accel_error_joint', zero_j)

    return {
        'APD': diversity,
        'ADE': ade,
        'FDE': fde,
        'MMADE': mmade,
        'MMFDE': mmfde,
        'joint_ade': joint_ade,
        'joint_fde': joint_fde,
        'time_ade': time_ade,
        'joint_time_error': joint_time_error,
        'whole_body_ade': whole_body_ade,
        'whole_body_fde': whole_body_fde,
        'bone_ade': bone_ade,
        'bone_fde': bone_fde,
        'bone_time_error': bone_time_error,
        'whole_body_ble': whole_body_ble,
        'whole_body_blf': whole_body_blf,
        'angle_ade': angle_ade,
        'angle_fde': angle_fde,
        'jitter': joint_jitter.mean(),
        'jitter_all': all_stat.get('jitter_joint', zero_j).mean(),
        'jitter_gt': gt_stat.get('jitter_joint', zero_j).mean(),
        'accel': best_stat.get('accel_joint', zero_j).mean(),
        'accel_gt': gt_stat.get('accel_joint', zero_j).mean(),
        'accel_error': joint_accel_error.mean(),
        'joint_jitter': joint_jitter,
        'joint_accel_error': joint_accel_error,
    }