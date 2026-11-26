import torch

# Joint indices below are AFTER the root/pelvis joint has been removed from the data.
# AthletePose3D / AthleticsPose: H36M-17 layout -> 16 joints
H36M_JOINT_NAMES = [
    'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist',
]

H36M_SKELETON_EDGES = [
    (0, 1), (1, 2),        # RHip-RKnee-RFoot
    (3, 4), (4, 5),        # LHip-LKnee-LFoot
    (6, 7), (7, 8), (8, 9),  # Spine-Thorax-Neck-Head
    (7, 10), (10, 11), (11, 12),  # Thorax-LShoulder-LElbow-LWrist
    (7, 13), (13, 14), (14, 15),  # Thorax-RShoulder-RElbow-RWrist
]

# WorldPose: SMPL-24 layout -> 23 joints
SMPL_23_JOINT_NAMES = [
    'LHip', 'RHip', 'Spine1', 'LKnee', 'RKnee', 'Spine2',
    'LAnkle', 'RAnkle', 'Spine3', 'LFoot', 'RFoot', 'Neck',
    'LCollar', 'RCollar', 'Head', 'LShoulder', 'RShoulder',
    'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHand', 'RHand',
]

SMPL_23_EDGES = [
    (0, 3), (3, 6), (6, 9),                          # LHip-LKnee-LAnkle-LFoot
    (1, 4), (4, 7), (7, 10),                         # RHip-RKnee-RAnkle-RFoot
    (2, 5), (5, 8), (8, 11),                         # Spine1-Spine2-Spine3-Neck
    (11, 14),                                        # Neck-Head
    (8, 12), (12, 15), (15, 17), (17, 19), (19, 21),  # Spine3-LCollar-LShoulder-LElbow-LWrist-LHand
    (8, 13), (13, 16), (16, 18), (18, 20), (20, 22),  # Spine3-RCollar-RShoulder-RElbow-RWrist-RHand
]

# end effectors used for the average end-effector error (AEE)
END_EFFECTOR_NAMES = ('LWrist', 'RWrist', 'LFoot', 'RFoot')

DATASET_FPS = {
    'ap': 60,
    'ap3d': 60,
    'wp': 50,
}


def get_joint_names(dataset_name, num_joints):
    if dataset_name in ('ap3d', 'ap') and num_joints == len(H36M_JOINT_NAMES):
        return list(H36M_JOINT_NAMES)
    if dataset_name == 'wp' and num_joints == len(SMPL_23_JOINT_NAMES):
        return list(SMPL_23_JOINT_NAMES)
    return [f'J{i}' for i in range(num_joints)]


def get_skeleton_edges(dataset_name):
    if dataset_name in ('ap3d', 'ap'):
        return H36M_SKELETON_EDGES
    elif dataset_name == 'wp':
        return SMPL_23_EDGES
    raise ValueError(f'no skeleton edges defined for dataset: {dataset_name}')


def get_bone_names(dataset_name, num_joints):
    jn = get_joint_names(dataset_name, num_joints)
    return [f'{jn[i]}-{jn[j]}' for i, j in get_skeleton_edges(dataset_name)]


def get_end_effector_indices(dataset_name, num_joints):
    jn = get_joint_names(dataset_name, num_joints)
    return [jn.index(n) for n in END_EFFECTOR_NAMES]


def get_fps(dataset_name, cfg=None):
    """cfg.fps if given, otherwise the dataset default"""
    if cfg is not None:
        fps = getattr(cfg, 'fps', None)
        if fps:
            return float(fps)
    if dataset_name not in DATASET_FPS:
        raise ValueError(f'no default fps for dataset: {dataset_name}, please set cfg.fps explicitly')
    return float(DATASET_FPS[dataset_name])


def _finite_diff(x, order, fps):
    """x: [..., T, J, 3] -> [..., T-order, J, 3], multiplied by fps**order"""
    for _ in range(order):
        x = x[..., 1:, :, :] - x[..., :-1, :, :]
    return x * (float(fps) ** order)


def compute_accel_jitter(xyz, fps, hist_xyz=None, gt_xyz=None):
    """
    xyz:      [..., T, J, 3]
    hist_xyz: [..., H, J, 3] optional, prepended to cover the observation/prediction seam
    gt_xyz:   [..., T, J, 3] optional, additionally returns accel_error_joint
    every returned entry has shape [..., J]
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


def compute_bone_length_error(pred_xyz, gt_xyz, edges):
    """pred_xyz, gt_xyz: [T, J, 3] -> absolute bone-length error [T, n_bones]"""
    bone_errors = []
    for p, c in edges:
        pred_len = torch.norm(pred_xyz[:, p] - pred_xyz[:, c], dim=-1)
        gt_len = torch.norm(gt_xyz[:, p] - gt_xyz[:, c], dim=-1)
        bone_errors.append((pred_len - gt_len).abs())
    return torch.stack(bone_errors, dim=1)


def compute_joint_angle(a, b, c):
    """angle (degree) at b of the triplet a-b-c, inputs [..., 3]"""
    ba = a - b
    bc = c - b

    ba = ba / (torch.norm(ba, dim=-1, keepdim=True) + 1e-8)
    bc = bc / (torch.norm(bc, dim=-1, keepdim=True) + 1e-8)

    cos_theta = torch.clamp((ba * bc).sum(dim=-1), -1.0, 1.0)

    return torch.rad2deg(torch.acos(cos_theta))


# Angle triplets are resolved by joint name. The H36M layout has no separate ankle joint
# (its 'Foot' joint is the ankle), so 'Ankle' falls back to 'Foot' only when it does not exist.
_ANGLE_TRIPLET_NAMES = {
    'RKnee': ('RHip', 'RKnee', 'RAnkle'),
    'LKnee': ('LHip', 'LKnee', 'LAnkle'),
    'LElbow': ('LShoulder', 'LElbow', 'LWrist'),
    'RElbow': ('RShoulder', 'RElbow', 'RWrist'),
}
_ANGLE_TRIPLET_NAME_ALIASES = {'RAnkle': 'RFoot', 'LAnkle': 'LFoot'}


def compute_joint_angle_error(pred_xyz, gt_xyz, dataset_name, num_joints):
    """pred_xyz, gt_xyz: [T, J, 3] -> {angle name: absolute angle error [T]}"""
    name_to_idx = {n: i for i, n in enumerate(get_joint_names(dataset_name, num_joints))}

    def _idx(n):
        return name_to_idx[n] if n in name_to_idx else name_to_idx[_ANGLE_TRIPLET_NAME_ALIASES[n]]

    angle_error = {}
    for key, (a_name, b_name, c_name) in _ANGLE_TRIPLET_NAMES.items():
        ia, ib, ic = _idx(a_name), _idx(b_name), _idx(c_name)
        pred_angle = compute_joint_angle(pred_xyz[:, ia], pred_xyz[:, ib], pred_xyz[:, ic])
        gt_angle = compute_joint_angle(gt_xyz[:, ia], gt_xyz[:, ib], gt_xyz[:, ic])
        angle_error[key] = (pred_angle - gt_angle).abs()

    return angle_error


def compute_all_metrics_detailed(pred, gt, gt_multi, num_joints, dataset_name, fps=None, hist=None):
    """
    pred:     [K, T, D] K predicted futures
    gt:       [1, T, D] ground-truth future
    gt_multi: [M, T, D] multimodal ground truth
    hist:     [H, D] optional, frames in front of the prediction (observation/prediction seam of jitter)
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
    # APD / ADE / FDE / MMADE / MMFDE
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
    # per-joint / per-bone / per-angle errors of the best (min-ADE) sample
    # =====================================================

    best_pred = pred[best_ade_idx]
    gt_single = gt.squeeze(0)

    pred_xyz = best_pred.reshape(pred.shape[1], num_joints, 3)
    gt_xyz = gt_single.reshape(pred.shape[1], num_joints, 3)

    joint_error = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)  # [T, J]
    joint_ade = joint_error.mean(dim=0)
    joint_fde = joint_error[-1]
    time_ade = joint_error.mean(dim=1)

    bone_error = compute_bone_length_error(pred_xyz, gt_xyz, edges=get_skeleton_edges(dataset_name))
    bone_ade = bone_error.mean(dim=0)
    bone_fde = bone_error[-1]

    angle_error = compute_joint_angle_error(pred_xyz, gt_xyz, dataset_name=dataset_name, num_joints=num_joints)
    angle_ade = {k: v.mean() for k, v in angle_error.items()}

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
        'bone_ade': bone_ade,
        'bone_fde': bone_fde,
        'angle_ade': angle_ade,
        'jitter': joint_jitter.mean(),
        'jitter_all': all_stat.get('jitter_joint', zero_j).mean(),
        'jitter_gt': gt_stat.get('jitter_joint', zero_j).mean(),
        'accel': best_stat.get('accel_joint', zero_j).mean(),
        'accel_gt': gt_stat.get('accel_joint', zero_j).mean(),
        'accel_error': joint_accel_error.mean(),
        'joint_jitter': joint_jitter,
        'joint_accel_error': joint_accel_error,
    }
