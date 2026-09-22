from scipy.spatial.distance import pdist
import numpy as np
import torch

# Both edge lists below are indexed AFTER the root/pelvis joint has already
# been dropped from the data (matching the `[..., 1:, :]` slicing used
# throughout data loading), i.e. into H36M_JOINT_NAMES / SMPL_23_JOINT_NAMES.
# The previous versions of these lists (still present in HumanMAC's sibling
# repos SMRNet-main/TransFusion-main) had bugs from treating the dropped
# root's children as if they connected to each other:
# - H36M_SKELETON_EDGES had two spurious extra edges, (0,3) and (0,6),
#   connecting RHip-LHip and RHip-Spine -- bones that don't physically exist
#   (RHip/LHip/Spine were only ever connected via the now-removed root).
# - SMPL_23_EDGES used an entirely different, non-matching index scheme
#   (index 0 treated as a hub/root rather than as LHip), silently computing
#   bone-length error between the wrong joint pairs for the 'wp' dataset.
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

WP24_NAMES = [
    "Pelvis",
    "LHip", "RHip", "Spine1",
    "LKnee", "RKnee", "Spine2",
    "LAnkle", "RAnkle", "Spine3",
    "LFoot", "RFoot",
    "Neck", "LCollar", "RCollar",
    "Head",
    "LShoulder", "RShoulder",
    "LElbow", "RElbow",
    "LWrist", "RWrist",
    "LHand", "RHand"
] # lfoot j9 rfoot j10

DATASET_FPS = {
    'ap': 60,
    'ap3d': 60,
    'wp': 50,
    'h36m': 50,
}

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

def get_joint_names(dataset_name, num_joints):
    if dataset_name in ('h36m', 'ap3d', 'ap') and num_joints == len(H36M_JOINT_NAMES):
        return list(H36M_JOINT_NAMES)
    if dataset_name == 'wp' and num_joints == len(SMPL_23_JOINT_NAMES):
        return list(SMPL_23_JOINT_NAMES)
    return [f'J{i}' for i in range(num_joints)]

def get_skeleton_edges(dataset_name):
    if dataset_name in ('h36m', 'ap3d', 'ap'):
        return H36M_SKELETON_EDGES
    elif dataset_name == 'wp':
        return SMPL_23_EDGES
    raise ValueError(f'no skeleton edges defined for dataset: {dataset_name}')

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


def compute_all_metrics(pred, gt, gt_multi):
    """
    calculate all metrics

    Args:
        pred: candidate prediction, shape as [50, t_pred, 3 * joints_num]
        gt: ground truth, shape as [1, t_pred, 3 * joints_num]
        gt_multi: multi-modal ground truth, shape as [multi_modal, t_pred, 3 * joints_num]

    Returns:
        diversity, ade, fde, mmade, mmfde
    """
    if pred.shape[0] == 1:
        diversity = 0.0
    dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist_diverse.mean()
    pred = pred[:, None, ...]

    gt_multi = torch.from_numpy(gt_multi).to('cuda')
    gt_multi_gt = torch.cat([gt_multi, gt], dim=0)
    gt_multi_gt = gt_multi_gt[None, ...]

    diff_multi = pred - gt_multi_gt
    dist = torch.linalg.norm(diff_multi, dim=3)
    # we can reuse 'dist' to optimize metrics calculation

    mmfde, _ = dist[:, :-1, -1].min(dim=0)
    mmfde = mmfde.mean()
    mmade, _ = dist[:, :-1].mean(dim=2).min(dim=0)
    mmade = mmade.mean()

    ade, _ = dist[:, -1].mean(dim=1).min(dim=0)
    fde, _ = dist[:, -1, -1].min(dim=0)
    ade = ade.mean()
    fde = fde.mean()

    return diversity, ade, fde, mmade, mmfde

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


# Angle triplets are resolved by joint NAME (via get_joint_names()) so that
# the correct indices are looked up dynamically per dataset instead of being
# hardcoded to a single (H36M) joint ordering.
_ANGLE_TRIPLET_NAMES = {
    'RKnee': ('RHip', 'RKnee', 'RAnkle'),
    'LKnee': ('LHip', 'LKnee', 'LAnkle'),
    'LElbow': ('LShoulder', 'LElbow', 'LWrist'),
    'RElbow': ('RShoulder', 'RElbow', 'RWrist'),
}
# H36M's reduced 16-joint skeleton has no separate ankle joint: its 'Foot'
# joint IS the ankle location. SMPL has both a distinct Ankle and a distinct
# Foot (toe) joint, so for SMPL we must NOT fall back to its 'Foot' -- only
# alias to 'Foot' when 'Ankle' truly isn't in the joint-name list (i.e. H36M).
_ANGLE_TRIPLET_NAME_ALIASES = {'RAnkle': 'RFoot', 'LAnkle': 'LFoot'}


def compute_joint_angle_error(pred_xyz, gt_xyz, dataset_name=None, num_joints=None):
    """
    pred_xyz, gt_xyz:
        [T,J,3]

    dataset_name, num_joints:
        used to resolve the correct joint indices for the current skeleton
        ordering via get_joint_names(). Without them, this hardcoded H36M
        joint indices (0,1,2 / 3,4,5 / 10,11,12 / 13,14,15) regardless of
        dataset, which silently computed angles between the WRONG joints for
        the 'wp' (SMPL-23) skeleton, whose joint ordering is different. If
        dataset_name/num_joints are not given, falls back to the legacy H36M
        indices for backward compatibility.
    """
    J = pred_xyz.shape[-2] if num_joints is None else num_joints

    if dataset_name is not None:
        joint_names = get_joint_names(dataset_name, J)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
    else:
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
