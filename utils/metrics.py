from scipy.spatial.distance import pdist
import numpy as np
import torch

H36M_SKELETON_EDGES = [
    (0,1),
    (1,2),

    (0,3),
    (3,4),
    (4,5),

    (0,6),
    (6,7),
    (7,8),
    (8,9),

    (7,10),
    (10,11),
    (11,12),

    (7,13),
    (13,14),
    (14,15)
]

SMPL_23_EDGES = [

    # spine
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),

    # left leg
    (0, 1),
    (1, 4),
    (4, 7),
    (7, 10),

    # right leg
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),

    # left arm
    (9, 13),
    (13, 16),
    (16, 18),
    (18, 20),

    # right arm
    (9, 14),
    (14, 17),
    (17, 19),
    (19, 21),

    # head
    (15, 22),
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


def compute_joint_angle_error(pred_xyz, gt_xyz):
    """
    pred_xyz:
        [T,J,3]

    gt_xyz:
        [T,J,3]
    """

    angle_error = {}

    # -------------------
    # knees
    # -------------------

    pred_rknee = compute_joint_angle(
        pred_xyz[:,0],
        pred_xyz[:,1],
        pred_xyz[:,2]
    )

    gt_rknee = compute_joint_angle(
        gt_xyz[:,0],
        gt_xyz[:,1],
        gt_xyz[:,2]
    )

    angle_error["RKnee"] = (
        pred_rknee - gt_rknee
    ).abs()

    pred_lknee = compute_joint_angle(
        pred_xyz[:,3],
        pred_xyz[:,4],
        pred_xyz[:,5]
    )

    gt_lknee = compute_joint_angle(
        gt_xyz[:,3],
        gt_xyz[:,4],
        gt_xyz[:,5]
    )

    angle_error["LKnee"] = (
        pred_lknee - gt_lknee
    ).abs()

    # -------------------
    # elbows
    # -------------------

    pred_lelbow = compute_joint_angle(
        pred_xyz[:,10],
        pred_xyz[:,11],
        pred_xyz[:,12]
    )

    gt_lelbow = compute_joint_angle(
        gt_xyz[:,10],
        gt_xyz[:,11],
        gt_xyz[:,12]
    )

    angle_error["LElbow"] = (
        pred_lelbow - gt_lelbow
    ).abs()

    pred_relbow = compute_joint_angle(
        pred_xyz[:,13],
        pred_xyz[:,14],
        pred_xyz[:,15]
    )

    gt_relbow = compute_joint_angle(
        gt_xyz[:,13],
        gt_xyz[:,14],
        gt_xyz[:,15]
    )

    angle_error["RElbow"] = (
        pred_relbow - gt_relbow
    ).abs()

    return angle_error



def compute_all_metrics_detailed(pred, gt, gt_multi, num_joints, dataset_name):
    """
    pred:
        [K, T, D]

    gt:
        [1, T, D]

    gt_multi:
        [M, T, D]
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

    if dataset_name == 'h36m' or dataset_name == 'ap3d' or dataset_name == 'ap':
        edges = H36M_SKELETON_EDGES
    elif dataset_name == 'wp':
        edges = SMPL_23_EDGES

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

    best_pred = pred[best_ade_idx]
    gt_single = gt.squeeze(0)
    pred_xyz = best_pred.reshape(
        pred.shape[1],
        num_joints,
        3
    )

    gt_xyz = gt_single.reshape(
        pred.shape[1],
        num_joints,
        3
    )

    angle_error = compute_joint_angle_error(
        pred_xyz,
        gt_xyz
    )

    angle_ade = {}

    angle_fde = {}

    for k, v in angle_error.items():
        angle_ade[k] = v.mean()

        angle_fde[k] = v[-1]


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
    }


