import torch
import torch.nn.functional as F

BONES = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9),
         (7, 10), (10, 11), (11, 12), (7, 13), (13, 14), (14, 15)]

SYMMETRY_PAIRS = [
    ((0, 1), (3, 4)),
    ((1, 2), (4, 5)),
    ((10, 11), (13, 14)),
    ((11, 12), (14, 15))
]


def velocity_loss(pred, gt):
    pred_vel = pred[:, :, 1:] - pred[:, :, :-1]
    gt_vel = gt[:, :, 1:] - gt[:, :, :-1]

    return F.mse_loss(pred_vel, gt_vel)


def acceleration_loss(pred, gt):
    pred_vel = pred[:, :, 1:] - pred[:, :, :-1]
    gt_vel = gt[:, :, 1:] - gt[:, :, :-1]

    pred_acc = pred_vel[:, :, 1:] - pred_vel[:, :, :-1]
    gt_acc = gt_vel[:, :, 1:] - gt_vel[:, :, :-1]

    return F.mse_loss(pred_acc, gt_acc)


def jerk_loss(pred, gt):
    pred_vel = pred[:, :, 1:] - pred[:, :, :-1]
    gt_vel = gt[:, :, 1:] - gt[:, :, :-1]

    pred_acc = pred_vel[:, :, 1:] - pred_vel[:, :, :-1]
    gt_acc = gt_vel[:, :, 1:] - gt_vel[:, :, :-1]

    pred_jerk = pred_acc[:, :, 1:] - pred_acc[:, :, :-1]
    gt_jerk = gt_acc[:, :, 1:] - gt_acc[:, :, :-1]

    return F.l1_loss(pred_jerk, gt_jerk)


# ============================================================
# bone loss
# ============================================================

def bone_length_loss(pred, gt):
    loss = 0.0

    for i, j in BONES:
        pred_len = torch.norm(
            pred[:, i] - pred[:, j],
            dim=-1
        )

        gt_len = torch.norm(
            gt[:, i] - gt[:, j],
            dim=-1
        )

        loss += F.l1_loss(
            pred_len,
            gt_len
        )

    return loss / len(BONES)


# ============================================================
# symmetry loss
# ============================================================

def symmetry_loss(pred):
    loss = 0.0

    for (a, b), (c, d) in SYMMETRY_PAIRS:
        left = torch.norm(pred[:, a] - pred[:, b], dim=-1)
        right = torch.norm(pred[:, c] - pred[:, d], dim=-1)
        loss += F.l1_loss(left, right)

    return loss / len(SYMMETRY_PAIRS)


# ============================================================
# biomechanics regularization
# ============================================================

def biomechanics_loss(pred_time, gt_time):
    vel = velocity_loss(pred_time, gt_time)

    acc = acceleration_loss(pred_time, gt_time)

    jerk = jerk_loss(pred_time, gt_time)

    bone = bone_length_loss(pred_time, gt_time)

    sym = symmetry_loss(pred_time)

    total = 0.01 * vel + 0.005 * acc + 0.002 * jerk + 0.01 * bone + 0.002 * sym

    return total * 10.0


def dct_vel_acc_loss(pred, gt, dct_m_vel, dct_m_acc, n_pre):
    pred_vel = pred[:, 1:] - pred[:, :-1]
    gt_vel = gt[:, 1:] - gt[:, :-1]

    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]

    vel_gt_dct = torch.matmul(dct_m_vel[:n_pre], gt_vel)
    vel_pred_dct = torch.matmul(dct_m_vel[:n_pre], pred_vel)

    acc_gt_dct = torch.matmul(dct_m_acc[:n_pre], gt_acc)
    acc_pred_dct = torch.matmul(dct_m_acc[:n_pre], pred_acc)

    l_vel = F.mse_loss(vel_pred_dct, vel_gt_dct)

    l_acc = F.mse_loss(acc_pred_dct, acc_gt_dct)

    return 0.1 * l_vel + 0.03 * l_acc


def compute_motion_losses(pred_traj, gt_traj):
    loss_traj = F.smooth_l1_loss(pred_traj, gt_traj)

    pred_vel = pred_traj[:, 1:] - pred_traj[:, :-1]
    gt_vel = gt_traj[:, 1:] - gt_traj[:, :-1]
    loss_vel = F.smooth_l1_loss(pred_vel, gt_vel)

    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
    loss_acc = F.smooth_l1_loss(pred_acc, gt_acc)

    pred_vel = pred_traj[:, 1:] - pred_traj[:, :-1]
    gt_vel = gt_traj[:, 1:] - gt_traj[:, :-1]

    pred_speed = torch.norm(pred_vel.reshape(*pred_vel.shape[:2], -1, 3), dim=-1)
    gt_speed = torch.norm(gt_vel.reshape(*gt_vel.shape[:2], -1, 3), dim=-1)
    loss_speed = F.smooth_l1_loss(pred_speed, gt_speed)

    pred_jerk = pred_acc[:, 1:] - pred_acc[:, :-1]
    gt_jerk = gt_acc[:, 1:] - gt_acc[:, :-1]
    loss_jerk = F.smooth_l1_loss(pred_jerk, gt_jerk)

    # EXP-A
    # loss =  0.1 * loss_traj + 1.0 * loss_vel + 0.5 * loss_speed

    # EXP-B
    # loss = 0.1 * loss_traj + 1.0 * loss_vel

    # EXP-C
    loss = 0.1 * loss_traj + 1.0 * loss_vel + 0.5 * loss_jerk
    return loss


h36m_joint_weight = torch.tensor([
    1.0,  # RHip
    1.2,  # RKnee
    2.0,  # RFoot

    1.0,  # LHip
    1.2,  # LKnee
    2.0,  # LFoot

    1.0,  # Spine
    1.0,  # Thorax
    1.0,  # Neck
    1.0,  # Head

    1.5,  # LShoulder
    2.0,  # LElbow
    3.0,  # LWrist

    1.5,  # RShoulder
    2.0,  # RElbow
    3.0  # RWrist
]).cuda()


def weighted_traj_loss(pred, gt, joint_weight=h36m_joint_weight):
    B, T, D = pred.shape

    J = joint_weight.shape[0]

    pred = pred.view(B, T, J, 3)
    gt = gt.view(B, T, J, 3)

    err = F.smooth_l1_loss(
        pred,
        gt,
        reduction='none'
    )

    err = err.mean(-1)

    err = err * joint_weight[None, None]

    return err.mean()


def weighted_velocity_loss(pred, gt, joint_weight=h36m_joint_weight):
    pred_vel = pred[:, 1:] - pred[:, :-1]
    gt_vel = gt[:, 1:] - gt[:, :-1]

    B, T, D = pred_vel.shape

    J = joint_weight.shape[0]

    pred_vel = pred_vel.view(B, T, J, 3)
    gt_vel = gt_vel.view(B, T, J, 3)

    err = F.smooth_l1_loss(
        pred_vel,
        gt_vel,
        reduction='none'
    )

    err = err.mean(-1)
    err = err * joint_weight[None, None]
    return err.mean()


END_JOINTS = [2, 5, 12, 15]


def end_effector_loss(pred, gt):
    B, T, D = pred.shape
    J = D // 3
    pred = pred.view(B, T, J, 3)
    gt = gt.view(B, T, J, 3)

    return F.smooth_l1_loss(pred[:, :, END_JOINTS], gt[:, :, END_JOINTS])


H36M_JOINT_WEIGHT = torch.tensor([
    1.0,  # RHip
    1.5,  # RKnee
    3.0,  # RFoot
    1.0,  # LHip
    1.5,  # LKnee
    4.0,  # LFoot
    1.0,  # Spine
    1.0,  # Thorax
    1.0,  # Neck
    1.0,  # Head
    1.0,  # LShoulder
    2.0,  # LElbow
    3.0,  # LWrist
    1.0,  # RShoulder
    2.0,  # RElbow
    4.0  # RWrist
])


class JointWeightedFlowMatchingLoss(torch.nn.Module):

    def __init__(self, joint_weight=H36M_JOINT_WEIGHT):
        super().__init__()

        self.register_buffer(
            "joint_weight",
            joint_weight.float()
        )

    def forward(self, pred_flow, target_flow):
        B, T, D = pred_flow.shape

        num_joints = self.joint_weight.shape[0]

        pred_flow = pred_flow.view(
            B,
            T,
            num_joints,
            3
        )

        target_flow = target_flow.view(
            B,
            T,
            num_joints,
            3
        )

        flow_error = (
                pred_flow -
                target_flow
        ).pow(2)

        flow_error = (
                flow_error *
                self.joint_weight[None, None, :, None]
        )

        loss = flow_error.mean()

        return loss


def limb_weighted_loss(pred_xyz, gt_xyz):
    B, T, D = pred_xyz.shape

    J = D // 3

    pred = pred_xyz.view(B, T, J, 3)
    gt = gt_xyz.view(B, T, J, 3)

    error = torch.norm(pred - gt, dim=-1)

    weight = H36M_JOINT_WEIGHT.to(
        pred_xyz.device
    )

    weight = weight / weight.mean()

    error = error * weight[None, None, :]

    return error.mean()


def joint_angle(a, b, c):
    """
    a-b-c

    [...,3]
    """

    ba = a - b
    bc = c - b

    dot = (ba * bc).sum(dim=-1)
    norm_ba = torch.norm(ba, dim=-1)
    norm_bc = torch.norm(bc, dim=-1)
    eps = 1e-6
    cos_theta = dot / (norm_ba * norm_bc + eps)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    return torch.acos(cos_theta)


ANGLE_TRIPLETS = [

    (0, 1, 2),
    (3, 4, 5),

    (10, 11, 12),
    (13, 14, 15),

    (7, 10, 11),
    (7, 13, 14),

    (6, 0, 1),
    (6, 3, 4),

    (7, 8, 9)
]


def full_body_angle_loss(pred_xyz, gt_xyz):
    B, T, D = pred_xyz.shape

    J = D // 3

    pred_xyz = pred_xyz.view(B, T, J, 3)
    gt_xyz = gt_xyz.view(B, T, J, 3)

    loss = 0

    for a, b, c in ANGLE_TRIPLETS:
        pred_angle = joint_angle(
            pred_xyz[:, :, a],
            pred_xyz[:, :, b],
            pred_xyz[:, :, c]
        )

        gt_angle = joint_angle(
            gt_xyz[:, :, a],
            gt_xyz[:, :, b],
            gt_xyz[:, :, c]
        )

        loss += F.mse_loss(
            pred_angle,
            gt_angle
        )

    return loss / len(ANGLE_TRIPLETS)


AP3D_BONE_EDGES = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(8,9),(7,10),(10,11),(11,12),(7,13),(13,14),(14,15)]

AP3D_BONE_WEIGHTS = torch.tensor([1.0,3.0,1.0,3.0,1.0,1.0,1.0,1.0,2.0,4.0,1.0,2.0,4.0], dtype=torch.float32)

AP3D_CHAIN_PAIRS = [(0,2),(3,5),(10,12),(13,15),(7,12),(7,15)]

AP3D_CHAIN_WEIGHTS = torch.tensor([2.0,2.5,3.0,3.0,2.0,2.0], dtype=torch.float32)


def to_xyz(x, num_joints=16):
    if x.dim() == 4:
        return x
    B, T, D = x.shape
    return x.view(B, T, num_joints, 3)


def slice_future(pred, gt, t_his=15, t_pred=60):
    if pred.shape[1] == t_his + t_pred:
        return pred[:, t_his:], gt[:, t_his:]
    return pred, gt


def safe_unit(v, eps=1e-6):
    return v / torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def bone_direction_loss(pred_traj, gt_traj, t_his=15, t_pred=60, num_joints=16):
    pred = to_xyz(pred_traj, num_joints)
    gt = to_xyz(gt_traj, num_joints)
    pred, gt = slice_future(pred, gt, t_his, t_pred)

    edges = torch.tensor(AP3D_BONE_EDGES, device=pred.device, dtype=torch.long)
    weights = AP3D_BONE_WEIGHTS.to(pred.device)
    weights = weights / weights.mean()

    p0 = pred[:, :, edges[:, 0]]
    p1 = pred[:, :, edges[:, 1]]
    g0 = gt[:, :, edges[:, 0]]
    g1 = gt[:, :, edges[:, 1]]

    pred_dir = safe_unit(p1 - p0)
    gt_dir = safe_unit(g1 - g0)

    cos = (pred_dir * gt_dir).sum(dim=-1).clamp(-1.0, 1.0)
    err = 1.0 - cos

    err = err * weights[None, None, :]
    return err.mean()


def chain_direction_loss(pred_traj, gt_traj, t_his=15, t_pred=60, num_joints=16):
    pred = to_xyz(pred_traj, num_joints)
    gt = to_xyz(gt_traj, num_joints)
    pred, gt = slice_future(pred, gt, t_his, t_pred)

    pairs = torch.tensor(AP3D_CHAIN_PAIRS, device=pred.device, dtype=torch.long)
    weights = AP3D_CHAIN_WEIGHTS.to(pred.device)
    weights = weights / weights.mean()

    p0 = pred[:, :, pairs[:, 0]]
    p1 = pred[:, :, pairs[:, 1]]
    g0 = gt[:, :, pairs[:, 0]]
    g1 = gt[:, :, pairs[:, 1]]

    pred_dir = safe_unit(p1 - p0)
    gt_dir = safe_unit(g1 - g0)

    cos = (pred_dir * gt_dir).sum(dim=-1).clamp(-1.0, 1.0)
    err = 1.0 - cos

    err = err * weights[None, None, :]
    return err.mean()


def kinematic_direction_loss(pred_traj, gt_traj, t_his=15, t_pred=60, num_joints=16):
    loss_bone_dir = bone_direction_loss(pred_traj, gt_traj, t_his, t_pred, num_joints)
    loss_chain_dir = chain_direction_loss(pred_traj, gt_traj, t_his, t_pred, num_joints)
    return loss_bone_dir, loss_chain_dir


def progressive_residual_fm_loss(
        flow_dict,
        target_v,
        w_r1=0.5,
        w_r2=0.25):

    v0 = flow_dict["v0"]

    r1 = flow_dict["r1"]

    r2 = flow_dict["r2"]

    v2 = flow_dict["v2"]

    # ------------------
    # main loss
    # ------------------

    loss_main = F.mse_loss(
        v2,
        target_v
    )

    # ------------------
    # residual-1
    # ------------------

    err0 = target_v - v0.detach()

    loss_r1 = F.mse_loss(
        r1,
        err0
    )

    # ------------------
    # residual-2
    # ------------------

    err1 = err0 - r1.detach()

    loss_r2 = F.mse_loss(
        r2,
        err1
    )

    loss = (
        loss_main
        + w_r1 * loss_r1
        + w_r2 * loss_r2
    )

    return loss


