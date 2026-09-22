import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from models.transformer import zero_module, timestep_embedding
from utils import util
from typing import Callable
from torch.nn import Module
from torch.nn.modules import activation




def get_activation(activation_name: str) -> nn.Module:
    """根据字符串名称获取对应的 PyTorch 激活函数层"""
    if not activation_name:
        return nn.Identity()  # 如果为空，则返回不做任何操作的恒等映射

    try:
        # 使用 getattr 从 torch.nn 中获取类名，并直接实例化
        activation_class = getattr(nn, activation_name)
        return activation_class()
    except AttributeError:
        raise ValueError(f"不支持的激活函数名称: {activation_name}")


# =============================================================================
# 核心常量定义（16个非根关节，彻底移除根关节）
# =============================================================================
# 原关节顺序：0=骨盆(已移除), 1=右髋, 2=右膝, 3=右踝, 4=左髋, 5=左膝, 6=左踝,
#             7=躯干, 8=脖子, 9=头, 10=头顶, 11=左肩, 12=左肘, 13=左腕,
#             14=右肩, 15=右肘, 16=右腕
# 新索引：0=右髋, 1=右膝, 2=右踝, 3=左髋, 4=左膝, 5=左踝,
#        6=躯干, 7=脖子, 8=头, 9=头顶, 10=左肩, 11=左肘, 12=左腕,
#        13=右肩, 14=右肘, 15=右腕
JOINT_NAMES = [
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "torso", "neck", "head", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist"
]
NUM_JOINTS = len(JOINT_NAMES)  # 16个非根关节

# 解剖学邻接矩阵（16×16，硬约束）
ANATOMY_ADJACENCY = torch.zeros(NUM_JOINTS, NUM_JOINTS, dtype=torch.bool)
# 右腿
ANATOMY_ADJACENCY[0, 1] = True  # 右髋→右膝
ANATOMY_ADJACENCY[1, 0] = True
ANATOMY_ADJACENCY[1, 2] = True  # 右膝→右踝
ANATOMY_ADJACENCY[2, 1] = True
# 左腿
ANATOMY_ADJACENCY[3, 4] = True  # 左髋→左膝
ANATOMY_ADJACENCY[4, 3] = True
ANATOMY_ADJACENCY[4, 5] = True  # 左膝→左踝
ANATOMY_ADJACENCY[5, 4] = True
# 躯干+头
ANATOMY_ADJACENCY[6, 7] = True  # 躯干→脖子
ANATOMY_ADJACENCY[7, 6] = True
ANATOMY_ADJACENCY[7, 8] = True  # 脖子→头
ANATOMY_ADJACENCY[8, 7] = True
ANATOMY_ADJACENCY[8, 9] = True  # 头→头顶
ANATOMY_ADJACENCY[9, 8] = True
# 左臂
ANATOMY_ADJACENCY[7, 10] = True  # 脖子→左肩
ANATOMY_ADJACENCY[10, 7] = True
ANATOMY_ADJACENCY[10, 11] = True  # 左肩→左肘
ANATOMY_ADJACENCY[11, 10] = True
ANATOMY_ADJACENCY[11, 12] = True  # 左肘→左腕
ANATOMY_ADJACENCY[12, 11] = True
# 右臂
ANATOMY_ADJACENCY[7, 13] = True  # 脖子→右肩
ANATOMY_ADJACENCY[13, 7] = True
ANATOMY_ADJACENCY[13, 14] = True  # 右肩→右肘
ANATOMY_ADJACENCY[14, 13] = True
ANATOMY_ADJACENCY[14, 15] = True  # 右肘→右腕
ANATOMY_ADJACENCY[15, 14] = True

# 对称关节对（运动生理学约束）
SYMMETRIC_JOINT_PAIRS = [
    (0, 3),  # 右髋 ↔ 左髋
    (1, 4),  # 右膝 ↔ 左膝
    (10, 13),  # 左肩 ↔ 右肩
    (11, 14)  # 左肘 ↔ 右肘
]

# 父关节索引（-1表示直接连接原点/根关节）
JOINT_PARENT = [
    -1, 0, 1,  # 右腿：右髋(-1)→右膝(0)→右踝(1)
    -1, 3, 4,  # 左腿：左髋(-1)→左膝(3)→左踝(4)
    -1, 6, 7, 8, 9,  # 躯干+头：躯干(-1)→脖子(6)→头(7)→头顶(8)
    7, 10, 11,  # 左臂：脖子(7)→左肩(10)→左肘(11)→左腕(12)
    7, 13, 14  # 右臂：脖子(7)→右肩(13)→右肘(14)→右腕(15)
]

KINETIC_CHAINS = [
    [2, 1, 0, 6, 7],  # right leg -> torso
    [5, 4, 3, 6, 7],  # left leg -> torso
    [6, 7, 10, 11, 12],  # left arm chain
    [6, 7, 13, 14, 15],  # right arm chain
]

BONES = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9),
         (7, 10), (10, 11), (11, 12), (7, 13), (13, 14), (14, 15)]

SYMMETRY_PAIRS = [
    ((0, 1), (3, 4)),
    ((1, 2), (4, 5)),
    ((10, 11), (13, 14)),
    ((11, 12), (14, 15))
]

LIMBS = {

    "right_leg": [0, 1, 2],
    "left_leg": [3, 4, 5],

    "spine": [6, 7, 8, 9],

    "left_arm": [10, 11, 12],
    "right_arm": [13, 14, 15]
}

JOINT_TO_LIMB = {

    0: 0, 1: 0, 2: 0,  # right_leg
    3: 1, 4: 1, 5: 1,  # left_leg

    6: 2, 7: 2, 8: 2, 9: 2,  # spine

    10: 3, 11: 3, 12: 3,  # left_arm
    13: 4, 14: 4, 15: 4  # right_arm
}


def project_limb_to_joint(
        limb_feat
):
    """
    limb_feat:
        [B,5,D]

    return:
        [B,16,D]
    """

    B, L, D = limb_feat.shape

    out = []

    for j in range(NUM_JOINTS):
        limb_idx = JOINT_TO_LIMB[j]

        out.append(
            limb_feat[:, limb_idx]
        )

    out = torch.stack(
        out,
        dim=1
    )

    return out


def build_anatomy_prior():
    A = torch.zeros(NUM_JOINTS, NUM_JOINTS)

    for i, j in BONES:
        A[i, j] = 1.0
        A[j, i] = 1.0

    A += torch.eye(NUM_JOINTS)

    return A


def build_kinetic_prior():
    prior = torch.zeros(NUM_JOINTS, NUM_JOINTS)

    for chain in KINETIC_CHAINS:
        for i in range(len(chain) - 1):
            a = chain[i]
            b = chain[i + 1]
            prior[a, b] = 1.0
            prior[b, a] = 1.0

    prior += torch.eye(NUM_JOINTS)

    return prior


# 预计算DCT基函数内积矩阵（全局常量，仅计算一次）
def precompute_dct_inner_product(dct_mat):
    """预计算C_{k,m} = sum_t dct_mat[k,t] * dct_mat[m,t]"""
    K = dct_mat.shape[0]
    C = torch.zeros(K, K, device=dct_mat.device)
    for k in range(K):
        for m in range(K):
            C[k, m] = torch.sum(dct_mat[k] * dct_mat[m])
    return C


def precompute_dct_derivative_matrix(T, K):
    """预计算DCT域一阶导数矩阵D，满足dct(dx/dt) = D @ dct(x)"""
    D = torch.zeros(K, K)
    for k in range(K):
        for m in range(K):
            if m > k:
                D[k, m] = 2 * torch.pi * m / T
    return D


# 全局预计算（训练前执行一次）
DCT_MAT, _ = util.get_dct_matrix(15)
DCT_INNER_PRODUCT = precompute_dct_inner_product(DCT_MAT)


def stable_kl_loss(sigma):
    """数值稳定的KL散度计算"""
    sigma_clamped = torch.clamp(sigma, min=0.1, max=10.0)
    log_sigma = torch.log(sigma_clamped + 1e-6)
    return 0.5 * torch.mean(log_sigma + 1 / (sigma_clamped ** 2 + 1e-6) - 1)


def total_vfm_loss(pred, v_gt, mu, sigma, x_t, DCT_DERIVATIVE_MAT, FREQ_WEIGHTS):
    """改进的变分流场匹配损失"""
    # 主损失：直接监督均值，更稳定
    main_loss = F.mse_loss(mu, v_gt)
    # 采样损失：辅助监督采样结果
    sample_loss = F.mse_loss(pred, v_gt)
    # KL散度：惩罚sigma偏离1
    kl_loss = stable_kl_loss(sigma)
    # 自适应物理损失
    phys_loss = dct_domain_second_order_phys_loss(pred, x_t, DCT_DERIVATIVE_MAT, FREQ_WEIGHTS)

    # 总损失权重
    return main_loss + 0.1 * sample_loss + 0.005 * kl_loss + phys_loss


def hard_example_weighted_loss(pred, v_gt, mu, sigma, x_t, DCT_DERIVATIVE_MAT, FREQ_WEIGHTS):
    loss = total_vfm_loss(pred, v_gt, mu, sigma, x_t, DCT_DERIVATIVE_MAT, FREQ_WEIGHTS)
    sample_loss = F.mse_loss(pred, v_gt, reduction='none').mean(dim=(1, 2))
    weight = torch.softmax(sample_loss / 0.1, dim=0) * len(sample_loss)
    return (loss * weight).mean()


def dct_domain_second_order_phys_loss(V_pred, X_t, DCT_DERIVATIVE_MAT, FREQ_WEIGHTS):
    """
    DCT域二阶动力学损失（速度+加速度约束）
    Args:
        V_pred: (B,15,48) DCT域速度预测
        X_t: (B,15,48) DCT域位置
    """
    B, K, _ = V_pred.shape
    J = NUM_JOINTS
    total_loss = 0.0

    # 重塑为关节×频率×坐标
    V_pred = V_pred.view(B, K, J, 3).permute(0, 2, 1, 3)  # (B,16,15,3)
    X_t = X_t.view(B, K, J, 3).permute(0, 2, 1, 3)  # (B,16,15,3)

    # 计算DCT域加速度：a = D @ v
    A_pred = torch.einsum('km,bjmc->bjkc', DCT_DERIVATIVE_MAT, V_pred)
    high_energy = torch.sum(torch.abs(X_t[:, :, 10:]), dim=(1, 2, 3))
    total_energy = torch.sum(torch.abs(X_t), dim=(1, 2, 3)) + 1e-6
    complexity = high_energy / total_energy
    w = 1e-5 * (1.0 - complexity).mean()

    loss = 0
    for j in range(J):
        p = JOINT_PARENT[j]
        if p == -1:
            Bx = X_t[:, j, :, 0]
            By = X_t[:, j, :, 1]
            Bz = X_t[:, j, :, 2]
            Vx = V_pred[:, j, :, 0]
            Vy = V_pred[:, j, :, 1]
            Vz = V_pred[:, j, :, 2]
            Ax = A_pred[:, j, :, 0]
            Ay = A_pred[:, j, :, 1]
            Az = A_pred[:, j, :, 2]
        else:
            Bx = X_t[:, j, :, 0] - X_t[:, p, :, 0]
            By = X_t[:, j, :, 1] - X_t[:, p, :, 1]
            Bz = X_t[:, j, :, 2] - X_t[:, p, :, 2]
            Vx = V_pred[:, j, :, 0] - V_pred[:, p, :, 0]
            Vy = V_pred[:, j, :, 1] - V_pred[:, p, :, 1]
            Vz = V_pred[:, j, :, 2] - V_pred[:, p, :, 2]
            Ax = A_pred[:, j, :, 0] - A_pred[:, p, :, 0]
            Ay = A_pred[:, j, :, 1] - A_pred[:, p, :, 1]
            Az = A_pred[:, j, :, 2] - A_pred[:, p, :, 2]

        iv = (Bx * Vx + By * Vy + Bz * Vz) * FREQ_WEIGHTS
        ia = (Bx * Ax + By * Ay + Bz * Az) * FREQ_WEIGHTS
        loss += F.mse_loss(iv.sum(1), torch.zeros_like(iv.sum(1)))
        loss += 0.5 * F.mse_loss(ia.sum(1), torch.zeros_like(ia.sum(1)))

    return w * loss / J


def dct_domain_phys_loss(V_pred, X_t):
    """
    DCT域原生物理一致性损失（无任何逆DCT变换）
    Args:
        V_pred: (B, K, 48) 模型预测的DCT域速度
        X_t: (B, K, 48) DCT域t时刻位置
    Returns:
        物理一致性损失
    """
    B, K, _ = V_pred.shape
    J = NUM_JOINTS
    loss = 0.0
    valid_joints = 0

    for j in range(J):
        p = JOINT_PARENT[j]
        # 提取骨骼的DCT系数（相对坐标，父关节为-1时直接取当前关节）
        if p == -1:
            Bx = X_t[:, :, j * 3]
            By = X_t[:, :, j * 3 + 1]
            Bz = X_t[:, :, j * 3 + 2]
            Vx = V_pred[:, :, j * 3]
            Vy = V_pred[:, :, j * 3 + 1]
            Vz = V_pred[:, :, j * 3 + 2]
        else:
            Bx = X_t[:, :, j * 3] - X_t[:, :, p * 3]
            By = X_t[:, :, j * 3 + 1] - X_t[:, :, p * 3 + 1]
            Bz = X_t[:, :, j * 3 + 2] - X_t[:, :, p * 3 + 2]
            Vx = V_pred[:, :, j * 3] - V_pred[:, :, p * 3]
            Vy = V_pred[:, :, j * 3 + 1] - V_pred[:, :, p * 3 + 1]
            Vz = V_pred[:, :, j * 3 + 2] - V_pred[:, :, p * 3 + 2]

        # DCT域内积约束（对应笛卡尔空间骨骼长度导数为0）
        inner_x = torch.einsum('bk,bm,km->b', Bx, Vx, DCT_INNER_PRODUCT)
        inner_y = torch.einsum('bk,bm,km->b', By, Vy, DCT_INNER_PRODUCT)
        inner_z = torch.einsum('bk,bm,km->b', Bz, Vz, DCT_INNER_PRODUCT)

        loss += F.mse_loss(inner_x, torch.zeros_like(inner_x))
        loss += F.mse_loss(inner_y, torch.zeros_like(inner_y))
        loss += F.mse_loss(inner_z, torch.zeros_like(inner_z))
        valid_joints += 1

    # 自适应权重：高频分量权重更高（对应快速运动）
    freq_weight = torch.linspace(1.0, 3.0, K, device=V_pred.device)
    freq_loss = F.mse_loss(V_pred * freq_weight.unsqueeze(0).unsqueeze(-1),
                           torch.zeros_like(V_pred))

    return 1e-4 * (loss / valid_joints) + 5e-5 * freq_loss


def velocity_physical_consistency_loss(v_pred, x_t):
    """
    速度域物理一致性损失（严格基于物理定律推导）
    Args:
        v_pred: (B, T, 16*3) 笛卡尔空间关节速度（由DCT域速度逆变换得到）
        x_t: (B, T, 16*3) 笛卡尔空间t时刻关节位置（由DCT域位置逆变换得到）
    Returns:
        物理一致性损失
    """
    B, T, _ = v_pred.shape
    J = NUM_JOINTS

    v_pred = v_pred.reshape(B, T, J, 3)
    x_t = x_t.reshape(B, T, J, 3)

    # 1. 骨骼长度不变性损失（核心）
    # 物理意义：相对速度在骨骼方向上的分量必须为零
    length_loss = 0.0
    valid_joints = 0
    for j in range(J):
        p = JOINT_PARENT[j]
        if p == -1:
            # 一级关节（髋、肩、躯干）：父关节是原点
            bone_dir = x_t[:, :, j]
            bone_dir_norm = torch.norm(bone_dir, dim=-1, keepdim=True) + 1e-6
            bone_dir_unit = bone_dir / bone_dir_norm
            rel_vel = v_pred[:, :, j]
        else:
            # 二级关节：父关节是其他关节
            bone_dir = x_t[:, :, j] - x_t[:, :, p]
            bone_dir_norm = torch.norm(bone_dir, dim=-1, keepdim=True) + 1e-6
            bone_dir_unit = bone_dir / bone_dir_norm
            rel_vel = v_pred[:, :, j] - v_pred[:, :, p]

        # 计算相对速度在骨骼方向上的分量（应该为0）
        vel_along_bone = torch.sum(rel_vel * bone_dir_unit, dim=-1)
        length_loss += F.mse_loss(vel_along_bone, torch.zeros_like(vel_along_bone))
        valid_joints += 1

    length_loss /= valid_joints

    # 2. 对称运动约束损失
    # 物理意义：对称关节的速度关于矢状面对称
    sym_loss = 0.0
    for (j1, j2) in SYMMETRIC_JOINT_PAIRS:
        v_j1_sym = v_pred[:, :, j1].clone()
        v_j1_sym[:, :, 0] = -v_j1_sym[:, :, 0]  # 翻转X坐标（矢状面）
        sym_loss += F.mse_loss(v_j1_sym, v_pred[:, :, j2])

    sym_loss /= len(SYMMETRIC_JOINT_PAIRS)

    # 权重：速度域损失权重远小于位置域（约1/1000）
    return 0.001 * length_loss + 0.0005 * sym_loss


# =============================================================================
# 基础工具模块
# =============================================================================
class SimpleResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, ffn_dim, dropout=0.2, activation='GELU'):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, output_dim))
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim, eps=1e-6)
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)
        y = self.dropout(self.linear2(self.activation(self.linear1(x))))
        return self.norm(residual + y)


class StateEncoder(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_hidden,
            dropout,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_hidden)
        )

    def forward(self, x):
        return self.net(x)


class SpectralMixer(nn.Module):

    def __init__(self, dct_n, dim, expansion=2):
        super().__init__()

        hidden = dct_n * expansion

        self.norm = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dct_n, hidden)
        self.fc2 = nn.Linear(hidden, dct_n)

    def forward(self, x):
        """
        x: [B,J,K,D]
        """

        residual = x

        x = self.norm(x)

        # [B,J,D,K]
        x = x.permute(0, 1, 3, 2)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        # [B,J,K,D]
        x = x.permute(0, 1, 3, 2)

        return residual + x


class SpectralDerivativeEmbedding(nn.Module):
    def __init__(self, dct_n):
        super().__init__()

        freq = torch.arange(dct_n).float()

        self.register_buffer("vel_scale", freq)
        self.register_buffer("acc_scale", freq ** 2)

    def forward(self, x):
        """
        x:
            [B,J,D]
        """
        vel = x * self.vel_scale[None, None, :, None]
        acc = x * self.acc_scale[None, None, :, None]

        feat = torch.cat([x, vel, acc], dim=-1)

        return feat


class FrequencyEmbedding(nn.Module):

    def __init__(self, dim_in, dim_hidden, dct_n):
        super().__init__()

        self.spectral_embed = SpectralDerivativeEmbedding(dct_n)

        self.net = nn.Sequential(
            nn.Linear(dim_in * 3, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_hidden)
        )

    def forward(self, x):
        x = self.spectral_embed(x)
        x = self.net(x)
        return x


# =============================================================================
# 骨骼引导自注意力（核心改进）
# =============================================================================
class SkeletonGuidedSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, num_joints, anatomy_adjacency, flash_attention=True):
        super().__init__()
        self.num_head = num_head
        self.dropout_p = dropout
        self.flash_attention = flash_attention
        self.num_joints = num_joints  # 每个注意力层处理的关节数

        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)

        # ✅ 每个注意力层使用独立的局部偏置矩阵
        self.anatomy_bias = nn.Parameter(torch.zeros(num_joints, num_joints))
        self.anatomy_bias.data = anatomy_adjacency.float() * 1.0

        self.limb_bias = nn.Parameter(torch.zeros(num_joints, num_joints))
        # 同一肢体内部所有关节偏置为0.5
        self.limb_bias.data.fill_(0.5)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim, eps=1e-6)

    def forward(self, x):
        B, T, D = x.shape
        H = self.num_head
        C = D // H

        x_norm = self.norm(x)
        q = self.query(x_norm).view(B, T, H, C).transpose(1, 2)
        k = self.key(x_norm).view(B, T, H, C).transpose(1, 2)
        v = self.value(x_norm).view(B, T, H, C).transpose(1, 2)

        # 计算原始注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(C, dtype=q.dtype))

        # ✅ 现在偏置矩阵大小与输入序列长度完全匹配
        total_bias = self.anatomy_bias + self.limb_bias
        attn_scores = attn_scores + total_bias.unsqueeze(0).unsqueeze(0)

        # 注意力计算
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        y = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, T, D)
        return self.norm(x + y)


class SkeletonGuidedSelfAttentionLite(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, flash_attention=True):
        super().__init__()
        self.num_head = num_head
        self.dropout_p = dropout
        self.flash_attention = flash_attention
        self.head_dim = latent_dim // num_head

        # 统一QKV投影（两种模式共享参数）
        self.qkv_proj = nn.Linear(latent_dim, 3 * latent_dim, bias=False)

        # 全局解剖学偏置（两种模式共享）
        self.anatomy_bias = nn.Parameter(torch.zeros(NUM_JOINTS, NUM_JOINTS))
        self.anatomy_bias.data = ANATOMY_ADJACENCY.float() * 0.1

        self.output_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim, eps=1e-6)

    def forward(self, x, attention_mask=None):
        """
        双模式注意力前向传播
        Args:
            x: (B, J, D) 关节特征
            attention_mask: (1, 1, J, J) 注意力掩码，True表示允许关注
        Returns:
            输出特征（两种模式数值一致）
        """
        B, T, D = x.shape
        H = self.num_head
        C = self.head_dim

        x_norm = self.norm(x)

        # 统一QKV计算（两种模式完全相同）
        qkv = self.qkv_proj(x_norm).view(B, T, 3, H, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, C)
        scale = 1.0 / torch.sqrt(torch.tensor(C, dtype=q.dtype))

        # 统一构建注意力偏置（解剖学偏置 + 掩码）
        attn_bias = self.anatomy_bias.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        if attention_mask is not None:
            mask_bias = torch.where(attention_mask, 0.0, -1e9).to(q.dtype)
            attn_bias = attn_bias + mask_bias

        if self.flash_attention:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_bias,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    scale=scale
                )
        else:
            # 仅启用标准数学注意力，禁用所有加速后端
            with sdpa_kernel([SDPBackend.MATH]):
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_bias,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    scale=scale
                )

        # 统一输出投影（两种模式完全相同）
        y = y.transpose(1, 2).reshape(B, T, D)
        y = self.output_proj(y)

        return self.norm(x + y)


class DynamicSoftMaskAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_p, flash_attention=True):
        super().__init__()
        self.num_head = num_head
        self.head_dim = latent_dim // num_head
        self.flash_attention = flash_attention
        self.dropout_p = dropout_p

        # 统一QKV投影
        self.qkv_proj = nn.Linear(latent_dim, 3 * latent_dim, bias=False)

        # 动态掩码预测器：基于DCT特征预测关节间交互权重
        self.mask_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, NUM_JOINTS),
            nn.Sigmoid()
        )

        # 基础解剖学偏置（保留先验，作为软约束）
        self.anatomy_bias = nn.Parameter(torch.zeros(NUM_JOINTS, NUM_JOINTS))
        self.anatomy_bias.data = ANATOMY_ADJACENCY.float() * 0.5

        self.output_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(latent_dim, eps=1e-6)

    def forward(self, x):
        """
        x: (B, J, D) 关节DCT特征
        """
        B, J, D = x.shape
        H = self.num_head
        C = self.head_dim

        x_norm = self.norm(x)

        # 预测动态软掩码：每个关节预测与其他所有关节的交互权重
        mask_logits = self.mask_predictor(x_norm)  # (B, J, J)
        # 融合基础解剖学偏置
        dynamic_mask = mask_logits * self.anatomy_bias.unsqueeze(0)
        # 转换为注意力偏置（0→-1e9，1→0）
        dynamic_bias = torch.where(dynamic_mask > 0.1, 0.0, -1e9)

        # QKV计算
        qkv = self.qkv_proj(x_norm).view(B, J, 3, H, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = 1.0 / torch.sqrt(torch.tensor(C, dtype=q.dtype))

        if self.flash_attention:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=dynamic_bias.unsqueeze(1),
                    dropout_p=self.dropout_p if self.training else 0.0,
                    scale=scale
                )
        else:
            with sdpa_kernel([SDPBackend.MATH]):
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=dynamic_bias.unsqueeze(1),
                    dropout_p=self.dropout_p if self.training else 0.0,
                    scale=scale
                )

        y = y.transpose(1, 2).reshape(B, J, D)
        y = self.output_proj(y)

        return self.norm(x + y)


class AnatomyDynamicsAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_p, flash_attention=True):
        super().__init__()
        self.num_head = num_head
        self.head_dim = latent_dim // num_head
        self.dropout_p = dropout_p
        self.flash_attention = flash_attention

        # 统一QKV投影（与原版SDFM完全相同）
        self.qkv_proj = nn.Linear(latent_dim, 3 * latent_dim, bias=False)

        prior = build_anatomy_prior()
        self.register_buffer(
            "prior",
            prior
        )

        self.learnable_bias = nn.Parameter(
            torch.zeros(
                NUM_JOINTS,
                NUM_JOINTS
            )
        )

        self.output_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        """
        x: (B, J=16, D=512) 关节特征
        """
        B, J, D = x.shape
        H = self.num_head
        C = self.head_dim
        res = x
        x_norm = self.norm(x)

        qkv = self.qkv_proj(x_norm).view(B, J, 3, H, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_bias = self.prior + self.learnable_bias

        # FlashAttention计算（序列长度=16，极致高效）
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.dropout_p if self.training else 0.0
            )

        y = y.transpose(1, 2).reshape(B, J, D)
        y = self.output_proj(y)
        out = res + self.dropout(y)
        return self.norm(out)


class KineticAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_p, flash_attention=True):
        super().__init__()
        self.num_head = num_head
        self.head_dim = latent_dim // num_head
        self.dropout_p = dropout_p
        self.flash_attention = flash_attention

        # 统一QKV投影（与原版SDFM完全相同）
        self.qkv_proj = nn.Linear(latent_dim, 3 * latent_dim, bias=False)

        prior = build_kinetic_prior()
        self.register_buffer("prior", prior)

        self.learnable_bias = nn.Parameter(torch.zeros(NUM_JOINTS, NUM_JOINTS))

        self.output_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        """
        x: (B, J=16, D=512) 关节特征
        """
        B, J, D = x.shape
        H = self.num_head
        C = self.head_dim
        res = x
        x_norm = self.norm(x)

        qkv = self.qkv_proj(x_norm).view(B, J, 3, H, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_bias = self.prior + self.learnable_bias
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn + attn_bias * 0.05
        attn = F.softmax(attn, dim=-1)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().reshape(B, J, D)
        y = self.output_proj(y)
        out = res + self.dropout(y)
        return out


class JointAttention(nn.Module):

    def __init__(self, latent_dim=512, heads=8, dropout=0.1):
        super().__init__()

        self.dim = latent_dim
        self.heads = heads
        self.head_dim = latent_dim // heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(latent_dim)

        self.qkv = nn.Linear(latent_dim, latent_dim * 3)
        self.proj = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)

        prior = build_kinetic_prior()

        self.register_buffer("prior", prior)

        self.bias = nn.Parameter(
            torch.zeros(NUM_JOINTS, NUM_JOINTS)
        )

    def forward(self, x):
        """
        x: [B,J,K,D]
        """

        B, J, K, D = x.shape

        residual = x

        x = self.norm(x)

        # [B,K,J,D]
        x = x.permute(0, 2, 1, 3)

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B, K, J, 3, self.heads, self.head_dim
        )

        qkv = qkv.permute(3, 0, 1, 4, 2, 5)

        q, k, v = qkv

        attn = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        attn = attn * self.scale

        # attn=attn+0.05*(self.prior+self.bias)

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)

        out = out.permute(0, 1, 3, 2, 4).contiguous()

        out = out.reshape(B, K, J, D)

        out = self.proj(out)

        out = out.permute(0, 2, 1, 3)

        out = residual + self.dropout(out)

        return out


# =============================================================================
# 层次化Transformer块（关节-肢体两级建模）
# =============================================================================
class HierarchicalSDFMBlock(nn.Module):
    def __init__(self, input_dim=512, ff_size=1024, num_heads=8, dropout=0.1):
        super().__init__()
        # 5个生理肢体（与人体结构完全一致）
        self.LIMBS = [
            [0, 1, 2],  # 右腿（3个关节）
            [3, 4, 5],  # 左腿（3个关节）
            [6, 7, 8, 9],  # 躯干+头（4个关节）
            [10, 11, 12],  # 左臂（3个关节）
            [13, 14, 15]  # 右臂（3个关节）
        ]
        self.num_limbs = len(self.LIMBS)

        # ✅ 为每个肢体创建独立的注意力层，使用局部邻接矩阵
        self.limb_attentions = nn.ModuleList()
        for limb_joints in self.LIMBS:
            L = len(limb_joints)
            # 生成该肢体的局部解剖学邻接矩阵
            local_adj = torch.zeros(L, L, dtype=torch.bool)
            for i in range(L):
                for j in range(L):
                    global_i = limb_joints[i]
                    global_j = limb_joints[j]
                    local_adj[i, j] = ANATOMY_ADJACENCY[global_i, global_j]
            # 创建对应大小的注意力层
            self.limb_attentions.append(
                SkeletonGuidedSelfAttention(
                    input_dim,
                    num_heads // 2,
                    dropout,
                    num_joints=L,
                    anatomy_adjacency=local_adj
                )
            )

        # ✅ 肢体间交叉注意力（处理全部16个关节）
        self.limb_cross_attn = SkeletonGuidedSelfAttention(
            input_dim,
            num_heads,
            dropout,
            num_joints=NUM_JOINTS,
            anatomy_adjacency=ANATOMY_ADJACENCY
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, x):
        B, J, D = x.shape
        limb_outputs = []

        # 第一步：每个肢体内部做自注意力
        for i, limb_joints in enumerate(self.LIMBS):
            limb_feat = x[:, limb_joints, :]
            limb_out = self.limb_attentions[i](limb_feat)
            limb_outputs.append(limb_out)

        # 拼接肢体特征
        x_limb = torch.zeros_like(x)
        for i, limb_joints in enumerate(self.LIMBS):
            x_limb[:, limb_joints, :] += limb_outputs[i]

        # 第二步：肢体间交叉注意力（建模全身协同）
        x_global = self.limb_cross_attn(x_limb)

        # 残差连接+前馈
        x = x + x_global
        x = x + self.ffn(self.norm(x))
        return x


class HierarchicalTransBlock(nn.Module):
    def __init__(self, input_dim=512, ff_size=1024, num_heads=8, dropout=0.2,
                 flash_attention=False):
        super().__init__()
        self.input_dim = input_dim
        self.ff_size = ff_size
        self.num_heads = num_heads

        # 预计算全局注意力掩码（无参数量，仅初始化一次）
        self.register_buffer('intra_limb_mask', self._build_intra_limb_mask())
        self.register_buffer('inter_limb_mask', self._build_inter_limb_mask())

        # 单注意力层同时处理内部和交叉注意力（与原版SDFM参数量完全相同）
        self.attention = SkeletonGuidedSelfAttentionLite(input_dim, num_heads, dropout, flash_attention)

        # FFN（与原版完全相同）
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)

    def _build_intra_limb_mask(self):
        """构建肢体内部注意力掩码：每个关节只能关注自己所在肢体的其他关节"""
        LIMBS = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12], [13, 14, 15]]
        mask = torch.zeros(NUM_JOINTS, NUM_JOINTS, dtype=torch.bool)
        for limb in LIMBS:
            for i in limb:
                for j in limb:
                    mask[i, j] = True
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,16,16)

    def _build_inter_limb_mask(self):
        """构建肢体间交叉注意力掩码：所有关节可以互相关注"""
        return torch.ones(1, 1, NUM_JOINTS, NUM_JOINTS, dtype=torch.bool)

    def forward(self, x):
        # 第一步：肢体内部注意力（掩码隔离不同肢体）
        x_intra = self.attention(x, self.intra_limb_mask)

        # 第二步：肢体间交叉注意力（全局协同建模）
        x_inter = self.attention(x_intra, self.inter_limb_mask)

        # 残差连接+FFN
        x = x + x_inter
        x = x + self.ffn(self.norm(x))
        return x


class DynamicsBlock(nn.Module):
    def __init__(
            self,
            latent_dim=512,
            ff_size=1024,
            num_heads=8,
            dropout=0.2,
            flash_attention=False,
            dct_l=15
    ):
        super().__init__()

        self.joint_attn = JointAttention(
            latent_dim, num_heads, dropout
        )
        self.spectral_mixer = SpectralMixer(dct_l, latent_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim)
        )

    def forward(self, x):
        x = self.joint_attn(x)
        x = self.spectral_mixer(x)
        x = x + self.ffn(x)

        return x


class JointFrequencyEmbedding(nn.Module):
    def __init__(self, num_joints=16, num_freq=15, latent_dim=512):
        super().__init__()
        self.proj = nn.Linear(num_freq * 3, latent_dim)
        self.his_encoder = nn.Sequential(nn.Linear(num_freq * 3, 256), nn.GELU(), nn.Linear(256, latent_dim))
        self.freq_gate = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Sigmoid())

    def forward(self, x, joint_emb, mod=None):
        """
        x: 当前帧DCT系数
        mod: 历史帧DCT系数（你的pad过的输入）
        """
        B, K, V3 = x.shape
        J = V3 // 3
        x_reshaped = x.view(B, K, J, 3).permute(0, 2, 1, 3).reshape(B, J, K * 3)
        feat = self.proj(x_reshaped) + joint_emb

        if mod is not None:
            mod_reshaped = mod.view(B, K, J, 3).permute(0, 2, 1, 3).reshape(B, J, K * 3)
            feat = feat + self.his_encoder(mod_reshaped)

        g = self.freq_gate(feat.mean(1, keepdim=True))
        return feat * g


class FrequencyStratifiedBlock(nn.Module):
    def __init__(self, latent_dim, ff_size, num_heads, dropout):
        super().__init__()
        self.attn = AnatomyDynamicsAttention(latent_dim, num_heads, dropout)

        # FFN（与原版完全相同）
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, latent_dim)
        )
        self.norm = nn.LayerNorm(latent_dim)

        self.res_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, residual=None):
        """
        x: 当前层特征 (B,16,512)
        low_freq_feat: 浅层低频特征 (B,16,512)
        high_freq_feat: 深层高频特征 (B,16,512)
        """
        x = x + self.attn(x)
        if residual is not None:
            x = x + torch.sigmoid(self.res_gate) * residual
        x = x + self.ffn(self.norm(x))
        return x


# =============================================================================
# 主模型：Skeleton-Guided SDFM (SG-SDFM)
# =============================================================================
class SGFormer(nn.Module):
    def __init__(self,
                 joint_num=16,
                 num_frames=15,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.2,
                 **kargs):
        super().__init__()
        self.joint_num = joint_num  # 16个非根关节
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.cfg = kargs.get('cfg', type('obj', (object,), {'dct_m_all': util.get_dct_matrix(num_frames)}))
        self.flash_attention = kargs.get('flash_attention', True)

        # 关节嵌入（16个非根关节）
        self.joint_embedding = nn.Parameter(torch.randn(1, self.joint_num, latent_dim))
        nn.init.normal_(self.joint_embedding, std=0.02)

        # 输入/时间/调制投影
        self.input_proj = SimpleResBlock(num_frames * 3, latent_dim, latent_dim, dropout)
        self.time_embed = SimpleResBlock(latent_dim, latent_dim, latent_dim, dropout)
        self.mod_proj = SimpleResBlock(num_frames * 3, latent_dim, latent_dim, dropout)

        # 层次化Transformer编码器
        self.layers = nn.ModuleList([
            HierarchicalSDFMBlock(latent_dim, ff_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 输出头（零初始化保证训练初期稳定）
        self.output_proj = zero_module(nn.Linear(latent_dim, num_frames * 3))

    def forward(self, x, timesteps, mod=None, **kwargs):
        """
        输入：
            x: (B, T, 16*3) 非根关节相对坐标（根关节已置0并移除）
            timesteps: (B,) 扩散时间步
            mod: (B, T, 16*3) 调制条件
        输出：
            output: (B, T, 16*3) 预测的非根关节相对坐标
        """
        B, T, V3 = x.shape
        V = V3 // 3
        assert V == self.joint_num, f"输入关节数错误：期望{self.joint_num}，实际{V}"
        timesteps = timesteps.flatten()

        # 输入特征处理：(B,T,48) → (B,16,45)
        x_reshaped = x.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)
        x_feat = self.input_proj(x_reshaped)

        # 时间步嵌入
        time_emb_base = timestep_embedding(timesteps, self.latent_dim)
        time_emb_encoded = self.time_embed(time_emb_base)
        time_emb = time_emb_encoded.unsqueeze(1).repeat(1, V, 1)

        # 调制特征融合
        if mod is not None and not torch.all(mod == 0):
            mod_reshaped = mod.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)
            mod_feat = self.mod_proj(mod_reshaped)
            x_feat = x_feat + mod_feat

        # 加入时间嵌入和关节嵌入
        h = x_feat + time_emb + self.joint_embedding

        prelist = []
        for i, module in enumerate(self.layers):
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h)
            elif i == (self.num_layers // 2) and self.num_layers % 2 == 1:
                h = module(h)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                h = module(h)
                h += prelist[-1]
                prelist.pop()

        # 输出预测
        output = self.output_proj(h).reshape(B, V, T, 3).permute(0, 2, 1, 3).reshape(B, T, V * 3)
        return output


class SGFormerLite(nn.Module):
    def __init__(self,
                 joint_num=16,
                 num_frames=15,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.2,
                 **kargs):
        super().__init__()
        self.joint_num = joint_num  # 16个非根关节
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.cfg = kargs.get('cfg', type('obj', (object,), {'dct_m_all': util.get_dct_matrix(num_frames)}))
        self.flash_attention = kargs.get('flash_attention', True)

        # 关节嵌入（16个非根关节）
        self.joint_embedding = nn.Parameter(torch.randn(1, self.joint_num, latent_dim))
        nn.init.normal_(self.joint_embedding, std=0.02)

        # 输入/时间/调制投影
        self.input_proj = SimpleResBlock(num_frames * 3, latent_dim, latent_dim, dropout)
        self.time_embed = SimpleResBlock(latent_dim, latent_dim, latent_dim, dropout)
        self.mod_proj = SimpleResBlock(num_frames * 3, latent_dim, latent_dim, dropout)

        # 层次化Transformer编码器
        self.layers = nn.ModuleList([
            HierarchicalTransBlock(latent_dim, ff_size, num_heads, dropout, self.flash_attention)
            for _ in range(num_layers)
        ])

        # 输出头（零初始化保证训练初期稳定）
        self.output_proj = zero_module(nn.Linear(latent_dim, num_frames * 3))

    def forward(self, x, timesteps, mod=None, **kwargs):
        """
        输入：
            x: (B, T, 16*3) 非根关节相对坐标（根关节已置0并移除）
            timesteps: (B,) 扩散时间步
            mod: (B, T, 16*3) 调制条件
        输出：
            output: (B, T, 16*3) 预测的非根关节相对坐标
        """
        B, T, V3 = x.shape
        V = V3 // 3
        assert V == self.joint_num, f"输入关节数错误：期望{self.joint_num}，实际{V}"
        timesteps = timesteps.flatten()

        # 输入特征处理：(B,T,48) → (B,16,45)
        x_reshaped = x.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)
        x_feat = self.input_proj(x_reshaped)

        # 时间步嵌入
        time_emb_base = timestep_embedding(timesteps, self.latent_dim)
        time_emb_encoded = self.time_embed(time_emb_base)
        time_emb = time_emb_encoded.unsqueeze(1).repeat(1, V, 1)

        # 调制特征融合
        if mod is not None and not torch.all(mod == 0):
            mod_reshaped = mod.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)
            mod_feat = self.mod_proj(mod_reshaped)
            x_feat = x_feat + mod_feat

        # 加入时间嵌入和关节嵌入
        h = x_feat + time_emb + self.joint_embedding

        prelist = []
        for i, module in enumerate(self.layers):
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h)
            elif i == (self.num_layers // 2) and self.num_layers % 2 == 1:
                h = module(h)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                h = module(h)
                h += prelist[-1]
                prelist.pop()

        # 输出预测
        output = self.output_proj(h).reshape(B, V, T, 3).permute(0, 2, 1, 3).reshape(B, T, V * 3)
        return output


class SGFormerLite2(nn.Module):
    def __init__(self,
                 joint_num=16,
                 num_frames=15,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.2,
                 **kargs):
        super().__init__()
        self.joint_num = joint_num  # 16个非根关节
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.cfg = kargs.get('cfg', type('obj', (object,), {'dct_m_all': util.get_dct_matrix(75)}))
        self.flash_attention = kargs.get('flash_attention', True)

        self.spectral_embed = SpectralDerivativeEmbedding(num_frames)

        # 关节嵌入（16个非根关节）
        self.joint_embedding = nn.Parameter(torch.randn(1, self.joint_num, 1, latent_dim))
        nn.init.normal_(self.joint_embedding, std=0.02)
        self.freq_embedding = nn.Parameter(torch.randn(1, 1, num_frames, latent_dim))
        nn.init.normal_(self.joint_embedding, std=0.02)

        self.input_proj = nn.Linear(9, latent_dim)

        self.time_embed = SimpleResBlock(latent_dim, latent_dim, latent_dim, dropout, "SiLU")

        # 频域分层Transformer块
        self.layers = nn.ModuleList([DynamicsBlock(latent_dim, ff_size, num_heads, dropout) for _ in range(num_layers)])

        self.output_proj = nn.Sequential(nn.LayerNorm(latent_dim),
                                         nn.Linear(latent_dim, 3))

    def forward(self, x_t, timesteps, cond=None, **kwargs):
        """
        输入：
            x: (B, T, 16*3) 非根关节相对坐标（根关节已置0并移除）
            timesteps: (B,) 扩散时间步
            mod: (B, T, 16*3) 调制条件
        输出：
            output: (B, T, 16*3) 预测的非根关节相对坐标
        """
        B, T, V3 = x_t.shape
        V = V3 // 3

        timesteps = timesteps.flatten()

        x_t = x_t.view(B, T, V, 3).permute(0, 2, 1, 3)

        if cond is not None and not torch.all(cond == 0):
            cond = cond.view(B, T, V, 3).permute(0, 2, 1, 3)
            x_t = x_t + cond

        x = self.spectral_embed(x_t)
        x = self.input_proj(x)

        time_emb_base = timestep_embedding(timesteps, self.latent_dim)
        time_emb_encoded = self.time_embed(time_emb_base)
        time_emb = time_emb_encoded.unsqueeze(1).unsqueeze(1).repeat(1, V, T, 1)

        x = x + time_emb + self.joint_embedding + self.freq_embedding

        for block in self.layers:
            x = block(x)

        out = self.output_proj(x)

        return out.permute(0, 2, 1, 3).reshape(B, T, V * 3)

        """
        prelist = []
        for i, module in enumerate(self.layers):
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h)
            elif i == (self.num_layers // 2) and self.num_layers % 2 == 1:
                h = module(h)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                res = prelist[-1]
                prelist.pop()
                h = module(h)
                h += res
        """

        result = self.output_proj(h).reshape(B, V, T, 3).permute(0, 2, 1, 3).reshape(B, T, V * 3)
        return result


class VKDFFormer(nn.Module):
    def __init__(self, joint_num=16, num_frames=15, latent_dim=512, ff_size=1024, num_layers=6, num_heads=8,
                 dropout=0.2, **kargs):
        super().__init__()
        self.emb = JointFrequencyEmbedding(joint_num, num_frames, latent_dim)
        self.joint_embedding = nn.Parameter(torch.randn(1, joint_num, latent_dim))
        nn.init.normal_(self.joint_embedding, std=0.02)
        self.time_emb = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))

        # 均值流场 (主要预测)
        self.mu_layers = nn.ModuleList(
            [FrequencyStratifiedBlock(latent_dim, ff_size, num_heads, dropout) for _ in range(num_layers)])
        # # 标准差流场 (变分多模态)
        # self.sigma_layers = nn.ModuleList(
        #     [FrequencyStratifiedBlock(latent_dim, 512, 4, dropout * 0.5) for _ in range(2)])

        self.mu_proj = zero_module(nn.Linear(latent_dim, num_frames * 3))
        self.sigma_proj = nn.Sequential(nn.Linear(latent_dim, num_frames * 3), nn.Softplus())
        self.mu = None
        self.sigma = None

    def forward(self, x, timesteps, mod=None, **kwargs):
        """
        完全兼容你的原有接口：
        - x: 当前帧DCT系数
        - timesteps: Flow Matching时间步
        - mod: 历史帧DCT系数（你的pad过的输入）
        - traj_his: 保留参数但不使用，兼容原有调用
        """
        B, K, V3 = x.shape
        J = V3 // 3
        timesteps = timesteps.flatten()

        feat = self.emb(x, self.joint_embedding, mod)
        t_emb = self.time_emb(timestep_embedding(timesteps, 512)).unsqueeze(1).repeat(1, J, 1)
        h = feat + t_emb

        # 均值预测
        h_mu = h
        res = []
        for layer in self.mu_layers:
            res.append(h_mu)
            h_mu = layer(h_mu, res[-1])
        mu = self.mu_proj(h_mu).view(B, J, K, 3).permute(0, 2, 1, 3).reshape(B, K, V3)

        # # 方差预测 (变分)
        # h_sigma = h
        # for layer in self.sigma_layers:
        #     h_sigma = layer(h_sigma, h_sigma)
        sigma = self.sigma_proj(h_mu).view(B, J, K, 3).permute(0, 2, 1, 3).reshape(B, K, V3)

        # 采样
        if self.training:
            z = torch.randn_like(mu)
            pred = mu + sigma * z
        else:
            pred = mu

        self.mu = mu
        self.sigma = sigma
        return pred

    def get_mu_sigma(self):
        return self.mu, self.sigma


# =============================================================================
# 测试代码（验证正确性）
# =============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_pred, t_his, n_pre = 60, 15, 15
    dct_m, _ = util.get_dct_matrix(t_pred + t_his)
    dct_m = dct_m.float().to(device)


    # 模拟配置对象
    class Cfg:
        def __init__(self):
            self.dct_m_all = dct_m


    cfg = Cfg()

    # 初始化模型
    model = SGFormer(
        num_frames=n_pre,
        latent_dim=512,
        ff_size=1024,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        cfg=cfg,
        flash_attention=True
    ).to(device)

    # 混合精度测试
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        B = 1920  # RTX A6000 48GB最大batch size
        x = torch.randn(B, n_pre, NUM_JOINTS * 3).to(device)
        timesteps = torch.randint(0, 1000, (B,)).to(device)
        mod = torch.randn(B, n_pre, NUM_JOINTS * 3).to(device)

        output = model(x, timesteps, mod)
        print(f"✅ 模型测试通过！")
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        assert output.shape == (B, n_pre, NUM_JOINTS * 3), f"输出形状错误：{output.shape}"

        # # 测试物理一致性损失
        # gt = torch.randn_like(output)
        # loss = physical_consistency_loss(output, gt)
        # print(f"物理一致性损失: {loss.item():.6f}")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 模型参数量: {total_params / 1e6:.2f}M")
    print("\n🎉 所有测试通过，模型可以开始训练！")
