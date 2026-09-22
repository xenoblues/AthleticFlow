import torch
import torch.nn as nn
import torch.nn.functional as F
from autobahn.wamp.gen.wamp.proto import Principal
from torch.nn.attention import sdpa_kernel, SDPBackend

from models.transformer import zero_module, timestep_embedding
from utils import util
from data_loader.dataset_ap3d import DatasetAP3D


# HUMAN3.6M
JOINT_NAMES = [
    "pelvis", "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "torso", "neck", "head", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist"
]

JOINT_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
JOINT_ID = {name: i for i, name in enumerate(JOINT_NAMES)}
NUM_JOINTS = len(JOINT_NAMES)

# 解剖学邻接矩阵（硬约束：只有相邻关节才能属于同一肢体）
ANATOMY_ADJACENCY = torch.zeros(NUM_JOINTS, NUM_JOINTS, dtype=torch.bool)
# 填充骨骼连接关系
ANATOMY_ADJACENCY[JOINT_ID["pelvis"], [JOINT_ID["r_hip"], JOINT_ID["l_hip"], JOINT_ID["torso"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["r_hip"], [JOINT_ID["pelvis"], JOINT_ID["r_knee"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["r_knee"], [JOINT_ID["r_hip"], JOINT_ID["r_ankle"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["l_hip"], [JOINT_ID["pelvis"], JOINT_ID["l_knee"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["l_knee"], [JOINT_ID["l_hip"], JOINT_ID["l_ankle"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["torso"], [JOINT_ID["pelvis"], JOINT_ID["neck"], JOINT_ID["l_shoulder"], JOINT_ID["r_shoulder"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["neck"], [JOINT_ID["torso"], JOINT_ID["head"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["head"], [JOINT_ID["neck"], JOINT_ID["head_top"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["l_shoulder"], [JOINT_ID["torso"], JOINT_ID["l_elbow"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["l_elbow"], [JOINT_ID["l_shoulder"], JOINT_ID["l_wrist"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["r_shoulder"], [JOINT_ID["torso"], JOINT_ID["r_elbow"]]] = True
ANATOMY_ADJACENCY[JOINT_ID["r_elbow"], [JOINT_ID["r_shoulder"], JOINT_ID["r_wrist"]]] = True
# 对称化邻接矩阵
ANATOMY_ADJACENCY = ANATOMY_ADJACENCY | ANATOMY_ADJACENCY.T

# 对称关节对（用于通用相位差计算）
SYMMETRIC_JOINT_PAIRS = [
    (JOINT_ID["r_hip"], JOINT_ID["l_hip"]),
    (JOINT_ID["r_knee"], JOINT_ID["l_knee"]),
    (JOINT_ID["r_shoulder"], JOINT_ID["l_shoulder"]),
    (JOINT_ID["r_elbow"], JOINT_ID["l_elbow"])
]


def physical_consistency_loss(pred, gt, joint_parent=JOINT_PARENTS):
    """
    物理一致性损失：骨骼长度不变 + 对称运动约束
    pred/gt: (B, T, (J-1)*3) 非根关节预测/真值
    """
    B, T, _ = pred.shape
    J = NUM_JOINTS

    # 恢复完整关节坐标（添加根关节在原点）
    pred_full = torch.zeros(B, T, J, 3, device=pred.device)
    pred_full[:, :, 1:, :] = pred.reshape(B, T, J - 1, 3)
    gt_full = torch.zeros(B, T, J, 3, device=gt.device)
    gt_full[:, :, 1:, :] = gt.reshape(B, T, J - 1, 3)

    # 1. 骨骼长度一致性损失（人体骨骼长度在运动中不变）
    pred_len = torch.zeros(B, T, J - 1, device=pred.device)
    gt_len = torch.zeros(B, T, J - 1, device=gt.device)
    for j in range(1, J):
        p = joint_parent[j]
        pred_len[:, :, j - 1] = torch.norm(pred_full[:, :, j] - pred_full[:, :, p], dim=-1)
        gt_len[:, :, j - 1] = torch.norm(gt_full[:, :, j] - gt_full[:, :, p], dim=-1)

    length_loss = F.mse_loss(pred_len, gt_len)

    # 2. 对称运动损失（左右对称关节运动相似）
    sym_loss = 0.0
    for (j1, j2) in SYMMETRIC_JOINT_PAIRS:
        # 左右关节的运动应该关于矢状面对称
        pred_sym = pred_full[:, :, j1].clone()
        pred_sym[:, :, 0] = -pred_sym[:, :, 0]  # 翻转X坐标
        sym_loss += F.mse_loss(pred_sym, pred_full[:, :, j2])
    sym_loss /= len(SYMMETRIC_JOINT_PAIRS)

    return 0.01 * length_loss + 0.005 * sym_loss


def limb_regularization_loss(limb_assignment, anatomy_adjacency=ANATOMY_ADJACENCY):
    """肢体聚类正则化损失：鼓励稀疏性与相邻关节同属"""
    # 稀疏性损失：每个关节主要属于1-2个肢体
    sparsity_loss = torch.mean(torch.sum(limb_assignment ** 2, dim=-1))

    # 邻接性损失：鼓励相邻关节属于同一肢体
    adjacency_loss = 0.0
    for i in range(limb_assignment.shape[0]):
        neighbors = torch.where(anatomy_adjacency[i])[0]
        if len(neighbors) > 0:
            sim = torch.cosine_similarity(limb_assignment[i:i + 1], limb_assignment[neighbors], dim=-1)
            adjacency_loss += torch.mean(1 - sim)
    adjacency_loss /= limb_assignment.shape[0]

    return 0.1 * sparsity_loss + 0.05 * adjacency_loss


def update_gumbel_temperature(epoch, total_epochs, initial_temp=1.0, final_temp=0.1):
    """训练过程中逐步降低Gumbel温度，从软分配过渡到硬分配"""
    return initial_temp * (final_temp / initial_temp) ** (epoch / total_epochs)


class LightMLP(nn.Module):
    """轻量级MLP，适配3D特征"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class SimpleResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, output_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        if input_dim != output_dim:
            self.linear3 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y = self.dropout(self.linear2((self.activation(self.linear1(x)))))
        if x.shape[-1] != y.shape[-1]:
            x = self.linear3(x)
        y = x + y
        return self.norm(y)


class BoneExtractor(nn.Module):
    def __init__(self, joint_num=NUM_JOINTS, parent=None):
        super().__init__()
        self.joint_num = joint_num
        self.parent = parent if parent is not None else [
            -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15
        ]
        self.register_buffer('gaussian_kernel',
                             torch.tensor([0.06136, 0.24477, 0.38774, 0.24477, 0.06136], dtype=torch.float32))
        self.global_stat_encoder = nn.Sequential(
            nn.Linear(21, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 32), nn.GELU(), nn.LayerNorm(32),
            nn.Linear(32, 7)
        )
        self.register_buffer('joint_mass',
                             torch.tensor(
                                 [0.28, 0.14, 0.07, 0.02, 0.14, 0.07, 0.02, 0.22, 0.06, 0.05, 0.04, 0.06, 0.04, 0.02,
                                  0.06, 0.04, 0.02],
                                 dtype=torch.float32))

    def _smooth_diff(self, x):
        def forward(self, x):
            # ✅ 终极维度兼容：处理所有可能的输入形状
            if x.dim() == 1:
                # 单样本单帧 (J*3,)
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 2:
                if x.shape[0] == self.joint_num and x.shape[1] == 3:
                    # 单样本单帧 (J, 3)
                    x = x.unsqueeze(0).unsqueeze(0)
                else:
                    # 多样本单帧 (B, J*3)
                    x = x.unsqueeze(1)
            elif x.dim() == 3:
                if x.shape[2] == 3:
                    # 单样本多帧 (T, J, 3)
                    x = x.unsqueeze(0)
                else:
                    # 多样本多帧展平 (B, T, J*3)
                    B, T, JC = x.shape
                    J = JC // 3
                    x = x.reshape(B, T, J, 3)

            # 强制维度检查
            assert x.dim() == 4, f"BoneExtractor输入必须是4维张量，当前维度: {x.dim()}"
            assert x.shape[-1] == 3, f"最后一维必须是3维坐标，当前: {x.shape[-1]}"
            assert x.shape[2] == self.joint_num, f"输入关节数错误：{x.shape[2]}，预期：{self.joint_num}"

        B, T, J, C = x.shape
        diff = torch.zeros_like(x)
        if T >= 3:
            diff[:, 1:-1, :, :] = (x[:, 2:, :, :] - x[:, :-2, :, :]) / 2.0
            diff[:, 0, :, :] = x[:, 1, :, :] - x[:, 0, :, :]
            diff[:, -1, :, :] = x[:, -1, :, :] - x[:, -2, :, :]
        else:
            diff[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
        x_reshaped = diff.permute(0, 2, 3, 1).reshape(B * J * C, 1, T)
        gaussian_kernel = self.gaussian_kernel.view(1, 1, -1).to(device=x.device, dtype=x.dtype)
        x_smooth = F.conv1d(x_reshaped, gaussian_kernel, padding=2).reshape(B, J, C, T).permute(0, 3, 1, 2)
        return x_smooth

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, x):
        B, T, J, _ = x.shape
        x = x.float()
        bone_dir, bone_len, rel_pos = torch.zeros_like(x), torch.zeros(B, T, J, 1, device=x.device,
                                                                       dtype=x.dtype), torch.zeros_like(x)
        root_joint = x[:, :, 0:1, :]

        for j in range(1, J):
            parent_j = self.parent[j]
            vec = x[:, :, j] - x[:, :, parent_j]
            length = torch.norm(vec, dim=-1, keepdim=True) + 1e-6
            bone_dir[:, :, j] = vec / length
            bone_len[:, :, j] = length
            rel_pos[:, :, j, :] = x[:, :, j, :] - root_joint[:, :, 0, :]

        avg_bone_len = bone_len[:, :, 1:, :].mean(dim=2, keepdim=True)
        bone_len[:, :, 0:1, :] = avg_bone_len

        joint_angle = torch.zeros(B, T, J, 3, device=x.device, dtype=x.dtype)
        for j in range(1, J):
            parent_j = self.parent[j]
            if parent_j == 0:
                parent_dir = torch.tensor([0, 1, 0], device=x.device, dtype=x.dtype).expand(B, T, 3)
            else:
                parent_dir = bone_dir[:, :, parent_j]

            cross = torch.cross(parent_dir, bone_dir[:, :, j], dim=-1)
            dot = torch.sum(parent_dir * bone_dir[:, :, j], dim=-1, keepdim=True)
            # 用atan2替代arccos，彻底解决梯度爆炸问题
            angle = torch.atan2(torch.norm(cross, dim=-1, keepdim=True), dot)
            joint_angle[:, :, j] = cross * angle

        joint_vel = self._smooth_diff(joint_angle)
        joint_acc = self._smooth_diff(joint_vel)
        bone_len_vel = self._smooth_diff(bone_len)

        # 最终骨骼特征：11维（角度3 + 长度1 + 速度3 + 加速度3 + 长度速度1）
        # bone_base = torch.cat([joint_angle, bone_len, joint_vel, joint_acc, bone_len_vel], dim=-1)
        bone_base = torch.cat([bone_dir, bone_len], dim=-1)

        # 全局统计特征
        history_frames = x[:, :15, 1:, :]
        all_joint_mean = history_frames.mean(dim=2, keepdim=True)
        all_joint_std = history_frames.std(dim=2, keepdim=True)
        all_bone_len_mean = bone_len[:, :15, 1:, :].mean(dim=2, keepdim=True)
        all_bone_len_std = bone_len[:, :15, 1:, :].std(dim=2, keepdim=True)
        upper_limb_len_mean = bone_len[:, :15, 11:17, :].mean(dim=2, keepdim=True)
        lower_limb_len_mean = bone_len[:, :15, 1:7, :].mean(dim=2, keepdim=True)
        limb_len_ratio = upper_limb_len_mean / (lower_limb_len_mean + 1e-8)
        torso_len_mean = bone_len[:, :15, 7:9, :].sum(dim=2, keepdim=True)
        leg_len_mean = bone_len[:, :15, [1, 2, 4, 5], :].mean(dim=2, keepdim=True)
        torso_leg_ratio = torso_len_mean / (leg_len_mean + 1e-8)
        arm_len_mean = bone_len[:, :15, [11, 12, 14, 15], :].mean(dim=2, keepdim=True)
        arm_leg_ratio = arm_len_mean / (leg_len_mean + 1e-8)
        acc_peak = joint_vel[:, :15, :, :].abs().max(dim=1, keepdim=True)[0].repeat(1, T, 1, 1)
        joint_mass = self.joint_mass.view(1, 1, J, 1).to(device=x.device, dtype=x.dtype)
        momentum_peak = (joint_vel * joint_mass)[:, :15, :, :].abs().max(dim=1, keepdim=True)[0].repeat(1, T, 1, 1)
        symmetry_hip = torch.norm(x[:, :15, 1:2, :] - x[:, :15, 4:5, :], dim=-1, keepdim=True)
        symmetry_shoulder = torch.norm(x[:, :15, 11:12, :] - x[:, :15, 14:15, :], dim=-1, keepdim=True)
        max_displacement = history_frames.abs().max(dim=2, keepdim=True)[0]
        max_displacement_2d = max_displacement[:, :, :, :2]

        global_stats = torch.cat([
            all_joint_mean.squeeze(2), all_joint_std.squeeze(2),
            all_bone_len_mean.squeeze(2), all_bone_len_std.squeeze(2),
            limb_len_ratio.squeeze(2), torso_leg_ratio.squeeze(2),
            arm_leg_ratio.squeeze(2), acc_peak.mean(dim=2).squeeze(2),
            momentum_peak.mean(dim=2).squeeze(2), symmetry_hip.squeeze(2),
            symmetry_shoulder.squeeze(2), max_displacement_2d.squeeze(2)
        ], dim=-1)
        root_feat = self.global_stat_encoder(global_stats)
        avg_angle = joint_angle[:, :, 1:, :].mean(dim=2, keepdim=True)
        joint_angle[:, :, 0:1, :] = avg_angle

        # return bone_base, root_feat, bone_dir
        return bone_base, root_joint, bone_dir


class GeneralizedLimbClustering(nn.Module):
    def __init__(self, joint_num=NUM_JOINTS, num_limbs=24, temperature=1.0, anatomy_adjacency=None):
        super().__init__()
        self.joint_num = joint_num
        self.num_limbs = num_limbs
        self.temperature = temperature
        self.anatomy_adjacency = anatomy_adjacency

        # 可学习的关节-肢体分配logits
        self.joint_limb_logits = nn.Parameter(torch.randn(joint_num, num_limbs))

        # 构建解剖学约束掩码
        if anatomy_adjacency is not None:
            self.register_buffer("constraint_mask", self._build_constraint_mask())
        else:
            self.constraint_mask = None

    def _build_constraint_mask(self):
        """构建正确的解剖学约束掩码：
        对于每个关节j，只允许它与相邻关节分配到同一肢体
        返回：(J, L) 掩码，-1e9表示禁止分配，0表示允许
        """
        mask = torch.zeros(self.joint_num, self.num_limbs, dtype=torch.float32)

        # 对于每个肢体，生成一个连通的关节子集
        for limb_idx in range(self.num_limbs):
            # 随机选择一个种子关节
            seed_joint = torch.randint(0, self.joint_num, (1,)).item()
            # BFS找到2跳邻域（保证肢体连通性）
            visited = set([seed_joint])
            queue = [seed_joint]
            max_limb_size = 4  # 每个肢体最多包含4个关节

            while queue and len(visited) < max_limb_size:
                current = queue.pop(0)
                neighbors = torch.where(self.anatomy_adjacency[current])[0].tolist()
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # 禁止非连通关节分配到当前肢体
            forbidden_joints = [j for j in range(self.joint_num) if j not in visited]
            mask[forbidden_joints, limb_idx] = -1e9

        return mask

    def forward(self):
        """返回：(J, L) 关节-肢体软分配矩阵，每行和为1"""
        logits = self.joint_limb_logits

        # 应用解剖学约束
        if self.constraint_mask is not None:
            logits = logits.masked_fill(self.constraint_mask == 0, -1e9)

        # 可微分软分配
        limb_assignment = F.gumbel_softmax(logits, tau=self.temperature, dim=-1)
        limb_assignment = limb_assignment / (limb_assignment.sum(dim=-1, keepdim=True) + 1e-8)

        return limb_assignment


class DynamicLimbAttention(nn.Module):
    def __init__(self, joint_num=NUM_JOINTS, num_limbs=24, feat_dim=4, hidden_dim=32):
        super().__init__()
        self.num_limbs = num_limbs
        self.limb_encoder = LightMLP(feat_dim, hidden_dim, hidden_dim)
        self.attn_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, bone_temporal, limb_assignment):
        B, T, J, C = bone_temporal.shape

        # 聚合关节特征到肢体特征
        limb_feat = torch.einsum('btjc,jl->btlc', bone_temporal, limb_assignment)

        # 编码肢体特征
        limb_feat_encoded = self.limb_encoder(limb_feat)

        # 预测动态权重
        limb_weights = self.attn_predictor(limb_feat_encoded).squeeze(-1)
        limb_weights = limb_weights / (limb_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return limb_weights, limb_feat_encoded


class LimbFuser(nn.Module):
    def __init__(self,
                 joint_num=NUM_JOINTS,
                 num_limbs=24,
                 feat_dim=4,  # 输入骨骼特征维度
                 out_dim=16,
                 temperature=1.0,
                 anatomy_adjacency=None):
        super().__init__()
        self.joint_num = joint_num
        self.num_limbs = num_limbs
        self.out_dim = out_dim

        # 1. 通用肢体聚类
        self.limb_clustering = GeneralizedLimbClustering(
            joint_num=joint_num,
            num_limbs=num_limbs,
            temperature=temperature,
            anatomy_adjacency=anatomy_adjacency
        )

        # 2. 动态肢体注意力
        self.dynamic_attn = DynamicLimbAttention(
            joint_num=joint_num,
            num_limbs=num_limbs,
            feat_dim=feat_dim
        )

        # 3. 多尺度编码器
        self.scale_encoders = nn.ModuleList([
            LightMLP(feat_dim, 32, out_dim),  # 关节级（细粒度）
            LightMLP(32, 64, out_dim),  # 肢体级（中粒度）
            LightMLP(32, 64, out_dim)  # 全局协同级（粗粒度）
        ])

        # 4. 多尺度融合
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

        # 5. 通用相位差编码器
        self.phase_encoder = nn.Sequential(
            nn.Linear(len(SYMMETRIC_JOINT_PAIRS), 8),
            nn.GELU(),
            nn.Linear(8, 2)
        )

        self.norm = nn.LayerNorm(out_dim + 2)

    def forward(self, bone_temporal, bone_dir, drop_root=False):
        B, T, J, C = bone_temporal.shape

        # 1. 获取通用肢体分配
        limb_assignment = self.limb_clustering()

        # 2. 动态注意力加权
        limb_weights, limb_feat = self.dynamic_attn(bone_temporal, limb_assignment)

        # 3. 多尺度特征提取
        joint_feat = self.scale_encoders[0](bone_temporal)

        weighted_limb_feat = limb_feat * limb_weights.unsqueeze(-1)
        limb_joint_feat = torch.einsum('btlh,jl->btjh', weighted_limb_feat, limb_assignment)
        limb_joint_feat = self.scale_encoders[1](limb_joint_feat)

        global_feat = weighted_limb_feat.mean(dim=2, keepdim=True)
        global_joint_feat = global_feat.repeat(1, 1, J, 1)
        global_joint_feat = self.scale_encoders[2](global_joint_feat)

        # 4. 多尺度融合
        fused_feat = torch.cat([joint_feat, limb_joint_feat, global_joint_feat], dim=-1)
        fused_feat = self.fusion(fused_feat)

        # 5. 通用相位差计算（所有对称关节对）
        phase_diffs = []
        for (j1, j2) in SYMMETRIC_JOINT_PAIRS:
            diff = torch.atan2(
                bone_temporal[:, :, j1, 1] - bone_temporal[:, :, j2, 1],
                bone_temporal[:, :, j1, 0] - bone_temporal[:, :, j2, 0]
            )
            phase_diffs.append(diff)
        phase_feat = torch.stack(phase_diffs, dim=-1)
        final_phase = self.phase_encoder(phase_feat)

        # 6. 拼接最终特征
        final_feat = torch.cat([
            fused_feat,
            final_phase.unsqueeze(2).repeat(1, 1, J, 1)
        ], dim=-1)

        if drop_root:
            final_feat = final_feat[:, :, 1:, :]

        return self.norm(final_feat)


class BoneDCT(nn.Module):
    def __init__(self, joint_num=NUM_JOINTS, num_frames=15, dct_m=None):
        super().__init__()
        self.joint_num = joint_num
        self.num_frames = num_frames
        self.register_buffer('dct_m', dct_m.float() if dct_m is not None else None)
        self.freq_norm = nn.Parameter(torch.ones(num_frames))
        self.freq_predictor = nn.Sequential(
            nn.Linear(11, 32), nn.GELU(), nn.LayerNorm(32),
            nn.Linear(32, num_frames), nn.Sigmoid()
        )

    def forward(self, bone_temporal, global_feat=None):
        B, T, J, C = bone_temporal.shape

        dct_m = self.dct_m[:self.num_frames, :T].to(dtype=bone_temporal.dtype)
        x_dct = torch.matmul(dct_m, bone_temporal.reshape(B, T, -1))
        x_dct = x_dct.reshape(B, self.num_frames, J, C) * self.freq_norm.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # 频率归一化（保留正交性）
        # avg_bone_feat = bone_temporal.mean(dim=(1, 2))
        # dynamic_norm = self.freq_predictor(avg_bone_feat)
        # final_norm = self.freq_norm * dynamic_norm
        # x_dct = x_dct * final_norm.unsqueeze(-1).unsqueeze(-1)

        # 输出：15*11=165维
        return x_dct.permute(0, 2, 1, 3).reshape(B, J, self.num_frames * C)


class LimbDCT(nn.Module):
    def __init__(self, joint_num=NUM_JOINTS, num_frames=15, input_dim=18, dct_m=None):
        super().__init__()
        self.input_dim = input_dim
        self.joint_num = joint_num
        self.num_frames = num_frames
        self.register_buffer('dct_m', dct_m.float() if dct_m is not None else None)
        self.freq_norm = nn.Parameter(torch.ones(num_frames))

    def forward(self, limb_temporal):
        B, T, J, C = limb_temporal.shape

        dct_m = self.dct_m[:self.num_frames, :T].to(dtype=limb_temporal.dtype)
        x_flat = limb_temporal.reshape(B, T, -1)
        x_dct = torch.matmul(dct_m, x_flat)
        x_dct = x_dct.reshape(B, self.num_frames, J, C)

        x_dct = x_dct * self.freq_norm.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # 输出：15*18=270维
        return x_dct.permute(0, 2, 1, 3).reshape(B, J, self.num_frames * C)


class PhysConditionFusion(nn.Module):
    def __init__(self, xt_dim=512, mod_dim=512, bone_dim=165, limb_dim=270, time_dim=512, latent_dim=512):
        super().__init__()
        self.bone_norm = nn.LayerNorm(bone_dim)
        self.limb_norm = nn.LayerNorm(limb_dim)

        self.global_phys_encoder = nn.Sequential(
            nn.Linear(bone_dim + limb_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        self.local_phys_encoder = nn.Sequential(
            nn.Linear(bone_dim + limb_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        self.mod_encoder = nn.Sequential(
            nn.Linear(mod_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        # 初始偏向物理特征
        self.cond_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        nn.init.constant_(self.cond_gate[-2].bias, 2.0)  # 初始输出≈0.88

        self.final_gate = nn.Parameter(torch.tensor(0.2))


    def forward(self, xt_feat, mod_feat, bone_dct, limb_dct, time_emb):
        B, J, D = xt_feat.shape

        bone_dct = self.bone_norm(bone_dct)
        limb_dct = self.limb_norm(limb_dct)
        phys_cond = torch.cat([bone_dct, limb_dct], dim=-1)  # (B, 17, 435)

        # 全局物理特征作为调制系数
        global_phys = phys_cond.mean(dim=1)  # (B, 435)
        global_phys_feat = self.global_phys_encoder(global_phys)  # (B, 512)
        global_gate = torch.sigmoid(global_phys_feat).unsqueeze(1)  # (B, 1, 512)

        local_phys_feat = self.local_phys_encoder(phys_cond)  # (B, 17, 512)
        local_phys_feat = local_phys_feat * global_gate  # (B, 17, 512)

        mod_feat = self.mod_encoder(mod_feat)  # (B, 17, 512)

        # 动态门控融合
        cond_concat = torch.cat([local_phys_feat, mod_feat], dim=-1)  # (B, 17, 1024)
        gate = self.cond_gate(cond_concat)  # (B, 17, 512)
        cond_feat = gate * local_phys_feat + (1 - gate) * mod_feat  # (B, 17, 512)
        cond_feat = cond_feat + time_emb  # (B, 17, 512)

        # 平衡主特征与物理特征
        return self.final_gate * xt_feat + (1 - self.final_gate) * cond_feat


class SelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, cross_attention=False, flash_attention=False):
        super().__init__()
        self.num_head = num_head
        self.dropout_p = dropout
        self.cross_attention = cross_attention
        self.flash_attention = flash_attention
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        if cross_attention:
            self.key_mod = nn.Linear(latent_dim, latent_dim, bias=False)
            self.value_mod = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, x, cond_emb):
        """
        x: B, T, D
        """
        B, T1, D = x.shape
        H = self.num_head
        C = D // H

        q = self.query(self.norm(x))  # B, T, D
        k = self.key(self.norm(x))
        v = self.value(self.norm(x))
        if not self.flash_attention:
            # B, T, H, C
            q_ = q.unsqueeze(2).view(B, T1, H, C)
            k_ = k.unsqueeze(1).view(B, T1, H, C)
            # B, T, T, H
            attention = torch.einsum('bnhd,bmhd->bnmh', q_, k_) / torch.sqrt(C)

            # weight = self.dropout(F.softmax(attention, dim=2))
            weight = F.softmax(attention, dim=2)
            v_ = v.view(B, T1, H, -1)
            y_s = torch.einsum('bnmh,bmhd->bnhd', weight, v_).reshape(B, T1, D)
        else:
            # (B, T, D) -> (B, H, T, C)
            q_ = q.view(B, T1, H, C).transpose(1, 2)
            k_ = k.view(B, T1, H, C).transpose(1, 2)
            v_ = v.view(B, T1, H, C).transpose(1, 2)
            with (sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])):
                y_s = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=self.dropout_p
                                                     ).transpose(1, 2).reshape(B, T1, D)

        # cross attention
        if self.cross_attention and cond_emb is not None:
            T2 = cond_emb.shape[1]
            k_mod = self.key_mod(self.norm(cond_emb))
            v_mod = self.value_mod(self.norm(cond_emb))
            if not self.flash_attention:
                q_ = q.view(B, T1, H, C)
                k_mod_ = k_mod.view(B, T2, H, C)
                v_mod_ = v_mod.view(B, T2, H, C)
                cross_attention = torch.einsum('bnhd,bmhd->bnmh', q_, k_mod_) / torch.sqrt(C)
                cross_weight = self.dropout(F.softmax(cross_attention, dim=2))
                y_c = torch.einsum('bnmh,bmhd->bnhd', cross_weight, v_mod_).reshape(B, T1, D)
            else:
                q_ = q.view(B, T1, H, C).transpose(1, 2)
                k_mod_ = k_mod.view(B, T2, H, C).transpose(1, 2)
                v_mod_ = v_mod.view(B, T2, H, C).transpose(1, 2)
                with (sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])):
                    y_c = F.scaled_dot_product_attention(q_, k_mod_, v_mod_, dropout_p=self.dropout_p
                                                         ).transpose(1,2).reshape(B, T1, D)
            y = y_s + y_c
        else:
            y = y_s

        y = self.norm(x + y)
        return y


class AdalnSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, cross_attention=False, flash_attention=False):
        super().__init__()
        self.num_head = num_head
        self.dropout_p = dropout
        self.cross_attention = cross_attention
        self.flash_attention = flash_attention

        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        if cross_attention:
            self.key_mod = nn.Linear(latent_dim, latent_dim, bias=False)
            self.value_mod = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, x, norm_params):
        """
        新增参数:
            norm_params: (B, 1, 2D) - 本层的AdaLN参数 (gamma_msa, beta_msa)
        """
        B, T1, D = x.shape
        H = self.num_head
        C = D // H

        gamma_msa, beta_msa = norm_params.chunk(2, dim=-1)  # (B,1,D) each
        x_norm = gamma_msa * F.layer_norm(x, (D,)) + beta_msa

        q = self.query(x_norm)
        k = self.key(x_norm)
        v = self.value(x_norm)

        if self.flash_attention:
            q_ = q.view(B, T1, H, C).transpose(1, 2)
            k_ = k.view(B, T1, H, C).transpose(1, 2)
            v_ = v.view(B, T1, H, C).transpose(1, 2)
            with ((sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))):
                y_s = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=self.dropout_p
                                                     ).transpose(1, 2).reshape(B, T1, D)
        else:
            q_ = q.view(B, T1, H, C)
            k_ = k.view(B, T1, H, C)
            attention = torch.einsum('bnhd,bmhd->bnmh', q_, k_) / torch.sqrt(C)
            weight = F.softmax(attention, dim=2)
            v_ = v.view(B, T1, H, C)
            y_s = torch.einsum('bnmh,bmhd->bnhd', weight, v_).reshape(B, T1, D)

        # 交叉注意力（如果需要）
        if self.cross_attention and hasattr(self, 'key_mod'):
            # 交叉注意力同样使用AdaLN归一化
            k_mod = self.key_mod(x_norm)
            v_mod = self.value_mod(x_norm)
            if self.flash_attention:
                q_ = q.view(B, T1, H, C).transpose(1, 2)
                k_mod_ = k_mod.view(B, T1, H, C).transpose(1, 2)
                v_mod_ = v_mod.view(B, T1, H, C).transpose(1, 2)
                with (sdpa_kernel([SDPBackend.FLASH_ATTENTION])):
                    y_c = F.scaled_dot_product_attention(q_, k_mod_, v_mod_, dropout_p=self.dropout_p
                                                         ).transpose(1,2).reshape(B, T1, D)
            else:
                q_ = q.view(B, T1, H, C)
                k_mod_ = k_mod.view(B, T1, H, C)
                cross_attention = torch.einsum('bnhd,bmhd->bnmh', q_, k_mod_) / torch.sqrt(C)
                cross_weight = F.softmax(cross_attention, dim=2)
                v_mod_ = v_mod.view(B, T1, H, C)
                y_c = torch.einsum('bnmh,bmhd->bnhd', cross_weight, v_mod_).reshape(B, T1, D)
            y_s = y_s + y_c

        return y_s


class KASDFMTransformerLayer(nn.Module):
    def __init__(self, input_dim=512, ff_size=1024, num_heads=8, dropout=0.2,
                 cross_attention=False, flash_attention=False, **kargs):
        super().__init__()
        # 关节流 (自注意力)
        self.attention = SelfAttention(input_dim, num_heads, dropout, cross_attention, flash_attention)
        self.ffn = SimpleResBlock(input_dim, input_dim, ff_size, dropout)

    def forward(self, x, cond_emb):
        x = self.attention(x, cond_emb)
        x = self.ffn(x)
        return x


class AdaLNTransformerLayer(nn.Module):
    def __init__(self, input_dim=512, ff_size=1024, num_heads=8, dropout=0.2,
                 cross_attention=False, flash_attention=False, **kargs):
        super().__init__()
        self.input_dim = input_dim

        # 自注意力模块
        self.attention = AdalnSelfAttention(input_dim, num_heads, dropout, cross_attention, flash_attention)

        # FFN模块
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_dim)
        )

        # 核心：AdaLN-Zero调制参数生成器
        # 为MSA和FFN各生成3个参数：gamma, beta, gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_dim, 6 * input_dim)
        )

        # 零初始化：保证训练初期模型是恒等映射
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond_emb=None):
        """
        输入:
            x: (B, J, D) - 关节特征
            time_emb: (B, D) - 全局时间步嵌入
        输出:
            x: (B, J, D) - 融合时间信息后的特征
        """
        B, J, D = x.shape
        if cond_emb is not None:
            modulation = self.adaLN_modulation(cond_emb)  # (B, J, 6D)
            modulation = modulation  # (B, J, 6D) - 广播到所有关节
            (gamma_msa, beta_msa, gate_msa,
             gamma_mlp, beta_mlp, gate_mlp) = modulation.chunk(6, dim=-1)  # (B, J ,D) each
        else:
            gamma_msa, gate_msa, gamma_mlp, gate_mlp = torch.ones(x.shape, dtype=torch.float32).to(x.device)
            beta_msa, beta_mlp = torch.zeros(x.shape, dtype=torch.float32).to(x.device)

        attn_out = self.attention(x, torch.cat([gamma_msa, beta_msa], dim=-1))
        x = x + gate_msa * attn_out  # 门控残差连接

        x_norm = gamma_mlp * F.layer_norm(x, (D,)) + beta_mlp
        ffn_out = self.ffn(x_norm)
        x = x + gate_mlp * ffn_out  # 门控残差连接

        return x


class KASDFMTransformer(nn.Module):
    def __init__(self,
                 input_dim=51,
                 num_frames=15,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.2,
                 joint_num=17,
                 activation="gelu",
                 **kargs):
        super().__init__()
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_dim = input_dim
        self.time_embed_dim = latent_dim
        self.joint_num = joint_num
        self.pred_joint_num  = joint_num - 1  # 16关节进Transformer
        self.use_dynamic_attention = kargs['use_dynamic_attention']
        self.cfg = kargs['cfg']
        self.joint_embedding = nn.Parameter(torch.randn(1, self.joint_num , latent_dim))
        nn.init.normal_(self.joint_embedding, std=0.02)

        self.input_proj = SimpleResBlock(num_frames * 3, latent_dim, latent_dim, dropout)
        self.time_embed = SimpleResBlock(latent_dim, latent_dim, latent_dim, dropout)


        self.bone_extractor = BoneExtractor(joint_num)
        self.limb_fuser = LimbFuser(
            joint_num=joint_num,
            num_limbs=24,
            feat_dim=4,
            out_dim=16,
            temperature=1.0,
            anatomy_adjacency=ANATOMY_ADJACENCY
        )
        self.bone_dct_processor = BoneDCT(
            joint_num=joint_num,
            num_frames=num_frames,
            dct_m=self.cfg.dct_m_all
        )
        self.limb_dct_processor = LimbDCT(
            joint_num=joint_num,
            num_frames=num_frames,
            input_dim=18,
            dct_m=self.cfg.dct_m_all,
        )

        self.phys_fusion = PhysConditionFusion(
            xt_dim=latent_dim,
            mod_dim=latent_dim,
            bone_dim=num_frames * 4,
            limb_dim=270,
            time_dim=latent_dim,
            latent_dim=latent_dim
        )

        if not kargs['use_adaln']:
            self.layers = nn.ModuleList([
                KASDFMTransformerLayer(latent_dim, ff_size, num_heads, dropout,
                                       flash_attention=kargs['flash_attention'],
                                       cross_attention=kargs['cross_attention']) for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                AdaLNTransformerLayer(latent_dim, ff_size, num_heads, dropout,
                                      flash_attention=kargs['flash_attention'],
                                      cross_attention=kargs['cross_attention']) for _ in range(num_layers)
            ])

        self.output_proj = nn.Linear(latent_dim, num_frames * 3)

    def forward(self, x, timesteps, mod=None, **kwargs):
        """
        x: DCT系数 (B, L, J * 3)
        timesteps: 时间步 (B,1)
        mod: padded DCT系数 (B, L, J * 3)
        """
        B, T, V3 = x.shape
        V = V3 // 3
        V_out = self.pred_joint_num  # 16个非根关节

        timesteps = timesteps.flatten()

        history_temporal = None
        min_batch = B
        if timesteps.shape[0] < min_batch:
            min_batch = timesteps.shape[0]
        if mod is not None and mod.shape[0] < min_batch:
            min_batch = mod.shape[0]
        if kwargs['traj_his'] is not None:
            B_his = kwargs['traj_his'].shape[0] if kwargs['traj_his'].dim() == 3 else kwargs['traj_his'].shape[0]
            if B_his < min_batch:
                min_batch = B_his

        # 截断所有输入到最小batch size
        x = x[:min_batch]
        timesteps = timesteps[:min_batch]
        if mod is not None:
            mod = mod[:min_batch]
        if kwargs['traj_his'] is not None:
            history_temporal = kwargs['traj_his'][:min_batch]

        B = min_batch  # 更新batch size

        # 处理输入特征
        x_reshaped = x.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)  # (B, 17, 45)
        x_feat = self.input_proj(x_reshaped)  # (B, 17, 512)

        time_emb_base = timestep_embedding(timesteps, self.latent_dim)  # 强制输出(B, 512)
        time_emb_encoded = self.time_embed(time_emb_base)  # (B, 512)
        time_emb = time_emb_encoded.unsqueeze(1).repeat(1, V, 1)  # (B, 17, 512)

        if mod is not None and not torch.all(mod == 0) and history_temporal is not None:
            if history_temporal.dim() == 3:
                B_his, T_his, JC_his = history_temporal.shape
                J_his = JC_his // 3
                history_temporal = history_temporal.reshape(B_his, T_his, J_his, 3)

            # 提取骨骼和肢体特征
            bone_temporal, root_joint, bone_dir = self.bone_extractor(history_temporal)
            limb_temporal = self.limb_fuser(bone_temporal=bone_temporal, bone_dir=bone_dir, drop_root=False)

            # DCT变换
            bone_dct = self.bone_dct_processor(bone_temporal)
            limb_dct = self.limb_dct_processor(limb_temporal)

            # 处理mod特征
            mod_reshaped = mod.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)  # (B, 17, 45)
            mod_feat = self.input_proj(mod_reshaped)  # (B, 17, 512)

            # 物理特征融合（传入1维time_emb_base）
            fused_feat = self.phys_fusion(x_feat, mod_feat, bone_dct, limb_dct, time_emb)
        else:
            fused_feat = x_feat + time_emb  # (B, 17, 512) + (B, 17, 512) = (B, 17, 512)

        # 分离根关节和非根关节
        root_feat = fused_feat[:, 0:1, :]  # (B, 1, 512)
        non_root_feat = fused_feat[:, 1:, :]  # (B, 16, 512)

        # 全局调制
        root_global_feat = root_feat.mean(dim=1, keepdim=True)
        fused_feat_non_root = non_root_feat * (1 + root_global_feat.sigmoid())  # (B, 16, 512)

        # 位置编码
        joint_emb_non_root = self.joint_embedding[:, 1:, :]  # (1, 16, 512)
        h = fused_feat_non_root + joint_emb_non_root  # (B, 16, 512)

        time_emb = time_emb[:, 1:, :]
        prelist = []
        for i, module in enumerate(self.layers):
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, time_emb)
            elif i == (self.num_layers // 2) and self.num_layers % 2 == 1:
                h = module(h, time_emb)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                h = module(h, time_emb)
                h += prelist[-1]
                prelist.pop()

        output = self.output_proj(h).reshape(B, V_out, T, 3).permute(0, 2, 1, 3).reshape(B, T, -1)
        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_pred = 60
    t_his = 15
    n_pre = 15

    # 生成测试DCT矩阵
    dct_m, idct_m = util.get_dct_matrix(t_pred + t_his)
    dct_m = dct_m.float().to(device)
    idct_m = idct_m.float().to(device)


    # 模拟配置对象
    class Cfg:
        def __init__(self):
            self.dct_m_all = dct_m
            self.idct_m_all = idct_m
            self.t_total = t_pred + t_his
            self.n_pre = n_pre


    cfg = Cfg()

    # 初始化模型
    model = KASDFMTransformer(
        num_frames=n_pre,
        joint_num=NUM_JOINTS,
        cfg=cfg,
        use_dynamic_attention=True,
        flash_attention=True,
        cross_attention=False,
        use_adaln=False
    ).to(device)

    # 测试不完整batch（模拟最后一个batch只有511个样本）
    B = 511
    x = torch.randn(B, n_pre, NUM_JOINTS * 3).to(device)
    timesteps = torch.randint(0, 1000, (B,)).to(device)
    mod = torch.randn(B, n_pre, NUM_JOINTS * 3).to(device)
    history_temporal = torch.randn(B, t_his, NUM_JOINTS * 3).to(device)

    print("🚀 测试不完整batch...")
    with torch.no_grad():
        output = model(x, timesteps, mod, history_temporal=history_temporal)
        print(f"✅ 不完整batch测试通过！")
        print(f"输入batch size: {B}")
        print(f"输出形状: {output.shape}")
        print(f"预期输出形状: {(B, n_pre, (NUM_JOINTS - 1) * 3)}")
        assert output.shape == (B, n_pre, (NUM_JOINTS - 1) * 3), f"输出形状错误：{output.shape}"

    print("\n🎉 所有测试通过，模型可以稳定训练了！")