import math
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import networkx as nx
from utils.util import compute_contact_free_athletic_state

LIMBS = []

H36M_LIMBS = [
    [0, 1, 2],  # right leg
    [3, 4, 5],  # left leg
    [6, 7, 8, 9],  # torso
    [10, 11, 12],  # left arm
    [13, 14, 15],  # right arm
]

ANCHORED_H36M_LIMBS = [[6, 0, 1, 2], [6, 3, 4, 5], [6, 7, 8, 9], [7, 10, 11, 12], [7, 13, 14, 15]]

ANCHORED_WP_LIMBS_23 = [
    [2, 0, 3, 6, 9],
    [2, 1, 4, 7, 10],
    [2, 5, 8, 11, 14],
    [8, 12, 15, 17, 19, 21],
    [8, 13, 16, 18, 20, 22],
]

CHAIN_SPECS_H36M = [
    ("right_leg", [6, 0, 1, 2]),
    ("left_leg", [6, 3, 4, 5]),
    ("torso", [6, 7, 8, 9]),
    ("left_arm", [7, 10, 11, 12]),
    ("right_arm", [7, 13, 14, 15]),
]

H36M_LIMB_CHAIN_EDGES = [(6, 0), (0, 1), (1, 2), (6, 3), (3, 4), (4, 5), (7, 10), (10, 11), (11, 12), (7, 13),
                         (13, 14), (14, 15)]

LIMB_CHAIN_ORDER_AP3D = [
    [6, 0, 1, 2],
    [6, 3, 4, 5],
    [7, 10, 11, 12],
    [7, 13, 14, 15]
]

LIMB_CHAIN_ORDER_AP3D = [
    [6, 0, 1, 2],
    [6, 3, 4, 5],
    [7, 10, 11, 12],
    [7, 13, 14, 15]
]

WP_LIMBS = [
    [0, 3], [3, 6], [6, 9],
    [1, 4], [4, 7], [7, 10],
    [2, 5], [5, 8], [8, 11], [11, 14],
    [12, 15], [15, 17], [17, 19], [19, 21],
    [13, 16], [16, 18], [18, 20], [20, 22]
]

WP_LIMBS2 = [
    # left leg
    [0, 3, 6, 9],

    # right leg
    [1, 4, 7, 10],

    # torso
    [2, 5, 8, 11, 14],

    # left arm
    [12, 15, 17, 19, 21],

    # right arm
    [13, 16, 18, 20, 22],
]

H36M_SKELETON_EDGES = [
    # right leg
    (0, 1), (1, 2),

    # left leg
    (3, 4), (4, 5),

    # spine
    (6, 7), (7, 8), (8, 9),

    # left arm
    (7, 10), (10, 11), (11, 12),

    # right arm
    (7, 13), (13, 14), (14, 15),

    # pelvis removed
    # reconnect legs to spine
    (6, 0), (6, 3), (0, 3), (7, 0), (7, 3)
]

COORD_EDGES = [
    # 左手 ↔ 右脚
    (12, 5),
    (11, 4),

    # 右手 ↔ 左脚
    (15, 2),
    (14, 1),

    # 双手
    (12, 15),
    (11, 14),

    # 双脚
    (2, 5),
    (1, 4),

    # 躯干协调
    (8, 2),
    (8, 5),
    (8, 12),
    (8, 15),
    (7, 2),
    (7, 5),
    (7, 12),
    (7, 15),
]

CHAIN_EDGES = [

    # right leg
    (0, 1),
    (1, 2),

    # left leg
    (3, 4),
    (4, 5),

    # lower body → spine
    (0, 6),
    (3, 6),

    # spine chain
    (6, 7),
    (7, 8),
    (8, 9),

    # left arm
    (7, 10),
    (10, 11),
    (11, 12),

    # right arm
    (7, 13),
    (13, 14),
    (14, 15),
]

JOINT_NAMES = [
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "torso", "neck", "head", "head_top",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist"
]
NUM_JOINTS = len(JOINT_NAMES)  # 16个非根关节

SUPER_LIMBS = [
    # anatomical
    [0, 1, 2],  # right leg
    [3, 4, 5],  # left leg
    [10, 11, 12],  # left arm
    [13, 14, 15],  # right arm

    # leg-spine
    [0, 1, 6, 7],
    [3, 4, 6, 7],

    # arm-spine
    [10, 11, 6, 7],
    [13, 14, 6, 7],

    # contralateral
    [11, 12, 1, 2],
    [14, 15, 4, 5],
]
N_SUPER = len(SUPER_LIMBS)

KEY_JOINTS = [
    6,  # spine
    12,  # left wrist
    15,  # right wrist
    5,  # left ankle
    2  # right ankle
]

LOCAL_EDGES = [
    (0,1),(1,2),      # right leg
    (3,4),(4,5),      # left leg

    (6,7),(7,8),(8,9), # torso

    (7,10),(10,11),(11,12), # left arm
    (7,13),(13,14),(14,15)  # right arm
]

LOCAL_NEIGHBORS = {
    0:[1],
    1:[0,2],
    2:[1],

    3:[4],
    4:[3,5],
    5:[4],

    6:[7],
    7:[6,8,10,13],
    8:[7,9],
    9:[8],

    10:[7,11],
    11:[10,12],
    12:[11],

    13:[7,14],
    14:[13,15],
    15:[14]
}

def _logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def build_limb_adj_matrix(num_joints=16):

    A = torch.eye(num_joints)

    intra_edges = [
        (0,1),(1,2),
        (3,4),(4,5),
        (6,7),(7,8),(8,9),
        (10,11),(11,12),
        (13,14),(14,15)
    ]

    for i,j in intra_edges:
        A[i,j] = 1.
        A[j,i] = 1.

    A = A / A.sum(dim=1, keepdim=True)

    return A


def build_graph_distance_matrix(num_joints=16):
    G = nx.Graph()
    G.add_nodes_from(range(num_joints))
    G.add_edges_from(H36M_SKELETON_EDGES)

    dist = torch.zeros(num_joints, num_joints, dtype=torch.int32)

    for i in range(num_joints):
        path = nx.single_source_shortest_path_length(G, i)

        for j in range(num_joints):
            dist[i, j] = path[j]

    return dist


def build_anatomy_graph():
    A = torch.zeros(
        NUM_JOINTS,
        NUM_JOINTS,
        dtype=torch.float32
    )

    for i, j in H36M_SKELETON_EDGES:
        A[i, j] = 1.
        A[j, i] = 1.

    A.fill_diagonal_(1.)

    return A


def build_chain_mask(num_joints=16):
    mask = torch.zeros(
        num_joints,
        num_joints
    )

    for i, j in CHAIN_EDGES:
        mask[i, j] = 1.
        mask[j, i] = 1.

    mask.fill_diagonal_(1.)
    return mask


def build_coord_mask():
    mask = torch.eye(NUM_JOINTS, dtype=torch.bool)

    for i, j in COORD_EDGES:
        mask[i, j] = True
        mask[j, i] = True

    return mask.unsqueeze(0).unsqueeze(0)


def build_multi_hop_graph(
        anatomy_graph,
        max_hop=3
):
    A = anatomy_graph

    hops = 0

    cur = A.clone()

    gamma = 0.8

    for i in range(max_hop):
        cur = (cur > 0).float()
        cur.fill_diagonal_(0)
        hops = hops + cur * (gamma ** i)
        cur = cur @ A

    return hops


def build_functional_graph():
    F = torch.zeros(
        NUM_JOINTS,
        NUM_JOINTS,
        dtype=torch.float32
    )

    pairs = [
        # hand-foot
        (12, 2),
        (15, 5),

        # bilateral hand
        (12, 15),

        # bilateral foot
        (2, 5),

        # shoulders
        (10, 13),

        # hips
        (0, 3),

        # shoulder-hip
        (10, 0),
        (13, 3),

        # arm-leg symmetry
        (12, 5),
        (15, 2),
    ]

    for i, j in pairs:
        F[i, j] = 1.
        F[j, i] = 1.

    return F


def build_intra_limb_mask():
    """构建肢体内部注意力掩码：每个关节只能关注自己所在肢体的其他关节"""
    global LIMBS
    mask = torch.zeros(NUM_JOINTS, NUM_JOINTS, dtype=torch.bool)
    for limb in LIMBS:
        for i in limb:
            for j in limb:
                mask[i, j] = True
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,16,16)


def build_global_limb_mask():
    return torch.ones(1, 1, NUM_JOINTS, NUM_JOINTS, dtype=torch.bool)


def build_inter_limb_mask():
    mask = torch.zeros(
        NUM_JOINTS,
        NUM_JOINTS,
        dtype=torch.bool
    )

    right_leg = LIMBS[0]
    left_leg = LIMBS[1]
    torso = LIMBS[2]
    left_arm = LIMBS[3]
    right_arm = LIMBS[4]

    def connect(group_a, group_b):
        for i in group_a:
            for j in group_b:
                mask[i, j] = True
                mask[j, i] = True

    # ------------------------
    # leg <-> leg
    # ------------------------
    connect(left_leg, right_leg)

    # ------------------------
    # arm <-> arm
    # ------------------------
    connect(left_arm, right_arm)

    # ------------------------
    # torso <-> legs
    # ------------------------
    connect(torso, left_leg)
    connect(torso, right_leg)

    # ------------------------
    # torso <-> arms
    # ------------------------
    connect(torso, left_arm)
    connect(torso, right_arm)

    return mask


def build_edges_from_chains(chains):
    edges = []
    seen = set()

    for chain in chains:
        for p, c in zip(chain[:-1], chain[1:]):
            if (p, c) not in seen:
                edges.append((p, c))
                seen.add((p, c))

    return edges


def build_chain_relative_index(num_joints=16, chains=ANCHORED_H36M_LIMBS):
    max_rel = max(len(c) for c in chains) - 1
    rel_offset = torch.zeros(num_joints, num_joints, dtype=torch.long)
    valid = torch.zeros(num_joints, num_joints, dtype=torch.bool)

    for chain in chains:
        for qi, i in enumerate(chain):
            for kj, j in enumerate(chain):
                offset = kj - qi
                if not valid[i, j]:
                    rel_offset[i, j] = offset
                    valid[i, j] = True
                else:
                    old = rel_offset[i, j].item()
                    if abs(offset) < abs(old):
                        rel_offset[i, j] = offset

    rel_index = rel_offset.clamp(-max_rel, max_rel) + max_rel
    return rel_index, valid, max_rel


def build_relation_matrix():
    rel = torch.full((16, 16), 3)

    for i in range(16):
        rel[i, i] = 0

    edges = [
        (0, 1), (1, 2),
        (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9),
        (10, 11), (11, 12),
        (13, 14), (14, 15)
    ]

    for i, j in edges:
        rel[i, j] = 1
        rel[j, i] = 1

    limbs = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ]

    for limb in limbs:
        for i in limb:
            for j in limb:
                if rel[i, j] != 1 and i != j:
                    rel[i, j] = 2
    return rel


def build_factorized_chain_indices(num_joints=16, chain_specs=CHAIN_SPECS_H36M):
    max_len = max(len(chain) for _, chain in chain_specs)
    max_rel = max_len - 1
    num_types = len(chain_specs)

    rel_offset = torch.zeros(num_joints, num_joints, dtype=torch.long)
    type_index = torch.zeros(num_joints, num_joints, dtype=torch.long)
    depth_pair = torch.zeros(num_joints, num_joints, dtype=torch.long)
    valid = torch.zeros(num_joints, num_joints, dtype=torch.bool)

    for type_id, (_, chain) in enumerate(chain_specs):
        for qi, i in enumerate(chain):
            for kj, j in enumerate(chain):
                offset = kj - qi
                rel = offset + max_rel
                dp = qi * max_len + kj

                if not valid[i, j]:
                    rel_offset[i, j] = rel
                    type_index[i, j] = type_id
                    depth_pair[i, j] = dp
                    valid[i, j] = True
                else:
                    old_rel = rel_offset[i, j].item() - max_rel
                    if abs(offset) < abs(old_rel):
                        rel_offset[i, j] = rel
                        type_index[i, j] = type_id
                        depth_pair[i, j] = dp

    return rel_offset, type_index, depth_pair, valid, max_rel, num_types, max_len


def build_factorized_chain_indices(num_joints=16, chain_specs=CHAIN_SPECS_H36M):
    max_len = max(len(chain) for _, chain in chain_specs)
    max_rel = max_len - 1
    num_types = len(chain_specs)

    rel_index = torch.zeros(num_joints, num_joints, dtype=torch.long)
    type_index = torch.zeros(num_joints, num_joints, dtype=torch.long)
    depth_index = torch.zeros(num_joints, num_joints, dtype=torch.long)
    valid = torch.zeros(num_joints, num_joints, dtype=torch.bool)

    for type_id, (_, chain) in enumerate(chain_specs):
        for qi, i in enumerate(chain):
            for kj, j in enumerate(chain):
                rel = kj - qi + max_rel
                depth = qi * max_len + kj

                if not valid[i, j]:
                    rel_index[i, j] = rel
                    type_index[i, j] = type_id
                    depth_index[i, j] = depth
                    valid[i, j] = True
                else:
                    old_rel = rel_index[i, j].item() - max_rel
                    new_rel = kj - qi
                    if abs(new_rel) < abs(old_rel):
                        rel_index[i, j] = rel
                        type_index[i, j] = type_id
                        depth_index[i, j] = depth

    return rel_index, type_index, depth_index, valid, max_rel, num_types, max_len


def build_chain_edges(chain_specs=CHAIN_SPECS_H36M):
    edges = []
    seen = set()

    for chain in chain_specs:
        for p, c in zip(chain[:-1], chain[1:]):
            if (p, c) not in seen:
                edges.append((p, c))
                seen.add((p, c))

    return edges

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_activation(activation_name: str) -> nn.Module:
    if not activation_name:
        return nn.Identity()

    try:
        activation_class = getattr(nn, activation_name)
        return activation_class()
    except AttributeError:
        raise ValueError(f"Unsupport Activation Func: {activation_name}")

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


class MotionConditionedChainBias(nn.Module):
    def __init__(self, dim, heads=8, num_joints=16, chains=CHAIN_SPECS_H36M,
                 n_chain_heads=2, hidden=None, dropout=0.2, init_gate=-4.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.num_joints = num_joints
        self.n_chain_heads = n_chain_heads

        rel_index, valid, max_rel = build_chain_relative_index(num_joints, chains)
        self.max_rel = max_rel
        self.num_rel = 2 * max_rel + 1

        self.register_buffer("rel_index", rel_index)
        self.register_buffer("chain_valid", valid)

        edges = build_chain_edges(chains)
        parent_ids = torch.tensor([p for p, c in edges], dtype=torch.long)
        child_ids = torch.tensor([c for p, c in edges], dtype=torch.long)

        self.register_buffer("parent_ids", parent_ids)
        self.register_buffer("child_ids", child_ids)

        head_mask = torch.zeros(heads)
        if n_chain_heads > 0:
            head_mask[-n_chain_heads:] = 1.0
        self.register_buffer("head_mask", head_mask)

        hidden = hidden or dim

        self.static_bias = nn.Parameter(torch.zeros(heads, self.num_rel))

        self.dynamic_mlp = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, heads * self.num_rel)
        )

        self.gate = nn.Parameter(torch.tensor(init_gate))

        nn.init.zeros_(self.dynamic_mlp[-1].weight)
        nn.init.zeros_(self.dynamic_mlp[-1].bias)

    def forward(self, x):
        B, J, D = x.shape

        global_feat = x.mean(dim=1)
        parent_feat = x[:, self.parent_ids].mean(dim=1)
        child_feat = x[:, self.child_ids].mean(dim=1)
        edge_feat = (x[:, self.child_ids] - x[:, self.parent_ids]).mean(dim=1)

        feat = torch.cat([global_feat, parent_feat, child_feat, edge_feat], dim=-1)

        dyn = self.dynamic_mlp(feat).view(B, self.heads, self.num_rel)

        bias = self.static_bias[None] + torch.sigmoid(self.gate) * dyn
        bias = bias * self.head_mask[None, :, None]

        rel_flat = self.rel_index.reshape(-1)
        pair_bias = bias[:, :, rel_flat].reshape(B, self.heads, J, J)

        return pair_bias

END_JOINTS_H36M = [2, 5, 12, 15]


class TimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device) / half
        )

        x = t[:, None] * freqs[None]
        emb = torch.cat([x.sin(), x.cos()], dim=-1)

        return self.mlp(emb)


class AdaptiveCondition(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x, cond):
        g = self.gate(torch.cat([x, cond], dim=-1))
        return x + g * cond


class HybridChainRelativeDualQKVAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.2, num_joints=16,
                 chains=ANCHORED_H36M_LIMBS, n_chain_heads=2):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.qkv_local = nn.Linear(dim, dim * 3, bias=False)
        self.out_local = nn.Linear(dim, dim, bias=False)

        self.qkv_global = nn.Linear(dim, dim * 3, bias=False)
        self.out_global = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        rel_index, valid, max_rel = build_chain_relative_index(num_joints, chains)
        self.register_buffer("chain_valid", valid[None, None])

        self.motion_chain_bias = MotionConditionedChainBias(dim=dim, heads=heads, num_joints=num_joints, chains=chains,
                                                            n_chain_heads=n_chain_heads, dropout=dropout, init_gate=-4.0)

        # self.motion_chain_bias = FactorizedChainRelativeBias(heads=heads, num_joints=num_joints,
        #                                                      chain_specs=CHAIN_SPECS_H36M,  n_chain_heads=2, init_gate=-1.5)
        # self.motion_chain_bias = MotionGatedFactorizedChainRelativeBias(dim=dim, heads=heads, num_joints=num_joints,
        #                                                      chain_specs=CHAIN_SPECS_H36M,  n_chain_heads=2, init_gate=-1.5)

    def _split_qkv(self, qkv):
        B, N, _ = qkv.shape
        qkv = qkv.view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _local_attn(self, q, k, v, x_for_bias, intra_mask):
        B, H, N, D = q.shape

        score = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        pair_bias = self.motion_chain_bias(x_for_bias).to(device=score.device, dtype=score.dtype)

        # FactorizedChainRelativeBias
        # pair_bias = self.motion_chain_bias(B).to(device=score.device, dtype=score.dtype)

        score = score + pair_bias

        if intra_mask is not None:
            mask = intra_mask.to(device=score.device, dtype=torch.bool)
        else:
            mask = self.chain_valid.to(device=score.device)

        score = score.masked_fill(~mask, -1e6)

        attn = torch.softmax(score.float(), dim=-1).to(dtype=q.dtype)
        attn = self.dropout(attn)

        return attn @ v

    def forward(self, x, intra_mask, inter_mask):
        B, N, D = x.shape

        x1 = self.norm1(x)
        q, k, v = self._split_qkv(self.qkv_local(x1))

        y = self._local_attn(q, k, v, x1, intra_mask)
        y = y.transpose(1, 2).reshape(B, N, D)

        x_local = x + self.out_local(y)

        x2 = self.norm2(x_local)
        q, k, v = self._split_qkv(self.qkv_global(x2))

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=inter_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim)
            )

        y = y.transpose(1, 2).reshape(B, N, D)

        return x_local + self.out_global(y)



class DualStageTransBlock(nn.Module):
    def __init__(
            self,
            input_dim=512,
            ff_size=1024,
            num_heads=8,
            dropout=0.2,
            chains=ANCHORED_H36M_LIMBS,
            n_chain_heads=2,
            num_joint=16
    ):
        super().__init__()

        self.attn = HybridChainRelativeDualQKVAttention(input_dim, num_heads, dropout, chains=chains,
                                                        n_chain_heads=n_chain_heads, num_joints=num_joint)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_dim)
        )

        self.norm = nn.LayerNorm(input_dim)

        self.register_buffer("intra_mask", build_intra_limb_mask())
        self.register_buffer("inter_mask", build_global_limb_mask())

    def forward(self, x):
        x = self.attn(x, self.intra_mask, self.inter_mask)
        x = x + self.ffn(self.norm(x))
        return x



class AthleticFlow(nn.Module):
    def __init__(
            self,
            joints=16,
            t_total=75,
            k=15,
            dim=128,
            depth=6,
            heads=8,
            dropout=0.1,
            cfg=None
    ):
        super().__init__()
        self.joints = joints
        self.k = k
        self.depth = depth
        self.cfg = cfg
        global LIMBS
        global NUM_JOINTS
        NUM_JOINTS = cfg.joint_num
        if cfg.dataset == 'ap3d' or cfg.dataset == 'ap':
            LIMBS = ANCHORED_H36M_LIMBS
        elif cfg.dataset == 'wp':
            LIMBS = ANCHORED_WP_LIMBS_23

        in_dim = k * 3

        self.x_proj = SimpleResBlock(in_dim, dim, dim, dropout)
        self.cond_proj = SimpleResBlock(in_dim, dim, dim, dropout)

        self.t_embed = TimeEmbedding(dim)
        self.joint_emb = nn.Parameter(torch.randn(1, joints, dim) * 0.02)

        self.cond_inject = nn.ModuleList([AdaptiveCondition(dim) for _ in range(depth)])

        head_list = [0, 0, 0, 2, 2, 2]
        self.blocks = nn.ModuleList(
            [DualStageTransBlock(input_dim=dim, dropout=dropout, num_heads=heads, ff_size=dim * 2, num_joint=joints,
                                 chains=LIMBS, n_chain_heads=head_list[i]) for i in range(depth)]
        )


        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, in_dim)

    def forward(self, x_t, t, cond=None, **kwargs):
        # x_t: [B,K,V*3]
        # cond: [B,K,V*3]
        B, T, V3 = x_t.shape
        V = V3 // 3

        x_t = x_t.reshape(B, T, V, 3).permute(0, 2, 1, 3)

        if cond is None:
            cond = torch.zeros_like(x_t)
        else:
            cond = cond.reshape(B, T, V, 3).permute(0, 2, 1, 3)

        x_t = x_t.reshape(B, V, -1)
        cond = cond.reshape(B, V, -1)

        # -------------------------------------------------
        # embeddings
        # -------------------------------------------------

        x = self.x_proj(x_t)
        c = self.cond_proj(cond)
        t_emb = self.t_embed(t).unsqueeze(1)

        x = x + t_emb + self.joint_emb

        features = []
        num_blocks = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            x = self.cond_inject[i](x, c)
            x = blk(x)

            if i < num_blocks // 2:
                features.append(x)
            else:
                skip = features[num_blocks - i - 1]
                x = x + skip


        x = self.norm(x)
        out = self.out(x)

        return out.reshape(B, V, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)

