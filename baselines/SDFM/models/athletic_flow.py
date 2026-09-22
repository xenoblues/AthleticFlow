import math
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from .SGTransformer import SimpleResBlock
import networkx as nx
from utils.util import compute_contact_free_athletic_state
from mamba_ssm import Mamba
from typing import Optional

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


def make_residual_head(dim, dropout=0.1, zero_last=False):
    head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
    if zero_last:
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    return head


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


"""
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

        # only static
        # bias = self.static_bias[None]

        # only dyn
        # bias = torch.sigmoid(self.gate) * dyn

        bias = bias * self.head_mask[None, :, None]

        rel_flat = self.rel_index.reshape(-1)
        pair_bias = bias[:, :, rel_flat].reshape(B, self.heads, J, J)

        return pair_bias
"""

class MotionConditionedChainBias(nn.Module):
    """
    Motion-conditioned kinematic chain-relative bias.

    Supported bias modes:
        - "both":    static bias + dynamic bias
        - "static":  static bias only
        - "dynamic": dynamic bias only
        - "none":    disable chain-relative bias
    """

    _MODE_ALIASES = {
        "both": "both",
        "full": "both",
        "static+dynamic": "both",
        "static": "static",
        "static_only": "static",
        "static-only": "static",
        "dynamic": "dynamic",
        "dynamic_only": "dynamic",
        "dynamic-only": "dynamic",
        "none": "none",
        "off": "none",
        "no_bias": "none",
        "w/o_bias": "none",
    }

    def __init__(
        self,
        dim,
        heads=8,
        num_joints=16,
        chains=CHAIN_SPECS_H36M,
        n_chain_heads=2,
        hidden=None,
        dropout=0.2,
        init_gate=-4.0,
        bias_mode="static",
        mask_invalid_pairs=True,
    ):
        super().__init__()

        if not 0 <= n_chain_heads <= heads:
            raise ValueError(
                f"n_chain_heads must be between 0 and {heads}, "
                f"but received {n_chain_heads}."
            )

        self.dim = dim
        self.heads = heads
        self.num_joints = num_joints
        self.n_chain_heads = n_chain_heads
        self.mask_invalid_pairs = mask_invalid_pairs

        self.bias_mode = self._canonicalize_mode(bias_mode)

        rel_index, valid, max_rel = build_chain_relative_index(
            num_joints,
            chains,
        )
        self.max_rel = max_rel
        self.num_rel = 2 * max_rel + 1

        self.register_buffer(
            "rel_index",
            rel_index.long(),
        )
        self.register_buffer(
            "chain_valid",
            valid.bool(),
        )

        edges = build_chain_edges(chains)

        if len(edges) == 0:
            raise ValueError("No valid kinematic-chain edges were constructed.")

        parent_ids = torch.tensor(
            [parent for parent, child in edges],
            dtype=torch.long,
        )
        child_ids = torch.tensor(
            [child for parent, child in edges],
            dtype=torch.long,
        )

        self.register_buffer("parent_ids", parent_ids)
        self.register_buffer("child_ids", child_ids)

        # Only the last n_chain_heads heads receive chain-relative bias.
        head_mask = torch.zeros(heads, dtype=torch.bool)
        if n_chain_heads > 0:
            head_mask[-n_chain_heads:] = True

        self.register_buffer("head_mask", head_mask)

        hidden = hidden or dim

        # Motion-independent relative-position bias.
        self.static_bias = nn.Parameter(
            torch.zeros(heads, self.num_rel)
        )

        # Motion-conditioned relative-position bias.
        self.dynamic_mlp = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, heads * self.num_rel),
        )

        # Controls the magnitude of the dynamic component.
        self.gate = nn.Parameter(
            torch.tensor(float(init_gate))
        )

        # Preserve the original zero initialization.
        nn.init.zeros_(self.dynamic_mlp[-1].weight)
        nn.init.zeros_(self.dynamic_mlp[-1].bias)

    @classmethod
    def _canonicalize_mode(cls, mode: str) -> str:
        """Convert aliases into a canonical bias mode."""
        if not isinstance(mode, str):
            raise TypeError(
                f"bias_mode must be a string, but received {type(mode)}."
            )

        normalized = mode.lower().strip()

        if normalized not in cls._MODE_ALIASES:
            supported = sorted(set(cls._MODE_ALIASES.values()))
            raise ValueError(
                f"Unsupported bias_mode '{mode}'. "
                f"Supported modes are {supported}."
            )

        return cls._MODE_ALIASES[normalized]

    def set_bias_mode(self, mode: str) -> None:
        """Change the active bias mode after initialization."""
        self.bias_mode = self._canonicalize_mode(mode)

    def _compute_motion_descriptor(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct the motion descriptor.

        Args:
            x: Joint features with shape [B, J, D].

        Returns:
            Motion descriptor with shape [B, 4D].
        """
        global_feat = x.mean(dim=1)

        parent_feat = x[:, self.parent_ids].mean(dim=1)
        child_feat = x[:, self.child_ids].mean(dim=1)

        edge_feat = (
            x[:, self.child_ids]
            -
            x[:, self.parent_ids]
        ).mean(dim=1)

        return torch.cat(
            [
                global_feat,
                parent_feat,
                child_feat,
                edge_feat,
            ],
            dim=-1,
        )

    def _compute_dynamic_bias(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the motion-conditioned bias table.

        Returns:
            Dynamic bias with shape [B, H, 2R+1].
        """
        batch_size = x.shape[0]

        motion_feat = self._compute_motion_descriptor(x)

        dynamic_bias = self.dynamic_mlp(motion_feat).view(
            batch_size,
            self.heads,
            self.num_rel,
        )

        dynamic_bias = (
            torch.sigmoid(self.gate)
            *
            dynamic_bias
        )

        return dynamic_bias

    def forward(
        self,
        x: torch.Tensor,
        bias_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Joint features with shape [B, J, D].
            bias_mode:
                Optional temporary mode override. When None, the mode
                specified during initialization is used.

        Returns:
            Pairwise attention bias with shape [B, H, J, J].
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [B, J, D], "
                f"but received shape {tuple(x.shape)}."
            )

        batch_size, num_joints, feature_dim = x.shape

        if num_joints != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joints, "
                f"but received {num_joints}."
            )

        if feature_dim != self.dim:
            raise ValueError(
                f"Expected feature dimension {self.dim}, "
                f"but received {feature_dim}."
            )

        mode = (
            self.bias_mode
            if bias_mode is None
            else self._canonicalize_mode(bias_mode)
        )

        # No-bias ablation.
        if mode == "none" or self.n_chain_heads == 0:
            return x.new_zeros(
                batch_size,
                self.heads,
                num_joints,
                num_joints,
            )

        if mode == "static":
            # Shape: [B, H, 2R+1]
            bias = self.static_bias.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )

        elif mode == "dynamic":
            # Shape: [B, H, 2R+1]
            bias = self._compute_dynamic_bias(x)

        elif mode == "both":
            static_bias = self.static_bias.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
            dynamic_bias = self._compute_dynamic_bias(x)

            bias = static_bias + dynamic_bias

        else:
            raise RuntimeError(f"Unexpected bias mode: {mode}")

        # Apply the bias only to selected attention heads.
        bias = bias * self.head_mask[
            None, :, None
        ].to(dtype=bias.dtype)

        # Convert relative-offset bias tables into pairwise bias matrices.
        rel_flat = self.rel_index.reshape(-1)

        pair_bias = bias[:, :, rel_flat].reshape(
            batch_size,
            self.heads,
            num_joints,
            num_joints,
        )

        # Pairs that do not belong to a common predefined chain receive
        # zero chain-relative bias.
        if self.mask_invalid_pairs:
            pair_bias = pair_bias.masked_fill(
                ~self.chain_valid[None, None, :, :],
                0.0,
            )

        return pair_bias

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"heads={self.heads}, "
            f"num_joints={self.num_joints}, "
            f"n_chain_heads={self.n_chain_heads}, "
            f"bias_mode='{self.bias_mode}'"
        )


class MotionGatedFactorizedChainRelativeBias(nn.Module):
    def __init__(self, dim, heads=8, num_joints=16, chain_specs=CHAIN_SPECS_H36M,
                 n_chain_heads=2, hidden=None, dropout=0.2, init_gate=-1.5):
        super().__init__()

        rel_index, type_index, depth_index, valid, max_rel, num_types, max_len = build_factorized_chain_indices(
            num_joints=num_joints,
            chain_specs=chain_specs
        )

        self.dim = dim
        self.heads = heads
        self.num_joints = num_joints
        self.n_chain_heads = n_chain_heads
        self.max_rel = max_rel
        self.num_rel = 2 * max_rel + 1
        self.num_types = num_types
        self.num_depth = max_len * max_len

        self.register_buffer("rel_index", rel_index)
        self.register_buffer("type_index", type_index)
        self.register_buffer("depth_index", depth_index)
        self.register_buffer("chain_valid", valid)

        head_mask = torch.zeros(heads)
        head_mask[-n_chain_heads:] = 1.0
        self.register_buffer("head_mask", head_mask)

        edges = build_chain_edges(chain_specs)
        parent_ids = torch.tensor([p for p, c in edges], dtype=torch.long)
        child_ids = torch.tensor([c for p, c in edges], dtype=torch.long)

        self.register_buffer("parent_ids", parent_ids)
        self.register_buffer("child_ids", child_ids)

        self.rel_bias = nn.Parameter(torch.zeros(heads, self.num_rel))
        self.type_bias = nn.Parameter(torch.zeros(heads, self.num_types))
        self.depth_bias = nn.Parameter(torch.zeros(heads, self.num_depth))

        hidden = hidden or dim

        self.motion_gate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, heads * 3)
        )

        self.base_gate = nn.Parameter(torch.tensor(init_gate))

        nn.init.zeros_(self.motion_gate[-1].weight)
        nn.init.zeros_(self.motion_gate[-1].bias)

    def forward(self, x):
        """
        x:
            [B, J, D]

        return:
            [B, H, J, J]
        """

        B, J, D = x.shape

        if J != self.num_joints:
            raise ValueError(f"Expected num_joints={self.num_joints}, got {J}")

        global_feat = x.mean(dim=1)

        parent_feat = x[:, self.parent_ids].mean(dim=1)
        child_feat = x[:, self.child_ids].mean(dim=1)

        edge_feat = child_feat - parent_feat

        motion_feat = torch.cat([global_feat, parent_feat + child_feat, edge_feat], dim=-1)

        gate = self.motion_gate(motion_feat).view(B, self.heads, 3)
        gate = torch.sigmoid(gate)

        rel_flat = self.rel_index.reshape(-1)
        type_flat = self.type_index.reshape(-1)
        depth_flat = self.depth_index.reshape(-1)

        rel_b = self.rel_bias[:, rel_flat].reshape(self.heads, J, J)
        type_b = self.type_bias[:, type_flat].reshape(self.heads, J, J)
        depth_b = self.depth_bias[:, depth_flat].reshape(self.heads, J, J)

        bias = (
            gate[:, :, 0, None, None] * rel_b[None]
            +
            gate[:, :, 1, None, None] * type_b[None]
            +
            gate[:, :, 2, None, None] * depth_b[None]
        )

        bias = bias * self.head_mask[None, :, None, None]
        bias = torch.sigmoid(self.base_gate) * bias

        return bias


class FactorizedChainRelativeBias(nn.Module):
    def __init__(self, heads=4, num_joints=16, chain_specs=CHAIN_SPECS_H36M, n_chain_heads=2, init_gate=-1.5):
        super().__init__()

        rel_index, type_index, depth_pair, valid, max_rel, num_types, max_len = build_factorized_chain_indices(num_joints, chain_specs)

        self.heads = heads
        self.num_joints = num_joints
        self.max_rel = max_rel
        self.num_rel = 2 * max_rel + 1
        self.num_types = num_types
        self.max_len = max_len
        self.num_depth_pair = max_len * max_len

        self.register_buffer("rel_index", rel_index)
        self.register_buffer("type_index", type_index)
        self.register_buffer("depth_pair", depth_pair)
        self.register_buffer("chain_valid", valid)

        head_mask = torch.zeros(heads)
        head_mask[-n_chain_heads:] = 1.0
        self.register_buffer("head_mask", head_mask)

        self.rel_bias = nn.Parameter(torch.zeros(heads, self.num_rel))
        self.type_bias = nn.Parameter(torch.zeros(heads, self.num_types))
        self.depth_bias = nn.Parameter(torch.zeros(heads, self.num_depth_pair))

        self.gate = nn.Parameter(torch.tensor(init_gate))

    def forward(self, batch_size):
        J = self.num_joints

        rel_flat = self.rel_index.reshape(-1)
        type_flat = self.type_index.reshape(-1)
        depth_flat = self.depth_pair.reshape(-1)

        rel_b = self.rel_bias[:, rel_flat].reshape(self.heads, J, J)
        type_b = self.type_bias[:, type_flat].reshape(self.heads, J, J)
        depth_b = self.depth_bias[:, depth_flat].reshape(self.heads, J, J)

        bias = rel_b + type_b + depth_b
        bias = bias * self.head_mask[:, None, None]
        bias = torch.sigmoid(self.gate) * bias

        return bias[None].expand(batch_size, -1, -1, -1)


class BiMambaLimbChainResidual(nn.Module):
    def __init__(self, dim, num_joints=16, chains=ANCHORED_H36M_LIMBS, dropout=0.1,
                 d_state=16, d_conv=2, expand=1, init_gate=-4.0, zero_init=True):
        super().__init__()
        self.dim = dim
        self.num_joints = num_joints
        self.chains = chains
        self.max_len = max(len(c) for c in chains)

        self.mamba_f = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_b = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.pos_emb = nn.Parameter(torch.randn(1, self.max_len, dim) * 0.02)

        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.gate = nn.Parameter(torch.tensor(init_gate))

        degree = torch.zeros(num_joints)
        joint_scale = torch.ones(num_joints)

        for chain in chains:
            for j in chain:
                degree[j] += 1.0

        joint_scale[[6, 7, 8, 9]] = 0.5
        joint_scale[[0, 1, 3, 4, 10, 11, 13, 14]] = 0.8
        joint_scale[[2, 5, 12, 15]] = 1.2

        self.register_buffer("degree", degree.clamp_min(1.0))
        self.register_buffer("joint_scale", joint_scale)

        if zero_init:
            nn.init.zeros_(self.out[-1].weight)
            nn.init.zeros_(self.out[-1].bias)

    def forward(self, x):
        B, J, D = x.shape

        if J != self.num_joints:
            raise ValueError(f"Expected num_joints={self.num_joints}, got {J}")

        input_dtype = x.dtype
        device = x.device

        agg = torch.zeros(B, J, D, device=device, dtype=input_dtype)

        for chain in self.chains:
            idx = torch.tensor(chain, dtype=torch.long, device=device)

            pos = self.pos_emb[:, :len(chain)].to(device=device, dtype=input_dtype)
            seq = x[:, idx] + pos

            y_f = self.mamba_f(seq)

            seq_rev = torch.flip(seq, dims=[1])
            y_b = torch.flip(self.mamba_b(seq_rev), dims=[1])

            y = 0.5 * (y_f + y_b)

            # AMP / BF16 safe
            y = y.to(dtype=input_dtype)

            agg.index_add_(1, idx, y)

        degree = self.degree.to(device=device, dtype=input_dtype)
        joint_scale = self.joint_scale.to(device=device, dtype=input_dtype)

        agg = agg / degree[None, :, None]

        delta = self.out(agg)
        delta = delta * joint_scale[None, :, None]

        out = torch.sigmoid(self.gate).to(dtype=input_dtype) * delta

        return out.to(dtype=input_dtype)


class CondSpectralMixer(nn.Module):
    def __init__(self, k=15, joints=16, init_scale=0.03):
        super().__init__()
        self.k = k
        self.joints = joints
        self.A = nn.Parameter(torch.eye(k) + 0.01 * torch.randn(k, k))
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, z):
        B, J, V, _ = z.shape
        x = z.permute(0, 2, 1, 3)
        A = torch.softmax(self.A, dim=-1)
        mixed = torch.einsum("kl,bljc->bkjc", A, x)
        x = x + torch.tanh(self.scale) * (mixed - x)
        return x.permute(0, 2, 1, 3)


class HybridChainRelativeDualQKVAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.2, num_joints=16,
                 chains=ANCHORED_H36M_LIMBS, n_chain_heads=2, use_ila=True, bias_mode='both'):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout_p = dropout
        self.use_ila = use_ila

        self.norm2 = nn.LayerNorm(dim)

        self.qkv_global = nn.Linear(dim, dim * 3, bias=False)
        self.out_global = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        if self.use_ila:
            self.norm1 = nn.LayerNorm(dim)

            self.qkv_local = nn.Linear(dim, dim * 3, bias=False)
            self.out_local = nn.Linear(dim, dim, bias=False)

            rel_index, valid, max_rel = build_chain_relative_index(num_joints, chains)
            self.register_buffer("chain_valid", valid[None, None])

            self.motion_chain_bias = MotionConditionedChainBias(dim=dim, heads=heads, num_joints=num_joints, chains=chains,
                                                                n_chain_heads=n_chain_heads, dropout=dropout, init_gate=-4.0,
                                                                bias_mode=bias_mode)

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

        if self.use_ila:
            x1 = self.norm1(x)
            q, k, v = self._split_qkv(self.qkv_local(x1))

            y = self._local_attn(q, k, v, x1, intra_mask)
            y = y.transpose(1, 2).reshape(B, N, D)

            x_local = x + self.out_local(y)
        else:
            x_local = x

        x2 = self.norm2(x_local)
        q, k, v = self._split_qkv(self.qkv_global(x2))

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=inter_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim)
            )

        y = y.transpose(1, 2).reshape(B, N, D)

        return x_local + self.out_global(y)


class ChainRelativeDualQKVAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.2, num_joints=16, chains=ANCHORED_H36M_LIMBS):
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

        rel_index, valid, max_rel = build_chain_relative_index(num_joints=num_joints, chains=chains)

        self.max_rel = max_rel
        self.register_buffer("rel_index", rel_index)
        self.register_buffer("chain_valid", valid[None, None])

        self.chain_rel_bias = nn.Parameter(torch.zeros(heads, 2 * max_rel + 1))

    def _split_qkv(self, qkv):
        B, N, _ = qkv.shape
        qkv = qkv.view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _local_attn_with_chain_bias(self, q, k, v, intra_mask):
        B, H, N, D = q.shape

        score = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        bias = self.chain_rel_bias[:, self.rel_index.reshape(-1)].reshape(H, N, N)
        bias = bias.to(dtype=score.dtype, device=score.device)

        score = score + bias[None]

        if intra_mask is not None:
            mask = intra_mask.to(device=score.device, dtype=torch.bool)
        else:
            mask = self.chain_valid.to(device=score.device)

        score = score.masked_fill(~mask, -1e4)

        attn = torch.softmax(score.float(), dim=-1).to(dtype=q.dtype)
        attn = self.dropout(attn)

        return attn @ v

    def forward(self, x, intra_mask, inter_mask):
        B, N, D = x.shape

        x1 = self.norm1(x)
        q, k, v = self._split_qkv(self.qkv_local(x1))

        y = self._local_attn_with_chain_bias(q, k, v, intra_mask)
        y = y.transpose(1, 2).reshape(B, N, D)

        x_local = x + self.out_local(y)

        x2 = self.norm2(x_local)
        q, k, v = self._split_qkv(self.qkv_global(x2))

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=inter_mask, dropout_p=self.dropout_p if self.training else 0.0, scale=1.0 / math.sqrt(self.head_dim))

        y = y.transpose(1, 2).reshape(B, N, D)

        out = x_local + self.out_global(y)

        return out


class LowRankCondAdaLN(nn.Module):
    def __init__(self, dim, rank=16, dropout=0.1, init_raw=0.05, init_mod=0.05, zero_init=True):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.x_down = nn.Linear(dim, rank, bias=False)
        self.c_down = nn.Linear(dim, rank, bias=False)
        self.g_down = nn.Linear(dim, rank, bias=False)
        self.to_mod = nn.Linear(rank, dim * 3)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.raw_scale = nn.Parameter(torch.tensor(init_raw))
        self.mod_scale = nn.Parameter(torch.tensor(init_mod))

        if zero_init:
            nn.init.zeros_(self.to_mod.weight)
            nn.init.zeros_(self.to_mod.bias)

    def forward(self, x, cond):
        xn = self.norm_x(x)
        cn = self.norm_c(cond)
        cg = cn.mean(dim=1, keepdim=True).expand_as(cn)

        h = self.x_down(xn) + self.c_down(cn) + self.g_down(cg)
        h = self.drop(self.act(h))

        gamma, beta, gate = self.to_mod(h).chunk(3, dim=-1)
        delta = torch.sigmoid(gate) * (gamma * xn + beta)

        raw_scale = torch.tanh(self.raw_scale)
        mod_scale = torch.tanh(self.mod_scale)

        return x + raw_scale * cond + mod_scale * delta


class LowRankCondAdaLNJointGate(nn.Module):
    def __init__(self, dim, rank=16, dropout=0.1, init_raw=0.05, init_mod=0.05, zero_init=True):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.x_down = nn.Linear(dim, rank, bias=False)
        self.c_down = nn.Linear(dim, rank, bias=False)
        self.g_down = nn.Linear(dim, rank, bias=False)
        self.to_mod = nn.Linear(rank, dim * 3)
        self.joint_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1), nn.Sigmoid())
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.raw_scale = nn.Parameter(torch.tensor(init_raw))
        self.mod_scale = nn.Parameter(torch.tensor(init_mod))

        if zero_init:
            nn.init.zeros_(self.to_mod.weight)
            nn.init.zeros_(self.to_mod.bias)

    def forward(self, x, cond):
        xn = self.norm_x(x)
        cn = self.norm_c(cond)
        cg = cn.mean(dim=1, keepdim=True).expand_as(cn)

        h = self.x_down(xn) + self.c_down(cn) + self.g_down(cg)
        h = self.drop(self.act(h))

        gamma, beta, gate = self.to_mod(h).chunk(3, dim=-1)
        delta = torch.sigmoid(gate) * (gamma * xn + beta)

        jgate = self.joint_gate(cn)
        raw_scale = torch.tanh(self.raw_scale)
        mod_scale = torch.tanh(self.mod_scale)

        return x + jgate * raw_scale * cond + jgate * mod_scale * delta


class ResidualFlowHead(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        hidden_dim=None
    ):
        super().__init__()

        hidden_dim = hidden_dim or dim

        self.v0_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.r1_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.r2_head = nn.Sequential(
            nn.Linear(dim + out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, feat):

        v0 = self.v0_head(feat)

        r1 = self.r1_head(feat)

        v1 = v0 + r1

        r2 = self.r2_head(
            torch.cat([feat, r1], dim=-1)
        )

        v = v1 + r2

        return v, v0, r1, r2


class LowRankCrossFuse(nn.Module):
    def __init__(self, dim, rank=16, dropout=0.1, init_scale=0.03):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_c = nn.LayerNorm(dim)
        self.x_down = nn.Linear(dim, rank, bias=False)
        self.c_down = nn.Linear(dim, rank, bias=False)
        self.g_down = nn.Linear(dim, rank, bias=False)
        self.value_up = nn.Linear(rank, dim)
        self.gate_up = nn.Linear(rank, dim)
        self.drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.act = nn.GELU()
        nn.init.constant_(self.gate_up.bias, -1.0)
        nn.init.zeros_(self.value_up.weight)
        nn.init.zeros_(self.value_up.bias)

    def forward(self, x, c):
        xn = self.norm_x(x)
        cn = self.norm_c(c)
        cg = cn.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        h = self.x_down(xn) + self.c_down(cn) + self.g_down(cg)
        h = self.drop(self.act(h))
        v = self.value_up(h)
        g = torch.sigmoid(self.gate_up(h))
        return self.scale * g * v


class ChainRefinementBlock(nn.Module):

    def __init__(self,
                 dim,
                 hidden=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x:
            (B,L,D)
        """
        center = x.mean(dim=1, keepdim=True)
        x = x + center
        x = x + self.mlp(self.norm(x))
        return x


class HierarchicalLimbRefinement(nn.Module):

    def __init__(self,
                 dim,
                 dct_dim):
        super().__init__()
        self.RIGHT_LEG = [0, 1, 2]
        self.LEFT_LEG = [3, 4, 5]
        self.LEFT_ARM = [7, 10, 11, 12]
        self.RIGHT_ARM = [7, 13, 14, 15]

        self.right_leg_block = ChainRefinementBlock(dim)

        self.left_leg_block = ChainRefinementBlock(dim)

        self.left_arm_block = ChainRefinementBlock(dim)

        self.right_arm_block = ChainRefinementBlock(dim)

        self.foot_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dct_dim)
        )

        self.wrist_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dct_dim)
        )

    def forward(self, feat):
        """
        feat:
            (B,J,D)
        """
        right_leg = feat[:, self.RIGHT_LEG]
        left_leg = feat[:, self.LEFT_LEG]
        left_arm = feat[:, self.LEFT_ARM]
        right_arm = feat[:, self.RIGHT_ARM]

        right_leg = self.right_leg_block(right_leg)
        left_leg = self.left_leg_block(left_leg)
        left_arm = self.left_arm_block(left_arm)
        right_arm = self.right_arm_block(right_arm)

        rfoot_feat = right_leg[:, -1]
        lfoot_feat = left_leg[:, -1]
        lwrist_feat = left_arm[:, -1]
        rwrist_feat = right_arm[:, -1]

        rfoot_res = self.foot_head(rfoot_feat)
        lfoot_res = self.foot_head(lfoot_feat)
        lwrist_res = self.wrist_head(lwrist_feat)
        rwrist_res = self.wrist_head(rwrist_feat)

        return {
            "rfoot": rfoot_res,
            "lfoot": lfoot_res,
            "lwrist": lwrist_res,
            "rwrist": rwrist_res
        }

END_JOINTS_H36M = [2, 5, 12, 15]
class GlobalEndEffectorFlowResidual(nn.Module):
    def __init__(self, dim, k=15, joints=16, end_joints=END_JOINTS_H36M, heads=4, dropout=0.1, init_gate=-5.0):
        super().__init__()
        self.dim = dim
        self.k = k
        self.joints = joints
        self.end_joints = end_joints
        self.num_end = len(end_joints)

        self.register_buffer("end_idx", torch.tensor(end_joints, dtype=torch.long))

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.end_to_all = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

        self.head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, k * 3)
        )

        self.gate = nn.Parameter(torch.tensor(init_gate))

        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, return_end=False):
        B, J, D = x.shape

        if J != self.joints:
            raise ValueError(f"Expected joints={self.joints}, got {J}")

        end_feat = x[:, self.end_idx]
        q = self.norm_q(end_feat)
        kv = self.norm_kv(x)

        ctx, _ = self.end_to_all(q, kv, kv, need_weights=False)

        global_feat = x.mean(dim=1, keepdim=True).expand(B, self.num_end, D)

        feat = torch.cat([ctx, global_feat], dim=-1)

        delta_end = self.head(feat)
        delta_end = torch.sigmoid(self.gate) * delta_end

        delta_joint = x.new_zeros(B, J, self.k * 3)
        delta_joint[:, self.end_idx] = delta_end

        if return_end:
            return delta_joint, delta_end

        return delta_joint

class GraphBias(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()

        dist = build_graph_distance_matrix(num_joints)

        self.register_buffer(
            "graph_dist",
            dist
        )

        self.alpha = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(self):
        return -self.alpha * self.graph_dist


class UnifiedAnatomyBias(nn.Module):

    def __init__(
            self,
            use_anatomy=True,
            use_multihop=False,
            use_functional=False,
            max_hop=3
    ):
        super().__init__()

        anatomy = build_anatomy_graph().float()

        self.anatomy_bias = nn.Parameter(anatomy)

        functional = build_functional_graph().float()
        self.functional_bias = nn.Parameter(functional)

        hops = build_multi_hop_graph(anatomy, max_hop)
        self.hops_bias = nn.Parameter(hops)

        self.use_anatomy = use_anatomy
        self.use_multihop = use_multihop
        self.use_functional = use_functional

        # -----------------------
        # learnable scales
        # -----------------------

        self.anatomy_scale = nn.Parameter(
            torch.tensor(1.0)
        )

        self.functional_scale = nn.Parameter(
            torch.tensor(0.2)
        )

        self.hop_weight = nn.Parameter(
            torch.tensor(
                [1.0, 0.5, 0.25][:max_hop]
            )
        )

    def forward(self):
        bias = torch.zeros(NUM_JOINTS, NUM_JOINTS, dtype=torch.float).cuda()
        if self.use_anatomy:
            bias = bias + self.anatomy_bias

        if self.use_multihop:
            # hop_bias = self.hop_weight[:, None, None] * self.hop_graph.sum(0)
            bias = bias + self.hops_bias

        if self.use_functional:
            bias = bias + self.functional_bias

        return bias


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


class LiteEdgeGCN(nn.Module):

    def __init__(self, dim):
        super().__init__()

        A = build_limb_adj_matrix()

        self.register_buffer(
            "A",
            A
        )

        self.proj = nn.Linear(
            dim,
            dim,
            bias=False
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x:
        [B,J,D]
        """

        y = torch.einsum(
            'ij,bjd->bid',
            self.A,
            x
        )

        y = self.proj(y)

        return self.norm(
            x + y
        )

class ChainTransformer(nn.Module):

    def __init__(self, dim, depth=2, heads=8, dropout=0.2):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

    def forward(self, x):
        return self.encoder(x)


class TorsoAwareChainTransformer(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.arm_chain = ChainTransformer(
            dim=dim,
            depth=1,
            heads=4
        )

        self.leg_chain = ChainTransformer(
            dim=dim,
            depth=1,
            heads=4
        )

    def refine_chain(self, x, idx, module):
        feat = x[:, idx]

        feat = feat + module(feat)

        x = x.clone()

        x[:, idx] = feat

        return x

    def forward(self, x):
        x = self.refine_chain(
            x,
            [6, 7, 8, 10, 11, 12],
            self.arm_chain
        )

        x = self.refine_chain(
            x,
            [6, 7, 8, 13, 14, 15],
            self.arm_chain
        )

        x = self.refine_chain(
            x,
            [0, 3, 4, 5],
            self.leg_chain
        )

        x = self.refine_chain(
            x,
            [0, 1, 2],
            self.leg_chain
        )

        return x


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


class AnatomySelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, flash_attention=True):
        super().__init__()
        self.num_head = num_head
        self.dropout_p = dropout
        self.flash_attention = flash_attention
        self.head_dim = latent_dim // num_head
        self.qkv_proj = nn.Linear(latent_dim, 3 * latent_dim, bias=False)
        # self.anatomy_bias = nn.Parameter(torch.zeros(NUM_JOINTS, NUM_JOINTS))
        # self.anatomy_bias.data = build_adjacency().float()

        self.bias_builder = UnifiedAnatomyBias(use_anatomy=False, use_multihop=False, use_functional=False)

        self.output_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim, eps=1e-6)

    def forward(self, x, attention_mask=None, dynamic_route=None):
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
        attn_bias = self.bias_builder().unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
        if attention_mask is not None:
            mask_bias = torch.where(attention_mask, 0.0, -1e9).to(q.dtype)
            attn_bias = attn_bias + mask_bias
        # attn_bias = None

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


class SharedKVDualQueryAttention(nn.Module):

    def __init__(
            self,
            dim,
            heads=8,
            dropout=0.1
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.norm = nn.LayerNorm(dim)

        # local query

        self.q_local = nn.Linear(
            dim,
            dim,
            bias=False
        )

        # global query

        self.q_global = nn.Linear(
            dim,
            dim,
            bias=False
        )

        # shared kv

        self.kv_proj = nn.Linear(
            dim,
            dim * 2,
            bias=False
        )

        self.out_proj = nn.Linear(
            dim,
            dim,
            bias=False
        )

        self.dropout = nn.Dropout(dropout)

    def _attention(
            self,
            q,
            k,
            v,
            mask=None
    ):
        score = (
                        q @ k.transpose(-2, -1)
                ) / math.sqrt(self.head_dim)

        if mask is not None:
            score = score.masked_fill(
                ~mask,
                -1e9
            )

        attn = torch.softmax(
            score,
            dim=-1
        )

        attn = self.dropout(attn)

        y = attn @ v

        return y

    def forward(
            self,
            x,
            intra_mask,
            inter_mask
    ):
        B, N, D = x.shape

        x0 = x

        x = self.norm(x)

        # ------------------
        # shared KV
        # ------------------

        kv = self.kv_proj(x)

        k, v = kv.chunk(2, dim=-1)

        k = k.view(
            B, N, self.heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, N, self.heads, self.head_dim
        ).transpose(1, 2)

        # ==================================================
        # stage1
        # ==================================================

        q_local = self.q_local(x)

        q_local = q_local.view(
            B, N, self.heads, self.head_dim
        ).transpose(1, 2)

        y_local = self._attention(
            q_local,
            k,
            v,
            intra_mask
        )

        y_local = (
            y_local
            .transpose(1, 2)
            .reshape(B, N, D)
        )

        x_local = x + self.out_proj(y_local)

        # ==================================================
        # stage2
        # ==================================================

        x_local_norm = self.norm(x_local)

        q_global = self.q_global(
            x_local_norm
        )

        q_global = q_global.view(
            B, N, self.heads, self.head_dim
        ).transpose(1, 2)

        y_global = self._attention(
            q_global,
            k,
            v,
            inter_mask
        )

        y_global = (
            y_global
            .transpose(1, 2)
            .reshape(B, N, D)
        )

        out = x_local + self.out_proj(
            y_global
        )

        return out


class SharedKDualQVAttention(nn.Module):

    def __init__(
            self,
            dim,
            heads=8,
            dropout=0.2
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.norm = nn.LayerNorm(dim)

        # -------------------------
        # shared key
        # -------------------------

        self.k_proj = nn.Linear(
            dim,
            dim,
            bias=False
        )

        # -------------------------
        # local qv
        # -------------------------

        self.q_local = nn.Linear(
            dim,
            dim,
            bias=False
        )

        self.v_local = nn.Linear(
            dim,
            dim,
            bias=False
        )

        # -------------------------
        # global qv
        # -------------------------

        self.q_global = nn.Linear(
            dim,
            dim,
            bias=False
        )

        self.v_global = nn.Linear(
            dim,
            dim,
            bias=False
        )

        self.out_local = nn.Linear(
            dim,
            dim,
            bias=False
        )

        self.out_global = nn.Linear(
            dim,
            dim,
            bias=False
        )

        self.dropout = nn.Dropout(dropout)

    def _attn(
            self,
            q,
            k,
            v,
            mask
    ):
        score = (
                        q @ k.transpose(-2, -1)
                ) / math.sqrt(self.head_dim)

        if mask is not None:
            score = score.masked_fill(
                ~mask,
                -1e9
            )

        attn = torch.softmax(
            score,
            dim=-1
        )

        attn = self.dropout(attn)

        return attn @ v

    def _reshape(self, x):
        B, N, D = x.shape

        return (
            x.view(
                B,
                N,
                self.heads,
                self.head_dim
            )
            .transpose(1, 2)
        )

    def forward(
            self,
            x,
            intra_mask,
            inter_mask
    ):
        B, N, D = x.shape

        x0 = x

        x = self.norm(x)

        # =====================
        # shared key
        # =====================

        k = self._reshape(
            self.k_proj(x)
        )

        # =====================
        # stage1
        # =====================

        q_local = self._reshape(
            self.q_local(x)
        )

        v_local = self._reshape(
            self.v_local(x)
        )

        y_local = self._attn(
            q_local,
            k,
            v_local,
            intra_mask
        )

        y_local = (
            y_local
            .transpose(1, 2)
            .reshape(B, N, D)
        )

        x_local = (
                x0 +
                self.out_local(y_local)
        )

        # =====================
        # stage2
        # =====================

        x2 = self.norm(x_local)

        q_global = self._reshape(
            self.q_global(x2)
        )

        v_global = self._reshape(
            self.v_global(x2)
        )

        y_global = self._attn(
            q_global,
            k,
            v_global,
            inter_mask
        )

        y_global = (
            y_global
            .transpose(1, 2)
            .reshape(B, N, D)
        )

        out = (
                x_local +
                self.out_global(y_global)
        )

        return out


class DualQKVAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dropout=0.2
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout_p = dropout
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # ------------------
        # local
        # ------------------
        self.qkv_local = nn.Linear(dim, dim * 3, bias=False)
        self.out_local = nn.Linear(dim, dim, bias=False)

        # ------------------
        # global
        # ------------------
        self.qkv_global = nn.Linear(dim, dim * 3, bias=False)
        self.out_global = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # graph_dist = build_graph_distance_matrix()
        # self.register_buffer("graph_dist",  graph_dist)
        # self.graph_alpha = nn.Parameter(torch.tensor(0.5))
        # max_dist = int(graph_dist.max())
        # self.graph_bias = nn.Embedding(max_dist + 1, heads)

    def _split_qkv(self, qkv):
        B, N, _ = qkv.shape
        qkv = (qkv.view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4))
        return qkv[0], qkv[1], qkv[2]

    def _attn(self, q, k, v, mask):
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            score = score.masked_fill(~mask, -1e9)

        attn = torch.softmax(score, dim=-1)
        attn = self.dropout(attn)
        return attn @ v

    # def _global_attn(self, q, k, v):
    #     score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    #     bias = self.graph_bias(self.graph_dist)
    #     bias = bias.permute(2, 0, 1)
    #     score = score + bias.unsqueeze(0)
    #     attn = torch.softmax(score, dim=-1)
    #     attn = self.dropout(attn)
    #     return attn @ v

    def forward(self, x, intra_mask, inter_mask):
        B, N, D = x.shape

        # ====================
        # local
        # ====================

        x1 = self.norm1(x)
        q, k, v = self._split_qkv(self.qkv_local(x1))

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=intra_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim)
            )

        y = y.transpose(1, 2).reshape(B, N, D)

        x_local = x + self.out_local(y)


        x2 = self.norm2(x_local)
        q, k, v = self._split_qkv(self.qkv_global(x2))

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=inter_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim)
            )
        # y = self._global_attn(q, k, v)

        y = y.transpose(1, 2).reshape(B, N, D)

        out = x_local + self.out_global(y)

        return out


class DualStageTransBlock(nn.Module):
    def __init__(
            self,
            input_dim=512,
            ff_size=1024,
            num_heads=8,
            dropout=0.2,
            chains=ANCHORED_H36M_LIMBS,
            n_chain_heads=2,
            num_joint=16,
            use_ila=True,
            bias_mode='both'
    ):
        super().__init__()

        self.attn = HybridChainRelativeDualQKVAttention(input_dim, num_heads, dropout, chains=chains,
                                                        n_chain_heads=n_chain_heads, num_joints=num_joint,
                                                        use_ila=use_ila, bias_mode=bias_mode)
        # self.attn = DualQKVAttention(input_dim, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_dim)
        )

        self.norm = nn.LayerNorm(input_dim)

        self.register_buffer("intra_mask", build_intra_limb_mask())
        self.register_buffer("inter_mask", build_global_limb_mask())

    def forward(self, x, coord_field=None):
        x = self.attn(x, self.intra_mask, self.inter_mask)
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
        self.register_buffer('intra_limb_mask', build_intra_limb_mask())
        self.register_buffer('inter_limb_mask', build_global_limb_mask())

        # 单注意力层同时处理内部和交叉注意力（与原版SDFM参数量完全相同）
        self.attention = AnatomySelfAttention(input_dim, num_heads, dropout, flash_attention)

        # FFN（与原版完全相同）
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, x):
        x_intra = self.attention(x, self.intra_limb_mask)
        x_inter = self.attention(x_intra, self.inter_limb_mask)
        x = x + x_inter
        x = x + self.ffn(self.norm(x))

        return x


class AdaptiveGCNGlobalAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dropout=0.2
    ):
        super().__init__()
        self.local_gcn = LiteEdgeGCN(dim)
        self.global_attn = AnatomySelfAttention(dim, heads, dropout)

    def forward(self, x):
        x = self.local_gcn(x)
        x = self.global_attn(x)
        return x


class TransBlock(nn.Module):
    def __init__(self, input_dim=512, ff_size=1024, num_heads=8, dropout=0.2,
                 flash_attention=True, mask_type="global"):
        super().__init__()
        self.input_dim = input_dim
        self.ff_size = ff_size
        self.num_heads = num_heads
        # 预计算全局注意力掩码（无参数量，仅初始化一次）
        if mask_type == "intra":
            self.register_buffer('mask', build_intra_limb_mask())
        elif mask_type == "inter":
            self.register_buffer('mask', build_inter_limb_mask())
        else:
            self.register_buffer('mask', build_global_limb_mask())

        # 单注意力层同时处理内部和交叉注意力（与原版SDFM参数量完全相同）
        self.attention = AnatomySelfAttention(input_dim, num_heads, dropout, flash_attention)
        # FFN（与原版完全相同）
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, x):
        x = x + self.attention(x, self.mask)
        x = x + self.ffn(self.norm(x))
        return x


class DualHierarchicalTransBlock(nn.Module):
    def __init__(self, input_dim=512, ff_size=1024, num_heads=8, dropout=0.2,
                 flash_attention=True):
        super().__init__()
        self.input_dim = input_dim
        self.ff_size = ff_size
        self.num_heads = num_heads
        # 预计算全局注意力掩码（无参数量，仅初始化一次）
        self.register_buffer('intra_limb_mask', build_intra_limb_mask())
        self.register_buffer('inter_limb_mask', build_global_limb_mask())
        # 单注意力层同时处理内部和交叉注意力（与原版SDFM参数量完全相同）
        self.attention1 = AnatomySelfAttention(input_dim, num_heads, dropout, flash_attention)
        self.attention2 = AnatomySelfAttention(input_dim, num_heads, dropout, flash_attention)
        # FFN（与原版完全相同）
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, x):
        x_intra = self.attention1(x, self.intra_limb_mask)
        x_inter = self.attention2(x_intra, self.inter_limb_mask)
        x = x + x_inter
        x = x + self.ffn(self.norm(x))
        return x


class GCNTransBlock(nn.Module):
    def __init__(
            self,
            input_dim=512,
            ff_size=1024,
            num_heads=8,
            dropout=0.2
    ):
        super().__init__()

        self.attn = AdaptiveGCNGlobalAttention(input_dim, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_dim)
        )

        self.norm = nn.LayerNorm(input_dim)


    def forward(self, x):
        x = self.attn(x)
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
        use_anchor = getattr(cfg, 'use_anchor', True)
        use_ila = getattr(cfg, 'use_ila', True)
        bias_mode = getattr(cfg, 'bias_mode', 'both')
        if cfg.dataset == 'ap3d' or cfg.dataset == 'ap':
            LIMBS = ANCHORED_H36M_LIMBS if use_anchor else H36M_LIMBS
        elif cfg.dataset == 'wp':
            LIMBS = ANCHORED_WP_LIMBS_23 if use_anchor else WP_LIMBS2

        in_dim = k * 3

        self.x_proj = SimpleResBlock(in_dim, dim, dim, dropout)
        self.cond_proj = SimpleResBlock(in_dim, dim, dim, dropout)

        self.t_embed = TimeEmbedding(dim)
        self.joint_emb = nn.Parameter(torch.randn(1, joints, dim) * 0.02)

        self.cond_inject = nn.ModuleList([AdaptiveCondition(dim) for _ in range(depth)])

        head_list = [0, 0, 0, 2, 2, 2]
        self.blocks = nn.ModuleList(
            # [TransBlock(input_dim=dim, ff_size=dim * 2, num_heads=heads) for _ in range(depth)]
            # [TransBlock(input_dim=dim, ff_size=dim * 2, num_heads=heads, mask_type='inter') for _ in range(2)] +
            [DualStageTransBlock(input_dim=dim, dropout=dropout, num_heads=heads, ff_size=dim * 2, num_joint=joints,
                                 chains=LIMBS, n_chain_heads=head_list[i], use_ila=use_ila, bias_mode=bias_mode) for i in range(depth)]
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

        # cond = self.cond_spectral_mixer(cond)

        x_t = x_t.reshape(B, V, -1)
        cond = cond.reshape(B, V, -1)

        # -------------------------------------------------
        # embeddings
        # -------------------------------------------------

        x = self.x_proj(x_t)
        c = self.cond_proj(cond)
        t_emb = self.t_embed(t).unsqueeze(1)

        # if kwargs['traj_his'] is not None:
        #     traj_his = kwargs['traj_his'].reshape(B, T, V, -1)
        # else:
        #     traj_his = torch.zeros((B, T, V, 3), dtype=x_t.dtype, device=x_t.device)

        # injector1
        # edge_x_bias, edge_c_bias = self.limb_chain_edge_inject(x_t, cond, t)
        # x = x + edge_x_bias
        # c = c + edge_c_bias

        # injector2
        # x = x + self.limb_edge_inject_x(x_t)
        # c = c + self.limb_edge_inject_c(cond)

        x = x + t_emb + self.joint_emb

        features = []
        num_blocks = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            x = self.cond_inject[i](x, c)
            # x = self.cond_inject(x, c)
            # x = blk(x + c + t_emb)
            x = blk(x)

            # if i >= num_blocks - 2:
            #     x = x + self.cross_fuse[i](x, c)

            if i < num_blocks // 2:
                features.append(x)
            else:
                skip = features[num_blocks - i - 1]
                x = x + skip

        """
        features = []
        delta_list = []
        num_blocks = len(self.blocks)

        for i, blk in enumerate(self.blocks):
            x = self.cond_inject[i](x, c)
            x = blk(x)

            if i < num_blocks // 2:
                features.append(x)
            else:
                skip = features[num_blocks - i - 1]
                x = x + skip

            delta_i = self.delta_heads[i](x)
            gate_i = torch.sigmoid(self.delta_gate[i])
            delta_list.append(gate_i * delta_i)

        base = self.out(self.norm(x))
        delta = torch.stack(delta_list, dim=0).sum(dim=0)
        out = base + delta
        """

        x = self.norm(x)
        out = self.out(x)

        # joint_corr, edge_res = self.edge_flow_correction(x, out)
        # out = out + joint_corr

        # v, v0, r1, r2 = self.res_head(x)
        # v = v.reshape(B, V, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)
        # v0 = v0.reshape(B, V, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)
        # r1 = r1.reshape(B, V, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)
        # r2 = r2.reshape(B, V, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)
        # return v, v0, r1, r2

        # refine = self.limb_refine(x)
        # out[:, 2, :] += refine["rfoot"]
        # out[:, 5, :] += refine["lfoot"]
        # out[:, 12, :] += refine["lwrist"]
        # out[:, 15, :] += refine["rwrist"]

        return out.reshape(B, V, T, -1).permute(0, 2, 1, 3).reshape(B, T, -1)

