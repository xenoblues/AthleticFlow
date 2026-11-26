import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# -----------------------------------------------------------------------------
# Kinematic chains (joint indices after the root joint has been removed)
# -----------------------------------------------------------------------------

# H36M-17 layout (AthletePose3D / AthleticsPose), 16 non-root joints
H36M_LIMBS = [
    [0, 1, 2],  # right leg
    [3, 4, 5],  # left leg
    [6, 7, 8, 9],  # torso
    [10, 11, 12],  # left arm
    [13, 14, 15],  # right arm
]

# torso-anchored chains: legs start at the spine, arms at the thorax
ANCHORED_H36M_LIMBS = [[6, 0, 1, 2], [6, 3, 4, 5], [6, 7, 8, 9], [7, 10, 11, 12], [7, 13, 14, 15]]

# SMPL-24 layout (WorldPose), 23 non-root joints
WP_LIMBS = [
    [0, 3, 6, 9],  # left leg
    [1, 4, 7, 10],  # right leg
    [2, 5, 8, 11, 14],  # torso
    [12, 15, 17, 19, 21],  # left arm
    [13, 16, 18, 20, 22],  # right arm
]

ANCHORED_WP_LIMBS = [
    [2, 0, 3, 6, 9],
    [2, 1, 4, 7, 10],
    [2, 5, 8, 11, 14],
    [8, 12, 15, 17, 19, 21],
    [8, 13, 16, 18, 20, 22],
]


def get_limbs(dataset, use_anchor=True):
    if dataset in ('ap3d', 'ap'):
        return ANCHORED_H36M_LIMBS if use_anchor else H36M_LIMBS
    elif dataset == 'wp':
        return ANCHORED_WP_LIMBS if use_anchor else WP_LIMBS
    raise ValueError(f'no kinematic chains defined for dataset: {dataset}')


def build_intra_limb_mask(limbs, num_joints):
    """Internal-limb attention mask: a joint only attends to joints of its own chain(s)."""
    mask = torch.zeros(num_joints, num_joints, dtype=torch.bool)
    for limb in limbs:
        for i in limb:
            for j in limb:
                mask[i, j] = True
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, J, J)


def build_global_limb_mask(num_joints):
    return torch.ones(1, 1, num_joints, num_joints, dtype=torch.bool)


def build_chain_relative_index(num_joints, chains):
    """Signed position offset along a common chain for every joint pair (shortest one if several)."""
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


def build_chain_edges(chains):
    edges = []
    seen = set()

    for chain in chains:
        for p, c in zip(chain[:-1], chain[1:]):
            if (p, c) not in seen:
                edges.append((p, c))
                seen.add((p, c))

    return edges


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

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
        raise ValueError(f"Unsupported activation function: {activation_name}")


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
    """
    Motion-conditioned kinematic chain-relative attention bias.

    Supported bias modes:
        - "both":    static bias + dynamic bias (AthleticFlow)
        - "static":  static bias only
        - "dynamic": dynamic bias only
        - "none":    no chain-relative bias
    """

    _MODE_ALIASES = {
        "both": "both",
        "full": "both",
        "static+dynamic": "both",
        "static": "static",
        "static_only": "static",
        "dynamic": "dynamic",
        "dynamic_only": "dynamic",
        "none": "none",
        "off": "none",
        "no_bias": "none",
    }

    def __init__(
        self,
        dim,
        heads=8,
        num_joints=16,
        chains=ANCHORED_H36M_LIMBS,
        n_chain_heads=2,
        hidden=None,
        dropout=0.2,
        init_gate=-4.0,
        bias_mode="both",
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

        rel_index, valid, max_rel = build_chain_relative_index(num_joints, chains)
        self.max_rel = max_rel
        self.num_rel = 2 * max_rel + 1

        self.register_buffer("rel_index", rel_index.long())
        self.register_buffer("chain_valid", valid.bool())

        edges = build_chain_edges(chains)
        if len(edges) == 0:
            raise ValueError("No valid kinematic-chain edges were constructed.")

        self.register_buffer("parent_ids", torch.tensor([parent for parent, child in edges], dtype=torch.long))
        self.register_buffer("child_ids", torch.tensor([child for parent, child in edges], dtype=torch.long))

        # Only the last n_chain_heads heads receive chain-relative bias.
        head_mask = torch.zeros(heads, dtype=torch.bool)
        if n_chain_heads > 0:
            head_mask[-n_chain_heads:] = True
        self.register_buffer("head_mask", head_mask)

        hidden = hidden or dim

        # Motion-independent (static) relative-position bias.
        self.static_bias = nn.Parameter(torch.zeros(heads, self.num_rel))

        # Motion-conditioned (dynamic) relative-position bias.
        self.dynamic_mlp = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, heads * self.num_rel),
        )

        # Controls the magnitude of the dynamic component.
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

        nn.init.zeros_(self.dynamic_mlp[-1].weight)
        nn.init.zeros_(self.dynamic_mlp[-1].bias)

    @classmethod
    def _canonicalize_mode(cls, mode: str) -> str:
        if not isinstance(mode, str):
            raise TypeError(f"bias_mode must be a string, but received {type(mode)}.")

        normalized = mode.lower().strip()
        if normalized not in cls._MODE_ALIASES:
            supported = sorted(set(cls._MODE_ALIASES.values()))
            raise ValueError(f"Unsupported bias_mode '{mode}'. Supported modes are {supported}.")

        return cls._MODE_ALIASES[normalized]

    def _compute_motion_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, J, D] -> motion descriptor [B, 4D]"""
        global_feat = x.mean(dim=1)
        parent_feat = x[:, self.parent_ids].mean(dim=1)
        child_feat = x[:, self.child_ids].mean(dim=1)
        edge_feat = (x[:, self.child_ids] - x[:, self.parent_ids]).mean(dim=1)
        return torch.cat([global_feat, parent_feat, child_feat, edge_feat], dim=-1)

    def _compute_dynamic_bias(self, x: torch.Tensor) -> torch.Tensor:
        """Motion-conditioned bias table [B, H, 2R+1]."""
        batch_size = x.shape[0]
        motion_feat = self._compute_motion_descriptor(x)
        dynamic_bias = self.dynamic_mlp(motion_feat).view(batch_size, self.heads, self.num_rel)
        return torch.sigmoid(self.gate) * dynamic_bias

    def forward(self, x: torch.Tensor, bias_mode: Optional[str] = None) -> torch.Tensor:
        """
        x: joint features [B, J, D]
        returns the pairwise attention bias [B, H, J, J]
        """
        batch_size, num_joints, feature_dim = x.shape
        if num_joints != self.num_joints or feature_dim != self.dim:
            raise ValueError(f"Expected x of shape [B, {self.num_joints}, {self.dim}], got {tuple(x.shape)}.")

        mode = self.bias_mode if bias_mode is None else self._canonicalize_mode(bias_mode)

        if mode == "none" or self.n_chain_heads == 0:
            return x.new_zeros(batch_size, self.heads, num_joints, num_joints)

        if mode == "static":
            bias = self.static_bias.unsqueeze(0).expand(batch_size, -1, -1)
        elif mode == "dynamic":
            bias = self._compute_dynamic_bias(x)
        else:  # both
            static_bias = self.static_bias.unsqueeze(0).expand(batch_size, -1, -1)
            bias = static_bias + self._compute_dynamic_bias(x)

        # apply the bias only to the selected attention heads
        bias = bias * self.head_mask[None, :, None].to(dtype=bias.dtype)

        # relative-offset bias tables -> pairwise bias matrices
        rel_flat = self.rel_index.reshape(-1)
        pair_bias = bias[:, :, rel_flat].reshape(batch_size, self.heads, num_joints, num_joints)

        # pairs that do not share a chain receive no chain-relative bias
        if self.mask_invalid_pairs:
            pair_bias = pair_bias.masked_fill(~self.chain_valid[None, None, :, :], 0.0)

        return pair_bias

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, heads={self.heads}, num_joints={self.num_joints}, "
                f"n_chain_heads={self.n_chain_heads}, bias_mode='{self.bias_mode}'")


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
    """Internal-limb attention (ILA) with chain-relative bias, followed by global joint attention."""

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

    def _split_qkv(self, qkv):
        B, N, _ = qkv.shape
        qkv = qkv.view(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _local_attn(self, q, k, v, x_for_bias, intra_mask):
        B, H, N, D = q.shape

        score = (q @ k.transpose(-2, -1)) / math.sqrt(D)

        pair_bias = self.motion_chain_bias(x_for_bias).to(device=score.device, dtype=score.dtype)
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

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_dim)
        )

        self.norm = nn.LayerNorm(input_dim)

        self.register_buffer("intra_mask", build_intra_limb_mask(chains, num_joint))
        self.register_buffer("inter_mask", build_global_limb_mask(num_joint))

    def forward(self, x):
        x = self.attn(x, self.intra_mask, self.inter_mask)
        x = x + self.ffn(self.norm(x))
        return x


# -----------------------------------------------------------------------------
# AthleticFlow velocity network
# -----------------------------------------------------------------------------

class AthleticFlow(nn.Module):
    """
    Joint-token velocity network of AthleticFlow.

    Each joint is a token holding its K temporal coefficients (DCT by default) of x, y and z.
    Ablation switches (read from cfg):
        use_ila        internal-limb attention (w/o ILA: global attention only, hence no chain bias)
        use_anchor     torso-anchored kinematic chains
        bias_mode      chain-relative bias: 'both' | 'static' | 'dynamic' | 'none'
        n_chain_heads  heads with chain-relative bias in the second half of the blocks (first half: none)
    """

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
        use_anchor = getattr(cfg, 'use_anchor', True)
        use_ila = getattr(cfg, 'use_ila', True)
        bias_mode = getattr(cfg, 'bias_mode', 'both')
        n_chain_heads = getattr(cfg, 'n_chain_heads', 2)
        limbs = get_limbs(cfg.dataset, use_anchor)

        in_dim = k * 3

        self.x_proj = SimpleResBlock(in_dim, dim, dim, dropout)
        self.cond_proj = SimpleResBlock(in_dim, dim, dim, dropout)

        self.t_embed = TimeEmbedding(dim)
        self.joint_emb = nn.Parameter(torch.randn(1, joints, dim) * 0.02)

        self.cond_inject = nn.ModuleList([AdaptiveCondition(dim) for _ in range(depth)])

        # chain-relative bias only in the second half of the blocks (default [0, 0, 0, 2, 2, 2])
        head_list = [0] * (depth // 2) + [n_chain_heads] * (depth - depth // 2)
        self.blocks = nn.ModuleList(
            [DualStageTransBlock(input_dim=dim, dropout=dropout, num_heads=heads, ff_size=dim * 2, num_joint=joints,
                                 chains=limbs, n_chain_heads=head_list[i], use_ila=use_ila, bias_mode=bias_mode)
             for i in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, in_dim)

    def forward(self, x_t, t, cond=None, **kwargs):
        # x_t:  [B, K, V*3] noisy coefficients
        # cond: [B, K, V*3] coefficients of the padded observation (None: unconditional)
        B, T, V3 = x_t.shape
        V = V3 // 3

        x_t = x_t.reshape(B, T, V, 3).permute(0, 2, 1, 3)

        if cond is None:
            cond = torch.zeros_like(x_t)
        else:
            cond = cond.reshape(B, T, V, 3).permute(0, 2, 1, 3)

        x_t = x_t.reshape(B, V, -1)
        cond = cond.reshape(B, V, -1)

        x = self.x_proj(x_t)
        c = self.cond_proj(cond)
        t_emb = self.t_embed(t).unsqueeze(1)

        x = x + t_emb + self.joint_emb

        # U-shaped long skip connections between the first and second half of the blocks
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
