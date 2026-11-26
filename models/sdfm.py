"""
SDFM baseline: joint-token transformer velocity network on truncated DCT coefficients
(the authors' previous flow-matching predictor; compared against AthleticFlow in the paper).

Module/attribute names are kept identical to the original implementation so that the released
checkpoints load unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = (torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
             .to(device=timesteps.device))
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class StylizationBlock(nn.Module):
    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        emb_out = self.emb_layers(emb)
        if len(emb.shape) == 2:
            emb_out = emb_out.unsqueeze(1)
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, input_dim, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        if input_dim != latent_dim:
            self.linear3 = nn.Linear(input_dim, latent_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, emb=None):
        y = self.dropout(self.linear2((self.activation(self.linear1(x)))))
        if x.shape[-1] != y.shape[-1]:
            x = self.linear3(x)
        if emb is not None:
            y = x + self.proj_out(y, emb)
        else:
            y = x + y
        return self.norm(y)


class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x + y


class JointSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.dropout_p = dropout
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb):
        """
        x: B, V, D (joint tokens), emb: B, V, D or B, D
        """
        B, T, D = x.shape
        H = self.num_head
        C = D // H
        q = self.query(self.norm(x))
        k = self.key(self.norm(x))
        v = self.value(self.norm(x))
        q_ = q.view(B, T, H, C).transpose(1, 2)
        k_ = k.view(B, T, H, C).transpose(1, 2)
        v_ = v.view(B, T, H, C).transpose(1, 2)
        # NOTE: as in the original SDFM implementation, attention dropout is applied in train AND eval mode
        # (the published numbers were obtained this way).
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            y = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=self.dropout_p).transpose(1, 2).reshape(B, T, D)

        if len(x.shape) == len(emb.shape):
            y = x + emb + y
        else:
            y = x + emb.unsqueeze(1) + y
        return y


class SDFMDecoderLayer(nn.Module):
    def __init__(self, latent_dim=32, ffn_dim=256, num_head=4, dropout=0.5):
        super().__init__()
        self.sa_block = JointSelfAttention(latent_dim, num_head, dropout)
        self.ffn = FFN(latent_dim, ffn_dim, dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, emb):
        x = self.norm(self.sa_block(x, emb))
        x = self.norm(self.ffn(x))
        return x


class SDFM(nn.Module):
    def __init__(self,
                 num_frames=15,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.2,
                 joint_num=16):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.time_embed_dim = latent_dim
        self.joint_num = joint_num
        self.joint_embedding = nn.Parameter(torch.randn(1, self.joint_num, latent_dim))

        # input embedding of the K coefficients (x, y, z) of every joint
        self.temporal_embed = ResBlock(num_frames * 3, latent_dim, ff_size, dropout, self.time_embed_dim)

        self.cond_embed = nn.Sequential(nn.Linear(self.num_frames * 3, latent_dim),
                                        nn.SiLU(),
                                        nn.Linear(latent_dim, latent_dim))

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim))

        self.temporal_decoder_blocks = nn.ModuleList(
            [SDFMDecoderLayer(latent_dim=latent_dim, ffn_dim=ff_size, num_head=num_heads, dropout=dropout)
             for _ in range(num_layers)])

        self.out = zero_module(nn.Linear(self.latent_dim, self.num_frames * 3))

    def forward(self, x, timesteps, mod=None, **kwargs):
        """
        x: B, K, V*3 noisy coefficients; mod: B, K, V*3 condition (None: unconditional)
        """
        B, T, V3 = x.shape
        V = V3 // 3
        x = x.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)).unsqueeze(1)

        if mod is not None:
            mod = mod.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)
            emb = emb + self.cond_embed(mod)
            emb = emb[:, -self.joint_num:, :]

        h = self.temporal_embed(x)[:, -self.joint_num:, :]
        h = h + self.joint_embedding

        # U-shaped skip connections
        prelist = []
        for i, module in enumerate(self.temporal_decoder_blocks):
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, emb)
            elif i == (self.num_layers // 2) and self.num_layers % 2 == 1:
                h = module(h, emb)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                h = module(h, emb)
                h += prelist[-1]
                prelist.pop()

        output = self.out(h).reshape(B, V, T, 3).permute(0, 2, 1, 3).reshape(B, T, -1)

        return output
