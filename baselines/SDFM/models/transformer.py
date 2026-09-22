import time

import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np

import math

from torch.nn.attention import sdpa_kernel, SDPBackend

from utils import *


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        '''
        平均更新Transformer模型
        '''
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


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


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


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
        # 时间嵌入维度扩大一倍
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
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb)
        if len(emb.shape) == 2:
            emb_out = emb_out.unsqueeze(1)

        # 分块 scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift  # 意义？
        h = self.out_layers(h)
        return h


class FFN_Sty(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim, out_dim=None, **kwargs):
        super().__init__()
        self.b_syt_block = kwargs['stylization_block']
        if out_dim is None:
            self.out_dim = latent_dim
        else:
            self.out_dim = out_dim
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, self.out_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if self.b_syt_block:
            self.proj_out = StylizationBlock(self.out_dim, time_embed_dim, dropout)
        if self.out_dim != latent_dim:
            self.linear3 = nn.Linear(latent_dim, self.out_dim)

    def forward(self, x, emb=None):
        '''
        :param x: B, T, V, latent_dim
        :param emb: B, time_embed_dim
        :return:
        '''
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        if x.shape[-1] != y.shape[-1]:
            x = self.linear3(x)
        if emb is not None and self.b_syt_block:
            y = x + self.proj_out(y, emb)
        else:
            y = x + y
        return y


class TemporalSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, time_embed_dim, cross_attention=False, stylization_block=False, flash_attention=False):
        super().__init__()
        self.num_head = num_head
        self.dropout_p = dropout
        self.cross_attention = cross_attention
        self.stylization_block = stylization_block
        self.flash_attention = flash_attention
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        if cross_attention:
            self.key_mod = nn.Linear(latent_dim, latent_dim, bias=False)
            self.value_mod = nn.Linear(latent_dim, latent_dim, bias=False)
        if stylization_block:
            self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb=None, mod_emb=None):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        C = D // H
        q = self.query(self.norm(x))    # B, T, D
        k = self.key(self.norm(x))
        v = self.value(self.norm(x))
        if not self.flash_attention:
            # B, T, H, C
            q_ = q.unsqueeze(2).view(B, T, H, C)
            k_ = k.unsqueeze(1).view(B, T, H, C)
            # B, T, T, H
            attention = torch.einsum('bnhd,bmhd->bnmh', q_, k_) / math.sqrt(C)
            # generate mask
            # subsequent_mask = torch.triu(torch.ones((T, T), device=query.device, dtype=torch.float32), diagonal=1)
            # subsequent_mask = subsequent_mask.unsqueeze(0).expand(B, -1, -1).gt(0.0)  # gt大于某个值
            # mask = subsequent_mask.repeat(H, 1, 1).contiguous().view(B, H, T, T).permute(0, 2, 3, 1)
            # attention = attention.masked_fill(mask, -np.inf)

            # weight = self.dropout(F.softmax(attention, dim=2))
            weight = F.softmax(attention, dim=2)
            v_ = v.view(B, T, H, -1)
            y_s = torch.einsum('bnmh,bmhd->bnhd', weight, v_).reshape(B, T, D)
        else:
            # (B, T, D) -> (B, H, T, C)
            q_ = q.view(B, T, H, C).transpose(1, 2)
            k_ = k.view(B, T, H, C).transpose(1, 2)
            v_ = v.view(B, T, H, C).transpose(1, 2)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                y_s = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=self.dropout_p).transpose(1, 2).reshape(B, T, D)

        # cross attention
        if self.cross_attention and mod_emb is not None:
            k_mod = self.key_mod(self.norm(mod_emb))
            v_mod = self.value_mod(self.norm(mod_emb))
            if not self.flash_attention:
                q_ = q.view(B, T, H, C)
                k_mod_ = k_mod.view(B, T, H, C)
                v_mod_ = v_mod.view(B, T, H, C)
                cross_attention = torch.einsum('bnhd,bmhd->bnmh', q_, k_mod_) / math.sqrt(C)
                cross_weight = self.dropout(F.softmax(cross_attention, dim=2))
                y_c = torch.einsum('bnmh,bmhd->bnhd', cross_weight, v_mod_).reshape(B, T, D)
            else:
                q_ = q.view(B, T, H, C).transpose(1, 2)
                k_mod_ = k_mod.view(B, T, H, C).transpose(1, 2)
                v_mod_ = v_mod.view(B, T, H, C).transpose(1, 2)
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    y_c = F.scaled_dot_product_attention(q_, k_mod_, v_mod_, dropout_p=self.dropout_p).transpose(1, 2).reshape(B, T, D)
            y = y_s + y_c
        else:
            y = y_s

        if emb is not None:
            if self.stylization_block:
                y = x + self.proj_out(y, emb)
            else:
                if len(x.shape) == len(emb.shape):
                    y = x + emb + y
                elif len(emb.shape) == 2:
                    y = x + emb.unsqueeze(1) + y
        else:
            y = x + y
        return y


class TemporalDiffusionTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 out_dim=None,
                 cross_attention=False,
                 **kwargs
                 ):
        super().__init__()
        self.sa_block = TemporalSelfAttention(latent_dim, num_head, dropout, time_embed_dim, cross_attention, kwargs['stylization_block'], kwargs['flash_attention'])
        self.ffn = FFN_Sty(latent_dim, ffn_dim, dropout, time_embed_dim, out_dim, stylization_block=kwargs['stylization_block'])

    def forward(self, x, emb, mod_emb=None):
        x = self.sa_block(x, emb, mod_emb)
        x = self.ffn(x, emb)
        return x


class TemporalDiffusionLinearDecoderLayer(nn.Module):
    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 out_dim=None
                 ):
        super().__init__()
        self.ffn1 = FFN_Sty(latent_dim, ffn_dim, dropout, time_embed_dim, out_dim)
        self.ffn2 = FFN_Sty(latent_dim, ffn_dim, dropout, time_embed_dim, out_dim)

    def forward(self, x, emb):
        x = self.ffn1(x, emb)
        x = self.ffn2(x, emb)
        return x


class MotionTransformerOrigin(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 cross_attention=False,
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.cross_attention = cross_attention
        self.sequence_embedding = nn.Parameter(torch.randn(self.num_frames, latent_dim))

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                TemporalDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                    cross_attention=self.cross_attention,
                    stylization_block=kargs['stylization_block'],
                    flash_attention=kargs['flash_attention']
                )
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))

        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1))
            emb = emb + mod_proj
            emb = emb

        # x: B, T, latent_dim
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        if self.cross_attention and mod is not None:
            mod_emb = self.joint_embed(mod)
            mod_emb = mod_emb + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        else:
            mod_emb = None

        prelist = []
        for i, module in enumerate(self.temporal_decoder_blocks):
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, emb, mod_emb)
            elif i == (self.num_layers // 2) and self.num_layers % 2 == 1:
                h = module(h, emb, mod_emb)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                h = module(h, emb, mod_emb)
                h += prelist[-1]
                prelist.pop()

        output = self.out(h).view(B, T, -1).contiguous()

        return output


if __name__ == '__main__':
    model = MotionTransformerOrigin(
            input_feats=3 * 10,  # 3 means x, y, z
            num_frames=10,
            num_layers=4,
            num_heads=8,
            latent_dim=1024,
            dropout=0.2,
            cross_attention=True,
            stylization_block=True,
            flash_attention=False
            ).to("cuda").eval()
    torch.initial_seed()
    x = torch.randn(1, 10, 30).to("cuda")
    x_m = torch.randn(1, 10, 30).to("cuda")
    t = torch.rand(x.shape[0]).to("cuda")
    # y1 = model(x, t, x_m, flash_attention=True)
    # y2 = model(x, t, x_m, flash_attention=False)
    # print(y1[0, 0, :2], y2[0, 0, :2])
    B = 1
    T = 10
    H = 4
    C = 32
    D = int(H * C)
    q = torch.randn(B, T, D).to("cuda")
    k = torch.randn(B, T, D).to("cuda")
    v = torch.randn(B, T, D).to("cuda")
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        q_ = q.view(B, T, H, C).transpose(1, 2)
        k_ = k.view(B, T, H, C).transpose(1, 2)
        v_ = v.view(B, T, H, C).transpose(1, 2)
        y_s1 = F.scaled_dot_product_attention(q_, k_, v_).reshape(B, T, D)

    q__ = q.view(B, T, H, -1)
    k__ = k.view(B, T, H, -1)
    attention = torch.einsum('bnhd,bmhd->bnmh', q__, k__) / math.sqrt(C)
    weight = F.softmax(attention, dim=2)
    v_ = v.view(B, T, H, -1)
    y_s2 = torch.einsum('bnmh,bmhd->bnhd', weight, v_).reshape(B, T, D)
    print(y_s2.shape)
    print(y_s1[0, 0, :5], y_s2[0, 0, :5])


