import torch
import torch.nn.functional as F
from numpy.ma.core import squeeze
from torch import layer_norm, nn
import numpy as np
import math
from functools import partial

from utils import *
from models.transformer import *
from models.transfusion import TemporalDiffusionTransformerDecoderLayer


class SpectralTemporalDualLoss(nn.Module):
    def __init__(
            self,
            vel_weight=1.0,
            acc_weight=0.5
    ):
        super().__init__()

        self.vel_weight = vel_weight
        self.acc_weight = acc_weight

    def forward(self, pred_x, gt_x):
        pred_vel = pred_x[:, 1:] - pred_x[:, :-1]
        gt_vel = gt_x[:, 1:] - gt_x[:, :-1]
        vel_loss = F.mse_loss(pred_vel, gt_vel)

        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        acc_loss = F.mse_loss(pred_acc, gt_acc)

        total_loss = self.vel_weight * vel_loss + self.acc_weight * acc_loss
        return total_loss


class SpectralDynamicsLoss(nn.Module):

    def __init__(
            self,
            vel_weight=1.e-3,
            jerk_weight=1.e-4
    ):
        super().__init__()

        self.vel_weight = vel_weight
        self.jerk_weight = jerk_weight

    def velocity_consistency_loss(self, pred_x):
        vel = (
                pred_x[:, 1:]
                - pred_x[:, :-1]
        )

        acc = (
                vel[:, 1:]
                - vel[:, :-1]
        )

        return acc.pow(2).mean()

    def jerk_loss(self, pred):
        vel = pred[:, 1:] - pred[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        jerk = acc[:, 1:] - acc[:, :-1]

        return (jerk ** 2).mean()

    def forward(self, pred, gt):
        vel_loss = self.velocity_consistency_loss(pred)
        jerk_loss = self.jerk_loss(pred)

        total = self.vel_weight * vel_loss + self.jerk_weight * jerk_loss
        return total


class StylizationBlock4D(nn.Module):
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
        h: B, T, V, D
        emb: B, D
        """
        # B, 1, 1, 2D
        emb_out = self.emb_layers(emb)
        # 分块 scale: B, 1, 1, D / shift: B, 1, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) * (1 + scale) + shift  # 意义？
        h = self.out_layers(h)
        return h


class FFN_Woemb(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x: B, T, V, latent_dim
        :return:
        '''
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + y
        return y


class FFN4D(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock4D(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        '''
        :param x: B, T, V, latent_dim
        :param emb: B, time_embed_dim
        :return:
        '''
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y


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
        '''
        :param x: B, V, D
        :param emb: B, time_embed_dim
        :return:
        '''
        y = self.dropout(self.linear2((self.activation(self.linear1(x)))))
        if x.shape[-1] != y.shape[-1]:
            x = self.linear3(x)
        if emb is not None:
            y = x + self.proj_out(y, emb)
        else:
            y = x + y
        return self.norm(y)


class GCN(nn.Module):

    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Conv2d(in_dim,
                              self.latent_dim,
                              kernel_size=(1, 1),
                              padding=(0, 0),
                              stride=(1, 1))

    def forward(self, x, A):
        B, T, V, D = x.shape
        x = x.permute(0, 3, 1, 2)  # B, D, T, V
        y = self.conv(x)
        y = torch.matmul(y, A).permute(0, 2, 3, 1)  # B, T, V, D
        return y


class GCN2D(nn.Module):
    def __init__(self, in_dim, latent_dim, adj_mat):
        super().__init__()
        self.latent_dim = latent_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_dim, latent_dim))
        self.bias = torch.nn.Parameter(torch.FloatTensor(latent_dim))
        self.adj_mat = nn.Parameter(adj_mat.clone().unsqueeze(0))
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        B, V, D = x.shape
        x = torch.matmul(x, self.weight)
        output = torch.matmul(self.adj_mat, x)
        y = output + self.bias
        return y


class GFFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim, out_dim=None, adj_mat=None):
        super().__init__()
        if out_dim is None:
            self.out_dim = latent_dim
        else:
            self.out_dim = out_dim
        self.linear1 = GCN2D(latent_dim, ffn_dim, adj_mat)
        self.linear2 = GCN2D(ffn_dim, ffn_dim, adj_mat)
        self.linear4 = zero_module(nn.Linear(ffn_dim, self.out_dim))
        self.norm = nn.LayerNorm(ffn_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(self.out_dim, time_embed_dim, dropout)
        if self.out_dim != latent_dim:
            self.linear3 = nn.Linear(latent_dim, self.out_dim)

    def forward(self, x, emb=None):
        '''
        :param x: B, T, V, latent_dim
        :param emb: B, time_embed_dim
        :return:
        '''
        y = self.activation(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        y = self.linear4(self.norm(y))
        if x.shape[-1] != y.shape[-1]:
            x = self.linear3(x)
        if emb is not None:
            y = x + self.proj_out(y, emb)
        else:
            y = x + y
        return y


class TemporalGraphConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, vertices_num):
        super(TemporalGraphConv, self).__init__()

        self.hidden_dim = hidden_dim
        self.w = nn.Linear(vertices_num, vertices_num)
        self.conv = nn.Conv2d(
            in_dim,
            self.hidden_dim,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1))

    def forward(self, x, A):
        B, T, V, D = x.shape
        # B, D, V, T
        y = self.conv(x.permute(0, 3, 2, 1))
        y = torch.matmul(y, A)
        y = self.w(y).permute(0, 3, 2, 1) + x  # B, T, V, D

        return y


class SpatialGraphConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, vertices_num):
        super(SpatialGraphConv, self).__init__()

        self.hidden_dim = hidden_dim
        self.w = nn.Linear(vertices_num, vertices_num)
        self.conv = nn.Conv2d(
            in_dim,
            self.hidden_dim,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1))

    def forward(self, x, A):
        B, T, V, D = x.shape
        # B, D, T, V
        y = self.conv(x.permute(0, 3, 1, 2))
        y = torch.matmul(y, A)
        y = self.w(y).permute(0, 2, 3, 1) + x  # B, T, V, D

        return y


class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, vertices_num, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.B = nn.Parameter(torch.zeros((vertices_num, vertices_num)))
        self.conv1 = nn.Conv1d(
            in_dim,
            self.hidden_dim,
            kernel_size=1)
        if in_dim == hidden_dim:
            self.conv2 = lambda x: x
        else:
            self.conv2 = nn.Sequential(
                nn.Conv1d(in_dim, self.hidden_dim, 1),
                nn.BatchNorm1d(self.hidden_dim)
            )
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.l1 = nn.Linear(in_dim, self.hidden_dim)
        self.l2 = nn.Linear(in_dim, self.hidden_dim)
        self.alpha1 = nn.Parameter(torch.tensor(0.1))
        self.alpha2 = nn.Parameter(torch.tensor(0.1))
        self.sm = nn.Softmax(-2)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(dropout)

    def forward(self, x, A):
        # B, D, V
        B, D, V = x.shape
        y = self.conv1(x)
        a = self.l1(x.permute(0, 2, 1))
        b = self.l2(x.permute(0, 2, 1)).permute(0, 2, 1)
        C = self.sm(a @ b)
        y = torch.matmul(y, A + self.alpha1 * self.B + self.alpha2 * C)
        y = self.relu(self.bn(y) + self.conv2(x))
        return y


class GroupGCNBlock(nn.Module):

    def __init__(self, in_dim, latent_dim, group_num, adj_matrices):
        super().__init__()
        self.latent_dim = latent_dim
        self.group_num = group_num
        self.gcn_groups = nn.ModuleList()
        self.adj_matrices = adj_matrices
        for i in range(group_num):
            self.gcn_groups.append(GCN(in_dim, latent_dim))

    def forward(self, x):
        B, T, V, D = x.shape
        y = self.gcn_groups[0](x, self.adj_matrices[0])
        for i in range(1, self.group_num):
            y += self.gcn_groups[i](x, self.adj_matrices[i])
        return y + x


class SpatialSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: B, T, V, D
        """
        B, T, V, D = x.shape
        H = self.num_head
        # BT, V, D -> BT, V, H, D//H
        query = self.query(self.norm(x)).contiguous().view(B * T, V, H, D // H).permute(0, 2, 1, 3)
        # BT, V, D -> BT, V, H, D//H
        key = self.key(self.norm(x)).contiguous().view(B * T, V, H, D // H).permute(0, 2, 1, 3)

        # BT, H, V, V
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(D // H)
        attention = self.dropout(F.softmax(attention, dim=-1))
        value = self.value(self.norm(x)).contiguous().view(B * T, V, H, D // H).permute(0, 2, 1, 3)
        # BT, H, V, D//H
        y = attention @ value
        y = x + y.transpose(1, 2).contiguous().view(B, T, V, D)
        return y


class GraphSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb, A):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)

        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        # generate mask
        # subsequent_mask = torch.triu(torch.ones((T, T), device=query.device, dtype=torch.float32), diagonal=1)
        # subsequent_mask = subsequent_mask.unsqueeze(0).expand(B, -1, -1).gt(0.0)  # gt大于某个值
        # mask = subsequent_mask.repeat(H, 1, 1).contiguous().view(B, H, T, T).permute(0, 2, 3, 1)
        # attention = attention.masked_fill(mask, -np.inf)

        # 注意力系数逐元素乘邻接矩阵
        attention = attention * A.unsqueeze(-1)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        if emb is not None:
            y = x + self.proj_out(y, emb)
        else:
            y = x + y
        return y


class MaskedTemporalSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock4D(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        """
        x: B, T, V, D
        """
        B, T, V, D = x.shape
        x = x.transpose(1, 2).contiguous().view(B * V, T, D)
        H = self.num_head
        # BV, T, D -> BV, T, H, D//H
        query = self.query(self.norm(x)).contiguous().view(B * V, T, H, D // H).permute(0, 2, 1, 3)
        # BV, T, D -> BV, T, H, D//H
        key = self.key(self.norm(x)).contiguous().view(B * V, T, H, D // H).permute(0, 2, 1, 3)

        # generate mask
        subsequent_mask = torch.triu(
            torch.ones((T, T), device=query.device, dtype=torch.float32), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(B * V, -1, -1).gt(0.0)  # gt():大于某个值
        mask = subsequent_mask.repeat(H, 1, 1).contiguous().view(B * V, H, T, T)

        # BV, H, T, T
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(D // H)
        attention = F.softmax(attention.masked_fill(mask, -np.inf), dim=-1)
        # attention = self.dropout(attention)
        value = self.value(self.norm(x)).contiguous().view(B * V, T, H, D // H).permute(0, 2, 1, 3)
        # BV, H, T, D//H
        y = attention @ value
        y = y.transpose(1, 2).contiguous().view(B * V, T, D).view(B, V, T, D).transpose(1, 2)
        y = self.proj_out(y, emb)
        # print(x.shape, y.shape)
        y = x.view(B, V, T, D).transpose(2, 1) + y
        return y


class SpatialDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 gcn_group_num=1,
                 adj_matrices=None
                 ):
        super().__init__()
        self.sa_block = SpatialSelfAttention(
            latent_dim, num_head, dropout)
        self.gcn = GroupGCNBlock(latent_dim, latent_dim, gcn_group_num, adj_matrices)

    def forward(self, x):
        '''
        :param x: B, T, V, D
        :param emb: B, D
        :return:
        '''
        x = self.sa_block(x)
        x = self.gcn(x)
        return x


class TemporalDiffusioniTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 out_dim=None,
                 **kwargs
                 ):
        super().__init__()
        self.sa_block = TemporalSelfAttention(latent_dim, num_head, dropout, time_embed_dim,
                                              flash_attention=kwargs['flash_attention'])
        self.ffn = FFN_Sty(latent_dim, ffn_dim, dropout, time_embed_dim, out_dim,
                           stylization_block=kwargs['stylization_block'])
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, emb=None):
        x = self.norm(self.sa_block(x, emb))
        x = self.norm(self.ffn(x, emb))
        return x


class GraphDiffusioniTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 out_dim=None,
                 adj_mat=None
                 ):
        super().__init__()
        self.sa_block = TemporalSelfAttention(latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = GFFN(latent_dim, ffn_dim, dropout, time_embed_dim, out_dim, adj_mat)
        self.norm = nn.LayerNorm(latent_dim)
        # self.adj_mat = nn.Parameter(adj_mat.unsqueeze(0).clone())

    def forward(self, x, emb=None):
        x = self.norm(self.sa_block(x, emb))
        x = self.norm(self.ffn(x, emb))
        return x

    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 out_dim=None,
                 adj_mat=None
                 ):
        super().__init__()
        self.sa_block = TemporalSelfAttention(latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = GFFN(latent_dim, ffn_dim, dropout, time_embed_dim, out_dim, adj_mat)
        self.norm = nn.LayerNorm(latent_dim)
        # self.adj_mat = nn.Parameter(adj_mat.unsqueeze(0).clone())

    def forward(self, x, emb):
        x = self.norm(self.sa_block(x, emb))
        x = self.norm(self.ffn(x, emb))
        return x


class MaskedTemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 ):
        super().__init__()
        self.sa_block = MaskedTemporalSelfAttention(latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN4D(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, emb):
        '''
        :param x: B, T, V, D
        :param emb: B, D
        :return:
        '''
        x = self.sa_block(x, emb)
        x = self.ffn(x, emb)
        return x


class SpatialTemporalCrossAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim * 2)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock4D(latent_dim, time_embed_dim, dropout)

    def forward(self, x_s, x_t, emb):
        """
        x_s: B, T, V, D
        x_t: B, T, V, D
        """
        B, T, V, D = x_s.shape
        x_s = x_s.contiguous().view(B, T * V, D)
        x_t = x_t.contiguous().view(B, T * V, D)
        N = x_s.shape[1]  # T * V
        H = self.num_head
        # B, N, 1, D
        query = self.query(self.norm(x_s)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.norm(x_t)).unsqueeze(1)
        query = query.view(B, N, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        # value = self.value(self.norm2(torch.cat((x_s, x_t), dim=-1))).view(B, N, H, -1)
        value = self.value(self.norm(x_s + x_t)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, N, D).view(B, T, V, D)
        y = (x_s + x_t).view(B, T, V, D) + self.proj_out(y, emb)
        return y


class TemporalCrossAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xf):
        """
        x: B, T, D
        xf: B, T, D
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        # print(x.shape, xf.shape)
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.text_norm(xf)).unsqueeze(2)
        # print(query.shape, key.shape)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + y
        return y


class STTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 sA=None,
                 tA=None
                 ):
        super().__init__()
        self.s_gcn = SpatialGraphConv(latent_dim, latent_dim, sA.shape[0])
        self.t_gcn = TemporalGraphConv(latent_dim, latent_dim, tA.shape[0])
        # self.st_attn = SpatialTemporalCrossAttention(latent_dim, num_head, dropout, time_embed_dim)
        self.t_attn = TemporalSelfAttention(latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN_Sty(latent_dim, ffn_dim, dropout, time_embed_dim)
        self.sA = sA
        self.tA = tA

    def forward(self, x, emb):
        # y_s = self.s_gcn(x, self.sA)
        # y_t = self.t_gcn(y_s, self.tA)
        # y = self.t_attn(y_s, y_t, emb)
        y_t = self.t_attn(x, emb)
        y = self.ffn(y_t, emb)
        return y


class TemporalDiffusionCrossTransformerLayer(nn.Module):
    def __init__(self,
                 latent_dim=32,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 ):
        super().__init__()
        self.ca_block = TemporalCrossAttention(
            latent_dim, num_head, dropout)
        self.ffn1 = FFN_Woemb(latent_dim, ffn_dim, dropout)
        self.ffn2 = FFN_Woemb(latent_dim, ffn_dim, dropout)

    def forward(self, x, global_feature):
        x = self.ffn1(x)
        x = self.ca_block(x, global_feature)
        x = self.ffn2(x)
        return x


class MotionTransformerUnetEncoder(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
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
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(self.num_frames, latent_dim))

        # Input Embedding
        self.joint_embed1 = nn.Linear(self.input_feats, latent_dim)
        self.joint_embed2 = nn.Linear(latent_dim, latent_dim // 4)

        self.temporal_encoder_blocks = nn.ModuleList([
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim // 4, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim // 2),
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim // 2, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim // 2),
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim // 2, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim),
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim),
        ])

    def forward(self, x, emb=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        # x: B, T, latent_dim
        h = self.joint_embed1(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        h = self.joint_embed2(h)
        i = 0
        for module in self.temporal_encoder_blocks:
            # print(i)
            h = module(h, emb)
            i += 1
        return h


class MotionTransformerUnet(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
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
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(self.num_frames, latent_dim))

        # Input Embedding
        self.joint_embed1 = nn.Linear(self.input_feats, latent_dim)
        self.joint_embed2 = nn.Linear(latent_dim, latent_dim // 4)

        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_encoder_blocks_up = nn.ModuleList([
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim // 4, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim // 2),
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim // 2, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim // 2),
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim // 2, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim),
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim)
        ])

        self.mid_blocks = TemporalDiffusionCrossTransformerLayer(
            latent_dim=latent_dim, ffn_dim=ff_size, num_head=num_heads, dropout=dropout)

        self.temporal_encoder_blocks_down = nn.ModuleList([
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim),
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim // 2),
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim // 2, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim // 2),
            TemporalDiffusionTransformerDecoderLayer(latent_dim=latent_dim // 2, time_embed_dim=self.time_embed_dim,
                                                     ffn_dim=ff_size, num_head=num_heads, dropout=dropout,
                                                     out_dim=latent_dim // 4)
        ])

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim // 4, self.input_feats))

    def forward(self, x, global_feature, timesteps, mod=None):
        """
        x: B, T, D   T = clip_total
        """
        B, T = x.shape[0], x.shape[1]

        emb = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1))
            emb = emb + mod_proj

        # x: B, T, latent_dim
        x = self.joint_embed1(x)
        x = x + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        x = self.joint_embed2(x)

        i = 0
        h = []
        for module in self.temporal_encoder_blocks_up:
            x = module(x, emb)
            h.append(x)
            i += 1

        x = self.mid_blocks(x, global_feature)

        j = 0
        for module in self.temporal_encoder_blocks_down:
            x = x + h.pop()
            # print("j:", j)
            x = module(x, emb)
            j += 1

        output = self.out(x)
        return output


class MotionTransformerParallel(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 graph_filters=None,
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.half_dim = latent_dim // 2
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(self.num_frames, latent_dim))
        self.A = torch.tensor(graph_filters[0], requires_grad=False, dtype=torch.float).cuda()
        self.vertices_num = self.A.shape[0]

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        # self.temporal_embed = nn.Parameter(torch.randn(25, self.num_frames))

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
                )
            )

        self.spatial_decoder_blocks = nn.ModuleList(
            [AdaptiveGraphConv(self.num_frames * 3, latent_dim // 2, vertices_num=self.vertices_num, dropout=dropout),
             AdaptiveGraphConv(latent_dim // 2, latent_dim // 2, vertices_num=self.vertices_num, dropout=dropout),
             AdaptiveGraphConv(latent_dim // 2, latent_dim, vertices_num=self.vertices_num, dropout=dropout),
             AdaptiveGraphConv(latent_dim, latent_dim, vertices_num=self.vertices_num, dropout=dropout),
             AdaptiveGraphConv(latent_dim, latent_dim, vertices_num=self.vertices_num, dropout=dropout),
             AdaptiveGraphConv(latent_dim, latent_dim // 2, vertices_num=self.vertices_num, dropout=dropout),
             AdaptiveGraphConv(latent_dim // 2, latent_dim // 2, vertices_num=self.vertices_num, dropout=dropout),
             AdaptiveGraphConv(latent_dim // 2, self.num_frames * 3, vertices_num=self.vertices_num, dropout=dropout)
             ]
        )

        # Output Module
        self.spatial_out = nn.Conv1d(latent_dim, self.num_frames * 3, 1)
        self.temporal_out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

        self.out = nn.Sequential(
            nn.Linear(self.input_feats, self.input_feats),
            nn.SiLU(),
            nn.Linear(self.input_feats, self.input_feats)
        )

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D
        """
        B, T, D = x.shape

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))

        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1))
            emb = emb + mod_proj

        # x: B, T, latent_dim
        # h = self.temporal_embed @ x
        t_h = self.joint_embed(x)
        t_h = t_h + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        # s_h = x.permute(0, 2, 1).view((B, self.vertices_num, 3, T)).contiguous().view((B, self.vertices_num, 3 * T)).permute(0, 2, 1)

        i = 0
        prelist = []

        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(t_h)
                t_h = module(t_h, emb)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                t_h = module(t_h, emb)
                t_h += prelist[-1]
                prelist.pop()
            i += 1

        t_h = self.temporal_out(t_h).view(B, T, -1).contiguous()

        s_h = t_h.reshape(B, T, self.vertices_num, 3).contiguous().permute(0, 2, 3, 1).reshape(B, self.vertices_num,
                                                                                               -1).permute(0, 2, 1)

        for module in self.spatial_decoder_blocks:
            s_h = module(s_h, self.A)

        # t_h = self.temporal_out(t_h).view(B, T, -1).contiguous()

        output = t_h + s_h.reshape(B, self.vertices_num, 3, T).permute(0, 3, 2, 1).reshape(B, T, -1)

        return output


class MotionTransformerMine(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 graph_filters=None,
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = latent_dim * 2
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats  # V * 3
        self.time_embed_dim = latent_dim
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))  # T, D
        self.jointpos_embedding = nn.Parameter(torch.randn(input_feats // 3, latent_dim))  # T, D
        self.graph_filters = nn.Parameter(torch.from_numpy(graph_filters).to(torch.float))
        self.group_num = graph_filters.shape[0]

        # Input Embedding
        self.joint_embed = nn.Linear(3, self.latent_dim)

        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.temporal_decoder_blocks = nn.ModuleList()

        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                SpatialDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                    gcn_group_num=self.group_num,
                    adj_matrices=self.graph_filters
                )
            ),

            self.temporal_decoder_blocks.append(
                MaskedTemporalDiffusionTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                )

            )

        # Output Module
        # self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

        self.out2 = zero_module(nn.Linear(self.latent_dim, 3))

    def set_graph_filters(self, filters):
        self.graph_filters = torch.tensor(filters)

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D D = 3V
        """
        B, T, D = x.shape[0], x.shape[1], x.shape[2]

        # B, latent_dim
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))

        # mod: DCT coefficients copy B, T, D
        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1))
            emb = emb + mod_proj

        # B, T, V, 3
        x = x.view(B, T, D // 3, 3).contiguous()
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0).unsqueeze(2)[:, :T, :, :]  # B, T, V, D

        i = 0
        prelist = []
        for module in self.temporal_decoder_blocks:
            if i < self.num_layers:
                prelist.append(h)
                '''
                if i % 2 == 0:
                    h = module(h)
                else:
                    h = module(h, emb)
                '''
                h = module(h, emb)
            elif i >= self.num_layers:
                '''
                if i % 2 == 0:
                    h = module(h)
                else:
                    h = module(h, emb)
                '''
                h = module(h, emb)
                h += prelist[-1]
                prelist.pop()
            i += 1

        # output = self.out(h).view(B, T, -1).contiguous()
        output = self.out2(h).view(B, T, -1)
        return output


class STCrossMotionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 spatial_graph=None,
                 temporal_graph=None,
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = latent_dim * 2
        self.dropout = dropout
        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        self.input_feats = input_feats  # V * 3
        self.time_embed_dim = latent_dim
        self.spatial_graph = nn.Parameter(torch.from_numpy(spatial_graph).to(torch.float))
        self.temporal_graph = nn.Parameter(torch.from_numpy(temporal_graph).to(torch.float))
        self.latent_num_frames = self.temporal_graph.shape[0]
        self.sequence_embedding = nn.Parameter(torch.randn(self.latent_num_frames, latent_dim))  # T, D
        # self.vertices_embedding = nn.init.normal_(nn.Parameter(torch.zeros(input_feats, latent_dim)), std=0.2)  # V, D

        # Input Embedding
        # 压缩序列长度
        # self.temporal_embed = nn.Linear(self.num_frames, self.latent_num_frames, bias=False)
        # self.temporal_embed = nn.Sequential(
        #     nn.Linear(self.num_frames, self.num_frames - 25, bias=False),
        #     nn.SiLU(),
        #     nn.Linear(self.num_frames - 25, self.num_frames - 50, bias=False),
        #     nn.SiLU(),
        #     nn.Linear(self.num_frames - 50, self.latent_num_frames, bias=False),
        # )

        self.joint_embed_4d = nn.Linear(3, self.latent_dim)
        self.joint_embed_3d = nn.Linear(self.input_feats, self.latent_dim)

        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.decoder_blocks = nn.ModuleList()

        for i in range(num_layers):
            self.decoder_blocks.append(
                STTransformerDecoderLayer(
                    latent_dim=latent_dim,
                    time_embed_dim=self.time_embed_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout,
                    sA=self.spatial_graph,
                    tA=self.temporal_graph
                )
            )

        # Output Module
        # self.temporal_out = nn.Linear(self.latent_num_frames, self.num_frames)
        # self.temporal_out = nn.Sequential(
        #     nn.Linear(self.latent_num_frames, self.num_frames - 50, bias=False),
        #     nn.SiLU(),
        #     nn.Linear(self.num_frames - 50, self.num_frames - 25, bias=False),
        #     nn.SiLU(),
        #     nn.Linear(self.num_frames - 25, self.num_frames, bias=False),
        # )
        # self.joint_out_4d = nn.Linear(self.latent_dim, 3)
        self.joint_out_3d = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D D = 3V
        """
        B, T, D = x.shape[0], x.shape[1], x.shape[2]

        # B, latent_dim DDPM步数嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))

        # mod: DCT coefficients copy B, T, D
        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1))
            emb = emb + mod_proj

        # B, T, V, 3 维度变化
        # x = x.view(B, T, D // 3, 3).contiguous().permute(0, 2, 3, 1)  # B, V, 3, T
        # x = self.activation(self.temporal_embed(x)).permute(0, 3, 1, 2)  # B, T, V, 3
        # x = x.view(B, T, D // 3, 3)
        # x = self.activation(self.temporal_embed(x.permute(0, 2, 1))).permute(0, 2, 1)
        h = self.joint_embed_3d(x)  # B, T, D
        # B, T, V, D 加上序列嵌入和顶点嵌入
        h = h + self.sequence_embedding.unsqueeze(0)[:, :, :]
        #     + self.vertices_embedding.unsqueeze(0).unsqueeze(1))

        i = 0
        prelist = []
        for module in self.decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, emb)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                h = module(h, emb)
                h += prelist[-1]
                prelist.pop()
            i += 1

        # 4D B, T, V, D
        # output = self.temporal_out(h.permute(0, 2, 3, 1))  # B, V, D, T
        # output = self.joint_out(output.permute(0, 3, 1, 2)).contiguous().view(B, T, -1)

        # 3D B, T, D
        # output = self.temporal_out(h.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.joint_out_3d(h).view(B, T, -1).contiguous()
        return output


class MotioniTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 joint_num=16,
                 activation="gelu",
                 spatial_graph=None,
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
        self.joint_num = joint_num
        self.joint_embedding = nn.Parameter(torch.randn(1, self.joint_num, latent_dim))
        if spatial_graph is not None:
            self.spatial_graph = torch.from_numpy(spatial_graph).to(torch.float)

        # Input Embedding
        self.temporal_embed = ResBlock(num_frames, latent_dim, ff_size, dropout, self.time_embed_dim)

        self.cond_embed = nn.Sequential(nn.Linear(self.num_frames, latent_dim),
                                        nn.SiLU(),
                                        nn.Linear(latent_dim, latent_dim))
        """
        self.vel_embed = nn.Sequential(nn.Linear(self.num_frames, latent_dim),
                                       nn.SiLU(),
                                       nn.Linear(latent_dim, latent_dim))
        self.acc_embed = nn.Sequential(nn.Linear(self.num_frames, latent_dim),
                                       nn.SiLU(),
                                       nn.Linear(latent_dim, latent_dim))
        """
        self.va_embed = nn.Sequential(nn.Linear(self.num_frames * 2, latent_dim),
                                      nn.SiLU(),
                                      nn.Linear(latent_dim, 50))
        self.vel_anchors = nn.Parameter(torch.randn(50, latent_dim))

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim))

        if spatial_graph is None:
            decoder_layer = partial(TemporalDiffusioniTransformerDecoderLayer,
                                    latent_dim=latent_dim,
                                    time_embed_dim=self.time_embed_dim,
                                    ffn_dim=ff_size,
                                    num_head=num_heads,
                                    dropout=dropout,
                                    flash_attention=kargs['flash_attention'])
        else:
            decoder_layer = partial(GraphDiffusioniTransformerDecoderLayer,
                                    latent_dim=latent_dim,
                                    time_embed_dim=self.time_embed_dim,
                                    ffn_dim=ff_size,
                                    num_head=num_heads,
                                    dropout=dropout)

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                decoder_layer() if (spatial_graph is None) else decoder_layer(adj_mat=self.spatial_graph)
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.num_frames))

    def forward(self, x, timesteps, mod=None, vel_acc=None):
        """
        x: B, T, V3
        """
        B, T, V = x.shape
        x = x.view(B, T, V // 3, 3).permute(0, 3, 2, 1).reshape(B * 3, V // 3, T)

        timesteps = torch.repeat_interleave(timesteps, 3, 0)
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)).unsqueeze(1)

        if mod is not None:
            mod = mod.view(B, T, V // 3, 3).permute(0, 3, 2, 1).reshape(B * 3, V // 3, T)
            mod_proj = self.cond_embed(mod)
            emb = emb + mod_proj

        if vel_acc is not None:
            """
            # vel_acc B T V 2
            vel = vel_acc[:, :, :, 0].permute(0, 2, 1)
            vel_proj = self.vel_embed(vel).repeat_interleave(3, 0)
            acc = vel_acc[:, :, :, 1].permute(0, 2, 1)
            acc_proj = self.vel_embed(acc).repeat_interleave(3, 0)
            emb = emb + vel_proj + acc_proj
            """
            vel_acc = vel_acc.permute(0, 2, 3, 1).reshape(B, V // 3, -1)
            va_proj = F.softmax(self.va_embed(vel_acc), dim=-1)
            va_emb = va_proj.matmul(self.vel_anchors)
            emb = emb + va_emb.repeat_interleave(3, 0)

        # x: B, V, latent_dim
        h = self.temporal_embed(x)

        h = h + self.joint_embedding

        i = 0
        prelist = []
        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, emb)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                h = module(h, emb)
                h += prelist[-1]
                prelist.pop()
            i += 1

        output = self.out(h).reshape(B, 3, V // 3, T).permute(0, 3, 2, 1).reshape(B, T, V)
        return output


class MotioniTransformer2(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 joint_num=16,
                 activation="gelu",
                 spatial_graph=None,
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
        self.joint_num = joint_num
        self.joint_embedding = nn.Parameter(torch.randn(1, self.joint_num, latent_dim))
        if spatial_graph is not None:
            self.spatial_graph = torch.from_numpy(spatial_graph).to(torch.float)

        # Input Embedding
        self.temporal_embed = ResBlock(num_frames * 3, latent_dim, ff_size, dropout, self.time_embed_dim)

        self.cond_embed = nn.Sequential(nn.Linear(self.num_frames * 3, latent_dim),
                                        nn.SiLU(),
                                        nn.Linear(latent_dim, latent_dim))
        """
        self.vel_embed = nn.Sequential(nn.Linear(self.num_frames, latent_dim),
                                       nn.SiLU(),
                                       nn.Linear(latent_dim, latent_dim))
        self.acc_embed = nn.Sequential(nn.Linear(self.num_frames, latent_dim),
                                       nn.SiLU(),
                                       nn.Linear(latent_dim, latent_dim))
        
        self.va_embed = nn.Sequential(nn.Linear(self.num_frames * 2, latent_dim),
                                       nn.SiLU(),
                                       nn.Linear(latent_dim, 50))
        
        self.vel_anchors = nn.Parameter(torch.randn(50, latent_dim))
        """

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim))

        if spatial_graph is None:
            decoder_layer = partial(TemporalDiffusioniTransformerDecoderLayer,
                                    latent_dim=latent_dim,
                                    time_embed_dim=self.time_embed_dim,
                                    ffn_dim=ff_size,
                                    num_head=num_heads,
                                    dropout=dropout,
                                    flash_attention=kargs['flash_attention'],
                                    stylization_block=kargs['stylization_block'])
        else:
            decoder_layer = partial(GraphDiffusioniTransformerDecoderLayer,
                                    latent_dim=latent_dim,
                                    time_embed_dim=self.time_embed_dim,
                                    ffn_dim=ff_size,
                                    num_head=num_heads,
                                    dropout=dropout)

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                decoder_layer() if (spatial_graph is None) else decoder_layer(adj_mat=self.spatial_graph)
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.num_frames * 3))

    def forward(self, x, timesteps, mod=None, **kwargs):
        """
        x: B, T, V3
        """
        B, T, V3 = x.shape
        V = V3 // 3
        x = x.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)

        # timesteps = torch.repeat_interleave(timesteps, 3, 0)
        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)).unsqueeze(1)

        if mod is not None:
            mod = mod.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)
            mod_proj = self.cond_embed(mod)
            emb = emb + mod_proj
            emb = emb[:, -self.joint_num:, :]

        """
        if vel_acc is not None:

            # vel_acc B T V 2
            vel = vel_acc[:, :, :, 0].permute(0, 2, 1)
            vel_proj = self.vel_embed(vel).repeat_interleave(3, 0)
            acc = vel_acc[:, :, :, 1].permute(0, 2, 1)
            acc_proj = self.vel_embed(acc).repeat_interleave(3, 0)
            emb = emb + vel_proj + acc_proj

            vel_acc = vel_acc.permute(0, 2, 3, 1).reshape(B, V // 3, -1)
            va_proj = F.softmax(self.va_embed(vel_acc), dim=-1)
            va_emb = va_proj.matmul(self.vel_anchors)
            emb = emb + va_emb.repeat_interleave(3, 0)
        """

        # x: B, V, latent_dim
        h = self.temporal_embed(x)[:, -self.joint_num:, :]

        h = h + self.joint_embedding

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


class MotioniTransformer3(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 joint_num=16,
                 activation="gelu",
                 spatial_graph=None,
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
        self.joint_num = joint_num
        self.joint_embedding = nn.Parameter(torch.randn(1, self.joint_num, latent_dim))
        if spatial_graph is not None:
            self.spatial_graph = torch.from_numpy(spatial_graph).to(torch.float)
        self.cross_attention = kargs['cross_attention']
        self.flash_attention = kargs['flash_attention']
        self.se_layer = kargs['se_layer']
        self.skip_type = kargs['skip_type']

        # Input Embedding
        self.temporal_embed = nn.Linear(self.num_frames * 3, self.latent_dim)

        self.cond_embed = nn.Linear(self.num_frames * 3, latent_dim)
        """
        self.vel_embed = nn.Sequential(nn.Linear(self.num_frames, latent_dim),
                                       nn.SiLU(),
                                       nn.Linear(latent_dim, latent_dim))
        self.acc_embed = nn.Sequential(nn.Linear(self.num_frames, latent_dim),
                                       nn.SiLU(),
                                       nn.Linear(latent_dim, latent_dim))

        self.va_embed = nn.Sequential(nn.Linear(self.num_frames * 2, latent_dim),
                                       nn.SiLU(),
                                       nn.Linear(latent_dim, 50))

        self.vel_anchors = nn.Parameter(torch.randn(50, latent_dim))
        """

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim))

        if spatial_graph is None:
            decoder_layer = partial(TemporalDiffusionTransformerDecoderLayer,
                                    latent_dim=latent_dim,
                                    time_embed_dim=self.time_embed_dim,
                                    ffn_dim=ff_size,
                                    num_head=num_heads,
                                    dropout=dropout,
                                    se_dim=self.joint_num,
                                    se_r=4,
                                    cross_attention=self.cross_attention,
                                    flash_attention=self.flash_attention,
                                    se_layer=self.se_layer,
                                    skip_type=self.skip_type,
                                    squeeze_dim=-1)
        else:
            decoder_layer = partial(GraphDiffusioniTransformerDecoderLayer,
                                    latent_dim=latent_dim,
                                    time_embed_dim=self.time_embed_dim,
                                    ffn_dim=ff_size,
                                    num_head=num_heads,
                                    dropout=dropout)

        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            if i >= num_layers // 2:
                b_skip = True
            else:
                b_skip = False
            self.temporal_decoder_blocks.append(
                decoder_layer(skip=b_skip) if (spatial_graph is None) else decoder_layer(adj_mat=self.spatial_graph,
                                                                                         skip=b_skip)
            )

        # Output Module
        self.out = zero_module(nn.Linear(self.latent_dim, self.num_frames * 3))

    def forward(self, x, timesteps, mod=None, vel_acc=None):
        """
        x: B, T, V3
        """
        B, T, V3 = x.shape
        V = V3 // 3
        x = x.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)).unsqueeze(1)

        if mod is not None:
            mod = mod.view(B, T, V, 3).permute(0, 2, 1, 3).reshape(B, V, T * 3)
            mod_proj = self.cond_embed(mod)
            emb = emb + mod_proj

        # x: B, V, latent_dim
        h = self.temporal_embed(x)

        h = h + self.joint_embedding

        i = 0
        prelist = []
        for module in self.temporal_decoder_blocks:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h, skip=None, mod_emb=emb)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                h = module(h, skip=prelist.pop(), mod_emb=emb)
            i += 1

        output = self.out(h).reshape(B, V, T, 3).permute(0, 2, 1, 3).reshape(B, T, V3)
        return output

