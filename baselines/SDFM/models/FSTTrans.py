import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as spn
from transformer_mine import FFN4D, ResBlock, zero_module
from transfusion import timestep_embedding


class FSTAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, time_embed_dim=0):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query_s = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key_s = nn.Linear(latent_dim, latent_dim, bias=False)
        self.query_t = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key_t = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.proj_out = StylizationBlock4D(latent_dim, time_embed_dim, dropout)

    def forward(self, x, adj_mat=None):
        """
        x: B, T, V, D
        """
        B, T, V, D = x.shape
        H = self.num_head
        # BT, V, D -> BT, V, H, D//H
        query_s = self.query_s(self.norm(x)).contiguous().view(B * T, V, H, D // H).permute(0, 2, 1, 3)
        # BT, V, D -> BT, V, H, D//H
        key_s = self.key_s(self.norm(x)).contiguous().view(B * T, V, H, D // H).permute(0, 2, 1, 3)
        # BT, H, V, V
        att_s = (query_s @ key_s.transpose(-2, -1)) / math.sqrt(D // H)
        if adj_mat is not None:
            att_s = F.softmax(att_s, dim=-1) * adj_mat
            att_s = self.dropout(att_s)
        else:
            att_s = self.dropout(F.softmax(att_s, dim=-1))  # BT, H, V, V

        x1 = x.transpose(1, 2).contiguous().view(B * V, T, D)
        # BV, T, D -> BV, T, H, D//H
        query_t = self.query_t(self.norm(x1)).contiguous().view(B * V, T, H, D // H).permute(0, 2, 1, 3)
        # BV, T, D -> BV, T, H, D//H
        key_t = self.key_t(self.norm(x1)).contiguous().view(B * V, T, H, D // H).permute(0, 2, 1, 3)
        att_t = (query_t @ key_t.transpose(-2, -1)) / math.sqrt(D // H)
        # att_t = F.softmax(att_t.masked_fill(mask, -np.inf), dim=-1)
        att_t = self.dropout(F.softmax(att_t, dim=-1))  # BV, H, T, T

        value = self.value(self.norm(x)).contiguous().view(B * T, V, H, D // H).permute(0, 2, 1, 3)
        # BT, H, V, D//H
        y = att_s @ value
        y = att_t @ y.view(B, T, H, V, D // H).permute(0, 3, 2, 1, 4).reshape(B * V, H, T, D // H)
        y = x + y.view(B, V, T, D).transpose(1, 2).contiguous()
        return y

class STTransformerLayerSimple(nn.Module):
    def __init__(self, latent_dim, ffn_dim, num_head, dropout, adj_mat=None):
        super().__init__()
        self.att = FSTAttention(latent_dim, num_head, dropout)
        self.ffn = FFN4D(latent_dim, ffn_dim, dropout, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        # self.adj_mat = nn.Parameter(adj_mat.clone())

    def forward(self, x, emb=None, adj_mat=None):
        # B, T, V, D = x.shape
        y = self.norm(self.att(x))
        y = x + self.norm(self.ffn(y, emb))
        return y

class FSTTransformer(nn.Module):
    def __init__(self, input_feats, latent_dim=512, n_head=8, total_len=125, n_joint=13, ff_size=1024, dropout=0.2, num_layers=8,
                 dct=False, adj_mat=None):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.n_head = n_head
        self.total_len = total_len
        self.n_joint = n_joint
        self.ff_size = ff_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.time_embed_dim = latent_dim
        self.dct = dct

        self.t_embedding = nn.Parameter(torch.randn((1, self.total_len, 1, self.latent_dim)))
        self.s_embedding = nn.Parameter(torch.randn((1, 1, self.n_joint, self.latent_dim)))

        self.input_embed = nn.Linear(3, self.latent_dim)
        # self.adj_mat = torch.tensor(adj_mat, dtype=torch.float, requires_grad=False).cuda()

        self.cond_embed = nn.Linear(n_joint * 3 * self.total_len, self.time_embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # self.noise_embed = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.ff_size),
        #     nn.SiLU(),
        #     nn.Linear(self.ff_size, self.hidden_size)
        # )

        # self.mu = nn.Linear(self.hidden_size, self.hidden_size)
        # self.log_var = nn.Linear(self.hidden_size, self.hidden_size)

        # self.linear1 = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # self.va_embed = nn.Sequential(nn.Linear(self.total_len * 2, self.hidden_size),
        #                               nn.SiLU(),
        #                               nn.Linear(self.hidden_size, 50))
        # self.vel_anchors = nn.Parameter(torch.randn(50, self.hidden_size))

        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(
                STTransformerLayerSimple(self.latent_dim, self.latent_dim, self.n_head, self.dropout))

        self.complete_out = zero_module(nn.Linear(self.latent_dim, 3))
        self.predict_out = zero_module(nn.Linear(self.latent_dim, 3))
        self.predict_out2 = zero_module(nn.Linear(self.latent_dim, 2))

    def forward(self, x, timesteps, mod=None, return_inter_feature=False):
        # B T V3
        B, T, V3 = x.shape
        x = x.view(B, T, V3 // 3, 3)

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)).unsqueeze(1).unsqueeze(1)

        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1)).unsqueeze(1).unsqueeze(1)
            emb = emb + mod_proj

        # x: B, V, latent_dim
        h = self.input_embed(x)

        # h = h + self.t_embedding + self.s_embedding

        i = 0
        prelist = []
        for module in self.encoder_layers:
            if i < (self.num_layers // 2):
                prelist.append(h)
                h = module(h + self.t_embedding + self.s_embedding, emb)
            elif i >= (self.num_layers // 2) and self.num_layers > 1:
                h = module(h + self.t_embedding + self.s_embedding, emb)
                h += prelist[-1]
                prelist.pop()
            i += 1

        output = self.complete_out(h).view(B, T, V3)
        # output2 = self.predict_out2(h).view(B, T, V3 // 3 * 2)  # 预测速度和加速度
        return output

    # def predict(self, x, x_p=None, va=None, va_p=None):
    #     B, T, V3 = x.shape
    #     h = self.encode(x)
    #     output1 = self.predict_out(h).view(B, T, V3)
    #     # output2 = self.predict_out2(h).view(B, T, V3 // 3 * 2)  # 预测速度和加速度
    #     return output1


if __name__ == '__main__':
    # mae = MAE()
    mask = torch.ones((1, 20, 39, 1))
    mask = torch.dropout(mask, 0.6, True)
    mask[mask > 0.0] = 1
    mask1 = mask.repeat(1, 1, 1, 3)
    x = torch.randn((64, 125, 39)).cuda()
    y = torch.randn((1, 20, 39, 3))
    # mse1 = nn.MSELoss()
    # loss1 = mse1(x[mask1.bool()], y[mask1.bool()])
    # mse2 = nn.MSELoss(reduction='sum')
    # loss2 = mse2(x * mask, y * mask) / (torch.count_nonzero(mask) * 3)
    # print(loss1, loss2)
    a = np.random.randn(13, 13)
    fsttrans = FSTTransformer(input_feats=13 * 3, latent_dim=128, total_len=125, num_layers=4, ff_size=256).cuda()
    total_params = sum(p.numel() for p in list(fsttrans.parameters())) / 1000000.0
    print(" params: {:.3f}M".format(total_params))
    t = torch.rand(x.shape[0]).cuda()
    mod = torch.randn(x.shape).cuda()
    y = fsttrans(x, t, mod)
    print(y.shape)
