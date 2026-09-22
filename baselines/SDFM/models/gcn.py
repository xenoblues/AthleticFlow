import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import math
from functools import partial
from utils import *
from models.transformer import timestep_embedding, zero_module


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 time_dim,
                 joints_dim
                 ):
        super(ConvTemporalGraphical, self).__init__()

        self.A = nn.Parameter(torch.FloatTensor(time_dim, joints_dim,
                                                joints_dim))  # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)
        '''
        self.prelu = nn.PReLU()

        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''
        self.joints_dim = joints_dim
        self.time_dim = time_dim

    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        ## x=self.prelu(x)
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous()


class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 time_dim,
                 joints_dim,
                 dropout,
                 bias=True,
                 version=0,
                 pose_info=None):

        super(ST_GCNN_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        '''
        if version == 0:
            self.gcn = ConvTemporalGraphical(time_dim, joints_dim)  # the convolution layer
        elif version == 1:
            self.gcn = ConvTemporalGraphicalV1(time_dim, joints_dim, pose_info=pose_info)
        '''
        self.gcn = ConvTemporalGraphical(time_dim, joints_dim)  # the convolution layer

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:

            self.residual = nn.Sequential(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x):
        #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.prelu(x)
        return x


class MotionSTGCN(nn.Module):
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
        self.joints_num = input_feats // 3
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))  # T, D
        # self.jointpos_embedding = nn.Parameter(torch.randn(input_feats // 3, latent_dim))  # T, D
        self.graph_filters = nn.Parameter(torch.from_numpy(graph_filters).to(torch.float))
        self.group_num = graph_filters.shape[0]

        # Input Embedding
        # self.joint_embed = nn.Linear(3, self.latent_dim)

        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, 64)

        self.time_embed = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        self.st_gcnns = nn.ModuleList()
        self.st_gcnns.append(ST_GCNN_layer(3, 64, [3, 1], 1, num_frames,
                                           self.joints_num, dropout))

        self.st_gcnns.append(ST_GCNN_layer(64, 32, [3, 1], 1, num_frames,
                                           self.joints_num, dropout))

        self.st_gcnns.append(ST_GCNN_layer(32, 64, [3, 1], 1, num_frames,
                                           self.joints_num, dropout))

        self.st_gcnns.append(ST_GCNN_layer(64, 32, [3, 1], 1, num_frames,
                                           self.joints_num, dropout))

        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A
        self.st_gcnns[-1].gcn.T = self.st_gcnns[-3].gcn.T

        self.st_gcnns.append(ST_GCNN_layer(32, 64, [3, 1], 1, num_frames,
                                           self.joints_num, dropout))
        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A
        self.st_gcnns[-1].gcn.T = self.st_gcnns[-3].gcn.T

        self.st_gcnns.append(ST_GCNN_layer(64, 32, [3, 1], 1, num_frames,
                                           self.joints_num, dropout))

        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A
        self.st_gcnns[-1].gcn.T = self.st_gcnns[-3].gcn.T

        self.st_gcnns.append(ST_GCNN_layer(32, 64, [3, 1], 1, num_frames,
                                           self.joints_num, dropout))

        self.st_gcnns[-1].gcn.A = self.st_gcnns[-3].gcn.A
        self.st_gcnns[-1].gcn.T = self.st_gcnns[-3].gcn.T

        self.st_gcnns.append(ST_GCNN_layer(64, 3, [3, 1], 1, num_frames,
                                           self.joints_num, dropout))

    def set_graph_filters(self, filters):
        self.graph_filters = torch.tensor(filters)

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D D = 3V
        """
        B, T, D = x.shape[0], x.shape[1], x.shape[2]

        # B, latent_dim
        emb = self.time_embed(timestep_embedding(timesteps, 64))

        # mod: DCT coefficients copy B, T, D
        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1))
            emb = emb + mod_proj

        # B, T, V, 3
        x = x.view(B, T, D // 3, 3).contiguous().permute(0, 3, 1, 2)

        i = 0
        prelist = []

        for gcn in self.st_gcnns[:5]:
            x = gcn(x)
        x = x + emb.unsqueeze(-1).unsqueeze(-1)

        for gcn in self.st_gcnns[5:]:
            x = gcn(x)

        return x.permute(0, 2, 3, 1).contiguous().view(B, T, -1)
