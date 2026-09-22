"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset_h36m.py
"""

import numpy as np
import os
import torch
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton
import utils.util
from torch import nn, optim
import time
import math


class DatasetH36M(Dataset):
    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False):
        self.use_vel = use_vel
        super().__init__(mode, t_his, t_pred, actions)
        if use_vel:
            self.traj_dim += 3

    def prepare_data(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.dirname(current_path)
        self.data_file = os.path.join(root_path, 'data', 'data_3d_h36m.npz')
        self.subjects_split = {'train': [1, 5, 6, 7, 8],
                               'test': [9, 11]}
        self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                 joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                 joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
        self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        self.kept_joints = np.array([x for x in range(32) if x not in self.removed_joints])
        self.skeleton.remove_joints(self.removed_joints)
        self.skeleton._parents[11] = 8
        self.skeleton._parents[14] = 8
        self.skeleton.gen_adj_mat()
        self.skeleton.gen_filters(4)

        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True, mmap_mode='r')['positions_3d'].item()
        self.S1_skeleton = data_o['S1']['Directions'][:1, self.kept_joints].copy()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        if self.actions != 'all':
            for key in list(data_f.keys()):
                data_f[key] = dict(filter(lambda x: all([a in x[0] for a in self.actions]), data_f[key].items()))
                if len(data_f[key]) == 0:
                    data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                if self.use_vel:
                    v = (np.diff(seq[:, :1], axis=0) * 50).clip(-5.0, 5.0)
                    v = np.append(v, v[[-1]], axis=0)
                seq[:, 1:] -= seq[:, :1]
                if self.use_vel:
                    seq = np.concatenate((seq, v), axis=1)
                data_s[action] = seq
        self.data = data_f
        print(self.data.keys())


if __name__ == '__main__':
    np.random.seed(0)
    actions = {'WalkDog'}
    start_time = time.time()
    dataset = DatasetH36M('train', use_vel=False)
    generator = dataset.sampling_generator()
    # dataset.normalize_data()
    generator = dataset.iter_generator()
    i = 0
    for data in generator:
        i += 1
    print(i)

    """
    mask_indices = torch.randint(0, 25, (64, int(25 * 0.8)))
    x = torch.ones((64, 25, 48)).cuda()
    x_n = torch.randn((64, 25, 48)).cuda()
    frame_mask = torch.ones((64, 25, 1)).cuda()
    x_n2 = torch.zeros((x_n.shape[0], 25, x_n.shape[2])).cuda()
    frame_mask = torch.dropout(frame_mask, 0.8, True)
    frame_mask[frame_mask > 0] = 1.0
    joint_mask = torch.ones((64, 25, 16)).cuda()
    joint_mask = torch.dropout(joint_mask, 0.8, True).repeat_interleave(3, -1)
    joint_mask[joint_mask > 0] = 1.0
    mask = frame_mask.mul(joint_mask)
    """
    # B, T, D
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    print(dataset.skeleton.num_joints())
    print(dataset.skeleton.parents())

    # for data, _ in generator:
        # data = np.multiply(data, mask) + np.multiply(data, 1 - mask)
        # print(data.shape)
