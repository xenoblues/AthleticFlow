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


class DatasetAsb(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100):
        self.seq_len = None
        super().__init__(mode, t_his, t_pred)
        self.dataset_name = 'assemble'

    def prepare_data(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.dirname(current_path)
        if self.mode == 'train':
            self.data_file = os.path.join(root_path, 'data/assemble_data', 'assemble_train_data.npz')
        elif self.mode == 'test':
            self.data_file = os.path.join(root_path, 'data/assemble_data', 'assemble_test_data.npz')
        # self.subjects_split = {'train': [1, 5, 6, 7, 8],
        #                        'test': [9, 11]}
        # self.subjects = ['S%d' % x for x in self.subjects_split[self.mode]]
        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11],
                                 joints_left=[1, 2, 3, 4, 5, 6],
                                 joints_right=[7, 8, 9, 10, 11, 12])
        # self.removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        self.kept_joints = np.array([x for x in range(13)])
        # self.skeleton.remove_joints(self.removed_joints)
        self.skeleton.gen_adj_mat()
        self.skeleton.gen_filters(4)
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True, mmap_mode='r')
        self.data = data_o["arr_0"] / 1000.0
        self.seq_len = data_o["arr_1"].astype(int)

    def sample(self):
        idx = np.random.randint(self.data.shape[0])
        # 随机截取一段t_total长的序列
        fr_start = np.random.randint(self.seq_len[idx] - self.t_total)[0]
        fr_end = fr_start + self.t_total
        traj = self.data[idx, fr_start: fr_end, :, :]
        return traj

    def iter_generator(self, step=25):
        for j in range(self.data.shape[0]):
            length = self.seq_len[j][0]
            for i in range(0, length - self.t_total, step):
                traj = self.data[j, i: i + self.t_total, :, :]
                yield traj


if __name__ == '__main__':
    np.random.seed(0)
    actions = {'WalkDog'}
    start_time = time.time()
    dataset = DatasetAsb('train')
    generator = dataset.sampling_generator(50000, 64, aug=False)
    # dataset.normalize_data()
    generator = dataset.iter_generator()
    for data in generator:
        print(data.shape)
        break

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

    # for data, mask in generator:
    #     data = np.multiply(data, mask) + np.multiply(data, 1 - mask)
    # print(data.shape)
    skelton = dataset.skeleton
