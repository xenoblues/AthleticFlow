"""
This code is adopted from:
https://github.com/wei-mao-2019/gsps/blob/main/motion_pred/utils/dataset_humaneva.py
"""


import numpy as np
import os

import torch

from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


class DatasetHumanEva(Dataset):
    def __init__(self, mode, t_his=15, t_pred=60, actions='all', **kwargs):
        self.b_redirect = kwargs.get('b_redirect', False)
        super().__init__(mode, t_his, t_pred, actions, **kwargs)


    def prepare_data(self, **kwargs):
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.dirname(current_path)
        self.data_file = os.path.join(root_path, 'data', 'data_3d_humaneva15.npz')
        self.subjects_split = {'train': ['Train/S1', 'Train/S2', 'Train/S3'],
                               'test': ['Validate/S1', 'Validate/S2', 'Validate/S3']}
        self.subjects = [x for x in self.subjects_split[self.mode]]

        self.skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
                                 joints_left=[2, 3, 4, 8, 9, 10],
                                 joints_right=[5, 6, 7, 11, 12, 13])
        self.kept_joints = np.arange(15)
        if not self.b_redirect:
            self.redicreted_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                              16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                                     joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                                     joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
            removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
            kept_joints = np.array([x for x in range(32) if x not in removed_joints])
            self.redicreted_skeleton.remove_joints(removed_joints)
            self.redicreted_skeleton._parents[11] = 8
            self.redicreted_skeleton._parents[14] = 8
            self.redicreted_skeleton.gen_adj_mat(masked_joints=[7, 9])
            self.redirection_dict = {0: 0, 1: 8, 2: 9, 3: 10, 4: 11, 5: 12, 6: 13, 8: 1, 10: 14, 11: 5, 12: 6, 13: 7, 14: 2,
                                 15: 3, 16: 4}
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)['positions_3d'].item()
        data_f = dict(filter(lambda x: x[0] in self.subjects, data_o.items()))
        # these takes have wrong head position, excluded from training and testing
        if self.mode == 'train':
            data_f['Train/S3'].pop('Walking 1 chunk0')
            data_f['Train/S3'].pop('Walking 1 chunk2')
        else:
            data_f['Validate/S3'].pop('Walking 1 chunk4')
        for key in list(data_f.keys()):
            data_f[key] = dict(filter(lambda x: (self.actions == 'all' or
                                                 all([a in x[0] for a in self.actions]))
                                                 and x[1].shape[0] >= self.t_total, data_f[key].items()))
            if len(data_f[key]) == 0:
                data_f.pop(key)
        for data_s in data_f.values():
            for action in data_s.keys():
                seq = data_s[action][:, self.kept_joints, :]
                seq[:, 1:] -= seq[:, :1]
                data_s[action] = seq
        self.data = data_f

    def redirect(self, data):
        redirected_data = np.zeros((data.shape[0], data.shape[1], self.redicreted_skeleton.num_joints(), 3))
        for k, v in self.redirection_dict.items():
            redirected_data[:, :, k, :] = data[:, :, v, :]
        return redirected_data

    def recovery(self, prediction_results):
        # prediction_results shape (B, T, 3(V-1))
        B, T, D = prediction_results.shape
        recovered_data = np.zeros((B, T, self.skeleton.num_joints(), 3))
        zero_data = np.zeros((B, T, 1, 3))
        prediction_results_padded = np.concatenate((zero_data, prediction_results.reshape(B, T, D // 3, 3)), axis=2)
        for k, v in self.redirection_dict.items():
            recovered_data[:, :, v, :] = prediction_results_padded[:, :, k, :]
        return recovered_data[:, :, 1:, :].reshape(B, T, -1)




if __name__ == '__main__':
    np.random.seed(0)
    actions = 'all'
    dataset = DatasetHumanEva('train', actions=actions)
    generator = dataset.sampling_generator(num_samples=51200, batch_size=1024)
    dataset.normalize_data()
    # generator = dataset.iter_generator()
    min_v = np.inf
    max_v = -np.inf
    for data, _ in generator:
        min_v = min(min_v, np.min(data))
        max_v = max(max_v, np.max(data))
    print(min_v, max_v)



