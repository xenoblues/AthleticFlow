import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data_loader.dataset_h36m import DatasetH36M
from utils.util import *


class Compressor(torch.nn.Module):
    def __init__(self, max_seq_length, dropout):
        super(Compressor, self).__init__()
        self.max_seq_length = max_seq_length
        dct_m, idct_m = get_dct_matrix(max_seq_length)
        self.transform_matrix = nn.Parameter(dct_m, requires_grad=True)
        self.inv_transform_matrix = nn.Parameter(idct_m, requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def transform(self, x, result_length):
        # tmp_mat = self.transform_matrix[:result_length, :]
        return torch.matmul(self.transform_matrix[:result_length, :], x)

    def inverse_transform(self, x, result_length):
        return torch.matmul(self.inv_transform_matrix[:, :result_length], x)

    def forward(self, x, result_length):
        compressed_x = self.dropout(self.transform(x, result_length))
        rec_x = self.inverse_transform(compressed_x, result_length)
        return rec_x

    def load(self, pt_path):
        self.load_state_dict(torch.load(pt_path))


def train():
    t_his = 25
    t_pred = 100
    batch_size = 256
    epochs = 500
    loss_fn = nn.MSELoss()
    dataset = DatasetH36M('train', t_his, t_pred, actions='all')
    dataset_test = DatasetH36M('test', t_his, t_pred, actions='all')
    compressor = Compressor(t_his + t_pred, 0.8).cuda()
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


    for e in tqdm(range(epochs)):
        sample_generator_train = dataset.sampling_generator(num_samples=50000, batch_size=batch_size)
        epoch_loss = []
        for traj_np, _ in sample_generator_train:
            result_length = torch.randint(low=1, high=t_his + t_pred, size=(1,)).cuda()
            # print(result_length)
            traj_tr = torch.from_numpy(traj_np).float().cuda().view(batch_size, t_his + t_pred, -1)
            y = compressor(traj_tr, result_length)
            # print(y[0, :5, 0], traj_tr[0, :5, 0])
            loss = loss_fn(y, traj_tr)
            epoch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sample_generator_val = dataset_test.sampling_generator(num_samples=5000, batch_size=batch_size)
        epoch_loss_val = []
        for traj_np, _ in sample_generator_val:
            result_length = torch.randint(low=1, high=t_his + t_pred, size=(1,)).cuda()
            # print(result_length)
            traj_tr = torch.from_numpy(traj_np).float().cuda().view(batch_size, t_his + t_pred, -1)
            y = compressor(traj_tr, result_length)
            loss = loss_fn(y, traj_tr)
            epoch_loss_val.append(loss.item())
        val_loss = np.mean(epoch_loss_val)
        scheduler.step(torch.tensor(val_loss))

        print("epoch %d, lr %f,  training loss %f, val loss %f" % (e, optimizer.state_dict()['param_groups'][0]['lr'], np.mean(epoch_loss), val_loss))
    torch.save(compressor.state_dict(), '../results/compressor.pt')


if __name__ == '__main__':
    train()

