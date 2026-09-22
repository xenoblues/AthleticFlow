import os
import sys
import csv
import math
import time
import argparse

import numpy
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torchvision import transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist, squareform

from data_utils.dataset_ap3d import DatasetAP3D
from data_utils.dataset_ap3d_multimodal import DatasetAP3D_multi
from data_utils.dataset_athleticspose import DatasetAthleticsPose
from data_utils.dataset_athleticspose_multimodal import DatasetAthleticsPose_multi
from data_utils.dataset_wp import DatasetWP
from data_utils.dataset_wp_multimodal import DatasetWP_multi

sys.path.append(os.getcwd())
from utils import *
from models.load_models import get_model
from models.fid_classifier import ClassifierForFID
from models.GaussianDiffusion import GaussianDiffusion
from utils.metrics import CMD_helper, CMD_pose, compute_detailed_metrics, get_joint_names, get_bone_names, get_fps
from utils.metrics import FID_helper, FID_pose
from utils.metrics import APD, APDE, ADE, FDE, MMADE, MMFDE
from data_utils.transforms import calculate_stats, load_stats
from data_utils.dataset_h36m import DatasetH36M, generate_h36_loss_weights
from data_utils.dataset_amass import DatasetAMASS, generate_amass_loss_weights
from data_utils.dataset_wp import generate_worldpose_loss_weights
from data_utils.datareader_ap3d import DataReaderAP3D
from data_utils.transforms import DataAugmentation



def generate_loss_weight(cfg):
    if cfg.dataset == 'h36m' or cfg.dataset == 'ap3d' or cfg.dataset == 'ap':
        gen_weight = generate_h36_loss_weights
    elif cfg.dataset == 'wp':
        gen_weight = generate_worldpose_loss_weights
    else:
        gen_weight = generate_amass_loss_weights
    # history recon weight
    in_weights = gen_weight(cfg.t_his, scale=cfg.loss_weight_scale)
    # future prediction weight
    out_weights = gen_weight(cfg.t_pred, scale=cfg.loss_weight_scale)
    loss_weights = torch.cat((in_weights, out_weights), dim=0)
    return loss_weights



class TrainerCustom(object):
    def __init__(
            self,
            dataset,
            dataset_multi,
            diffusion_model,
            cfg,
            train_batch_size=16,
            train_lr=1e-4,
            weight_decay=0,
            actions='all',
    ):
        super().__init__()

        self.model = diffusion_model
        self.device = next(self.model.parameters()).device

        self.cfg = cfg

        self.batch_size = train_batch_size
        self.input_n = cfg.t_his
        self.output_n = cfg.t_pred
        self.dtype = torch.float32 if self.cfg.dtype == 'float32' else torch.float64

        # dataset and dataloader initialization
        # transform = transforms.Compose([DataAugmentation(cfg.rota_prob)])
        # test_transform = None
        # stat_dataset = dataset('train', self.input_n, self.output_n, augmentation=cfg.augmentation, stride=cfg.stride,
        #                        transform=transform, dtype=cfg.dtype)

        print('Preparing datasets...')
        self.train_dataset = dataset(mode='train', t_his=cfg.t_his, t_pred=cfg.t_pred)
        self.eval_dataset = dataset(mode='test', t_his=cfg.t_his, t_pred=cfg.t_pred)

        if dataset_multi is not None:
            self.dataset_multi_test = dataset_multi(mode='test', t_his=cfg.t_his, t_pred=cfg.t_pred)

        # multimodal GT
        print('Calculating mmGT...')
        self.multimodal_traj, self.eval_gt_group, self.eval_data_group = self.get_multimodal_gt()

        self.generator_train = self.train_dataset.sampling_generator(num_samples=50000,
                                                                     batch_size=self.cfg.batch_size, aug=True)

        # optimizer
        self.opt = AdamW(self.model.parameters(), lr=train_lr, betas=(0.9, 0.99),
                         weight_decay=weight_decay)  # weight_decay is 0, same as Adam Pytorch Implementation
        self.scheduler = get_scheduler(self.opt, policy=cfg.sched_policy, nepoch_fix=cfg.num_epoch_fix_lr,
                                       nepoch=cfg.train_epoch)

        # epoch counter state
        self.epoch = 0
        self.train_loss_list = []
        print('Trainer initialization done.')

    def save(self, to_save_path):
        data = {
            'epoch': self.epoch,
            'train_loss_list': self.train_loss_list,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'sched': self.scheduler.state_dict(),
        }
        torch.save(data, to_save_path)
        return

    def load(self, to_load_path):
        device = self.device
        data = torch.load(to_load_path, map_location=device)
        self.epoch = data['epoch']
        self.train_loss_list = data['train_loss_list']
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['sched'])
        print(">>> finish loading model ckpt from path '{}'".format(to_load_path))
        return

    def train(self):
        self.model.train()
        t_s = time.time()
        epoch_loss = 0.
        epoch_iter = 0
        epoch_loss_info = {}
        self.generator_train = self.train_dataset.sampling_generator(num_samples=50000,
                                                                     batch_size=self.cfg.batch_size, aug=True)
        for traj_np in self.generator_train:
            traj = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
            traj = torch.tensor(traj, dtype=self.dtype, device=self.device)
            loss, loss_info = self.model(traj, None, div_k=self.cfg.div_k, uncond=True, mmgt=None)
            for key, value in loss_info.items():
                if key not in epoch_loss_info:
                    epoch_loss_info[key] = value
                else:
                    epoch_loss_info[key] += value
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            epoch_iter += 1

        self.scheduler.step()
        self.epoch += 1
        epoch_loss /= epoch_iter
        for key, value in epoch_loss_info.items():
            epoch_loss_info[key] /= epoch_iter
        lr = self.opt.param_groups[0]['lr']
        dt = time.time() - t_s
        self.train_loss_list.append(epoch_loss)
        return lr, epoch_loss, epoch_loss_info, dt

    def get_multimodal_gt(self):
        """
        return list of tensors of shape [[num_similar, t_pred, NC]]
        """
        data_gen_multi_test = self.dataset_multi_test.iter_generator(step=self.cfg.t_his)
        traj_gt_arr = []
        data_group = []
        num_mult = []

        for data, multi_traj in data_gen_multi_test:
            data_group.append(data)
            # 形状：(K, T_pred, (J-1)*3) → 3维，与gt维度完全一致
            traj_gt_arr.append(
                multi_traj[:, self.cfg.t_his:, 1:, :].reshape(len(multi_traj), self.cfg.t_pred, -1)
            )
            num_mult.append(len(multi_traj))

        data_group = np.concatenate(data_group, axis=0)
        all_data = data_group[..., 1:, :].reshape(data_group.shape[0], data_group.shape[1], -1)
        gt_group = all_data[:, self.cfg.t_his:, :]

        return traj_gt_arr, gt_group, data_group

    def get_prediction(self, data, sample_num, uncond, use_ema=True, concat_hist=False):
        """
        data: [batch_size, total_len, num_joints=17, 3]
        act:  [batch_size]
        sample_num: how many samples to generate for one data entry
        """
        data = torch.from_numpy(data).to(self.device).to(self.dtype)
        if data.dim() == 3:
            data = data.unsqueeze(0)
        traj = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)  # [b, t_total, 16x3]

        # process x_0_history: [b*sample_num, t_pred, nc]
        x_0_history = torch.repeat_interleave(traj[:, :-self.output_n, :], sample_num, dim=0)
        total_sample_num = x_0_history.shape[0]
        Y = self.model.sample(x_0_history, None, batch_size=total_sample_num, clip_denoised=False,
                              uncond=uncond)  # [b*sample_num, t_pred, nc]

        if concat_hist:
            Y = torch.cat((x_0_history, Y), dim=1)
        Y = Y.contiguous()

        return Y

    @torch.no_grad()
    def compute_stats(self):
        """
        return: dic [stat_name, stat_val] NOTE: val.avg is standard
        """
        self.model.eval()

        def get_gt(data, input_n):
            data = torch.from_numpy(data).to(self.device).to(self.dtype)
            if data.dim() == 3:
                data = data.unsqueeze(0)
            gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)

            return gt[:, input_n:, :]

        def get_hist(data, input_n, n_frame=3):
            data = torch.from_numpy(data).to(self.device).to(self.dtype)
            if data.dim() == 3:
                data = data.unsqueeze(0)
            h = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
            return h[:, max(input_n - n_frame, 0):input_n, :]

        # all quantitative results in paper
        stats_func = {'APD': APD, 'ADE': ADE, 'FDE': FDE, 'MMADE': MMADE, 'MMFDE': MMFDE}
        angle_keys = ['RKnee', 'LKnee', 'LElbow', 'RElbow']
        scalar_names = list(stats_func.keys()) + [
            'whole_body_ade', 'whole_body_fde', 'whole_body_ble', 'whole_body_blf',
            'jitter', 'jitter_all', 'jitter_gt', 'accel', 'accel_gt', 'accel_error',
        ] + [f'angle_{m}_{k}' for m in ('ade', 'fde') for k in angle_keys]
        vector_names = ['joint_ade', 'joint_fde', 'time_ade', 'joint_time_error',
                        'bone_ade', 'bone_fde', 'bone_time_error',
                        'joint_jitter', 'joint_accel_error']
        stats_names = scalar_names + vector_names
        stats_meter = {x: AverageMeterTorch() for x in stats_names}

        fps = get_fps(self.cfg.dataset, self.cfg)

        counter = 0
        all_pred = []
        for i, data in enumerate(self.eval_data_group):
            t = time.time()
            gt = get_gt(data, self.input_n).to(self.device).to(self.dtype)  # [b, t_pred, NC]
            hist = get_hist(data, self.input_n).to(self.device).to(self.dtype)
            pred = self.get_prediction(data, sample_num=self.cfg.eval_sample_num, uncond=True,
                                       concat_hist=False).detach()

            pred = rearrange(pred, '(b s) ... -> b s ...', b=gt.shape[0])
            all_pred.append(pred)
            gt_multi = torch.tensor(self.multimodal_traj[i])
            gt_multi = [t_.to(self.device).to(self.dtype) for t_ in gt_multi]

            for stats in stats_func:
                val = stats_func[stats](pred, gt, gt_multi)
                stats_meter[stats].update(val)

            detail = compute_detailed_metrics(pred, gt, dataset_name=self.cfg.dataset,
                                              fps=fps, hist=hist)
            for k, v in detail.items():
                if k in stats_meter:
                    stats_meter[k].update(v)

            counter += gt.shape[0]

            print('-' * 80)
            print('Num in multi_GT: ', len(gt_multi))
            for stats in scalar_names:
                str_stats = f'{counter - gt.shape[0]:04d} {stats:<18}: ' + f'({stats_meter[stats].avg:.4f})'
                print(str_stats)
            print('eval time: ', time.time() - t)

        self.print_detailed_stats(stats_meter)

        all_pred = torch.cat(all_pred, dim=0).cpu().numpy()
        pred_path = os.path.join(self.cfg.result_dir, self.cfg.dataset + "_" + self.cfg.model_id + "_pred.npy")
        np.save(pred_path, all_pred)

        return stats_meter

    def print_detailed_stats(self, stats_meter):
        joint_ade = stats_meter['joint_ade'].avg
        n_joint = joint_ade.shape[0]
        joint_names = get_joint_names(self.cfg.dataset, n_joint)

        for title, key in (('Joint ADE', 'joint_ade'),
                           ('Joint FDE', 'joint_fde'),
                           ('Joint Jitter (unit/s^3)', 'joint_jitter'),
                           ('Joint Accel Error (unit/s^2)', 'joint_accel_error')):
            print('=' * 80)
            print(title)
            print('=' * 80)
            for name, err in zip(joint_names, stats_meter[key].avg.cpu().numpy()):
                print(f'{name:<15}: {err:.6f}')

        err_np = joint_ade.cpu().numpy()
        idx = np.argsort(err_np)[::-1]
        print('=' * 80)
        print('Worst Joint ADE')
        print('=' * 80)
        for k in range(min(5, len(idx))):
            print(f'{joint_names[idx[k]]:<15}: {err_np[idx[k]]:.6f}')

        bone_ade = stats_meter['bone_ade'].avg.cpu().numpy()
        bone_names = get_bone_names(self.cfg.dataset, n_joint)
        if len(bone_names) != len(bone_ade):
            bone_names = [f'B{i}' for i in range(len(bone_ade))]
        print('=' * 80)
        print('Bone Length ADE')
        print('=' * 80)
        for name, err in zip(bone_names, bone_ade):
            print(f'{name:<24}: {err:.6f}')

        print('=' * 80)
        print('Joint Angle ADE (degree)')
        print('=' * 80)
        for k in ['RKnee', 'LKnee', 'LElbow', 'RElbow']:
            key = f'angle_ade_{k}'
            if key in stats_meter:
                print(f'{k:<10}: {stats_meter[key].avg:.4f}')

        print('=' * 80)
        print('Smoothness')
        print('=' * 80)
        print(f"jitter(best)={stats_meter['jitter'].avg:.4f}  "
              f"jitter(all K)={stats_meter['jitter_all'].avg:.4f}  "
              f"jitter(GT)={stats_meter['jitter_gt'].avg:.4f}")
        print(f"accel={stats_meter['accel'].avg:.4f}  "
              f"accel(GT)={stats_meter['accel_gt'].avg:.4f}  "
              f"accel_error={stats_meter['accel_error'].avg:.4f}")

        print('=' * 80)
        print('Time ADE')
        print('=' * 80)
        for t, err in enumerate(stats_meter['time_ade'].avg.cpu().numpy()):
            print(f't={t + 1:03d}: {err:.6f}')

    def evaluation(self):
        """NOTE: can be only called once"""
        stats_dic = self.compute_stats()
        return {x: y.avg for x, y in stats_dic.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="ap3d")
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--run_dir', type=str, default=None,
                        help='existing results/<cfg>_<timestamp> dir holding the checkpoint to '
                             'evaluate; required with --test. Test outputs (eval_stats.csv, '
                             'pred.npy) are written back into this same directory.')
    args = parser.parse_args()

    if args.test and args.run_dir is None:
        raise ValueError('--run_dir is required with --test '
                          '(path to the results/<cfg>_<timestamp> directory to evaluate)')

    """setup"""
    cfg = Config(args.cfg, test=args.test, run_dir=args.run_dir)
    set_global_seed(args.seed)
    dtype = torch.float32 if cfg.dtype == 'float32' else torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    """parameter"""
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    node_n = cfg.node_n

    """data"""
    dataset_multi_test = None
    if cfg.dataset == 'h36m':
        dataset_cls = DatasetH36M
    elif cfg.dataset == 'amass':
        dataset_cls = DatasetAMASS
    elif cfg.dataset == 'ap3d':
        dataset_cls = DatasetAP3D
        dataset_multi_test = DatasetAP3D_multi
    elif cfg.dataset == 'ap':
        dataset_cls = DatasetAthleticsPose
        dataset_multi_test = DatasetAthleticsPose_multi
    elif cfg.dataset == 'wp':
        dataset_cls = DatasetWP
        dataset_multi_test = DatasetWP_multi

    action = 'all'

    """loss weight"""
    loss_weights = generate_loss_weight(cfg)  # [t_all/r_pred, NxC]

    """model"""
    model = get_model(cfg).to(dtype).to(device)
    diffuser = GaussianDiffusion(
        model=model,
        cfg=cfg,
        future_motion_size=(t_pred, node_n),  # [T_pred, N*C=num_nodes]
        timesteps=cfg.diffuse_steps,
        loss_type=cfg.loss_type,
        objective=cfg.objective,
        beta_schedule=cfg.beta_schedule,
        history_weight=cfg.history_weight,
        future_weight=cfg.future_weight,
        st_loss_weight=loss_weights,
    ).to(dtype).to(device)

    """trainer"""
    trainer = TrainerCustom(
        dataset=dataset_cls,
        dataset_multi=dataset_multi_test,
        diffusion_model=diffuser,
        train_batch_size=cfg.batch_size,
        train_lr=cfg.train_lr,
        weight_decay=cfg.weight_decay,
        actions=action,
        cfg=cfg,
    )

    start_epoch = 0
    print(">>> model on:", device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # For testing only
    if args.test:
        file_name = 'ckpt_' + cfg.id + '.pth.tar'
        trainer.load(os.path.join(cfg.model_dir, file_name))
    else:
        # For continuous training
        if args.load:
            file_name = 'ckpt_' + cfg.id + '.pth.tar'
            trainer.load(os.path.join(cfg.model_dir, file_name))
            start_epoch = trainer.epoch

        # Training
        for epoch in range(start_epoch, cfg.train_epoch):
            ret_log = np.array([epoch + 1])
            head = np.array(['epoch'])
            lr, epoch_loss, epoch_loss_info, dt = trainer.train()

            print(">>> epoch: ", epoch)

            ret_log = np.append(ret_log, [lr, dt, epoch_loss])
            head = np.append(head, ['lr', 'dt', 't_l'])

            for key, value in epoch_loss_info.items():
                head = np.append(head, key)
                ret_log = np.append(ret_log, value)

            # update log file and save checkpoint
            is_create = False
            if not args.load:
                if epoch == 0:
                    is_create = True
            save_csv_log(cfg, head, ret_log, is_create, file_name=cfg.id + '_log')

            # checkpoint, info saving
            file_name = 'ckpt_' + cfg.id + '.pth.tar'
            save_ckpt(cfg, trainer, file_name=file_name)

    print('Compute final stats...')
    stats = trainer.evaluation()
    print(stats)

    with open('%s/eval_stats.csv' % (cfg.result_dir), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, stats.keys())
        writer.writeheader()
        writer.writerow(stats)
    print('Done.')


if __name__ == '__main__':
    main()