import copy
import time
import numpy as np
import ptwt
import torch
from flow_matching.solver import ODESolver
from pytorch_kinematics.urdf_parser_py.urdf import TransmissionJoint
from sympy.physics.units import frequency
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel

from models import athletic_flow
from utils.visualization import render_animation
from models.transformer import EMA
from utils import *
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator
from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from torch.nn.modules import Module
# from adan import Adan
from models.athletic_flow import *
# from models.StochasticDistalOrientationResidual import *


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    t = 1 / (1 + sigma)
    t = torch.clip(t, min=0.0001, max=1.0)
    return t

def second_diff(x):
    return x[:,2:] - 2*x[:,1:-1] + x[:,:-2]

class Trainer_fm:
    def __init__(self,
                 model,
                 generator,
                 dataset,
                 cfg,
                 multimodal_dict,
                 logger,
                 tb_logger):
        super().__init__()

        self.generator_val = None
        self.val_losses = None
        self.t_s = None
        self.train_losses = None
        self.val_min_loss = None

        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None
        self.his_mask = None
        self.scaler = None

        self.model = model
        self.generator = generator
        self.dataset = dataset
        self.multimodal_dict = multimodal_dict
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger
        self.iter = 0
        self.lrs = []
        self.resume = False

        if self.cfg.ema is True:
            self.ema = EMA(0.999)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False).cuda()
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None

        # flow matching
        self.path = CondOTProbPath()

    def loop(self):
        self.before_train()
        if self.iter == -1:
            self.iter = 0
        for self.iter in range(self.iter, self.cfg.num_epoch):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()

        # save last epoch model
        if self.cfg.ema is True:
            torch.save(self.ema_model.state_dict(),
                       os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))

    def before_train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        # self.optimizer = Adan(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.02)

        self.iter = -1
        if self.cfg.resume:
            loaded_ckpt = torch.load(self.cfg.ckpt_path, map_location='cuda')
            self.model.load_state_dict(loaded_ckpt, strict=False)
            if self.cfg.ckpt_path[-6] == '_':
                self.iter = int(self.cfg.ckpt_path[-5:-3])
            else:
                self.iter = int(self.cfg.ckpt_path[-6:-3])
            milestone = np.asarray(self.cfg.milestone)
            if self.iter > milestone[-1]:
                power = milestone.shape[0]
            else:
                power = int(min(np.argwhere(milestone > self.iter)))
            last_lr = self.cfg.lr * (self.cfg.gamma ** power)
            self.optimizer = optim.AdamW([{"params": self.model.parameters(),
                                                "initial_lr": self.cfg.lr}],
                                                lr=last_lr)

        # self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,
        #                                                    gamma=self.cfg.gamma, last_epoch=self.iter)

        iters_per_epoch = int(self.cfg.num_data_sample // self.cfg.batch_size)
        warmup_iters = int(50 * iters_per_epoch)
        warmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.05, end_factor=1.0,
                                                       total_iters=warmup_iters)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                T_max=int(self.cfg.num_epoch * iters_per_epoch - warmup_iters),
                                                                eta_min=1.e-6)

        self.lr_scheduler = optim.lr_scheduler.SequentialLR(self.optimizer,
                                                            schedulers=[warmup_scheduler, cosine_scheduler],
                                                            milestones=[warmup_iters])
        self.scaler = torch.amp.GradScaler()
        self.val_min_loss = MinMeter()
        self.his_mask = torch.zeros(
            [self.cfg.batch_size, self.cfg.t_his + self.cfg.t_pred, self.dataset['train'].traj_dim]).to(self.cfg.device)
        for i in range(0, self.cfg.t_his):
            self.his_mask[:, i, :] = 1

        self.hdct = self.cfg.hdct.to(self.cfg.device)
        # self.kin = self.cfg.kin
        self.kin = None
        if self.cfg.mode == 'fine_tune':
            loaded_ckpt = torch.load(self.cfg.ckpt_path, map_location='cuda')
            self.model.load_state_dict(loaded_ckpt, strict=False)
            self.refiner_optimizer = optim.AdamW(self.generator.refiner.parameters(), lr=self.cfg.lr)
            self.refiner_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.refiner_optimizer,
                                                                    T_max=int(self.cfg.num_epoch * iters_per_epoch),
                                                                    eta_min=1.e-6, last_epoch=-1)


    def before_train_step(self):

        if self.cfg.mode == 'fine_tune':
            self.generator.refiner.train()
        else:
            self.model.train()
        self.generator_train = self.dataset['train'].sampling_generator(num_samples=self.cfg.num_data_sample,
                                                                        batch_size=self.cfg.batch_size,
                                                                        aug=True)
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):
        for traj_np, mask in self.generator_train:
            # ep_start = time.time()
            with (torch.no_grad()):
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                if self.cfg.dataset == 'assemble' or not self.cfg.remove_root:
                    traj_np = traj_np.reshape([traj_np.shape[0], self.cfg.t_total, -1])
                else:
                    traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_total, -1])

                traj = torch.tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_his = traj[:, :self.cfg.t_his, :]
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

                if self.kin:
                    traj = self.kin.encode(traj)
                    traj_his = self.kin.encode(traj_his)
                    traj_pad = self.kin.encode(traj_pad)

                traj_dct = traj.clone()
                traj_mod = traj_pad
                input_traj = None


                if self.cfg.b_frequency_transform:
                    if self.cfg.use_dct:
                        traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                        traj_pad_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                        # traj_dct = self.hdct.encode(traj)
                        # traj_pad_dct = self.hdct.encode(traj_pad)

                    elif self.cfg.use_dwt:
                        traj_frequency_components = ptwt.wavedec(traj, wavelet=self.cfg.dwt_wave,
                                                                 level=self.cfg.dwt_level, axis=1, mode='constant')
                        traj_pad_frequency_components = ptwt.wavedec(traj_pad, wavelet=self.cfg.dwt_wave,
                                                                     level=self.cfg.dwt_level, axis=1, mode='constant')

                        if traj_frequency_components[0].shape[1] > self.cfg.n_pre:  # 41 > n_pre 30
                            traj_dct = traj_frequency_components[0][:, :self.cfg.n_pre, :]
                            traj_pad_dct = traj_pad_frequency_components[0][:, :self.cfg.n_pre, :]

                        else:
                            len_ca = traj_frequency_components[0].shape[1]  # 20
                            diff_len = self.cfg.n_pre - len_ca  # 10
                            traj_cd_n = traj_frequency_components[1][:, :diff_len, :]
                            traj_dct = torch.cat((traj_frequency_components[0], traj_cd_n), dim=1)
                            traj_pad_cd_n = traj_frequency_components[1][:, :diff_len, :]
                            traj_pad_dct = torch.cat((traj_pad_frequency_components[0], traj_pad_cd_n), dim=1)

                    elif self.cfg.use_fft:
                        traj_dct = torch.fft.fft(traj, dim=1).real
                        traj_dct = traj_dct[:, :self.cfg.n_pre, :]
                        traj_pad_dct = torch.fft.fft(traj_pad, dim=1).real
                        traj_pad_dct = traj_pad_dct[:, :self.cfg.n_pre, :]

                    if np.random.random() > self.cfg.mod_train:
                        if self.cfg.parallel:
                            traj_mod = torch.zeros_like(traj_pad_dct).to(self.cfg.device)
                            traj_his[...] = None
                        else:
                            traj_mod = None
                            traj_his = None

                    else:
                        traj_mod = traj_pad_dct
                    noise = torch.randn(traj_dct.shape).to(self.cfg.device)
                    # if self.cfg.use_dct:
                    #     noise = self.hdct.mask(noise)

                    # traj_pad_dct_noised = traj_pad_dct + noise
                    # traj_pad_noised = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], traj_pad_dct_noised[:, :self.cfg.n_pre])
                    # input_traj = torch.mul(self.his_mask, traj) + torch.mul(1 - self.his_mask, traj_pad_noised)
                    # input_traj = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], input_traj)
                    input_traj = noise

                else:
                    noise = torch.randn(traj_pad.shape).to(self.cfg.device)
                    input_traj = noise
                    if np.random.random() > self.cfg.mod_train:
                        if self.cfg.parallel:
                            traj_mod = torch.zeros_like(traj_pad).to(self.cfg.device)
                            traj_his[...] = 0.0
                        else:
                            traj_mod = None
                            traj_his = None

            if self.cfg.skewed_timesteps:
                t = skewed_timestep_sample(input_traj.shape[0], device=self.cfg.device)
            else:
                t = torch.rand(input_traj.shape[0]).to(self.cfg.device)

            path_sample = self.path.sample(t=t, x_0=input_traj, x_1=traj_dct)
            x_t = path_sample.x_t
            # x_t = self.hdct.mask(x_t)
            u_t = path_sample.dx_t
            if not self.cfg.remove_root:
                u_t = u_t[:, :, 3:]

            # t2 = time.time()
            # print("数据处理时间:{:.5f}秒".format(t2 - ep_start))

            if self.cfg.res_fm:
                traj_reshaped = traj.reshape(traj.shape[0], self.cfg.t_total, self.cfg.joint_num, -1).clone()
                traj_feature = traj_reshaped[:, self.cfg.t_his:]
                traj_mod = traj_reshaped[:, :self.cfg.t_his]
                # x_t, u_t, t = sample_training_tuple(traj_feature, traj_mod)
                # u_t = u_t.reshape(traj.shape[0], self.cfg.t_pred, -1)
                if np.random.random() > self.cfg.mod_train:
                    traj_mod = torch.zeros_like(traj_mod)

            if self.cfg.mode == 'train':
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.refiner_optimizer.zero_grad(set_to_none=True)

            with (torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda")):
                if self.cfg.mode == 'fine_tune':
                    self.model.requires_grad_(False)
                    with torch.no_grad():
                        pred_ut, latent = self.model(x_t, t, traj_mod, traj_his=traj_his)
                else:
                    pred_ut = self.model(x_t, t, traj_mod, traj_his=traj_his)

                loss = torch.pow(pred_ut.float() - u_t.float(), 2).mean()
                # pred_ut = self.hdct.mask(pred_ut)
                # u_t = self.hdct.mask(u_t)
                # loss = self.hdct.masked_mse(pred_ut.float(), u_t.float())

                pred_x1 = x_t.float() + pred_ut * (1.0 - t[:, None, None])
                # loss_bio = biomechanics_loss(pred_x1.float(), traj, self.cfg.idct_m_all, self.cfg.n_pre)
                pred_traj = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], pred_x1[:, :self.cfg.n_pre])
                # loss = loss + loss_bio


            # t3 = time.time()
            # print("前向计算耗时:{:.5f}秒".format(t3 - t2))

            # with torch.autograd.detect_anomaly():  # 反向传播时：在求导时开启侦测
            self.scaler.scale(loss).backward()
            if self.cfg.mode == 'train':
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
            else:
                self.scaler.unscale_(self.refiner_optimizer)
                torch.nn.utils.clip_grad_norm_(self.generator.refiner.parameters(), max_norm=1.0)
                self.scaler.step(self.refiner_optimizer)

            self.scaler.update()

            if self.cfg.mode == 'train':
                self.lr_scheduler.step()
            else:
                self.refiner_scheduler.step()

            # t4 = time.time()
            # print("反向传播耗时:{:.5f}秒".format(t4 - t3))
            # print("前向+反向训练耗时:{:.5f}秒".format(t4 - t2))

            # if self.iter >= 5:
            #     for name, params in self.model.named_parameters():
            #         if params.requires_grad:
            #             print("name", name, 'params:', params, "grad:", params.grad)

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]
            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            self.train_losses.update(loss.item())
            self.tb_logger.add_scalar('Loss/train', loss.item(), self.iter)
            end_time5 = time.time()
            # print("epoch耗时:{:.5f}秒".format(end_time5 - start_time))

            del loss, traj, traj_dct, traj_mod, traj_pad, traj_np, mask


    def after_train_step(self):
        if self.cfg.mode == 'train':
            self.lrs.append(self.optimizer.param_groups[0]['lr'])
        else:
            self.lrs.append(self.refiner_optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.4f} Train Loss: {:.8f} lr: {:.8f}'.format(self.iter,
                                                                                time.time() - self.t_s,
                                                                                self.train_losses.avg,
                                                                                self.lrs[-1]))
        # if self.iter % self.cfg.save_gif_interval == 0:
        #     pose_gen = pose_generator(self.dataset['train'], self.model, self.diffusion, self.cfg, mode='gif')
        #     render_animation(self.dataset['train'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his, ncol=4,
        #                      output=os.path.join(self.cfg.gif_dir, f'training_{self.iter}.gif'))

    def before_val_step(self):
        self.model.eval()
        if self.cfg.mode == 'fine_tune':
            self.generator.refiner.eval()

        self.t_s = time.time()
        self.val_losses = AverageMeter()
        self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
                                                                     batch_size=self.cfg.batch_size * 4)
        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        for traj_np, mask in self.generator_val:
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                if self.cfg.dataset == 'assemble' or not self.cfg.remove_root:
                    traj_np = traj_np.reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                else:
                    traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])

                traj = torch.tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_his = traj[:, :self.cfg.t_his, :]
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

                if self.kin:
                    traj = self.kin.encode(traj)
                    traj_his = self.kin.encode(traj_his)
                    traj_pad = self.kin.encode(traj_pad)

                # [n_pre × (t_his + t_pre)] matmul [(t_his + t_pre) × 3 * (joints - 1)]
                traj_dct = traj
                traj_mod = traj_pad

                if self.cfg.b_frequency_transform:
                    if self.cfg.use_dct:
                        traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                        traj_pad_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                        # traj_dct = self.hdct.encode(traj)
                        # traj_pad_dct = self.hdct.encode(traj_pad)

                    elif self.cfg.use_dwt:
                        traj_frequency_components = ptwt.wavedec(traj, wavelet=self.cfg.dwt_wave,
                                                                 level=self.cfg.dwt_level, axis=1, mode='constant')
                        traj_pad_frequency_components = ptwt.wavedec(traj_pad, wavelet=self.cfg.dwt_wave,
                                                                     level=self.cfg.dwt_level, axis=1, mode='constant')

                        if traj_frequency_components[0].shape[1] > self.cfg.n_pre:
                            traj_dct = traj_frequency_components[0][:, :self.cfg.n_pre, :]
                            traj_pad_dct = traj_pad_frequency_components[0][:, :self.cfg.n_pre, :]
                        else:
                            len_ca = traj_frequency_components[0].shape[1]
                            diff_len = self.cfg.n_pre - len_ca
                            traj_cd_n = traj_frequency_components[1][:, :diff_len, :]
                            traj_dct = torch.cat((traj_frequency_components[0], traj_cd_n), dim=1)
                            traj_pad_cd_n = traj_frequency_components[1][:, :diff_len, :]
                            traj_pad_dct = torch.cat((traj_pad_frequency_components[0], traj_pad_cd_n), dim=1)
                    elif self.cfg.use_fft:
                        traj_dct = torch.fft.fft(traj, dim=1).real
                        traj_dct = traj_dct[:, :self.cfg.n_pre, :]
                        traj_pad_dct = torch.fft.fft(traj_pad, dim=1).real
                        traj_pad_dct = traj_pad_dct[:, :self.cfg.n_pre, :]

                    if np.random.random() > self.cfg.mod_train:
                        if self.cfg.parallel:
                            traj_mod = torch.zeros_like(traj_pad).to(self.cfg.device)
                            traj_his[...] = 0.0
                        else:
                            traj_mod = None
                            traj_his = None
                    else:
                        traj_mod = traj_pad_dct

                    noise = torch.randn(traj_dct.shape).to(self.cfg.device)
                    # if self.cfg.use_dct:
                    #     noise = self.hdct.mask(noise)

                    # traj_pad_dct_noised = traj_pad_dct + noise
                    # traj_pad_noised = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], traj_pad_dct_noised[:, :self.cfg.n_pre])
                    # input_traj = torch.mul(self.his_mask, traj) + torch.mul(1 - self.his_mask, traj_pad_noised)
                    # input_traj = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], input_traj)

                    input_traj = noise
                else:
                    noise = torch.randn(traj_dct.shape).to(self.cfg.device)
                    input_traj = noise
                    if np.random.random() > self.cfg.mod_train:
                        if self.cfg.parallel:
                            traj_mod = torch.zeros_like(traj_pad).to(self.cfg.device)
                            traj_his[...] = 0.0
                        else:
                            traj_mod = None
                            traj_his = None

                if self.cfg.skewed_timesteps:
                    t = skewed_timestep_sample(input_traj.shape[0], device=self.cfg.device)
                else:
                    t = torch.rand(input_traj.shape[0]).to(self.cfg.device)

                path_sample = self.path.sample(t=t, x_0=input_traj, x_1=traj_dct)

                x_t = path_sample.x_t
                # x_t = self.hdct.mask(x_t)
                u_t = path_sample.dx_t
                if not self.cfg.remove_root:
                    u_t = u_t[:, :, 3:]

                if self.cfg.res_fm:
                    traj_reshaped = traj.reshape(traj.shape[0], self.cfg.t_total, self.cfg.joint_num, -1).clone()
                    traj_feature = traj_reshaped[:, self.cfg.t_his:]
                    traj_mod = traj_reshaped[:, :self.cfg.t_his]
                    # x_t, u_t, t = sample_training_tuple(traj_feature, traj_mod)
                    # u_t = u_t.reshape(traj.shape[0], self.cfg.t_pred, -1)
                    if np.random.random() > self.cfg.mod_train:
                        traj_mod = torch.zeros_like(traj_mod)

                if self.cfg.mode == 'fine_tune':
                    self.model.requires_grad_(False)
                    with torch.no_grad():
                        pred_ut = self.model(x_t, t, traj_mod, traj_his=traj_his)
                else:
                    pred_ut = self.model(x_t, t, traj_mod, traj_his=traj_his)
                # pred_ut, v0, r1, r2 = self.model(x_t, t, traj_mod, traj_his=traj_his)
                loss = torch.pow(pred_ut - u_t, 2).mean()
                # pred_ut = self.hdct.mask(pred_ut)
                # u_t = self.hdct.mask(u_t)
                # loss = self.hdct.masked_mse(pred_ut.float(), u_t.float()


                self.val_losses.update(loss.item())
                self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_mod, traj_pad, traj_np, input_traj

    def after_val_step(self):
        self.val_min_loss.update(self.iter, self.val_losses.avg)
        self.logger.info('====> Epoch: {} Time: {:.4f} Val Loss: {:.8f}'.format(self.iter, time.time() - self.t_s,
                                                                                self.val_losses.avg))
        self.logger.info(
            '====> Min Val Loss: {} Epoch: {}'.format(self.val_min_loss.min_loss, self.val_min_loss.min_iter))
        if self.iter % self.cfg.save_gif_interval == 0:
            if self.cfg.ema is True:
                pose_gen = pose_generator(self.dataset['test'], self.ema_model, self.generator, self.cfg, mode='gif')
            else:
                pose_gen = pose_generator(self.dataset['test'], self.model, self.generator, self.cfg, mode='gif')
            render_animation(self.dataset['test'].skeleton, pose_gen, ['FM'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'val_{self.iter}.gif'),
                             dataset_name=self.cfg.dataset)

        if self.cfg.save_model_interval > 0 and (self.iter + 1) % self.cfg.save_model_interval == 0:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))

            if self.cfg.mode == 'fine_tune':
                torch.save(self.generator.refiner.state_dict(), os.path.join(self.cfg.model_path, f"refiner_ckpt_{self.iter + 1}.pt"))

        if (self.iter + 1) >= 400 and self.iter == self.val_min_loss.min_iter:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))
            self.logger.info('====> Save Current Min Val Loss Epoch: {}'.format(self.val_min_loss.min_iter))

        if self.iter % self.cfg.save_metrics_interval == 0 and self.iter != 0:
            if self.cfg.ema is True:
                compute_stats(self.generator, self.multimodal_dict, self.ema_model, self.logger, self.cfg)
            else:
                compute_stats(self.generator, self.multimodal_dict, self.model, self.logger, self.cfg)
