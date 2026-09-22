import copy
import time

import numpy as np
import torch
from torch import optim, nn

from utils.visualization import render_animation
from models.transformer import EMA
from utils import *
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator


class Trainer:
    def __init__(self,
                 model,
                 diffusion,
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

        self.model = model
        self.diffusion = diffusion
        self.dataset = dataset
        self.multimodal_dict = multimodal_dict
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger

        self.iter = 0

        self.lrs = []

        self.resume = True

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False).cuda()
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None

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

    def before_train(self):
        # torch.autograd.set_detect_anomaly(True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        if self.cfg.frame_mask:
            self.criterion = nn.MSELoss(reduction='sum')
        else:
            self.criterion = nn.MSELoss()

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
            self.optimizer = optim.Adam([{"params": self.model.parameters(), "initial_lr": self.cfg.lr}],
                                        lr=last_lr)

        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,
                                                           gamma=self.cfg.gamma, last_epoch=self.iter)
        self.val_min_loss = MinMeter()

    def before_train_step(self):
        self.model.train()
        self.generator_train = self.dataset['train'].sampling_generator(num_samples=self.cfg.num_data_sample,
                                                                        batch_size=self.cfg.batch_size, aug=False)
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):
        train_n_pre = self.cfg.n_pre + 5
        for traj_np, mask in self.generator_train:
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                """
                if not self.cfg.iTransformer:
                    traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                else:
                    # N3, t_his + t_pre, joints - 1
                    traj_np = traj_np[..., 1, :].transpose(0, 3, 1, 2).reshape(
                        [traj_np.shape[0] * 3, self.cfg.t_his + self.cfg.t_pred, -1])
                """
                if self.cfg.dataset == 'assemble' or not self.cfg.remove_root:
                    traj_np = traj_np.reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                else:
                    traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])

                traj = torch.tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                vel_acc_pad = None
                if self.cfg.residual_data:  # 使用速度作为输入
                    # res_traj = torch.zeros(traj.shape, device=self.cfg.device, dtype=self.cfg.dtype)
                    # res_traj[:, 1:, :] = traj[:, 1:, :] - traj[:, :-1, :]
                    # traj = res_traj

                    # B, T-1, V, 3
                    if np.random.random() > self.cfg.mod_train:
                        vel_acc = None
                    else:
                        vel_acc = cal_vel_acc(traj)
                        vel_acc_pad = padding_vel(vel_acc, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

                traj_dct = traj
                traj_dct_mod = traj_pad

                if self.cfg.random_sample:
                    input_traj = torch.zeros((traj.shape[0], self.cfg.n_pre, traj.shape[-1]))
                    for i in range(traj.shape[0]):
                        a = np.arange(traj.shape[1])
                        np.random.shuffle(a)
                        a = np.sort(a[:self.cfg.n_pre])
                        input_traj[i, :, :] = traj[i, a, :]

                if self.cfg.use_dct:
                    traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                    traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                    if np.random.random() > self.cfg.mod_train:
                        traj_dct_mod = None
                    input_traj = traj_dct
                else:
                    input_traj = traj
                    if np.random.random() > self.cfg.mod_train:
                        traj_dct_mod = None

            # train
            """
            if not self.cfg.iTransfomer:
                t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
            else:
                t = self.diffusion.sample_timesteps(traj.shape[0] // 3).to(self.cfg.device)
                t = torch.repeat_interleave(t, 3, dim=0)
            """
            t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)

            x_t, noise = self.diffusion.noise_motion(input_traj, t)

            start_time = time.time()
            frame_mask = torch.tensor(0.0).cuda().detach()
            joint_mask = torch.tensor(0.0).cuda().detach()
            if self.cfg.frame_mask:
                # 随机掩膜，选择被加噪声的帧
                frame_mask = torch.ones((x_t.shape[0], self.cfg.n_pre, 1)).cuda().detach()
                frame_mask = torch.dropout(frame_mask, 0.8, True)
                frame_mask[frame_mask > 0] = 1.0
            if self.cfg.joint_mask:
                joint_mask = torch.ones((x_t.shape[0], self.cfg.n_pre, x_t.shape[-1] // 3)).cuda().detach()
                joint_mask = torch.dropout(joint_mask, 0.8, True).repeat_interleave(3, -1)
                joint_mask[joint_mask > 0] = 1.0

            mask = frame_mask.mul(joint_mask).cuda().detach()
            x_t = input_traj.mul(mask) + x_t.mul(1.0 - mask)

            end_time1 = time.time()
            # print("生成Mask耗时:{:.5f}秒".format(end_time1 - start_time))

            predicted_noise = self.model(x_t, t, mod=traj_dct_mod)

            end_time2 = time.time()
            # print("模型计算耗时:{:.5f}秒".format(end_time2 - end_time1))

            if self.cfg.frame_mask or self.cfg.joint_mask:
                predicted_noise = predicted_noise.mul(1.0 - mask)
                noise = noise.mul(1.0 - mask)
                loss = self.criterion(predicted_noise, noise) / torch.count_nonzero(1.0 - mask)
            else:
                loss = self.criterion(predicted_noise, noise)

            end_time3 = time.time()
            # print("loss计算耗时:{:.5f}秒".format(end_time3 - end_time2))

            self.optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():  # 反向传播时：在求导时开启侦测
            loss.backward()
            self.optimizer.step()
            end_time4 = time.time()
            # print("反向传播耗时:{:.5f}秒".format(end_time4 - end_time3))

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

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np, mask

    def after_train_step(self):
        self.lr_scheduler.step()
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} lr: {:.5f}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.train_losses.avg,
                                                                            self.lrs[-1]))
        if self.iter % self.cfg.save_gif_interval == 0:
            pose_gen = pose_generator(self.dataset['train'], self.model, self.diffusion, self.cfg, mode='gif')
            render_animation(self.dataset['train'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'training_{self.iter}.gif'))

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
                                                                     batch_size=self.cfg.batch_size)
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
                vel_acc_pad = None
                if self.cfg.residual_data:  # 使用速度作为输入
                    # res_traj = torch.zeros(traj.shape, device=self.cfg.device, dtype=self.cfg.dtype)
                    # res_traj[:, 1:, :] = traj[:, 1:, :] - traj[:, :-1, :]
                    # traj = res_traj

                    # B, T-1, V, 3
                    if np.random.random() > self.cfg.mod_train:
                        vel_acc_pad = None
                    else:
                        vel_acc = cal_vel_acc(traj)
                        vel_acc_pad = padding_vel(vel_acc, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)  #
                # [n_pre × (t_his + t_pre)] matmul [(t_his + t_pre) × 3 * (joints - 1)]
                traj_dct = traj
                traj_dct_mod = traj_pad

                if self.cfg.use_dct:
                    traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                    traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                    if np.random.random() > self.cfg.mod_train:
                        traj_dct_mod = None
                    input_traj = traj_dct
                else:
                    if np.random.random() > self.cfg.mod_train:
                        traj_dct_mod = None
                    input_traj = traj

                t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
                x_t, noise = self.diffusion.noise_motion(input_traj, t)

                predicted_noise = self.model(x_t, t, mod=traj_dct_mod)

                if self.cfg.frame_mask or self.cfg.joint_mask:
                    loss = self.criterion(predicted_noise, noise) / noise.numel()
                else:
                    loss = self.criterion(predicted_noise, noise)

                self.val_losses.update(loss.item())
                self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np, input_traj

    def after_val_step(self):
        self.val_min_loss.update(self.iter, self.val_losses.avg)
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.val_losses.avg))
        self.logger.info('====> Min Val Loss: {} Epoch: {}'.format(self.val_min_loss.min_loss,
                                                                   self.val_min_loss.min_iter))
        if self.iter % self.cfg.save_gif_interval == 0:
            if self.cfg.ema is True:
                pose_gen = pose_generator(self.dataset['test'], self.ema_model, self.diffusion, self.cfg, mode='gif')
            else:
                pose_gen = pose_generator(self.dataset['test'], self.model, self.diffusion, self.cfg, mode='gif')
            render_animation(self.dataset['test'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'val_{self.iter}.gif'))

        if self.cfg.save_model_interval > 0 and (self.iter + 1) % self.cfg.save_model_interval == 0:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))

        if (self.iter + 1) >= 400 and self.iter == self.val_min_loss.min_iter:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))
            self.logger.info('====> Save Current Min Val Loss Epoch: {}'.format(self.val_min_loss.min_iter))

        if self.iter % self.cfg.save_metrics_interval == 0 and self.iter != 0:
            if self.cfg.ema is True:
                compute_stats(self.diffusion, self.multimodal_dict, self.ema_model, self.logger, self.cfg)
            else:
                compute_stats(self.diffusion, self.multimodal_dict, self.model, self.logger, self.cfg)
