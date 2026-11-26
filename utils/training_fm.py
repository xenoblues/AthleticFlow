import copy
import os
import time
import numpy as np
import torch
from torch import optim
from adan import Adan
from flow_matching.path import CondOTProbPath

from models.ema import EMA
from utils import AverageMeter, MinMeter, padding_traj
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator
from utils.visualization import render_animation


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    t = 1 / (1 + sigma)
    t = torch.clip(t, min=0.0001, max=1.0)
    return t


class Trainer_fm:
    """Conditional flow matching (optimal-transport path) on the truncated temporal coefficients."""

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

        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None
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
        for self.iter in range(self.iter, self.cfg.num_epoch):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()

        # save last epoch model
        self.save_model(f"{self.iter + 1}")

    def save_model(self, suffix):
        if self.cfg.ema is True:
            torch.save(self.ema_model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_ema_{suffix}.pt"))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{suffix}.pt"))

    def before_train(self):
        self.optimizer = Adan(self.model.parameters(), lr=self.cfg.lr, weight_decay=0.02)
        self.iter = 0

        # linear warm-up (50 epochs) followed by cosine annealing, stepped every iteration
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

    def prepare_batch(self, traj_np):
        """
        (N, t_his + t_pred, joints, 3) -> flow-matching tuple (x_t, u_t, t) and condition.
        The order of the random draws is kept identical to the code that produced the released checkpoints.
        """
        # discard the root joint and combine xyz coordinate
        traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_total, -1])

        traj = torch.tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
        traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)

        traj_dct = traj.clone()
        traj_mod = traj_pad

        if self.cfg.use_dct:
            traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
            traj_pad_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)

            # classifier-free guidance: drop the condition with probability 1 - mod_train
            if np.random.random() > self.cfg.mod_train:
                traj_mod = None
            else:
                traj_mod = traj_pad_dct
            input_traj = torch.randn(traj_dct.shape).to(self.cfg.device)
        else:
            input_traj = torch.randn(traj_pad.shape).to(self.cfg.device)
            if np.random.random() > self.cfg.mod_train:
                traj_mod = None

        if self.cfg.skewed_timesteps:
            t = skewed_timestep_sample(input_traj.shape[0], device=self.cfg.device)
        else:
            t = torch.rand(input_traj.shape[0]).to(self.cfg.device)

        path_sample = self.path.sample(t=t, x_0=input_traj, x_1=traj_dct)
        return path_sample.x_t, path_sample.dx_t, t, traj_mod

    def before_train_step(self):
        self.model.train()
        self.generator_train = self.dataset['train'].sampling_generator(num_samples=self.cfg.num_data_sample,
                                                                        batch_size=self.cfg.batch_size,
                                                                        aug=True)
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):
        for traj_np, mask in self.generator_train:
            with torch.no_grad():
                x_t, u_t, t, traj_mod = self.prepare_batch(traj_np)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                pred_ut = self.model(x_t, t, traj_mod)
                loss = torch.pow(pred_ut.float() - u_t.float(), 2).mean()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]
            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            self.train_losses.update(loss.item())
            self.tb_logger.add_scalar('Loss/train', loss.item(), self.iter)

            del loss, x_t, u_t, traj_mod, traj_np, mask

    def after_train_step(self):
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.4f} Train Loss: {:.8f} lr: {:.8f}'.format(self.iter,
                                                                                time.time() - self.t_s,
                                                                                self.train_losses.avg,
                                                                                self.lrs[-1]))

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
                                                                     batch_size=self.cfg.batch_size * 4)
        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        for traj_np, mask in self.generator_val:
            with torch.no_grad():
                x_t, u_t, t, traj_mod = self.prepare_batch(traj_np)
                pred_ut = self.model(x_t, t, traj_mod)
                loss = torch.pow(pred_ut - u_t, 2).mean()

                self.val_losses.update(loss.item())
                self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)

            del loss, x_t, u_t, traj_mod, traj_np

    def after_val_step(self):
        self.val_min_loss.update(self.iter, self.val_losses.avg)
        self.logger.info('====> Epoch: {} Time: {:.4f} Val Loss: {:.8f}'.format(self.iter, time.time() - self.t_s,
                                                                                self.val_losses.avg))
        self.logger.info(
            '====> Min Val Loss: {} Epoch: {}'.format(self.val_min_loss.min_loss, self.val_min_loss.min_iter))
        eval_model = self.ema_model if self.cfg.ema is True else self.model

        if self.iter % self.cfg.save_gif_interval == 0:
            pose_gen = pose_generator(self.dataset['test'], eval_model, self.generator, self.cfg, mode='gif')
            render_animation(self.dataset['test'].skeleton, pose_gen, ['FM'], self.cfg.t_his, ncol=4,
                             output=os.path.join(self.cfg.gif_dir, f'val_{self.iter}.gif'),
                             dataset_name=self.cfg.dataset)

        if self.cfg.save_model_interval > 0 and (self.iter + 1) % self.cfg.save_model_interval == 0:
            self.save_model(f"{self.iter + 1}")

        if (self.iter + 1) >= 400 and self.iter == self.val_min_loss.min_iter:
            self.save_model(f"{self.iter + 1}")
            self.logger.info('====> Save Current Min Val Loss Epoch: {}'.format(self.val_min_loss.min_iter))

        if self.iter % self.cfg.save_metrics_interval == 0 and self.iter != 0:
            compute_stats(self.generator, self.multimodal_dict, eval_model, self.logger, self.cfg)
