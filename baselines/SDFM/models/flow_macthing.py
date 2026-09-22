import copy
import torch
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from utils import *
import numpy as np
import math
from copy import deepcopy
import pywt
import ptwt


Heun2_butcher_tableu = [[0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1 / 2, 1 / 2]]


class SamplingModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, cfg_scale: float = None, dct_mod: torch.Tensor = None,
                traj_his = None, b_parallel=False):
        module = (
            self.model.module
            if isinstance(self.model, DistributedDataParallel)
            else self.model
        )

        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale == 0.0 or cfg_scale is None:
            with torch.no_grad():
                result = self.model(x, t, traj_his=None)
        elif cfg_scale == 1.0:
            with torch.no_grad():
                result = self.model(x, t, dct_mod, traj_his=traj_his)
        else:
            with torch.no_grad():
                B = x.shape[0]
                if not b_parallel:
                    conditional = self.model(x, t, dct_mod, traj_his=traj_his)
                    condition_free = self.model(x, t, traj_his=None)
                    result = cfg_scale * conditional + (1.0 - cfg_scale) * condition_free
                else:
                    x_ = x.clone()
                    x_double = torch.concatenate((x, x_), dim=0)
                    t_ = t.clone()
                    t_double = torch.concatenate((t, t_), dim=0)
                    dct_mod_zero = torch.zeros_like(dct_mod)
                    dct_mod_double = torch.concatenate((dct_mod, dct_mod_zero), dim=0)
                    y = self.model(x_double, t_double, dct_mod_double)
                    result = cfg_scale * y[:B, :, :] + (1 - cfg_scale) * y[B:, :, :]

        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


class FlowMatchingCustom:
    def __init__(self, net_model, cfg, refiner=None):
        self.sampling_model = SamplingModel(net_model)
        self.cfg = cfg
        self.refiner = refiner

    def update_sampling_model(self, new_model):
        self.sampling_model.model = new_model

    @staticmethod
    def grid_constructor(t, step_size):
        start_time = t[0]
        end_time = t[-1]

        niters = torch.ceil((end_time - start_time) / step_size + 1).item()
        t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
        t_infer[-1] = t[-1]
        return t_infer

    def traj_fusion(self, traj_ori, y_t, z_dct, t, mask):  # 拼接原始历史
        # z_temp = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], z_dct[:, :self.cfg.n_pre])
        if self.cfg.b_frequency_transform:  # 逆变换至空域
            if self.cfg.use_dct:
                y_t_ori = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], y_t[:, :self.cfg.n_pre])
                # y_t_ori = self.cfg.hdct.decode(y_t)

            elif self.cfg.use_dwt:
                inputs = []
                for i in range(0, len(self.cfg.dwt_lens)):
                    inputs.append(torch.zeros(y_t.shape[0], self.cfg.dwt_lens[i], y_t.shape[2],
                                              device=self.cfg.device))

                if self.cfg.dwt_lens[0] >= self.cfg.n_pre:  # dwt_lens[0] 41 >  n_pre 30
                    inputs[0][:, :self.cfg.n_pre, :] = y_t.clone()
                else:                                       # dwt_lens[0] 21 < n_pre 30
                    inputs[0][:, :, :] = y_t[:, :self.cfg.dwt_lens[0], :].clone()
                    diff_len = self.cfg.n_pre - self.cfg.dwt_lens[0]
                    inputs[1][:, :diff_len, :] = y_t[:, self.cfg.dwt_lens[0]:, :].clone()

                y_t_ori = ptwt.waverec(inputs, wavelet=self.cfg.dwt_wave, axis=1)[:, :self.cfg.t_total, :]

            elif self.cfg.use_fft:
                zeros = torch.zeros((y_t.shape[0], self.cfg.t_total - self.cfg.n_pre, y_t.shape[2]),
                                    device=self.cfg.device, dtype=torch.float32)
                dft_coeffs = torch.cat((y_t, zeros), dim=1)
                y_t_ori = torch.fft.ifft(dft_coeffs, axis=1).real
        else:
            y_t_ori = y_t

        # traj_ori_noised = z_temp * (1 - t) + traj_ori * t
        y_mid = torch.mul(mask, traj_ori) + torch.mul((1 - mask), y_t_ori)  # mask拼接原始历史 空域

        if self.cfg.b_frequency_transform:  # 再转频域
            if self.cfg.use_dct:
                y_mid = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], y_mid)  # DCT变换到n_pre帧
                # y_mid = self.cfg.hdct.encode(y_mid)
            elif self.cfg.use_dwt:
                y_mid_frequency_components = ptwt.wavedec(y_mid, wavelet=self.cfg.dwt_wave,
                                                          level=self.cfg.dwt_level, axis=1, mode='constant')

                if y_mid_frequency_components[0].shape[1] > self.cfg.n_pre:  # dwt_lens[0] 41 >  n_pre 30
                    y_mid = y_mid_frequency_components[0][:, :self.cfg.n_pre, :].clone()
                else:  # dwt_lens[0] 21 < n_pre 30
                    len_ca = y_mid_frequency_components[0].shape[1]
                    diff_len = self.cfg.n_pre - len_ca
                    cd_n = y_mid_frequency_components[1][:, :diff_len, :].clone()
                    y_mid = torch.cat((y_mid_frequency_components[0], cd_n), dim=1)
            elif self.cfg.use_fft:
                y_mid = torch.fft.fft(y_mid, dim=1).real
                y_mid = y_mid[:, :self.cfg.n_pre, :]

        return y_mid

    def sample_fm(self, mode_dict, traj_dct, traj_dct_mod, traj_pad, fusion_traj_till=-1.0,
                  mod_till=1.0, pre_inject=False):
        with torch.set_grad_enabled(False):
            if self.cfg.edm_schedule:
                t = get_time_discretization(nfes=self.cfg.ode_options["nfe"])
            else:
                t = torch.tensor([0.0, 1.0], device=self.cfg.device)

            step_size = self.cfg.ode_options["step_size"] if "step_size" in self.cfg.ode_options else None
            assert step_size is not None

            time_grid = self.grid_constructor(t, step_size)
            assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

            z_dct = torch.randn_like(traj_dct).to(traj_dct.device)  # dct field noise
            y0 = z_dct
            # y0 = self.cfg.hdct.mask(y0)

            solution = torch.empty(len(time_grid), *y0.shape, dtype=y0.dtype, device=y0.device)
            solution[0] = y0

            # iDCT变换至时域
            # traj_pad = torch.matmul(self.cfg.idct_m_all[:, :self.cfg.n_pre], traj_dct_mod[:, :self.cfg.n_pre])
            traj_pad = traj_pad
            traj_his = traj_pad[:, :self.cfg.t_his:, :]

            if pre_inject:
                y0 = self.traj_fusion(traj_pad, y0, z_dct, 0.0, mode_dict['mask'])

            for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
                dt = t1 - t0
                if self.cfg.ode_method == 'euler':
                    if t0 <= mod_till:
                        vt = self.sampling_model(y0, t0, cfg_scale=self.cfg.cfg_scale, dct_mod=traj_dct_mod,
                                                      traj_his=traj_his, b_parallel=self.cfg.parallel)
                    else:
                        vt = self.sampling_model(y0, t0, b_parallel=self.cfg.parallel)

                    dy = dt * vt

                    if not self.cfg.remove_root:
                        zero_root = torch.zeros((y0.shape[0], y0.shape[1], 3), device=self.cfg.device, dtype=torch.float32)
                        dy = torch.cat((zero_root, dy), dim=-1)

                    # dy = self.cfg.hdct.mask(dy)
                    y1 = y0 + dy
                    # y1 = self.cfg.hdct.mask(y1)

                    if t0 <= fusion_traj_till:
                        y1 = self.traj_fusion(traj_pad, y1, z_dct, t0, mode_dict['mask'])

                    y0 = y1

                elif self.cfg.ode_method == 'midpoint':
                    half_dt = 0.5 * dt
                    f0 = self.sampling_model(y0, t0, cfg_scale=self.cfg.cfg_scale, dct_mod=traj_dct_mod,
                                             b_parallel=self.cfg.parallel)

                    y_mid = y0 + f0 * half_dt

                    if t0 <= fusion_traj_till:
                        y_mid = self.traj_fusion(traj_pad, y_mid, z_dct, t0, mode_dict['mask'])
                    dy = dt * self.sampling_model(y_mid, t0 + half_dt, cfg_scale=self.cfg.cfg_scale,
                                                  dct_mod=traj_dct_mod, b_parallel=self.cfg.parallel)
                    y1 = y0 + dy

                    if t0 <= fusion_traj_till:
                        y1 = self.traj_fusion(traj_pad, y1, z_dct, t0, mode_dict['mask'])
                    y0 = y1

                elif self.cfg.ode_method == 'heun2':
                    f0 = self.sampling_model(y0, t0, cfg_scale=self.cfg.cfg_scale, dct_mod=traj_dct_mod,
                                             b_parallel=self.cfg.parallel)
                    k1 = f0
                    y_mid = y0 + dt * k1 * Heun2_butcher_tableu[1][1]

                    if t0 <= fusion_traj_till:
                        y_mid = self.traj_fusion(traj_pad, y_mid, z_dct, t0, mode_dict['mask'])

                    k2 = self.sampling_model(y_mid, t0 + dt * Heun2_butcher_tableu[1][0],
                                             cfg_scale=self.cfg.cfg_scale, dct_mod=traj_dct_mod,
                                             b_parallel=self.cfg.parallel)
                    # y1 = y0 + dt * (k1 * Heun2_butcher_tableu[2][1] + k2 * Heun2_butcher_tableu[2][2])

                    if t0 <= fusion_traj_till:
                        y1 = self.traj_fusion(traj_pad, y1, z_dct, t0, mode_dict['mask'])
                    y0 = y1

            sampled_motion = y0
            return sampled_motion

    def sample_res_fm(self, mode_dict, traj_dct, traj_dct_mod, traj_pad, pre_inject=False):
        B, T, _ = traj_dct.shape
        with torch.set_grad_enabled(False):
            if self.cfg.edm_schedule:
                t = get_time_discretization(nfes=self.cfg.ode_options["nfe"])
            else:
                t = torch.tensor([0.0, 1.0], device=self.cfg.device)

            step_size = self.cfg.ode_options["step_size"] if "step_size" in self.cfg.ode_options else None
            assert step_size is not None

            time_grid = self.grid_constructor(t, step_size)
            assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

            y0 = torch.randn_like(traj_dct).to(traj_dct.device)
            traj_his = traj_pad[:, :self.cfg.t_his].reshape(traj_dct.shape[0], self.cfg.t_his, self.cfg.joint_num, -1)
            for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
                dt = t1 - t0
                dy = dt * self.sampling_model(y0, t0, cfg_scale=self.cfg.cfg_scale, dct_mod=traj_his,
                                            traj_his=traj_his, b_parallel=self.cfg.parallel)
                y0 = y0 + dy

        return y0