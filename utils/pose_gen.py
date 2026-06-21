import ptwt
import torch
from torch import tensor
from utils import *
from utils.script import sample_preprocessing
from utils.evaluation import *


def pose_generator(data_set, model_select, generator_model, cfg, mode=None,
                   action=None, nrow=1, encoder_select=None):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    traj_np = None
    j = None
    while True:
        poses = {}
        draw_order_indicator = -1
        for k in range(0, nrow):
            if mode == 'switch':
                data = data_set.sample_all_action()
            elif mode == 'pred':
                if cfg.dataset == 'h36m' or cfg.dataset == 'humaneva' or cfg.dataset == 'assemble':
                    data = data_set.sample_iter_action(action, cfg.dataset)
                else:
                    data = data_set.sample()
            elif mode == 'gif' or 'fix' in mode or mode == 'gif_regression':
                data = data_set.sample()
            elif mode == 'zero_shot':
                data = data_set[np.random.randint(0, data_set.shape[0])].copy()
                data = np.expand_dims(data, axis=0)
            else:
                raise NotImplementedError(f"unknown pose generator mode: {mode}")

            # gt
            if cfg.dataset == 'assemble' or not cfg.remove_root:
                gt = data.copy()
            else:
                gt = data[0].copy()
                gt[:, :1, :] = 0
                data[:, :, :1, :] = 0

            if mode == 'switch':
                poses = {}
                traj_np = data[..., 1:, :].reshape([data.shape[0], cfg.t_his + cfg.t_pred, -1])

            elif mode == 'pred' or mode == 'gif' or 'fix' in mode or mode == 'zero_shot' or mode == 'gif_regression':
                if draw_order_indicator == -1:
                    poses['context'] = gt
                    poses['gt'] = gt
                else:
                    poses[f'HumanMAC_{draw_order_indicator + 1}'] = gt
                    poses[f'HumanMAC_{draw_order_indicator + 2}'] = gt

                gt = np.expand_dims(gt, axis=0)

                if cfg.dataset == 'assemble' or not cfg.remove_root:
                    traj_np = gt[..., :, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred, -1])
                else:
                    traj_np = gt[..., 1:, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred, -1])

            ori_traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)
            traj = ori_traj.clone()
            mode_dict, traj_dct, traj_dct_cond, vel_acc_pad, traj_pad = sample_preprocessing(traj, cfg, mode=mode)

            sampled_motion = generator_model.sample_fm(mode_dict, traj_dct, traj_dct_cond, traj_pad,
                                                                   pre_inject=True)

            if cfg.b_frequency_transform:
                if cfg.use_dct:
                    if cfg.mode != 'fine_tune':
                        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
                    else:
                        traj_est = sampled_motion

                elif cfg.use_dwt:
                    inputs = []
                    for i in range(0, len(cfg.dwt_lens)):
                        inputs.append(torch.zeros(sampled_motion.shape[0], cfg.dwt_lens[i], sampled_motion.shape[2], device=cfg.device))

                    if cfg.dwt_lens[0] >= cfg.n_pre: # 41 > 30
                        inputs[0][:, :cfg.n_pre, :] = sampled_motion.clone()
                    else:                                       # 21 < 30
                        inputs[0][:, :, :] = sampled_motion[:, :cfg.dwt_lens[0], :].clone()
                        diff_len = cfg.n_pre - cfg.dwt_lens[0]
                        inputs[1][:, :diff_len, :] = sampled_motion[:, cfg.dwt_lens[0]:, :].clone()
                    traj_est = ptwt.waverec(inputs, wavelet=cfg.dwt_wave, axis=1)[:, :cfg.t_total, :]

                elif cfg.use_fft:
                    zeros = torch.zeros((sampled_motion.shape[0], cfg.t_total - cfg.n_pre, sampled_motion.shape[2]),
                                        device=cfg.device, dtype=torch.float32)
                    dft_coeffs = torch.cat((sampled_motion, zeros), dim=1)
                    traj_est = torch.fft.ifft(dft_coeffs, axis=1).real
            else:
                traj_est = sampled_motion

            traj_est = traj_est.cpu().numpy()
            traj_est = post_process(traj_est, cfg)

            if k == 0:
                for j in range(traj_est.shape[0]):
                    poses[f'{j}'] = traj_est[j]
            else:
                for j in range(traj_est.shape[0]):
                    poses[f'{j + draw_order_indicator + 2 + 1}'] = traj_est[j]

            if draw_order_indicator == -1:
                draw_order_indicator = j
            else:
                draw_order_indicator = j + draw_order_indicator + 2 + 1

        yield poses
