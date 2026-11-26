import numpy as np
import torch
from torch import tensor
from utils import post_process
from utils.script import sample_preprocessing


def pose_generator(data_set, model_select, generator_model, cfg, mode=None, nrow=1):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    j = None
    while True:
        poses = {}
        draw_order_indicator = -1
        for k in range(0, nrow):
            if mode == 'pred':
                sampling_generator = data_set.sampling_generator(num_samples=256, batch_size=256, aug=False)
                data, _ = next(sampling_generator)
            elif mode == 'gif':
                data = data_set.sample()
            else:
                raise NotImplementedError(f"unknown pose generator mode: {mode}")

            # gt (root joint at the origin)
            gt = data[0].copy()
            gt[:, :1, :] = 0
            data[:, :, :1, :] = 0

            if draw_order_indicator == -1:
                poses['context'] = gt
                poses['gt'] = gt
            else:
                poses[f'HumanMAC_{draw_order_indicator + 1}'] = gt
                poses[f'HumanMAC_{draw_order_indicator + 2}'] = gt

            gt = np.expand_dims(gt, axis=0)
            traj_np = gt[..., 1:, :].reshape([gt.shape[0], cfg.t_his + cfg.t_pred, -1])

            traj = tensor(traj_np, device=cfg.device, dtype=cfg.dtype)
            mode_dict, traj_dct, traj_dct_cond, traj_pad = sample_preprocessing(traj, cfg, mode=mode)

            generator_model.update_sampling_model(model_select)
            sampled_motion = generator_model.sample_fm(mode_dict, traj_dct, traj_dct_cond, traj_pad, pre_inject=True)

            if cfg.use_dct:
                traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
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
