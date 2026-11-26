import copy

import numpy as np
import torch

from data_loader.dataset_ap3d import DatasetAP3D, DatasetAP3D_multi
from data_loader.dataset_athleticspose import DatasetAthleticsPose, DatasetAthleticsPose_multi
from data_loader.dataset_wp import DatasetWP, DatasetWP_multi
from models.diffusion import Diffusion
from models.transformer import MotionTransformer
from models.transformer_hmac import HMACMotionTransformer
from utils.util import padding_traj

DATASETS = {
    'ap3d': (DatasetAP3D, DatasetAP3D_multi),
    'ap': (DatasetAthleticsPose, DatasetAthleticsPose_multi),
    'wp': (DatasetWP, DatasetWP_multi),
}


def create_model_and_diffusion(cfg):
    """TransFusion denoiser ('tf', model_name: None) or HumanMAC denoiser ('hmac') + DDIM diffusion."""
    model_cls = MotionTransformer if cfg.model_name is None else HMACMotionTransformer
    model = model_cls(
        input_feats=3 * cfg.joint_num,  # 3 means x, y, z
        num_frames=cfg.n_pre,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        latent_dim=cfg.latent_dims,
        dropout=cfg.dropout,
    ).to(cfg.device)

    diffusion = Diffusion(
        noise_steps=cfg.noise_steps,
        motion_size=(cfg.n_pre, 3 * cfg.joint_num),  # 3 means x, y, z
        device=cfg.device, padding=cfg.padding,
        EnableComplete=cfg.Complete,
        ddim_timesteps=cfg.ddim_timesteps,
        scheduler=cfg.scheduler,
        mod_test=cfg.mod_test,
        dct=cfg.dct_m_all,
        idct=cfg.idct_m_all,
        n_pre=cfg.n_pre
    )
    return model, diffusion


def dataset_split(cfg, train=True):
    """Returns ({'train', 'test'} training/validation datasets or None, multimodal test set)."""
    base, multi = DATASETS[cfg.dataset]
    dataset = None
    if train:
        dataset = {'train': base('train', cfg.t_his, cfg.t_pred, data_root=cfg.data_root),
                   'test': base('test', cfg.t_his, cfg.t_pred, data_root=cfg.data_root)}
    dataset_multi_test = multi(cfg.t_his, cfg.t_pred, cfg.data_root, cfg.multimodal_path, cfg.data_candi_path)
    return dataset, dataset_multi_test


def get_multimodal_gt_full(logger, dataset_multi_test, cfg):
    logger.info('preparing full evaluation dataset...')
    traj_gt_arr = []
    data_group = []
    num_mult = []

    for data, multi_traj in dataset_multi_test.iter_generator():
        data_group.append(data)
        # (K, t_pred, (J-1)*3), root joint dropped
        traj_gt_arr.append(multi_traj[:, cfg.t_his:, 1:, :].reshape(len(multi_traj), cfg.t_pred, -1))
        num_mult.append(len(multi_traj))

    data_group = np.concatenate(data_group, axis=0)
    all_data = data_group[..., 1:, :].reshape(data_group.shape[0], data_group.shape[1], -1)
    gt_group = all_data[:, cfg.t_his:, :]

    num_mult = np.array(num_mult)
    logger.info('=' * 80)
    logger.info(f'#windows: {len(num_mult)}')
    logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{len(num_mult)}')
    logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{len(num_mult)}')
    logger.info(f'Average #future: {np.mean(num_mult):.1f}')
    logger.info('done...')
    logger.info('=' * 80)

    return {
        'traj_gt_arr': traj_gt_arr,
        'data_group': data_group,
        'gt_group': gt_group,
        'num_samples': len(num_mult),
    }


def display_exp_setting(logger, cfg):
    logger.info('=' * 80)
    log_dict = cfg.__dict__.copy()
    for key in list(log_dict):
        if 'dir' in key or 'dct' in key or key in ('zero_index', 'idx_pad', 'model_path'):
            del log_dict[key]
    logger.info(log_dict)
    logger.info('=' * 80)


def sample_preprocessing(traj, cfg, mode):
    """
    Preprocess traj for sample_ddim().
    output: mode dict, traj_dct, traj_dct_mod
    """
    if mode != 'metrics':
        raise NotImplementedError(f"unknown purpose for sampling: {mode}")
    n = traj.shape[0]

    mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
    for i in range(0, cfg.t_his):
        mask[:, i, :] = 1

    traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)

    traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
    traj_dct_mod = copy.deepcopy(traj_dct)
    if np.random.random() > cfg.mod_test:
        traj_dct_mod = None

    return {'mask': mask,
            'sample_num': n,
            'mode': 'metrics'}, traj_dct, traj_dct_mod
