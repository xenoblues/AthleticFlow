import copy
import os

import numpy as np
import torch

from data_loader.dataset_ap3d import DatasetAP3D
from data_loader.dataset_ap3d_multimodal import DatasetAP3D_multi
from data_loader.dataset_athleticspose import DatasetAthleticsPose
from data_loader.dataset_athleticspose_multimodal import DatasetAthleticsPose_multi
from utils import padding_traj
from utils.visualization import render_animation
from models.predictor import Noise_predictor
from models.diffusion import Diffusion

from data_loader.dataset_h36m import DatasetH36M
from data_loader.dataset_humaneva import DatasetHumanEva
from data_loader.dataset_h36m_multimodal import DatasetH36M_multi
from data_loader.dataset_humaneva_multimodal import DatasetHumanEva_multi
from data_loader.dataset_wp import DatasetWP
from data_loader.dataset_wp_multimodal import DatasetWP_multi

from scipy.spatial.distance import pdist, squareform


def create_model_and_diffusion(cfg):
    """
    create TransLinear model and Diffusion
    """
    # model = MotionTransformer(
    #     input_feats=3 * cfg.joint_num,  # 3 means x, y, z
    #     num_frames=cfg.n_pre,
    #     num_layers=cfg.num_layers,
    #     num_heads=cfg.num_heads,
    #     latent_dim=cfg.latent_dims,
    #     dropout=cfg.dropout,
    # ).to(cfg.device)
    model = Noise_predictor(
        input_feats=3 * cfg.joint_num,
        n_pre=cfg.n_pre,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        latent_dim=cfg.latent_dims,
        dropout=cfg.dropout
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


def dataset_split(cfg):

    # ===================== 核心：数据集-类映射表（扩展新数据集只需修改这里） =====================
    DATASET_CLASS_MAP = {
        "h36m": {
            "base": DatasetH36M,
            "multi": DatasetH36M_multi,
            "actions": "all"  # H36M/HumanEva 需要指定动作
        },
        "humaneva": {
            "base": DatasetHumanEva,
            "multi": DatasetHumanEva_multi,
            "actions": "all"
        },
        "ap3d": {
            "base": DatasetAP3D,
            "multi": DatasetAP3D_multi,
            "actions": None
        },
        "ap": {
            "base": DatasetAthleticsPose,
            "multi": DatasetAthleticsPose_multi,
            "actions": None
        },
        "wp": {
            "base": DatasetWP,
            "multi": DatasetWP_multi,
            "actions": None
        }
    }

    # 校验数据集合法性
    if cfg.dataset not in DATASET_CLASS_MAP:
        raise ValueError(f"不支持的数据集: {cfg.dataset}，支持列表: {list(DATASET_CLASS_MAP.keys())}")

    # 读取当前数据集配置
    dataset_cfg = DATASET_CLASS_MAP[cfg.dataset]
    BaseDataset = dataset_cfg["base"]
    MultiDataset = dataset_cfg["multi"]
    actions = dataset_cfg["actions"]

    # ===================== 1. 创建 训练/测试 普通数据集 =====================
    # 带动作参数的数据集（H36M/HumanEva）
    if actions is not None:
        dataset = BaseDataset('train', cfg.t_his, cfg.t_pred, actions=actions)
        dataset_test = BaseDataset('test', cfg.t_his, cfg.t_pred, actions=actions)
    # 无动作参数的数据集（AP3D/Assemble）
    else:
        dataset = BaseDataset('train', cfg.t_his, cfg.t_pred)
        dataset_test = BaseDataset('test', cfg.t_his, cfg.t_pred)

    dataset_dict = {'train': dataset, 'test': dataset_test}

    # ===================== 2. 创建 多模态测试数据集（无则为None） =====================
    dataset_multi_test = None
    if MultiDataset is not None:
        dataset_multi_test = MultiDataset(
            'test',
            cfg.t_his,
            cfg.t_pred,
            multimodal_path=cfg.multimodal_path,
            data_candi_path=cfg.data_candi_path
        )

    # ===================== 3. 统一归一化逻辑（所有数据集均生效） =====================
    if cfg.dct_norm_enable:
        dataset.normalize_data()
        dataset_test.normalize_data()
        if dataset_multi_test is not None:
            dataset_multi_test.normalize_data()

    return dataset_dict, dataset_multi_test


def get_multimodal_gt_full(logger, dataset_multi_test, args, cfg):
    """
    calculate the multi-modal data
    """
    logger.info('preparing full evaluation dataset...')
    data_group = []
    num_samples = 0
    data_gen_multi_test = dataset_multi_test.iter_generator(step=cfg.t_his)
    for data, _ in data_gen_multi_test:
        num_samples += 1
        data_group.append(data)
    data_group = np.concatenate(data_group, axis=0)
    all_data = data_group[..., 1:, :].reshape(data_group.shape[0], data_group.shape[1], -1)
    gt_group = all_data[:, cfg.t_his:, :]

    all_start_pose = all_data[:, cfg.t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, cfg.t_his:, :])
        num_mult.append(len(ind[0]))
    # np.savez_compressed('./data/data_3d_h36m_test.npz',data=all_data)
    # np.savez_compressed('./data/data_3d_humaneva15_test.npz',data=all_data)
    num_mult = np.array(num_mult)
    logger.info('=' * 80)
    logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}')
    logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}')
    logger.info('done...')
    logger.info('=' * 80)
    return {'traj_gt_arr': traj_gt_arr,
            'data_group': data_group,
            'gt_group': gt_group,
            'num_samples': num_samples}


def get_multimodal_gt_full_ap(logger, dataset_multi_test, args, cfg):
    logger.info('preparing full evaluation dataset...')
    traj_gt_arr = []
    data_group = []
    num_mult = []

    data_gen_multi_test = dataset_multi_test.iter_generator(step=cfg.t_his)

    for data, multi_traj in data_gen_multi_test:
        data_group.append(data)
        # 形状：(K, T_pred, (J-1)*3) → 3维，与gt维度完全一致
        traj_gt_arr.append(
            multi_traj[:, cfg.t_his:, 1:, :].reshape(len(multi_traj), cfg.t_pred, -1)
        )
        num_mult.append(len(multi_traj))

    data_group = np.concatenate(data_group, axis=0)
    all_data = data_group[..., 1:, :].reshape(data_group.shape[0], data_group.shape[1], -1)
    gt_group = all_data[:, cfg.t_his:, :]

    num_mult = np.array(num_mult)
    logger.info('=' * 80)
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
        'dataset_multi_test': dataset_multi_test
    }



def display_exp_setting(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 80)
    log_dict = cfg.__dict__.copy()
    for key in list(log_dict):
        if 'dir' in key or 'path' in key or 'dct' in key:
            del log_dict[key]
    del log_dict['zero_index']
    del log_dict['idx_pad']
    logger.info(log_dict)
    logger.info('=' * 80)


def sample_preprocessing(traj, cfg, mode):
    """
    This function is used to preprocess traj for sample_ddim().
    input : traj_seq, cfg, mode
    output: a dict for specific mode,
            traj_dct,
            traj_dct_mod
    """
    if mode == 'gif':
        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)

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
                'mode': 'gif'}, traj_dct, traj_dct_mod

    elif mode == 'pred':
        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)

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
                'mode': 'pred'}, traj_dct, traj_dct_mod


    elif mode == 'metrics':
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
    else:
        raise NotImplementedError(f"unknown purpose for sampling: {mode}")
