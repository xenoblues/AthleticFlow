import copy
import numpy as np
import torch

from data_loader.dataset_ap3d import DatasetAP3D
from data_loader.dataset_ap3d_multimodal import DatasetAP3D_multi
from data_loader.dataset_athleticspose import DatasetAthleticsPose
from data_loader.dataset_athleticspose_multimodal import DatasetAthleticsPose_multi
from data_loader.dataset_wp import DatasetWP
from data_loader.dataset_wp_multimodal import DatasetWP_multi
from models.flow_macthing import FlowMatchingCustom
from models.athletic_flow import AthleticFlow
from models.sdfm import SDFM
from utils import padding_traj

DATASET_CLASS_MAP = {
    "ap3d": {"base": DatasetAP3D, "multi": DatasetAP3D_multi},  # AthletePose3D
    "ap": {"base": DatasetAthleticsPose, "multi": DatasetAthleticsPose_multi},  # AthleticsPose
    "wp": {"base": DatasetWP, "multi": DatasetWP_multi},  # WorldPose
}


def create_model_and_generator(cfg):
    """
    create the velocity network and the flow-matching sampler
    """
    # number of temporal coefficients per joint and axis
    num_frames = cfg.n_pre if cfg.use_dct else cfg.t_total

    if cfg.model_name in ('AF', 'athleticflow'):
        model = AthleticFlow(
            joints=cfg.joint_num,
            t_total=cfg.t_total,
            k=num_frames,
            dim=cfg.latent_dims,
            depth=cfg.num_layers,
            heads=cfg.num_heads,
            dropout=cfg.dropout,
            cfg=cfg
        ).to(cfg.device)
    elif cfg.model_name in ('SDFM', 'MotioniTransformer2'):
        model = SDFM(
            num_frames=num_frames,
            latent_dim=cfg.latent_dims,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            joint_num=cfg.joint_num
        ).to(cfg.device)
    else:
        raise ValueError(f"unknown model_name: {cfg.model_name}")

    generator = FlowMatchingCustom(model, cfg)

    return model, generator


def dataset_split(cfg):
    if cfg.dataset not in DATASET_CLASS_MAP:
        raise ValueError(f"unsupported dataset: {cfg.dataset}, supported: {list(DATASET_CLASS_MAP.keys())}")

    BaseDataset = DATASET_CLASS_MAP[cfg.dataset]["base"]
    MultiDataset = DATASET_CLASS_MAP[cfg.dataset]["multi"]

    # training / validation sequences
    dataset = BaseDataset('train', cfg.t_his, cfg.t_pred)
    dataset_test = BaseDataset('test', cfg.t_his, cfg.t_pred)
    dataset_dict = {'train': dataset, 'test': dataset_test}

    # test windows with their multimodal ground truth (evaluation protocol)
    dataset_multi_test = MultiDataset(
        'test',
        cfg.t_his,
        cfg.t_pred,
        multimodal_path=cfg.multimodal_path,
        data_candi_path=cfg.data_candi_path
    )

    return dataset_dict, dataset_multi_test


def get_multimodal_gt_full(logger, dataset_multi_test, cfg):
    logger.info('preparing full evaluation dataset...')
    traj_gt_arr = []
    data_group = []
    num_mult = []

    data_gen_multi_test = dataset_multi_test.iter_generator(step=cfg.t_his)

    for data, multi_traj in data_gen_multi_test:
        data_group.append(data)
        # (M, t_pred, (J-1)*3), root joint removed
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
    Prepare the inputs of FlowMatchingCustom.sample_fm().
    input : traj [*, t_his + t_pred, 3 * joints_num], cfg, mode ('metrics' | 'gif' | 'pred')
    output: mode dict (observation mask, ...),
            traj_dct: coefficients of the padded observation (shape template of the noise),
            traj_dct_mod: condition (None: unconditional),
            traj_pad: padded observation in the time domain
    """
    if mode == 'metrics':
        n = traj.shape[0]
    elif mode in ('gif', 'pred'):
        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)
    else:
        raise NotImplementedError(f"unknown purpose for sampling: {mode}")

    mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
    for i in range(0, cfg.t_his):  # observed frames
        mask[:, i, :] = 1

    traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)  # last-frame padding

    if cfg.use_dct:
        traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
        traj_dct_mod = copy.deepcopy(traj_dct)
    else:
        traj_dct = copy.deepcopy(traj_pad)
        traj_dct_mod = copy.deepcopy(traj_dct)

    if np.random.random() > cfg.mod_test:
        traj_dct_mod = None

    return {'mask': mask,
            'sample_num': n,
            'mode': mode}, traj_dct, traj_dct_mod, traj_pad
