import copy
import os
import time
import numpy as np
import torch
from torch.nn.modules import dropout

from data_loader.dataset_ap3d import DatasetAP3D
from data_loader.dataset_ap3d_multimodal import DatasetAP3D_multi
from data_loader.dataset_athleticspose import DatasetAthleticsPose
from data_loader.dataset_athleticspose_multimodal import DatasetAthleticsPose_multi
from data_loader.dataset_sportspose import DatasetSP
from data_loader.dataset_sportspose_multimodal import DatasetSP_multi
from data_loader.dataset_wp import DatasetWP
from data_loader.dataset_wp_multimodal import DatasetWP_multi
from models.KAFormer import KASDFMTransformer
from models.flow_macthing import FlowMatchingCustom
from models.transfusion import MotionTransformerTransfusion
from models.SGTransformer import *
from models.athletic_flow import *
from utils import padding_traj
from utils.visualization import render_animation
from models.transformer import *
from models.transformer_mine import *
from models.diffusion import Diffusion
from models.gcn import MotionSTGCN

from data_loader.dataset_h36m import DatasetH36M
from data_loader.dataset_humaneva import DatasetHumanEva
from data_loader.dataset_h36m_multimodal import DatasetH36M_multi
from data_loader.dataset_humaneva_multimodal import DatasetHumanEva_multi
from data_loader.dataset_assemble import DatasetAsb
from scipy.spatial.distance import pdist, squareform
from models.bone_kinematics import RootBoneKinematics
import multiprocessing
import pywt
import ptwt
from tqdm import tqdm
from models.StochasticDistalOrientationResidual import *


def create_model_and_diffusion(cfg, filters, temporal_graph):
    """
    create TransLinear model and Diffusion
    """
    if cfg.use_dct or cfg.use_fft:
        num_frames = cfg.n_pre
    elif cfg.use_dwt:
        data = torch.randn(1, cfg.t_total)
        coeffs = ptwt.wavedec(data, cfg.dwt_wave, level=cfg.dwt_level, mode='constant', axis=1)
        for i in range(len(coeffs)):
            cfg.dwt_lens.append(coeffs[i].shape[-1])
        # num_frames = cfg.dwt_lens[0]
        num_frames = cfg.n_pre
    else:
        if not cfg.autoregression:
            num_frames = cfg.t_total
        else:
            num_frames = cfg.clip_total

    if not cfg.autoregression:
        if cfg.model_name == 'MotioniTransformer':
            model = MotioniTransformer(
                input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                joint_num=cfg.joint_num,
                spatial_graph=None,
                flash_attention=cfg.flash_attention
            ).to(cfg.device)
            if cfg.knowledge_distill:
                t_model = MotioniTransformer(
                    input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                    num_frames=num_frames,
                    num_layers=cfg.t_num_layers,
                    num_heads=cfg.t_num_heads,
                    latent_dim=cfg.t_latent_dims,
                    dropout=cfg.dropout,
                    spatial_graph=None,
                    flash_attention=cfg.flash_attention,
                ).to(cfg.device)
            else:
                t_model = None
        elif cfg.model_name == 'MotioniTransformer2':
            model = MotioniTransformer2(
                input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                joint_num=cfg.joint_num,
                spatial_graph=None,
                flash_attention=cfg.flash_attention,
                stylization_block=cfg.stylization_block
            ).to(cfg.device)
            if cfg.knowledge_distill:
                t_model = MotioniTransformer2(
                    input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                    num_frames=num_frames,
                    num_layers=cfg.t_num_layers,
                    num_heads=cfg.t_num_heads,
                    latent_dim=cfg.t_latent_dims,
                    dropout=cfg.dropout,
                    spatial_graph=None,
                    flash_attention=cfg.flash_attention,
                    stylization_block=cfg.stylization_block
                ).to(cfg.device)
            else:
                t_model = None
        elif cfg.model_name == 'MotionTransformerOrigin':
            model = MotionTransformerOrigin(
                input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                cross_attention=cfg.cross_attention,
                stylization_block=cfg.stylization_block,
                flash_attention=cfg.flash_attention
            ).to(cfg.device)
            if cfg.knowledge_distill:
                t_model = MotionTransformerOrigin(
                    input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                    num_frames=num_frames,
                    num_layers=cfg.t_num_layers,
                    num_heads=cfg.t_num_heads,
                    latent_dim=cfg.t_latent_dims,
                    dropout=cfg.dropout,
                    cross_attention=cfg.cross_attention,
                    stylization_block=cfg.stylization_block,
                    flash_attention=cfg.flash_attention,
                ).to(cfg.device)
            else:
                t_model = None
        elif cfg.model_name == 'MotionTransformerTransfusion':
            model = MotionTransformerTransfusion(
                input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                cross_attention=cfg.cross_attention,
                stylization_block=cfg.stylization_block,
                flash_attention=cfg.flash_attention,
                se_layer=cfg.se_layer,
                skip_type=cfg.skip_type
            ).to(cfg.device)
            if cfg.knowledge_distill:
                t_model = MotionTransformerTransfusion(
                    input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                    num_frames=num_frames,
                    num_layers=cfg.t_num_layers,
                    num_heads=cfg.t_num_heads,
                    latent_dim=cfg.t_latent_dims,
                    dropout=cfg.dropout,
                    cross_attention=cfg.cross_attention,
                    stylization_block=cfg.stylization_block,
                    flash_attention=cfg.flash_attention,
                    se_layer=cfg.se_layer,
                    skip_type=cfg.skip_type
                ).to(cfg.device)
            else:
                t_model = None
            """
            model = STCrossMotionTransformer(
                input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                spatial_graph=filters[0],
                temporal_graph=temporal_graph
            ).to(cfg.device)
    
            model = MotionTransformerParallel(
                input_feats=3 * cfg.joint_num,  # 3 means x, y, z
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                graph_filters=filters[0]
            ).to(cfg.device)
            """
        elif cfg.model_name == 'KASDFMTransformer':
            model = KASDFMTransformer(
                input_dim=3 * cfg.joint_num,  # 3 means x, y, z
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                joint_num=cfg.joint_num,
                flash_attention=cfg.flash_attention,
                cross_attention=cfg.cross_attention,
                cfg=cfg,
                use_dynamic_attention=True,
                use_adaln=False
            ).to(cfg.device)
            if cfg.knowledge_distill:
                t_model = KASDFMTransformer(
                    input_dim=3 * cfg.joint_num,  # 3 means x, y, z
                    num_frames=num_frames,
                    num_layers=cfg.t_num_layers,
                    num_heads=cfg.t_num_heads,
                    latent_dim=cfg.t_latent_dims,
                    dropout=cfg.dropout,
                    joint_num=cfg.joint_num,
                    flash_attention=cfg.flash_attention,
                    cross_attention=cfg.cross_attention,
                    cfg=cfg,
                    use_dynamic_attention=True,
                    use_adaln=False
                ).to(cfg.device)
            else:
                t_model = None
        elif cfg.model_name == 'SGFormer':
            model = SGFormer(
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                joint_num=cfg.joint_num,
                flash_attention=cfg.flash_attention,
                cross_attention=cfg.cross_attention,
                cfg=cfg
            ).to(cfg.device)
            if cfg.knowledge_distill:
                t_model = SGFormer(
                    num_frames=num_frames,
                    num_layers=cfg.t_num_layers,
                    num_heads=cfg.t_num_heads,
                    latent_dim=cfg.t_latent_dims,
                    dropout=cfg.dropout,
                    joint_num=cfg.joint_num,
                    flash_attention=cfg.flash_attention,
                    cross_attention=cfg.cross_attention,
                    cfg=cfg,
                ).to(cfg.device)
            else:
                t_model = None
        elif cfg.model_name == 'SGFormerLite':
            model = SGFormerLite(
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                joint_num=cfg.joint_num,
                flash_attention=cfg.flash_attention,
                cross_attention=cfg.cross_attention,
                cfg=cfg
            ).to(cfg.device)
            if cfg.knowledge_distill:
                t_model = SGFormerLite(
                    num_frames=num_frames,
                    num_layers=cfg.t_num_layers,
                    num_heads=cfg.t_num_heads,
                    latent_dim=cfg.t_latent_dims,
                    dropout=cfg.dropout,
                    joint_num=cfg.joint_num,
                    flash_attention=cfg.flash_attention,
                    cross_attention=cfg.cross_attention,
                    cfg=cfg,
                ).to(cfg.device)
            else:
                t_model = None
        elif cfg.model_name == 'SGFormerLite2':
            model = SGFormerLite2(
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                joint_num=cfg.joint_num,
                flash_attention=cfg.flash_attention,
                cross_attention=cfg.cross_attention,
                cfg=cfg
            ).to(cfg.device)
            if cfg.knowledge_distill:
                t_model = SGFormerLite2(
                    num_frames=num_frames,
                    num_layers=cfg.t_num_layers,
                    num_heads=cfg.t_num_heads,
                    latent_dim=cfg.t_latent_dims,
                    dropout=cfg.dropout,
                    joint_num=cfg.joint_num,
                    flash_attention=cfg.flash_attention,
                    cross_attention=cfg.cross_attention,
                    cfg=cfg,
                ).to(cfg.device)
            else:
                t_model = None
        elif cfg.model_name == 'VKDFFormer':
            model = VKDFFormer(
                num_frames=num_frames,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                latent_dim=cfg.latent_dims,
                dropout=cfg.dropout,
                joint_num=cfg.joint_num,
                flash_attention=cfg.flash_attention,
                cross_attention=cfg.cross_attention,
                cfg=cfg
            ).to(cfg.device)
            if cfg.knowledge_distill:
                t_model = VKDFFormer(
                    num_frames=num_frames,
                    num_layers=cfg.num_layers,
                    num_heads=cfg.num_heads,
                    latent_dim=cfg.latent_dims,
                    dropout=cfg.dropout,
                    joint_num=cfg.joint_num,
                    flash_attention=cfg.flash_attention,
                    cross_attention=cfg.cross_attention,
                    cfg=cfg,
                ).to(cfg.device)
            else:
                t_model = None
        elif cfg.model_name == 'RTLDFM':
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
            if cfg.knowledge_distill:
                t_model = AthleticFlow(
                    joints=cfg.joint_num,
                    t_total=cfg.t_total,
                    k=num_frames,
                    dim=cfg.latent_dims,
                    depth=cfg.num_layers,
                    heads=cfg.num_heads,
                    dropout=cfg.dropout,
                    cfg=cfg
                ).to(cfg.device)
            else:
                t_model = None
        encoder = None
    else:
        model = MotionTransformerUnet(
            input_feats=3 * cfg.joint_num,  # 3 means x, y, z
            num_frames=num_frames,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            latent_dim=cfg.latent_dims,
            dropout=cfg.dropout,
            graph_filters=filters[0]
        ).to(cfg.device)

        encoder = MotionTransformerUnetEncoder(
            input_feats=3 * cfg.joint_num,  # 3 means x, y, z
            num_frames=num_frames,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            latent_dim=cfg.latent_dims,
            dropout=cfg.dropout,
            graph_filters=filters[0]
        ).to(cfg.device)

    if cfg.mode == 'fine_tune' or cfg.mode == 'ft_eval':
        distal_refiner = WholeLimbChainOrientationRefiner().to(cfg.device)
    else:
        distal_refiner = None

    if cfg.generator == 'diffusion':
        generator = Diffusion(
            noise_steps=cfg.noise_steps,
            motion_size=(num_frames, 3 * cfg.joint_num),  # 3 means x, y, z
            device=cfg.device, padding=cfg.padding,
            EnableComplete=cfg.Complete,
            ddim_timesteps=cfg.ddim_timesteps,
            scheduler=cfg.scheduler,
            mod_test=cfg.mod_test,
            dct=cfg.dct_m_all,
            idct=cfg.idct_m_all,
            n_pre=cfg.n_pre,
            use_dct=cfg.use_dct
        )
    elif cfg.generator == 'flow_matching':
        generator = FlowMatchingCustom(model, cfg, distal_refiner)

    return model, generator, encoder, t_model


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
        "sp": {
            "base": DatasetSP,
            "multi": DatasetSP_multi,
            "actions": None
        },
        "wp": {
            "base": DatasetWP,
            "multi": DatasetWP_multi,
            "actions": None
        },
        "assemble": {
            "base": DatasetAsb,
            "multi": None,  # assemble 无多模态数据集
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

    # all_start_pose = all_data[:, cfg.t_his - 1, :]
    # pd = squareform(pdist(all_start_pose))
    # traj_gt_arr = []
    # num_mult = []
    # for i in range(pd.shape[0]):
    #     ind = np.nonzero(pd[i] < args.multimodal_threshold)
    #     traj_gt_arr.append(all_data[ind][:, cfg.t_his:, :])
    #     num_mult.append(len(ind[0]))
    #
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

def get_multimodal_gt_full_custom(logger, dataset_test, args, cfg):
    """
    calculate the multi-modal data
    """
    logger.info('preparing full evaluation dataset...')
    data_group = []
    num_samples = 0
    data_gen_test = dataset_test.iter_generator(step=cfg.t_his)
    for data in data_gen_test:
        num_samples += 1
        data_group.append(data)
    data_group = np.asarray(data_group)
    all_data = data_group.reshape(data_group.shape[0], data_group.shape[1], -1)
    gt_group = all_data[:, cfg.t_his:, :]

    all_start_pose = all_data[:, cfg.t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []

    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, cfg.t_his:, :])
        num_mult.append(len(ind[0]))

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
    vel_acc_pad = None

    if mode.split('_')[0] == 'fix':

        # skeleton joints in h36m
        fix_list = [
            [0, 1, 2],  #
            [3, 4, 5],
            [6, 7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        ]

        index = int(mode.split('_')[1])

        # [ ['right_leg'], ['left_leg'], ['torso'], [‘left_arm’], ['right_arm'] ]
        joint_fix_lb = fix_list[index][0] * 3
        joint_fix_ub = fix_list[index][-1] * 3 + 3

        traj_fix = traj[:, cfg.idx_pad, :]
        traj_fix[:, :, joint_fix_lb:joint_fix_ub] = traj[:, :, joint_fix_lb:joint_fix_ub]

        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        mask_fix = copy.deepcopy(mask)
        mask_fix[:, :, joint_fix_lb:joint_fix_ub] = 1

        if cfg.residual_data:
            # B, T-1, V, 3
            if np.random.random() > cfg.mod_test:
                vel_acc_pad = None
            else:
                vel_acc = cal_vel_acc(traj)
                vel_acc_pad = padding_vel(vel_acc, cfg.padding, cfg.idx_pad, cfg.zero_index)

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)
        if cfg.kin:
            traj_pad = cfg.kin.encode(traj_pad)

        if cfg.b_frequency_transform:
            if cfg.use_dct:
                traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
                # traj_dct = cfg.hdct.encode(traj_pad)
                traj_dct_mod = copy.deepcopy(traj_dct)

            elif cfg.use_dwt:
                traj_pad_frequency_components = ptwt.wavedec(traj_pad, wavelet=cfg.dwt_wave,
                                                             level=cfg.dwt_level, axis=1, mode='constant')
                if traj_pad_frequency_components[0].shape[1] > cfg.n_pre:
                    traj_dct = traj_pad_frequency_components[0][:, :cfg.n_pre, :]
                else:
                    len_ca = traj_pad_frequency_components[0].shape[1]
                    diff_len = cfg.n_pre - len_ca
                    traj_pad_cd_n = traj_pad_frequency_components[1][:, :diff_len, :]
                    traj_dct = torch.cat((traj_pad_frequency_components[0], traj_pad_cd_n), dim=1)
                traj_dct_mod = copy.deepcopy(traj_dct)
            elif cfg.use_fft:
                traj_dct = torch.fft.fft(traj_pad, dim=1).real
                traj_dct = traj_dct[:, :cfg.n_pre, :]
                traj_dct_mod = copy.deepcopy(traj_dct)

        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'traj_fix': traj_fix,
                'mask': mask_fix,
                'sample_num': n,
                'mode': 'control'}, traj_dct, traj_dct_mod, vel_acc_pad, traj_pad

    elif mode == 'gif_regression':
        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)
        mask = torch.zeros([n, cfg.clip_total, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.clip_his):
            mask[:, i, :] = 1

        traj_dct = traj
        traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'autoregression'}, traj_dct, traj_dct_mod, None, None

    elif mode == 'autoregression' or cfg.autoregression:
        n = traj.shape[0]
        mask = torch.zeros([n, cfg.clip_total, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.clip_his):
            mask[:, i, :] = 1

        traj_dct = traj
        traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'autoregression'}, traj_dct, traj_dct_mod, None, None

    elif mode == 'switch':
        n = traj.shape[0]
        traj_switch = traj[:, (cfg.t_pred - cfg.t_his):(cfg.t_pred + cfg.t_his), :]
        direct_current = traj[:, (cfg.t_pred - cfg.t_his), :].unsqueeze(1).repeat(1, cfg.t_pred - cfg.t_his,
                                                                                  1)
        traj_switch = torch.cat([direct_current, traj_switch], dim=1)

        traj_switch = traj_switch[0].unsqueeze(0).repeat(n, 1, 1)
        # traj_switch = torch.roll(traj_switch, 1, 0) # make the target traj be various

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        mask_end = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(cfg.t_pred - cfg.t_his, cfg.t_his + cfg.t_pred):
            mask_end[:, i, :] = 1

        if cfg.residual_data:
            # B, T-1, V, 3
            if np.random.random() > cfg.mod_test:
                vel_acc_pad = None
            else:
                vel_acc = cal_vel_acc(traj)
                vel_acc_pad = padding_vel(vel_acc, cfg.padding, cfg.idx_pad, cfg.zero_index)

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)
        if cfg.kin:
            traj_pad = cfg.kin.encode(traj_pad)

        if cfg.b_frequency_transform:
            if cfg.use_dct:
                traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
                # traj_dct = cfg.hdct.encode(traj_pad)
                traj_dct_mod = copy.deepcopy(traj_dct)
            elif cfg.use_dwt:
                traj_pad_frequency_components = ptwt.wavedec(traj_pad, wavelet=cfg.dwt_wave,
                                                             level=cfg.dwt_level, axis=1, mode='constant')
                if traj_pad_frequency_components[0].shape[1] > cfg.n_pre:
                    traj_dct = traj_pad_frequency_components[0][:, :cfg.n_pre, :]
                else:
                    len_ca = traj_pad_frequency_components[0].shape[1]
                    diff_len = cfg.n_pre - len_ca
                    traj_pad_cd_n = traj_pad_frequency_components[1][:, :diff_len, :]
                    traj_dct = torch.cat((traj_pad_frequency_components[0], traj_pad_cd_n), dim=1)
                traj_dct_mod = copy.deepcopy(traj_dct)
            elif cfg.use_fft:
                traj_dct = torch.fft.fft(traj_pad, dim=1).real
                traj_dct = traj_dct[:, :cfg.n_pre, :]
                traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'traj_switch': traj_switch,
                'mask_end': mask_end,
                'mask': mask,
                'sample_num': n,
                'mode': 'switch'}, traj_dct, traj_dct_mod, vel_acc_pad, None

    elif mode == 'gif':
        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        if cfg.residual_data:
            # B, T-1, V, 3
            if np.random.random() > cfg.mod_test:
                vel_acc_pad = None
            else:
                vel_acc = cal_vel_acc(traj)
                vel_acc_pad = padding_vel(vel_acc, cfg.padding, cfg.idx_pad, cfg.zero_index)

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)
        if cfg.kin:
            traj_pad = cfg.kin.encode(traj_pad)

        if cfg.b_frequency_transform:
            if cfg.use_dct:
                traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
                # traj_dct = cfg.hdct.encode(traj_pad)
                traj_dct_mod = copy.deepcopy(traj_dct)
            elif cfg.use_dwt:
                traj_pad_frequency_components = ptwt.wavedec(traj_pad, wavelet=cfg.dwt_wave,
                                                             level=cfg.dwt_level, axis=1, mode='constant')
                if traj_pad_frequency_components[0].shape[1] > cfg.n_pre:
                    traj_dct = traj_pad_frequency_components[0][:, :cfg.n_pre, :]
                else:
                    len_ca = traj_pad_frequency_components[0].shape[1]
                    diff_len = cfg.n_pre - len_ca
                    traj_pad_cd_n = traj_pad_frequency_components[1][:, :diff_len, :]
                    traj_dct = torch.cat((traj_pad_frequency_components[0], traj_pad_cd_n), dim=1)
                traj_dct_mod = copy.deepcopy(traj_dct)
            elif cfg.use_fft:
                traj_dct = torch.fft.fft(traj_pad, dim=1).real
                traj_dct = traj_dct[:, :cfg.n_pre, :]
                traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'gif'}, traj_dct, traj_dct_mod, vel_acc_pad, traj_pad

    elif mode == 'pred':
        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        if cfg.residual_data:
            # B, T-1, V, 3
            if np.random.random() > cfg.mod_test:
                vel_acc_pad = None
            else:
                vel_acc = cal_vel_acc(traj)
                vel_acc_pad = padding_vel(vel_acc, cfg.padding, cfg.idx_pad, cfg.zero_index)

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)
        if cfg.kin:
            traj_pad = cfg.kin.encode(traj_pad)

        if cfg.b_frequency_transform:
            if cfg.use_dct:
                traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
                # traj_dct = cfg.hdct.encode(traj_pad)
                traj_dct_mod = copy.deepcopy(traj_dct)
            elif cfg.use_dwt:
                traj_pad_frequency_components = ptwt.wavedec(traj_pad, wavelet=cfg.dwt_wave,
                                                             level=cfg.dwt_level, axis=1, mode='constant')
                if traj_pad_frequency_components[0].shape[1] > cfg.n_pre:  # 41 > 30
                    traj_dct = traj_pad_frequency_components[0][:, :cfg.n_pre, :]
                else:
                    len_ca = traj_pad_frequency_components[0].shape[1]
                    diff_len = cfg.n_pre - len_ca
                    traj_pad_cd_n = traj_pad_frequency_components[1][:, :diff_len, :]
                    traj_dct = torch.cat((traj_pad_frequency_components[0], traj_pad_cd_n), dim=1)
                traj_dct_mod = copy.deepcopy(traj_dct)
            elif cfg.use_fft:
                traj_dct = torch.fft.fft(traj_pad, dim=1).real
                traj_dct = traj_dct[:, :cfg.n_pre, :]
                traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_pad)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'pred'}, traj_dct, traj_dct_mod, vel_acc_pad, traj_pad

    elif mode == 'zero_shot':
        n = cfg.vis_col
        traj = traj.repeat(n, 1, 1)

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)
        for i in range(0, cfg.t_his):
            mask[:, i, :] = 1

        if cfg.residual_data:
            # B, T-1, V, 3
            if np.random.random() > cfg.mod_test:
                vel_acc_pad = None
            else:
                vel_acc = cal_vel_acc(traj)
                vel_acc_pad = padding_vel(vel_acc, cfg.padding, cfg.idx_pad, cfg.zero_index)

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)
        if cfg.kin:
            traj_pad = cfg.kin.encode(traj_pad)

        if cfg.b_frequency_transform:
            if cfg.use_dct:
                traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
                # traj_dct = cfg.hdct.encode(traj_pad)
                traj_dct_mod = copy.deepcopy(traj_dct)
            elif cfg.use_dwt:
                traj_pad_frequency_components = ptwt.wavedec(traj_pad, wavelet=cfg.dwt_wave,
                                                             level=cfg.dwt_level, axis=1, mode='constant')
                if traj_pad_frequency_components[0].shape[1] > cfg.n_pre:
                    traj_dct = traj_pad_frequency_components[0][:, :cfg.n_pre, :]
                else:
                    len_ca = traj_pad_frequency_components[0].shape[1]
                    diff_len = cfg.n_pre - len_ca
                    traj_pad_cd_n = traj_pad_frequency_components[1][:, :diff_len, :]
                    traj_dct = torch.cat((traj_pad_frequency_components[0], traj_pad_cd_n), dim=1)
                traj_dct_mod = copy.deepcopy(traj_dct)
            elif cfg.use_fft:
                traj_dct = torch.fft.fft(traj_pad, dim=1).real
                traj_dct = traj_dct[:, :cfg.n_pre, :]
                traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'zero_shot'}, traj_dct, traj_dct_mod, vel_acc_pad, traj_pad

    elif mode == 'metrics':
        n = traj.shape[0]

        mask = torch.zeros([n, cfg.t_his + cfg.t_pred, traj.shape[-1]]).to(cfg.device)

        for i in range(0, cfg.t_his):  # 历史帧标记为1
            mask[:, i, :] = 1

        if cfg.residual_data:
            # B, T-1, V, 3
            if np.random.random() > cfg.mod_test:
                vel_acc_pad = None
            else:
                vel_acc = cal_vel_acc(traj)
                vel_acc_pad = padding_vel(vel_acc, cfg.padding, cfg.idx_pad, cfg.zero_index)

        traj_pad = padding_traj(traj, cfg.padding, cfg.idx_pad, cfg.zero_index)  # last frame padding
        if cfg.kin:
            traj_pad = cfg.kin.encode(traj_pad)

        if cfg.b_frequency_transform:
            if cfg.use_dct:
                traj_dct = torch.matmul(cfg.dct_m_all[:cfg.n_pre], traj_pad)
                # traj_dct = cfg.hdct.encode(traj_pad)
                traj_dct_mod = copy.deepcopy(traj_dct)

            elif cfg.use_dwt:
                traj_pad_frequency_components = ptwt.wavedec(traj_pad, wavelet=cfg.dwt_wave,
                                                             level=cfg.dwt_level, axis=1, mode='constant')
                if traj_pad_frequency_components[0].shape[1] > cfg.n_pre:
                    traj_dct = traj_pad_frequency_components[0][:, :cfg.n_pre, :]
                else:
                    len_ca = traj_pad_frequency_components[0].shape[1]
                    diff_len = cfg.n_pre - len_ca
                    traj_pad_cd_n = traj_pad_frequency_components[1][:, :diff_len, :]
                    traj_dct = torch.cat((traj_pad_frequency_components[0], traj_pad_cd_n), dim=1)
                traj_dct_mod = copy.deepcopy(traj_dct)

            elif cfg.use_fft:
                traj_dct = torch.fft.fft(traj_pad, dim=1).real
                traj_dct = traj_dct[:, :cfg.n_pre, :]
                traj_dct_mod = copy.deepcopy(traj_dct)
        else:
            traj_dct = copy.deepcopy(traj_pad)
            traj_dct_mod = copy.deepcopy(traj_dct)

        if np.random.random() > cfg.mod_test:
            traj_dct_mod = None

        return {'mask': mask,
                'sample_num': n,
                'mode': 'metrics'}, traj_dct, traj_dct_mod, vel_acc_pad, traj_pad

    else:
        raise NotImplementedError(f"unknown purpose for sampling: {mode}")
