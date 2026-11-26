import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from utils.logger import AverageMeter
from utils.metrics import compute_all_metrics_detailed, get_bone_names, get_fps, get_joint_names, get_skeleton_edges
from utils.script import sample_preprocessing

K = 50  # samples per test window
END_EFFECTORS = ('LWrist', 'RWrist', 'LFoot', 'RFoot')  # AEE joints
MEM_THRESHOLD_GB = 12  # 'auto' sampling: whole test set in one batch if this much GPU memory is free


def _predict(data, model, diffusion, cfg, mode='whole', slice_num=10):
    """One prediction (all t_his + t_pred frames, root removed) for every test window."""
    traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
    traj = torch.tensor(traj_np, device=cfg.device, dtype=torch.float32)
    traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
    # traj: [N, t_his + t_pred, 3 * joint_num]

    if mode == 'whole':
        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
        sampled_motion = diffusion.sample_ddim(model, traj_dct, traj_dct_cond, mode_dict)
        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
    else:
        traj_est = torch.zeros(traj.shape, device=cfg.device, dtype=torch.float32)
        n = traj.shape[0] // slice_num
        for s in range(slice_num):
            sl = slice(s * n, (s + 1) * n) if s != slice_num - 1 else slice(s * n, None)
            # every slice is sampled (the original code only sampled the last one)
            mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj[sl], cfg, mode='metrics')
            sampled_motion = diffusion.sample_ddim(model, traj_dct, traj_dct_cond, mode_dict)
            traj_est[sl] = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)

    return traj_est.cpu().numpy()[None, ...]


def _sampling_mode(cfg):
    if getattr(cfg, 'sample_mode', 'auto') != 'auto':
        return cfg.sample_mode
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        return 'whole' if free_bytes / 1024 ** 3 >= MEM_THRESHOLD_GB else 'sliced'
    return 'sliced'


def compute_stats_cs(diffusion, multimodal_dict, model, logger, cfg, save_pred=False):
    """
    Best-of-K evaluation (K=50) on the multimodal test set.
    Logs APD, ADE, FDE, MMADE, MMFDE, AEE, ABE and smoothness metrics, writes
    results/stats.csv and the per-window metrics results/per_window.npz.
    """
    st = time.time()
    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    traj_gt_arr = multimodal_dict['traj_gt_arr']
    num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE',
                   'jitter', 'jitter_all', 'jitter_gt', 'accel', 'accel_gt', 'accel_error']
    stats_meter = {x: AverageMeter() for x in stats_names}
    fps = get_fps(cfg.dataset, cfg)
    joint_names = get_joint_names(cfg.dataset, cfg.joint_num)
    bone_names = get_bone_names(cfg.dataset, cfg.joint_num)
    ee_idx = [joint_names.index(n) for n in END_EFFECTORS]

    pred = []
    for i in tqdm(range(K), position=0):
        pred.append(_predict(data_group, model, diffusion, cfg, mode=_sampling_mode(cfg)))
    pred = np.concatenate(pred, axis=0)  # [K, N, t_his + t_pred, 3 * joint_num]
    pred_all = pred if save_pred else None

    if cfg.jitter_seam == 'observed':
        # last 3 observed frames (identical for every sample and every method)
        _gh = data_group[:, max(cfg.t_his - 3, 0):cfg.t_his, 1:, :].reshape(data_group.shape[0], -1, pred.shape[-1])
        hist = np.ascontiguousarray(np.broadcast_to(_gh[None], (pred.shape[0],) + _gh.shape)).astype(pred.dtype)
    elif cfg.jitter_seam == 'predicted':
        hist = pred[:, :, max(cfg.t_his - 3, 0):cfg.t_his, :]
    else:
        raise ValueError(f'unknown jitter_seam: {cfg.jitter_seam}')
    pred = pred[:, :, cfg.t_his:, :]

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt_group = torch.from_numpy(gt_group).to(dev)
    pred = torch.from_numpy(pred).to(dev)
    hist = torch.from_numpy(hist).to(dev)

    joint_ade_sum = 0
    joint_fde_sum = 0
    time_ade_sum = 0
    joint_jitter_sum = 0
    bone_ade_sum = torch.zeros(len(get_skeleton_edges(cfg.dataset)), device=dev)
    angle_ade_sum = {k: 0.0 for k in ('RKnee', 'LKnee', 'LElbow', 'RElbow')}
    per_window = {k: [] for k in ('APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'jitter', 'jitter_all', 'jitter_gt',
                                  'accel', 'accel_gt', 'accel_error', 'joint_ade', 'bone_ade')}

    for j in range(num_samples):
        metrics = compute_all_metrics_detailed(
            pred[:, j, :, :],
            gt_group[j][np.newaxis, ...],
            traj_gt_arr[j],
            num_joints=cfg.joint_num,
            dataset_name=cfg.dataset,
            fps=fps,
            hist=hist[0, j],
        )
        for k in per_window:
            per_window[k].append(metrics[k].detach().float().cpu().numpy())
        for k in stats_names:
            stats_meter[k].update(metrics[k])
        joint_ade_sum = joint_ade_sum + metrics['joint_ade']
        joint_fde_sum = joint_fde_sum + metrics['joint_fde']
        time_ade_sum = time_ade_sum + metrics['time_ade']
        joint_jitter_sum = joint_jitter_sum + metrics['joint_jitter']
        bone_ade_sum += metrics['bone_ade']
        for k in angle_ade_sum:
            angle_ade_sum[k] += metrics['angle_ade'][k]

    joint_ade_mean = (joint_ade_sum / num_samples).cpu().numpy()
    joint_fde_mean = (joint_fde_sum / num_samples).cpu().numpy()
    time_ade_mean = (time_ade_sum / num_samples).cpu().numpy()
    joint_jitter_mean = (joint_jitter_sum / num_samples).cpu().numpy()
    bone_ade_mean = (bone_ade_sum / num_samples).cpu().numpy()
    summary = {k: float(stats_meter[k].avg.item()) for k in stats_names}
    summary['AEE'] = float(joint_ade_mean[ee_idx].mean())
    summary['ABE'] = float(bone_ade_mean.mean())

    for k in stats_names:
        logger.info(f'{k}: {summary[k]:.4f}')
    logger.info(f"AEE: {summary['AEE']:.4f}")
    logger.info(f"ABE: {summary['ABE']:.4f}")

    def section(title, names, values, fmt='.6f'):
        logger.info('=' * 80)
        logger.info(title)
        logger.info('=' * 80)
        for name, v in zip(names, values):
            logger.info(f'{name:<15}: {v:{fmt}}')

    section('Joint ADE', joint_names, joint_ade_mean)
    section('Joint FDE', joint_names, joint_fde_mean)
    section('Joint Jitter (unit/s^3)', joint_names, joint_jitter_mean)
    section('Bone Length ADE', bone_names, bone_ade_mean)
    section('Joint Angle ADE (degree)', angle_ade_sum.keys(),
            [float(v / num_samples) for v in angle_ade_sum.values()], fmt='.4f')
    section('Time ADE', [f't={t + 1:03d}' for t in range(len(time_ade_mean))], time_ade_mean)
    logger.info(f'{time.time() - st:.4f} seconds')

    with open(os.path.join(cfg.result_dir, 'stats.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for k in ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'AEE', 'ABE'] + stats_names[5:]:
            writer.writerow([k, summary[k]])
    np.savez_compressed(os.path.join(cfg.result_dir, 'per_window.npz'),
                        **{k: np.stack(v) for k, v in per_window.items()})
    if save_pred:
        np.save(os.path.join(cfg.result_dir, 'pred_all.npy'), pred_all)
        np.save(os.path.join(cfg.result_dir, 'data_all.npy'), data_group)
    return summary
