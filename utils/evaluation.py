import os
import csv
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.logger import AverageMeter
from utils.metrics import (compute_all_metrics_detailed, get_fps, get_joint_names, get_bone_names,
                           get_skeleton_edges, get_end_effector_indices)
from utils.script import sample_preprocessing

tensor = torch.tensor

K_SAMPLES = 50  # number of predicted futures per test window


def compute_stats(generator_model, multimodal_dict, model, logger, cfg, save_results=False):
    """
    Evaluate K = 50 predicted futures for every test window.

    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    Printed summary: APD, ADE, FDE, MMADE, MMFDE, AEE (mean ADE of the wrists and feet), ABE (mean bone-length
    error) and jitter (best sample), all of the best-of-K sample except APD / jitter_all.
    """

    def get_prediction(data, mode='whole', slice_num=10):
        # root joint removed: [N, t_total, J+1, 3] -> [N, t_total, 3 * J]
        traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)

        traj_est = torch.zeros(traj.shape, device=cfg.device, dtype=torch.float32)

        if mode == 'whole':
            slices = [(0, traj.shape[0])]
        else:
            n = traj.shape[0] // slice_num
            slices = [(s * n, (s + 1) * n if s != slice_num - 1 else traj.shape[0]) for s in range(slice_num)]

        generator_model.update_sampling_model(model)
        for st, ed in slices:
            mode_dict, traj_dct, traj_dct_cond, traj_pad = sample_preprocessing(traj[st:ed], cfg, mode='metrics')
            sampled_motion = generator_model.sample_fm(mode_dict, traj_dct, traj_dct_cond, traj_pad, pre_inject=True)
            if cfg.use_dct:
                traj_est[st:ed] = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
            else:
                traj_est[st:ed] = sampled_motion

        # [1, N, t_total, 3 * J]
        return traj_est.cpu().numpy()[None, ...]

    st = time.time()
    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    traj_gt_arr = multimodal_dict['traj_gt_arr']
    num_samples = multimodal_dict['num_samples']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE',
                   'jitter', 'jitter_all', 'jitter_gt',
                   'accel', 'accel_gt', 'accel_error']
    stats_meter = {x: AverageMeter() for x in stats_names}

    fps = get_fps(cfg.dataset, cfg)
    save_per_window = getattr(cfg, 'save_per_window', False)
    save_windows = set(int(x) for x in str(getattr(cfg, 'save_windows', '') or '').split(',') if x.strip())

    joint_ade_sum = None
    joint_fde_sum = None
    time_ade_sum = None
    joint_jitter_sum = None
    joint_accel_error_sum = None
    n_bones = len(get_skeleton_edges(cfg.dataset))
    bone_ade_sum = torch.zeros(n_bones, device='cuda')
    angle_ade_sum = {"RKnee": 0.0, "LKnee": 0.0, "LElbow": 0.0, "RElbow": 0.0}

    pred = []
    for i in tqdm(range(0, K_SAMPLES), position=0):
        # one prediction for every test window per iteration
        if torch.cuda.is_available():
            # memory-saving sliced mode on GPUs with less than 12 GB free memory
            # (the sampled noise, hence the numbers, differ slightly between both modes)
            free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info()
            mode_ = 'whole' if free_memory_bytes / (1024 ** 3) >= 12 else 'sliced'
        else:
            mode_ = 'sliced'

        pred.append(get_prediction(data_group, mode=mode_, slice_num=10))

    # pred [K, N, t_total, 3 * J]
    pred = np.concatenate(pred, axis=0)
    pred_all = pred if save_results else None
    if cfg.jitter_seam == 'observed':
        # the last 3 observed ground-truth frames (identical for every sample and method)
        _gh = data_group[:, max(cfg.t_his - 3, 0):cfg.t_his, 1:, :].reshape(data_group.shape[0], -1, pred.shape[-1])
        hist = np.ascontiguousarray(np.broadcast_to(_gh[None], (pred.shape[0],) + _gh.shape)).astype(pred.dtype)
    else:
        # 'predicted': the model's reconstruction of the last 3 observed frames
        hist = pred[:, :, max(cfg.t_his - 3, 0):cfg.t_his, :]
    pred = pred[:, :, cfg.t_his:, :]
    # Use GPU to accelerate
    gt_group = torch.from_numpy(gt_group).to('cuda')
    pred = torch.from_numpy(pred).to('cuda')
    hist = torch.from_numpy(hist).to('cuda')

    per_window = {k: [] for k in ('APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'jitter', 'jitter_all', 'jitter_gt',
                                  'accel', 'accel_gt', 'accel_error', 'joint_ade', 'bone_ade',
                                  'ee_vpeak_gt', 'ee_vpeak_best', 'ee_vpeak_all',
                                  'ee_apeak_gt', 'ee_apeak_best', 'ee_apeak_all')}
    selected = {}
    ee_idx = get_end_effector_indices(cfg.dataset, cfg.joint_num)
    # end effectors of the per-window peak speed / acceleration, columns of the ee_* arrays in per_window.npz:
    # ap3d / ap: LWrist, RWrist, RFoot, LFoot; wp: LWrist, RWrist, LFoot, RFoot
    ee_peak_idx = [19, 20, 9, 10] if cfg.dataset == 'wp' else [12, 15, 2, 5]

    def _peaks(x):
        # peak speed / acceleration of the end effectors over the predicted frames: [..., 4]
        v = torch.linalg.norm((x[..., 1:, :, :] - x[..., :-1, :, :]) * fps, dim=-1)[..., ee_peak_idx]
        a = torch.linalg.norm((x[..., 2:, :, :] - 2 * x[..., 1:-1, :, :] + x[..., :-2, :, :]) * fps ** 2,
                              dim=-1)[..., ee_peak_idx]
        return v.amax(dim=-2), a.amax(dim=-2)

    # pred [K, N, t_pred, 3 * J]
    for j in range(num_samples):

        metrics = compute_all_metrics_detailed(
            pred[:, j, :, :],
            gt_group[j][np.newaxis, ...],
            traj_gt_arr[j],
            num_joints=cfg.joint_num,
            dataset_name=cfg.dataset,
            fps=fps,
            hist=hist[0, j]
        )

        for stats in stats_names:
            stats_meter[stats].update(metrics[stats])

        if save_per_window:
            for _k in per_window:
                if not _k.startswith('ee_'):
                    _v = metrics[_k]
                    per_window[_k].append(_v.detach().float().cpu().numpy() if torch.is_tensor(_v)
                                          else np.asarray(_v, dtype=np.float32))
            _p = pred[:, j].reshape(pred.shape[0], cfg.t_pred, -1, 3).float()
            _g = gt_group[j].reshape(cfg.t_pred, -1, 3).float()
            _bi = torch.linalg.norm(_p - _g[None], dim=-1).mean(dim=(1, 2)).argmin()
            _vg, _ag = _peaks(_g)
            _vb, _ab = _peaks(_p[_bi])
            _va, _aa = _peaks(_p)
            for _k, _v in (('ee_vpeak_gt', _vg), ('ee_vpeak_best', _vb), ('ee_vpeak_all', _va.mean(0)),
                           ('ee_apeak_gt', _ag), ('ee_apeak_best', _ab), ('ee_apeak_all', _aa.mean(0))):
                per_window[_k].append(_v.detach().cpu().numpy())

        if j in save_windows:
            selected[f'pred_{j}'] = pred[:, j].float().cpu().numpy()
            selected[f'data_{j}'] = np.asarray(data_group[j])

        if joint_ade_sum is None:
            joint_ade_sum = metrics['joint_ade'].clone()
            joint_fde_sum = metrics['joint_fde'].clone()
            time_ade_sum = metrics['time_ade'].clone()
            joint_jitter_sum = metrics['joint_jitter'].clone()
            joint_accel_error_sum = metrics['joint_accel_error'].clone()
        else:
            joint_ade_sum += metrics['joint_ade']
            joint_fde_sum += metrics['joint_fde']
            time_ade_sum += metrics['time_ade']
            joint_jitter_sum += metrics['joint_jitter']
            joint_accel_error_sum += metrics['joint_accel_error']

        bone_ade_sum += metrics['bone_ade']

        for k in angle_ade_sum.keys():
            angle_ade_sum[k] += metrics["angle_ade"][k]

    joint_ade_mean = joint_ade_sum / num_samples
    joint_fde_mean = joint_fde_sum / num_samples
    time_ade_mean = time_ade_sum / num_samples
    joint_jitter_mean = joint_jitter_sum / num_samples
    joint_accel_error_mean = joint_accel_error_sum / num_samples
    bone_ade_mean = bone_ade_sum / num_samples

    joint_names = get_joint_names(cfg.dataset, cfg.joint_num)
    joint_ade_np = joint_ade_mean.cpu().numpy()
    bone_ade_np = bone_ade_mean.cpu().numpy()
    aee = float(np.mean(joint_ade_np[ee_idx]))
    abe = float(np.mean(bone_ade_np))

    for stats in stats_names[:5]:
        logger.info(f'{stats}: {stats_meter[stats].avg.item():.4f}')
    logger.info(f'AEE: {aee:.4f}')
    logger.info(f'ABE: {abe:.4f}')
    for stats in stats_names[5:]:
        logger.info(f'{stats}: {stats_meter[stats].avg.item():.4f}')

    def log_table(title, names, values, fmt='.6f'):
        logger.info('=' * 80)
        logger.info(title)
        logger.info('=' * 80)
        for name, err in zip(names, values):
            logger.info(f'{name:<15}: {err:{fmt}}')

    log_table('Joint ADE', joint_names, joint_ade_np)
    log_table('Joint FDE', joint_names, joint_fde_mean.cpu().numpy())
    log_table('Joint Jitter (unit/s^3)', joint_names, joint_jitter_mean.cpu().numpy())
    log_table('Joint Accel Error (unit/s^2)', joint_names, joint_accel_error_mean.cpu().numpy())

    idx = np.argsort(joint_ade_np)[::-1][:5]
    log_table('Worst Joint ADE', [joint_names[j] for j in idx], joint_ade_np[idx])
    jitter_np = joint_jitter_mean.cpu().numpy()
    idx = np.argsort(jitter_np)[::-1][:5]
    log_table('Worst Joint Jitter', [joint_names[j] for j in idx], jitter_np[idx])

    logger.info("=" * 80)
    logger.info("Bone Length ADE")
    logger.info("=" * 80)
    for name, err in zip(get_bone_names(cfg.dataset, cfg.joint_num), bone_ade_np):
        logger.info(f"{name:<20}: {err:.6f}")

    logger.info("=" * 80)
    logger.info("Joint Angle ADE (degree)")
    logger.info("=" * 80)
    for k, v in angle_ade_sum.items():
        logger.info(f"{k:<10}: {(v / num_samples).item():.4f}")

    logger.info('=' * 80)
    logger.info('Smoothness (fps=%.1f)' % fps)
    logger.info('=' * 80)
    logger.info(f"jitter(best)   : {stats_meter['jitter'].avg.item():.4f}")
    logger.info(f"jitter(all K)  : {stats_meter['jitter_all'].avg.item():.4f}")
    logger.info(f"jitter(GT)     : {stats_meter['jitter_gt'].avg.item():.4f}")
    logger.info(f"accel          : {stats_meter['accel'].avg.item():.4f}")
    logger.info(f"accel(GT)      : {stats_meter['accel_gt'].avg.item():.4f}")
    logger.info(f"accel_error    : {stats_meter['accel_error'].avg.item():.4f}")

    logger.info('=' * 80)
    logger.info('Time ADE')
    logger.info('=' * 80)
    for t, err in enumerate(time_ade_mean.cpu().numpy()):
        logger.info(f't={t + 1:03d}: ' f'{err:.6f}')

    et = time.time()
    logger.info(f'{et - st:.4f} seconds')

    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'

    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric', 'Value'])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            value = meter.avg
            if torch.is_tensor(value):
                value = value.item()
            writer.writerow({'Metric': stats, 'Value': float(value)})
        writer.writerow({'Metric': 'AEE', 'Value': aee})
        writer.writerow({'Metric': 'ABE', 'Value': abe})

    df1 = pd.read_csv(file_latest % cfg.result_dir)
    if not os.path.exists(file_stat % cfg.result_dir):
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2, df1], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)

    np.save(os.path.join(cfg.result_dir, 'joint_ade.npy'), joint_ade_np)
    np.save(os.path.join(cfg.result_dir, 'joint_fde.npy'), joint_fde_mean.cpu().numpy())
    np.save(os.path.join(cfg.result_dir, 'time_ade.npy'), time_ade_mean.cpu().numpy())
    np.save(os.path.join(cfg.result_dir, 'joint_jitter.npy'), jitter_np)
    np.save(os.path.join(cfg.result_dir, 'joint_accel_error.npy'), joint_accel_error_mean.cpu().numpy())
    if save_per_window:
        np.savez_compressed(os.path.join(cfg.result_dir, 'per_window.npz'),
                            **{k: np.stack(v) for k, v in per_window.items()})
    if selected:
        np.savez_compressed(os.path.join(cfg.result_dir, 'selected_windows.npz'), **selected)
    if save_results:
        np.save(os.path.join(cfg.result_dir, 'pred_all.npy'), pred_all)
        np.save(os.path.join(cfg.result_dir, 'data_all.npy'), data_group)
