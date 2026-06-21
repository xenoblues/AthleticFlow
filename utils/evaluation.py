import csv
import numpy as np
import pandas as pd
import ptwt
import torch

from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing
import time

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros

BONE_NAMES = [
    "RHip-RKnee",
    "RKnee-RFoot",

    "LHip-LKnee",
    "LKnee-LFoot",

    "Hip-Spine",
    "Spine-Thorax",
    "Thorax-Neck",
    "Neck-Head",

    "Thorax-LShoulder",
    "LShoulder-LElbow",
    "LElbow-LWrist",

    "Thorax-RShoulder",
    "RShoulder-RElbow",
    "RElbow-RWrist"
]


def clips_prediction(traj_data, cfg, diffusion, encoder_model, model_select, mode_dict):
    traj_est = traj_data.clone()
    clip_num = (cfg.t_total - cfg.clip_his) // cfg.clip_pred  # 总clip数量
    clip_res = (cfg.t_total - cfg.clip_his) % cfg.clip_pred
    nopred_clip_num = (cfg.t_his - cfg.clip_his) // cfg.clip_pred  # 不用做预测的clip数量
    nopred_res = (cfg.t_his - cfg.clip_his) % cfg.clip_pred
    his_clip_num = cfg.t_his // cfg.clip_his  # 有可用历史真值的clip
    last_feature = None
    pred_prev_clip = None
    if clip_res:
        clip_num += 1
    if nopred_res:
        nopred_clip_num += 1
    for c in range(clip_num):
        if clip_res:
            if c < clip_num - 1:
                ground_clip = traj_est[:, c * cfg.clip_pred: c * cfg.clip_pred + cfg.clip_total, :].clone()
            else:
                ground_clip = traj_est[:, -cfg.clip_total, :].clone()
        else:
            ground_clip = traj_est[:, c * cfg.clip_pred: c * cfg.clip_pred + cfg.clip_total, :].clone()

        if c < nopred_clip_num:  # 历史帧不进行预测，融合clip特征
            if c == 0:
                pass
            else:
                global_feature = encoder_model(prev_clip).detach()
                if last_feature is None:
                    last_feature = global_feature
                else:
                    last_feature = (global_feature + last_feature) / 2.0
            """
            elif nopred_clip_num <= c < his_clip_num:
                global_feature = encoder_model(prev_clip).detach()
                last_feature = (global_feature + last_feature) / 2.0
    
                # last his frame padding
                his = cfg.t_his - c * cfg.clip_his
                clip_cond = ground_clip[:, :, :].clone()
                mask = torch.zeros_like(ground_clip).to(cfg.device)
                his_range = [i for i in range(his)]
                mask[:, his_range, :] = 1
    
                pred_prev_clip = diffusion.sample_ddim_autoregression(model_select,
                                                                      clip_cond,
                                                                      last_feature,
                                                                      mode_dict,
                                                                      mask)
                print('clip: ', c, 'traj_est.shape: ', traj_est.shape, 'pred_prev_clip.shape: ', pred_prev_clip.shape)
                traj_est[:, cfg.t_his: c * cfg.clip_his + cfg.clip_total - cfg.t_his, :] \
                    = pred_prev_clip[:, -(c * cfg.clip_his + cfg.clip_total - cfg.t_his):, :]
            """
        else:
            # 输入条件帧和全局特征进行预测
            global_feature = encoder_model(prev_clip).detach()
            last_feature = (global_feature + last_feature) / 2.0

            if c * cfg.clip_pred < cfg.t_his < c * cfg.clip_pred + cfg.clip_total:
                his = cfg.t_his - c * cfg.clip_pred
            else:
                his = cfg.clip_his
            his_ind = [i for i in range(his)]

            clip_cond = ground_clip[:, :, :].clone()
            mask = torch.zeros_like(ground_clip).to(cfg.device)
            mask[:, his_ind, :] = 1
            # mask包含在mode_dict中
            pred_prev_clip = diffusion.sample_ddim_autoregression(model_select,
                                                                  clip_cond,
                                                                  last_feature,
                                                                  mode_dict,
                                                                  mask)
            if clip_res:
                if c != clip_num:
                    if c * cfg.clip_pred < cfg.t_his < c * cfg.clip_pred + cfg.clip_total:
                        traj_est[:, cfg.t_his: c * cfg.clip_his + cfg.clip_total - cfg.t_his, :] = pred_prev_clip[
                            :, -(c * cfg.clip_his + cfg.clip_total - cfg.t_his):, :]
                    else:
                        traj_est[:, c * cfg.clip_his: c * cfg.clip_his + cfg.clip_total, :] = pred_prev_clip
                else:
                    traj_est[: -cfg.clip_his:, :] = pred_prev_clip
            else:
                # print('c:', c, traj_est.shape, pred_prev_clip.shape)
                if c * cfg.clip_pred < cfg.t_his < c * cfg.clip_pred + cfg.clip_total:
                    traj_est[:, cfg.t_his: c * cfg.clip_pred + cfg.clip_total, :] = pred_prev_clip[
                        :, -(c * cfg.clip_pred + cfg.clip_total - cfg.t_his):, :]
                else:
                    traj_est[:, c * cfg.clip_his: c * cfg.clip_his + cfg.clip_total, :] = pred_prev_clip

        if c <= nopred_clip_num:
            prev_clip = ground_clip
        else:
            # clip包含已知历史帧，则
            if (c - 1) * cfg.clip_pred < cfg.t_his < (c - 1) * cfg.clip_pred + cfg.clip_total:
                gt_len = cfg.t_his - (c - 1) * cfg.clip_his
                prev_clip = torch.cat((ground_clip[:gt_len], pred_prev_clip[gt_len:]))
            else:
                prev_clip = pred_prev_clip

    return traj_est


def compute_stats(generator_model, multimodal_dict, model, logger, cfg, encoder_model=None, save_results=False):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    def get_prediction(data, model_select, mode='whole', slice_num=5):
        if cfg.dataset == 'assemble' or not cfg.remove_root:
            traj_np = data.reshape([data.shape[0], cfg.t_total, -1])
            traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        else:
            traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
            traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
            traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)

        # traj.shape: [*, t_his + t_pre, 3 * joints_num]
        traj_est = torch.zeros(traj.shape, device=cfg.device, dtype=torch.float32)

        if mode == 'whole':
            mode_dict, traj_dct, traj_dct_cond, vel_acc_pad, traj_pad = sample_preprocessing(traj, cfg, mode='metrics')

            if cfg.generator == 'diffusion':
                sampled_motion = generator_model.sample_ddim(model_select,
                                                             traj_dct,
                                                             traj_dct_cond,
                                                             mode_dict,
                                                             vel_acc_pad=vel_acc_pad)

            elif cfg.generator == 'flow_matching':
                generator_model.update_sampling_model(model)
                if not cfg.res_fm:
                    sampled_motion = generator_model.sample_fm(mode_dict, traj_dct, traj_dct_cond, traj_pad,
                                                               pre_inject=True)
                else:
                    sampled_motion = generator_model.sample_res_fm(mode_dict, traj_dct, traj_dct_cond, traj_pad,
                                                                   pre_inject=True)

            if cfg.use_dct:
                if cfg.mode != 'fine_tune' and cfg.mode != 'ft_eval':
                    traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
                    # traj_est = apply_limb_chain_direction_oracle(traj_est, traj, num_joints=cfg.joint_num)
                else:
                    traj_est = sampled_motion
                # traj_est = cfg.hdct.decode(sampled_motion)

            elif cfg.use_dwt:
                inputs = []
                for i in range(0, len(cfg.dwt_lens)):
                    inputs.append(torch.zeros(sampled_motion.shape[0], cfg.dwt_lens[i], sampled_motion.shape[2],
                                              device=cfg.device))

                if cfg.dwt_lens[0] >= cfg.n_pre:  # 41 > 30
                    inputs[0][:, :cfg.n_pre, :] = sampled_motion.clone()
                else:  # 21 < 30
                    inputs[0][:, :, :] = sampled_motion[:, :cfg.dwt_lens[0], :].clone()
                    diff_len = cfg.n_pre - cfg.dwt_lens[0]
                    inputs[1][:, :diff_len, :] = sampled_motion[:, cfg.dwt_lens[0]:, :].clone()
                traj_est = ptwt.waverec(inputs, wavelet=cfg.dwt_wave, axis=1)[:, :cfg.t_total, :]

            elif cfg.use_fft:
                zeros = torch.zeros((sampled_motion.shape[0], cfg.t_total - cfg.n_pre, sampled_motion.shape[2]),
                                    device=cfg.device, dtype=torch.float32)
                dft_coeffs = torch.cat((sampled_motion, zeros), dim=1).cpu().numpy()
                traj_est = torch.tensor(np.fft.ifft(dft_coeffs, axis=1).real, device=cfg.device, dtype=torch.float32)

            else:
                traj_est = sampled_motion

        else:
            n = traj.shape[0] // slice_num
            for s in range(slice_num):
                if s != slice_num - 1:
                    traj_tmp = traj[s * n:(s + 1) * n, :, :]
                else:
                    traj_tmp = traj[s * n:, :, :]

                # traj_dct padding过后的dct系数
                mode_dict, traj_dct, traj_dct_cond, vel_acc_pad, traj_pad = sample_preprocessing(traj_tmp, cfg,
                                                                                                 mode='metrics')

                if cfg.generator == 'diffusion':
                    sampled_motion = generator_model.sample_ddim(model_select,
                                                                 traj_dct,
                                                                 traj_dct_cond,
                                                                 mode_dict,
                                                                 vel_acc_pad=vel_acc_pad)
                elif cfg.generator == 'flow_matching':
                    generator_model.update_sampling_model(model)
                    if not cfg.res_fm:
                        sampled_motion = generator_model.sample_fm(mode_dict, traj_dct, traj_dct_cond, traj_pad,
                                                                   pre_inject=True)
                    else:
                        sampled_motion = generator_model.sample_res_fm(mode_dict, traj_dct, traj_dct_cond, traj_pad,
                                                                       pre_inject=True)

                if cfg.use_dct:
                    if cfg.mode != 'fine_tune' and cfg.mode != 'ft_eval':
                        traj_est_tmp = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
                        # traj_est = apply_limb_chain_direction_oracle(traj_est, traj, num_joints=cfg.joint_num)
                    else:
                        traj_est = sampled_motion

                elif cfg.use_dwt:
                    inputs = []
                    for i in range(0, len(cfg.dwt_lens)):
                        inputs.append(torch.zeros(sampled_motion.shape[0], cfg.dwt_lens[i], sampled_motion.shape[2],
                                                  device=cfg.device))

                    if cfg.dwt_lens[0] >= cfg.n_pre:  # 41 > 30
                        inputs[0][:, :cfg.n_pre, :] = sampled_motion.clone()
                    else:  # 21 < 30
                        inputs[0][:, :, :] = sampled_motion[:, :cfg.dwt_lens[0], :].clone()
                        diff_len = cfg.n_pre - cfg.dwt_lens[0]
                        inputs[1][:, :diff_len, :] = sampled_motion[:, cfg.dwt_lens[0]:, :].clone()
                    traj_est = ptwt.waverec(inputs, wavelet=cfg.dwt_wave, axis=1)[:, :cfg.t_total, :]
                elif cfg.use_fft:
                    zeros = torch.zeros((sampled_motion.shape[0], cfg.t_total - cfg.n_pre, sampled_motion.shape[2]),
                                        device=cfg.device, dtype=torch.float32)
                    dft_coeffs = torch.cat((sampled_motion, zeros), dim=1).cpu().numpy()
                    traj_est = torch.tensor(np.fft.ifft(dft_coeffs, axis=1).real, device=cfg.device,
                                            dtype=torch.float32)
                else:
                    traj_est_tmp = sampled_motion

                if s != slice_num - 1:
                    traj_est[s * n:(s + 1) * n, :, :] = traj_est_tmp
                else:
                    traj_est[s * n:, :, :] = traj_est_tmp

        # traj_est.shape (K, 125, 48)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        if not cfg.remove_root:
            traj_est = traj_est[..., 3:]

        # if cfg.kin:
        #     traj_est = cfg.kin.decode(traj_est)

        return traj_est

    def get_prediction_autoregression(data, model_select, mode='whole', slice_num=10):
        traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]
        mode_dict, traj_dct, traj_dct_cond, traj_pad = sample_preprocessing(traj, cfg, mode='autoregression')

        traj_est = clips_prediction(traj_dct, cfg, generator_model, encoder_model, model_select, mode_dict)

        # traj_est.shape (K, 125, 48)
        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est

    st = time.time()
    if multimodal_dict is None:
        gt_group = None
        data_group = None
        traj_gt_arr = None
        num_samples = None
        dataset_multi_test = None
    else:
        gt_group = multimodal_dict['gt_group']
        data_group = multimodal_dict['data_group']
        traj_gt_arr = multimodal_dict['traj_gt_arr']
        num_samples = multimodal_dict['num_samples']
        dataset_multi_test = multimodal_dict['dataset_multi_test']

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE']
    stats_meter = {x: AverageMeter() for x in stats_names}

    joint_ade_sum = None
    joint_fde_sum = None
    time_ade_sum = None
    joint_time_error_sum = None
    bone_ade_sum = torch.zeros(cfg.joint_num - 1, device='cuda')
    bone_fde_sum = torch.zeros(cfg.joint_num - 1, device='cuda')
    bone_time_sum = torch.zeros(cfg.t_pred,  device='cuda')
    angle_ade_sum = {
        "RKnee": 0.0,
        "LKnee": 0.0,
        "LElbow": 0.0,
        "RElbow": 0.0
    }

    angle_fde_sum = {
        "RKnee": 0.0,
        "LKnee": 0.0,
        "LElbow": 0.0,
        "RElbow": 0.0
    }

    K = 50
    pred = []
    pred_all = []
    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        if not cfg.autoregression:
            if torch.cuda.is_available():
                # 获取当前GPU的空闲/总显存（单位：字节）
                free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info()
                free_memory_gb = free_memory_bytes / (1024 ** 3)  # 转换为 GB

                # 显存阈值：≥12GB 用 whole（大模式），<12GB 用 sliced（显存节省模式）
                # 可根据你的显卡自由调整阈值：16GB显卡设12，24GB设20
                MEM_THRESHOLD = 12
                if free_memory_gb >= MEM_THRESHOLD:
                    mode_ = 'whole'
                else:
                    mode_ = 'sliced'
            else:
                # 无GPU(CPU环境)默认使用节省显存的 sliced 模式
                mode_ = 'sliced'

            pred_i_nd = get_prediction(data_group, model, mode=mode_, slice_num=10)

            if cfg.dct_norm_enable:
                pred_i_nd = dataset_multi_test.denormalize(pred_i_nd)

        else:
            pred_i_nd = get_prediction_autoregression(data_group, model, mode='whole')

        pred.append(pred_i_nd)

        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
            # pred [50, 5187, 125, 48] in h36m
            pred_all.append(pred)
            pred = pred[:, :, cfg.t_his:, :]
            # Use GPU to accelerate
            try:
                gt_group = torch.from_numpy(gt_group).to('cuda')
            except:
                pass
            try:
                pred = torch.from_numpy(pred).to('cuda')
            except:
                pass

            # pred [50, 5187, 100, 48]
            for j in range(num_samples):

                metrics = compute_all_metrics_detailed(
                    pred[:, j, :, :],
                    gt_group[j][np.newaxis, ...],
                    traj_gt_arr[j],
                    num_joints=cfg.joint_num,
                    dataset_name=cfg.dataset
                )

                stats_meter['APD'].update(metrics['APD'])
                stats_meter['ADE'].update(metrics['ADE'])
                stats_meter['FDE'].update(metrics['FDE'])
                stats_meter['MMADE'].update(metrics['MMADE'])
                stats_meter['MMFDE'].update(metrics['MMFDE'])

                if joint_ade_sum is None:
                    joint_ade_sum = metrics['joint_ade'].clone()
                    joint_fde_sum = metrics['joint_fde'].clone()
                    time_ade_sum = metrics['time_ade'].clone()
                    joint_time_error_sum = metrics['joint_time_error'].clone()
                else:
                    joint_ade_sum += metrics['joint_ade']
                    joint_fde_sum += metrics['joint_fde']
                    time_ade_sum += metrics['time_ade']
                    joint_time_error_sum += metrics['joint_time_error']

                bone_ade_sum += metrics['bone_ade']
                bone_fde_sum += metrics['bone_fde']
                bone_time_sum += metrics['bone_time_error']

                for k in angle_ade_sum.keys():
                    angle_ade_sum[k] += metrics["angle_ade"][k]
                    angle_fde_sum[k] += metrics["angle_fde"][k]

            for stats in stats_names:
                str_stats = f'{stats}: ' + ' '.join([f'{stats_meter[stats].avg.item():.4f}'])
                logger.info(str_stats)

            pred = []
            joint_ade_mean = joint_ade_sum / num_samples
            joint_fde_mean = joint_fde_sum / num_samples
            time_ade_mean = time_ade_sum / num_samples
            joint_time_error_mean = joint_time_error_sum / num_samples
            bone_ade_mean = bone_ade_sum / num_samples
            bone_fde_mean = bone_fde_sum / num_samples
            bone_time_mean = bone_time_sum / num_samples

            if cfg.joint_num == 16:
                joint_names = [
                    'RHip',
                    'RKnee',
                    'RFoot',
                    'LHip',
                    'LKnee',
                    'LFoot',
                    'Spine',
                    'Thorax',
                    'Neck',
                    'Head',
                    'LShoulder',
                    'LElbow',
                    'LWrist',
                    'RShoulder',
                    'RElbow',
                    'RWrist'
                ]
            else:
                joint_names = [f'J{i}' for i in range(cfg.joint_num)]

            logger.info('=' * 80)
            logger.info('Joint ADE')
            logger.info('=' * 80)

            for name, err in zip(joint_names, joint_ade_mean.cpu().numpy()):
                logger.info(f'{name:<15}: {err:.6f}')

            logger.info('=' * 80)
            logger.info('Joint FDE')
            logger.info('=' * 80)

            for name, err in zip(joint_names, joint_fde_mean.cpu().numpy()):
                logger.info(f'{name:<15}: {err:.6f}')

            joint_err_np = joint_ade_mean.cpu().numpy()

            idx = np.argsort(joint_err_np)[::-1]

            logger.info('=' * 80)
            logger.info('Worst Joint ADE')
            logger.info('=' * 80)
            for k in range(min(5, len(idx))):
                j = idx[k]
                logger.info(f'{joint_names[j]:<15}: ' f'{joint_err_np[j]:.6f}')

            logger.info("=" * 80)
            logger.info("Bone Length ADE")
            logger.info("=" * 80)
            for name, err in zip(BONE_NAMES, bone_ade_mean):
                logger.info(f"{name:<20}: {err.item():.6f}")

            logger.info("=" * 80)
            logger.info("Joint Angle ADE (degree)")
            logger.info("=" * 80)
            for k, v in angle_ade_sum.items():
                logger.info(f"{k:<10}: {(v / num_samples).item():.4f}")

            logger.info('=' * 80)
            logger.info('Time ADE')
            logger.info('=' * 80)
            time_np = time_ade_mean.cpu().numpy()
            for t, err in enumerate(time_np):
                logger.info(f't={t + 1:03d}: ' f'{err:.6f}')

            et = time.time()
            logger.info(f'{et - st:.4f} seconds')

            file_latest = '%s/stats_latest.csv'
            file_stat = '%s/stats.csv'

            with open(file_latest % cfg.result_dir, 'w') as csv_file:
                writer = csv.DictWriter(csv_file,fieldnames=['Metric', 'Value'])
                writer.writeheader()

                for stats, meter in stats_meter.items():
                    value = meter.avg

                    if torch.is_tensor(value):
                        value = value.item()

                    writer.writerow({'Metric': stats, 'Value': float(value)})

            df1 = pd.read_csv(file_latest % cfg.result_dir)

            if not os.path.exists(file_stat % cfg.result_dir):
                df1.to_csv(file_stat % cfg.result_dir, index=False)
            else:
                df2 = pd.read_csv(file_stat % cfg.result_dir)
                df = pd.concat([df2, df1], axis=1, ignore_index=True)

                df.to_csv(file_stat % cfg.result_dir, index=False)

            np.save(os.path.join(cfg.result_dir, 'joint_ade.npy'), joint_ade_mean.cpu().numpy())
            np.save(os.path.join(cfg.result_dir, 'joint_fde.npy'), joint_fde_mean.cpu().numpy())
            np.save(os.path.join(cfg.result_dir, 'time_ade.npy'), time_ade_mean.cpu().numpy())
            np.save(os.path.join(cfg.result_dir, 'joint_time_error.npy'), joint_time_error_mean.cpu().numpy())

            if save_results:
                pred_all = np.concatenate(pred_all, axis=0)
                np.save(os.path.join(cfg.result_dir, 'pred_all.npy'), pred_all)
                np.save(os.path.join(cfg.result_dir, 'data_all.npy'), data_group)


