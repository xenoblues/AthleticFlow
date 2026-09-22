import csv
import time
import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

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

def compute_stats(diffusion, multimodal_dict, model, logger, cfg,
                  save_results=False):
    """
     The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
     several (K=50) times.
     """

    def get_prediction(data, model_select, mode='whole', slice_num=10):
        traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        traj_est = torch.zeros(traj.shape, device=cfg.device, dtype=torch.float32)

        if mode == 'whole':
            mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
            sampled_motion = diffusion.sample_ddim(model_select,
                                                   traj_dct,
                                                   traj_dct_cond,
                                                   mode_dict)
            traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)

        else:
            n = traj.shape[0] // slice_num
            for s in range(slice_num):
                if s != slice_num - 1:
                    traj_tmp = traj[s * n:(s + 1) * n, :, :]
                else:
                    traj_tmp = traj[s * n:, :, :]
            mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj_tmp, cfg, mode='metrics')
            sampled_motion = diffusion.sample_ddim(model_select,
                                                   traj_dct,
                                                   traj_dct_cond,
                                                   mode_dict)
            traj_est_tmp = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)

            if s != slice_num - 1:
                traj_est[s * n:(s + 1) * n, :, :] = traj_est_tmp
            else:
                traj_est[s * n:, :, :] = traj_est_tmp

        traj_est = traj_est.cpu().numpy()
        traj_est = traj_est[None, ...]
        return traj_est

    gt_group = multimodal_dict['gt_group']
    data_group = multimodal_dict['data_group']
    traj_gt_arr = multimodal_dict['traj_gt_arr']
    num_samples = multimodal_dict['num_samples']
    dataset_multi_test = multimodal_dict['dataset_multi_test']

    # stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'ADE-m', 'FDE-m', 'MMADE-m', 'MMFDE-m', 'ADE-w', 'FDE-w',
    #                'MMADE-w', 'MMFDE-w']
    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE',
                   'jitter', 'jitter_all', 'jitter_gt',
                   'accel', 'accel_gt', 'accel_error']
    stats_meter = {x: AverageMeter() for x in stats_names}

    fps = get_fps(cfg.dataset, cfg)

    joint_ade_sum = None
    joint_fde_sum = None
    time_ade_sum = None
    joint_time_error_sum = None
    joint_jitter_sum = None
    joint_accel_error_sum = None
    n_bones = len(get_skeleton_edges(cfg.dataset))
    bone_ade_sum = torch.zeros(n_bones, device='cuda')
    bone_fde_sum = torch.zeros(n_bones, device='cuda')
    bone_time_sum = torch.zeros(cfg.t_pred, device='cuda')
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

    st = time.time()
    K = 50
    pred = []
    pred_all = []
    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
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

        pred.append(pred_i_nd)
        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
            pred_all.append(pred)
            # pred [50, 5187, 125, 48] in h36m
            hist = pred[:, :, max(cfg.t_his - 3, 0):cfg.t_his, :]
            pred = pred[:, :, cfg.t_his:, :]
            print('Got 50 predictions')
            # Use GPU to accelerate
            try:
                gt_group = torch.from_numpy(gt_group).to('cuda')
            except:
                pass
            try:
                pred = torch.from_numpy(pred).to('cuda')
            except:
                pass
            try:
                hist = torch.from_numpy(hist).to('cuda')
            except:
                pass

            for j in range(num_samples):

                metrics = compute_all_metrics_detailed(
                    pred[:, j, :, :],
                    gt_group[j][np.newaxis, ...],
                    traj_gt_arr[j],
                    num_joints=cfg.joint_num,
                    dataset_name=cfg.dataset,
                    fps=fps,
                    hist=hist[0, j] if hist is not None else None
                )

                stats_meter['APD'].update(metrics['APD'])
                stats_meter['ADE'].update(metrics['ADE'])
                stats_meter['FDE'].update(metrics['FDE'])
                stats_meter['MMADE'].update(metrics['MMADE'])
                stats_meter['MMFDE'].update(metrics['MMFDE'])
                stats_meter['jitter'].update(metrics['jitter'])
                stats_meter['jitter_all'].update(metrics['jitter_all'])
                stats_meter['jitter_gt'].update(metrics['jitter_gt'])
                stats_meter['accel'].update(metrics['accel'])
                stats_meter['accel_gt'].update(metrics['accel_gt'])
                stats_meter['accel_error'].update(metrics['accel_error'])

                if joint_ade_sum is None:
                    joint_ade_sum = metrics['joint_ade'].clone()
                    joint_fde_sum = metrics['joint_fde'].clone()
                    time_ade_sum = metrics['time_ade'].clone()
                    joint_time_error_sum = metrics['joint_time_error'].clone()
                    joint_jitter_sum = metrics['joint_jitter'].clone()
                    joint_accel_error_sum = metrics['joint_accel_error'].clone()
                else:
                    joint_ade_sum += metrics['joint_ade']
                    joint_fde_sum += metrics['joint_fde']
                    time_ade_sum += metrics['time_ade']
                    joint_time_error_sum += metrics['joint_time_error']
                    joint_jitter_sum += metrics['joint_jitter']
                    joint_accel_error_sum += metrics['joint_accel_error']

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
            joint_jitter_mean = joint_jitter_sum / num_samples
            joint_accel_error_mean = joint_accel_error_sum / num_samples
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

            logger.info('=' * 80)
            logger.info('Joint Jitter (unit/s^3)')
            logger.info('=' * 80)

            for name, err in zip(joint_names, joint_jitter_mean.cpu().numpy()):
                logger.info(f'{name:<15}: {err:.6f}')

            logger.info('=' * 80)
            logger.info('Joint Accel Error (unit/s^2)')
            logger.info('=' * 80)

            for name, err in zip(joint_names, joint_accel_error_mean.cpu().numpy()):
                logger.info(f'{name:<15}: {err:.6f}')

            joint_err_np = joint_ade_mean.cpu().numpy()

            idx = np.argsort(joint_err_np)[::-1]

            logger.info('=' * 80)
            logger.info('Worst Joint ADE')
            logger.info('=' * 80)
            for k in range(min(5, len(idx))):
                j = idx[k]
                logger.info(f'{joint_names[j]:<15}: ' f'{joint_err_np[j]:.6f}')

            jitter_err_np = joint_jitter_mean.cpu().numpy()
            jitter_idx = np.argsort(jitter_err_np)[::-1]

            logger.info('=' * 80)
            logger.info('Worst Joint Jitter')
            logger.info('=' * 80)
            for k in range(min(5, len(jitter_idx))):
                j = jitter_idx[k]
                logger.info(f'{joint_names[j]:<15}: ' f'{jitter_err_np[j]:.6f}')

            logger.info("=" * 80)
            logger.info("Bone Length ADE")
            logger.info("=" * 80)
            for name, err in zip(get_bone_names(cfg.dataset, cfg.joint_num), bone_ade_mean):
                logger.info(f"{name:<20}: {err.item():.6f}")

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
            time_np = time_ade_mean.cpu().numpy()
            for t, err in enumerate(time_np):
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
            np.save(os.path.join(cfg.result_dir, 'joint_jitter.npy'), joint_jitter_mean.cpu().numpy())
            np.save(os.path.join(cfg.result_dir, 'joint_accel_error.npy'), joint_accel_error_mean.cpu().numpy())

            if save_results:
                pred_all = np.concatenate(pred_all, axis=0)
                np.save(os.path.join(cfg.result_dir, 'pred_all.npy'), pred_all)
                np.save(os.path.join(cfg.result_dir, 'data_all.npy'), data_group)

