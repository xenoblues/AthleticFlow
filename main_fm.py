import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.script import create_model_and_generator, dataset_split, get_multimodal_gt_full, display_exp_setting
from utils.training_fm import Trainer_fm
from config import Config, update_config
import torch
from tensorboardX import SummaryWriter
from utils.evaluation import compute_stats


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='ap3d_af',
                        help="config name under cfg/ (e.g. ap3d_af, ap_af, wp_af, ap3d_sdfm, ablation/ap3d_af_nobias) "
                             "or a path to a .yml file")
    parser.add_argument('--mode', default='train', help='train / eval / pred')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--save_model_interval', type=int, default=50)
    parser.add_argument('--save_gif_interval', type=int, default=100)
    parser.add_argument('--save_metrics_interval', type=int, default=-1,
                        help='train: evaluate every N epochs (-1: only after the last epoch)')
    parser.add_argument('--ckpt', type=str, default='./checkpoints/ap3d_af.pt', help='eval / pred: checkpoint to load')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_col', type=int, default=5)
    parser.add_argument('--vis_row', type=int, default=1)
    parser.add_argument("--ode_method", default="euler", choices=["euler", "midpoint"],
                        help="ODE solver used to generate samples.")
    parser.add_argument("--ode_options", default='{"step_size": 0.1}', type=json.loads,
                        help="ODE solver options, e.g. the step size of the fixed-grid solvers.")
    parser.add_argument("--skewed_timesteps", action="store_true",
                        help="Use skewed timestep sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.")
    parser.add_argument("--edm_schedule", default=False, action="store_true",
                        help="Use the alternative time discretization during sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.")
    # evaluation outputs
    parser.add_argument('--save_pred', action='store_true',
                        help='eval: save all K samples to <result_dir>/pred_all.npy (large)')
    parser.add_argument('--save_per_window', action='store_true',
                        help='eval: save per-window metrics to <result_dir>/per_window.npz')
    parser.add_argument('--save_windows', type=str, default='',
                        help='eval: comma-separated test-window indices whose K samples are saved to '
                             '<result_dir>/selected_windows.npz')
    args = parser.parse_args()

    """setup"""
    seed_set(args.seed)

    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))
    dataset, dataset_multi_test = dataset_split(cfg)
    """logger"""
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)

    """model"""
    model, generator = create_model_and_generator(cfg)

    total_params = sum(p.numel() for p in list(model.parameters())) / 1000000.0
    logger.info(">>> total params: {:.2f}M".format(total_params))

    if args.mode == 'train':
        # prepare full evaluation dataset
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, cfg)
        trainer = Trainer_fm(
            model=model,
            generator=generator,
            dataset=dataset,
            cfg=cfg,
            multimodal_dict=multimodal_dict,
            logger=logger,
            tb_logger=tb_logger)
        trainer.loop()

    elif args.mode == 'eval':
        ckpt = torch.load(args.ckpt, map_location=cfg.device)
        model.load_state_dict(ckpt)
        model.eval()
        # prepare full evaluation dataset
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, cfg)
        compute_stats(generator, multimodal_dict, model, logger, cfg, save_results=args.save_pred)

    elif args.mode == 'pred':
        ckpt = torch.load(args.ckpt, map_location=cfg.device)
        model.load_state_dict(ckpt)
        model.eval()
        demo_visualize(args.mode, cfg, model, generator, dataset)

    else:
        raise NotImplementedError(f"unknown mode: {args.mode}")
