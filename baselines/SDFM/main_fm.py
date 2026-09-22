import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  
# os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
sys.path.append(os.getcwd())
import argparse
import json
import time
from torchdiffeq._impl.odeint import SOLVERS
import utils
from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.script import *
from utils.training_fm import Trainer_fm
from config import Config, update_config
import torch
from tensorboardX import SummaryWriter
from utils.evaluation import compute_stats
from utils.draw import render_pictures, draw_diversity_comparison
from utils.training_fm_kd import Trainer_fm_kd
from thop import profile
from thop import clever_format


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
    parser.add_argument('--cfg', default='ap3d_sdfm',
                        help='h36m or humaneva' or 'assemble' or 'ap3d_wodct_itrans2l4_wosty' or 'he_dct15_itrans2l4_wosty')  # step_size 参数0.1
    # parser.add_argument('--cfg', default='h36m_dct30l8', help='h36m or humaneva' or 'assemble') # step_size 参数0.05
    parser.add_argument('--generator', default='flow_matching', type=str, help='flow_matching' or 'diffusion')
    parser.add_argument('--mode', default='train', help='train / eval / pred / switch/ control/ zero_shot/ draw/ fine_tune / ft_eval')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # parser.add_argument('--model_name', type=str, default='MotionTransformerTransfusion',
    #                     help='MotionTransformerOrigin'or 'MotionTransformerTransfusion' or 'MotioniTransformer')
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    # parser.add_argument('--milestone', type=list, default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--save_model_interval', type=int, default=50)
    parser.add_argument('--save_gif_interval', type=int, default=100)
    parser.add_argument('--save_metrics_interval', type=int, default=-1)
    parser.add_argument('--ckpt', type=str,
                        default='./results/ap_rtldfm_train_20260616-162050/models/ckpt_ema_500.pt',
                        help="h36m_dct30l8_train_2025-03-25-19-47/models/ckpt_ema_1000.pt" or
                             "he_bs256_dct15_train_2025-03-25-00-37/models/ckpt_ema_500.pt")
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_switch_num', type=int, default=10)
    parser.add_argument('--vis_col', type=int, default=5)
    parser.add_argument('--vis_row', type=int, default=1)
    parser.add_argument("--ode_method", default="euler", choices=list(SOLVERS.keys()) + ["edm_heun"],
                        help="ODE solver used to generate samples.")
    parser.add_argument("--ode_options", default='{"step_size": 0.1}', type=json.loads,
                        help="ODE solver options. Eg. the midpoint solver requires step-size, dopri5 has no options to set.")
    # parser.add_argument("--cfg_scale", default=0.0, type=float, help="Classifier-free guidance scale for generating samples.")
    parser.add_argument("--skewed_timesteps", action="store_true",
                        help="Use skewed timestep sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.")
    parser.add_argument("--edm_schedule", default=False, action="store_true",
                        help="Use the alternative time discretization during sampling proposed in the EDM paper: https://arxiv.org/abs/2206.00364.")
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

    temporal_graph = utils.get_temporal_graph(cfg.n_pre)

    """model"""
    # encoder for autoregression model
    model, generator, encoder, t_model = create_model_and_diffusion(cfg, dataset["train"].skeleton.multiscale_filters(),
                                                                    temporal_graph)

    # input1 = torch.randn(1, cfg.n_pre, 3 * cfg.joint_num).cuda()
    # input2 = torch.randn(1).cuda()
    # input3 = torch.rand_like(input1).cuda()
    # flops, params = profile(model, inputs=(input1, input2, input3))
    # flops, params = clever_format([flops, params], '%.3f')
    # print(f"运算量：{flops}, 参数量：{params}")

    total_params = sum(p.numel() for p in list(model.parameters())) / 1000000.0
    if encoder:
        total_params += sum(p.numel() for p in list(encoder.parameters())) / 1000000.0
    logger.info(">>> total params: {:.2f}M".format(total_params))

    if args.mode == 'train' or args.mode == 'fine_tune':
        # prepare full evaluation dataset
        if dataset_multi_test is not None:
            multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        else:
            multimodal_dict = get_multimodal_gt_full_custom(logger, dataset['test'], args, cfg)
        if not cfg.knowledge_distill:
            trainer = Trainer_fm(
                model=model,
                generator=generator,
                dataset=dataset,
                cfg=cfg,
                multimodal_dict=multimodal_dict,
                logger=logger,
                tb_logger=tb_logger)
        else:
            t_models = []
            for i in range(len(cfg.teacher_pt_paths)):
                t_model_ = copy.deepcopy(t_model).eval().requires_grad_(False).to(cfg.device)
                t_model_.load_state_dict(torch.load(cfg.teacher_pt_paths[i]))
                t_models.append(t_model_)
            del t_model
            trainer = Trainer_fm_kd(
                s_model=model,
                t_models=t_models,
                generator=generator,
                dataset=dataset,
                cfg=cfg,
                multimodal_dict=multimodal_dict,
                logger=logger,
                tb_logger=tb_logger)
        trainer.loop()
    elif args.mode == 'eval' or args.mode == 'ft_eval':
        ckpt = torch.load(args.ckpt)
        load_result = model.load_state_dict(ckpt, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            logger.info(f"load_state_dict(strict=False): missing_keys={load_result.missing_keys}, "
                        f"unexpected_keys={load_result.unexpected_keys}")
        model.eval()
        if generator.refiner and cfg.refiner_ckpt_path:
            ckpt = torch.load(cfg.refiner_ckpt_path)
            generator.refiner.load_state_dict(ckpt)
            generator.refiner.eval()
        # prepare full evaluation dataset
        if dataset_multi_test is not None:
            multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        else:
            multimodal_dict = get_multimodal_gt_full_custom(logger, dataset['test'], args, cfg)
        compute_stats(generator, multimodal_dict, model, logger, cfg, save_results=False)
    elif args.mode == 'draw':
        # render_pictures(cfg.dataset, dataset['test'].skeleton, cfg.t_his, fix_0=True, azim=0.0, output=None, mode='pred', size=2,
        #                     ncol=12, bitrate=3000, fix_index=None)
        draw_diversity_comparison(cfg.dataset, dataset['test'].skeleton, cfg.t_his, fix_0=True, azim=0.0, output=None,
                                  mode='pred', size=4,
                                  ncol=12, bitrate=3000, fix_index=None)
    else:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        demo_visualize(args.mode, cfg, model, generator, dataset)
