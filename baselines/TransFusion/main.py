import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import sys
from utils import create_logger, seed_set
from utils.demo_visualize import demo_visualize
from utils.script import *
import numpy as np
sys.path.append(os.getcwd())
from config import Config, update_config
import torch
from tensorboardX import SummaryWriter
from utils.training import Trainer
from utils.evaluation import compute_stats, compute_stats_cs

from data_loader.dataset_amass import DatasetAMASS
from thop import profile
from thop import clever_format

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', default='wp_tf', help='h36m or humaneva or amass')
    parser.add_argument('--mode', default='train', help='train / eval / pred')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--milestone', type=list, default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--save_model_interval', type=int, default=100)
    parser.add_argument('--save_metrics_interval', type=int, default=-1)
    parser.add_argument('--ckpt', type=str, default='./results/wp_tf_train_20260613-144522/models/ckpt_ema_1000.pt')
    parser.add_argument('--ema', type=bool, default=True)
    parser.add_argument('--vis_col', type=int, default=5)
    parser.add_argument('--vis_row', type=int, default=1)
    args = parser.parse_args()

    """setup"""
    seed_set(args.seed)

    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))

    if cfg.dataset == 'amass':
        dataset = {'train': DatasetAMASS('train'), 'test': DatasetAMASS('test')}
    else:
        dataset, dataset_multi_test = dataset_split(cfg)


    """logger"""
    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)
    """model"""
    model, diffusion = create_model_and_diffusion(cfg)
    if cfg.dataset == 'amass':
        multimodal_dict = get_multimodal_gt_full(logger, dataset['test'], args, cfg)
    if cfg.dataset == 'ap' or cfg.dataset == 'ap3d' or cfg.dataset == 'wp':
        multimodal_dict = get_multimodal_gt_full_ap(logger, dataset_multi_test, args, cfg)
    else:
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
    
    # input1 = torch.randn(1, cfg.n_pre, 3 * cfg.joint_num).cuda()
    # input2 = torch.randn(1).cuda()
    # input3 = torch.rand_like(input1).cuda()
    #
    # flops, params = profile(model, inputs=(input1, input2, input3))
    # flops, params = clever_format([flops, params], '%.3f')
    #
    # print(f"运算量：{flops}, 参数量：{params}")

    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.mode == 'train':
        trainer = Trainer(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            cfg=cfg,
            logger=logger,
            tb_logger=tb_logger,
            multimodal_dict=multimodal_dict)
        trainer.loop()

    elif args.mode == 'eval':
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        model.eval()
        if cfg.dataset == 'amass':
            multimodal_dict = get_multimodal_gt_full(logger, dataset['test'], args, cfg)
        if cfg.dataset == 'ap' or cfg.dataset == 'ap3d' or cfg.dataset == 'wp':
            multimodal_dict = get_multimodal_gt_full_ap(logger, dataset_multi_test, args, cfg)
        else:
            multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)

        if cfg.dataset == 'ap' or cfg.dataset == 'ap3d' or cfg.dataset == 'wp':
            compute_stats_cs(diffusion, multimodal_dict, model, logger, cfg, save_results=True)
        else:
            compute_stats(diffusion, multimodal_dict, model, logger, cfg)

    else:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        demo_visualize(args.mode, cfg, model, diffusion, dataset)
