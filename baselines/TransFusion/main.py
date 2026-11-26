import argparse
import os

import torch
from tensorboardX import SummaryWriter

from config import ROOT, Config, update_config
from utils import create_logger, seed_set
from utils.evaluation import compute_stats_cs
from utils.script import create_model_and_diffusion, dataset_split, display_exp_setting, get_multimodal_gt_full
from utils.training import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='ap_tf', help='{ap3d,ap,wp}_{tf,hmac}')
    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_root', type=str, default=None,
                        help='dataset folder (default: <AthleticFlow repo>/data)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='eval checkpoint (default: weights/<dataset>/<tf|hmac>.pt)')
    parser.add_argument('--sample_mode', default='auto', choices=['auto', 'whole', 'sliced'],
                        help="evaluation batching; 'auto' = whole test set per batch if >= 12 GB GPU memory is free")
    parser.add_argument('--save_pred', action='store_true', help='also save all K predictions (large)')
    parser.add_argument('--milestone', type=int, nargs='+',
                        default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--save_model_interval', type=int, default=100)
    parser.add_argument('--save_metrics_interval', type=int, default=-1)
    args = parser.parse_args()
    args.device = torch.device(args.device)

    seed_set(args.seed)

    cfg = Config(args.cfg, test=(args.mode != 'train'))
    cfg = update_config(cfg, vars(args))

    dataset, dataset_multi_test = dataset_split(cfg, train=(args.mode == 'train'))

    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)

    model, diffusion = create_model_and_diffusion(cfg)
    multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, cfg)
    logger.info('>>> total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    if args.mode == 'train':
        Trainer(model=model, diffusion=diffusion, dataset=dataset, cfg=cfg, logger=logger,
                tb_logger=tb_logger, multimodal_dict=multimodal_dict).loop()
    else:
        ckpt = args.ckpt or os.path.join(ROOT, 'weights', cfg.dataset, f'{cfg.variant}.pt')
        logger.info(f'checkpoint: {ckpt}')
        model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
        model.eval()
        compute_stats_cs(diffusion, multimodal_dict, model, logger, cfg, save_pred=args.save_pred)
