import os
import time

import torch
import yaml

from utils import generate_pad
from utils.util import get_dct_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))
# main AthleticFlow repository data folder (baselines/TransFusion/../../data)
DEFAULT_DATA_ROOT = os.path.normpath(os.path.join(ROOT, '..', '..', 'data'))
JOINT_NUM = {'ap3d': 16, 'ap': 16, 'wp': 23}  # without the root joint


class Config:

    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = os.path.join(ROOT, 'cfg', f'{cfg_id}.yml')
        if not os.path.exists(cfg_name):
            raise FileNotFoundError(f"Config file doesn't exist: {cfg_name}")
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        self.base_dir = os.path.join(ROOT, 'inference' if test else 'results')
        os.makedirs(self.base_dir, exist_ok=True)

        # data
        self.dataset = cfg['dataset']
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']
        self.multimodal_path = cfg['multimodal_path']  # relative to data_root
        self.data_candi_path = cfg['data_candi_path']  # relative to data_root
        # 'observed': the jitter of the first predicted frames uses the observed history
        # 'predicted': it uses the model's reconstruction of the history (protocol of the WorldPose table)
        self.jitter_seam = cfg.get('jitter_seam', 'observed')

        # training
        self.batch_size = cfg['batch_size']
        self.num_epoch = cfg['num_epoch']
        self.num_data_sample = cfg['num_data_sample']
        self.num_val_data_sample = cfg['num_val_data_sample']
        self.lr = cfg['lr']

        # diffusion / model
        self.n_pre = cfg['n_pre']
        self.padding = cfg['padding']
        self.Complete = cfg['Complete']
        self.noise_steps = cfg['noise_steps']
        self.ddim_timesteps = cfg['ddim_timesteps']
        self.scheduler = cfg['scheduler']
        self.num_layers = cfg['num_layers']
        self.latent_dims = cfg['latent_dims']
        self.dropout = cfg['dropout']
        self.num_heads = cfg['num_heads']
        self.mod_train = cfg['mod_train']
        self.mod_test = cfg['mod_test']
        self.model_name = cfg.get('model_name', None)  # None: TransFusion, 'hmac': HumanMAC

        self.joint_num = JOINT_NUM[self.dataset]
        self.idx_pad, self.zero_index = generate_pad(self.padding, self.t_his, self.t_pred)

    @property
    def variant(self):
        return 'tf' if self.model_name is None else self.model_name


def update_config(cfg, args_dict):
    """merge args into cfg, build DCT matrices and output directories"""
    for k, v in args_dict.items():
        setattr(cfg, k, v)
    if cfg.data_root is None:
        cfg.data_root = DEFAULT_DATA_ROOT

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfg.dtype = dtype

    cfg.dct_m, cfg.idct_m = get_dct_matrix(cfg.t_pred + cfg.t_his)
    cfg.dct_m_all = cfg.dct_m.float().to(cfg.device)
    cfg.idct_m_all = cfg.idct_m.float().to(cfg.device)

    stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    cfg.cfg_dir = os.path.join(cfg.base_dir, f"{args_dict['cfg']}_{args_dict['mode']}_seed{args_dict['seed']}_{stamp}")
    cfg.model_dir = os.path.join(cfg.cfg_dir, 'models')
    cfg.result_dir = os.path.join(cfg.cfg_dir, 'results')
    cfg.log_dir = os.path.join(cfg.cfg_dir, 'log')
    cfg.tb_dir = os.path.join(cfg.cfg_dir, 'tb')
    for d in (cfg.model_dir, cfg.result_dir, cfg.log_dir, cfg.tb_dir):
        os.makedirs(d, exist_ok=True)
    cfg.model_path = cfg.model_dir

    if cfg.save_metrics_interval == -1:
        cfg.save_metrics_interval = cfg.num_epoch - 1

    return cfg
