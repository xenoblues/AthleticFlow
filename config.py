import yaml
import os
import time
import torch
from utils import util, generate_pad

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

JOINT_NUM = {'ap3d': 16, 'ap': 16, 'wp': 23}  # joints without the root


def resolve_cfg_path(cfg_id):
    """'ap3d_af' -> cfg/ap3d_af.yml, 'ablation/ap3d_af_nobias' -> cfg/ablation/ap3d_af_nobias.yml, or a .yml path."""
    if cfg_id.endswith('.yml') or cfg_id.endswith('.yaml'):
        return cfg_id
    return os.path.join(ROOT_DIR, 'cfg', '%s.yml' % cfg_id)


def update_config(cfg, args_dict):
    """
    update some configuration related to args
        - merge args to cfg
        - dct, idct matrix
        - save path dir
    """
    for k, v in args_dict.items():
        setattr(cfg, k, v)

    cfg.dtype = torch.float32

    if cfg.temporal_basis == 'dct':
        cfg.dct_m, cfg.idct_m = util.get_dct_matrix(cfg.t_pred + cfg.t_his)
    elif cfg.temporal_basis == 'keyframe':
        # time-domain control: n_pre uniformly spaced keyframes + linear interpolation
        cfg.dct_m, cfg.idct_m = util.get_keyframe_matrices(cfg.t_pred + cfg.t_his, cfg.n_pre)
    else:
        raise ValueError(f"unknown temporal_basis: {cfg.temporal_basis}")
    cfg.dct_m_all = cfg.dct_m.float().to(cfg.device)
    cfg.idct_m_all = cfg.idct_m.float().to(cfg.device)

    # seed + PID suffix avoids collisions when several runs start within the same second
    formatted_time = "_" + time.strftime('%Y%m%d-%H%M%S', time.localtime()) + \
        "_seed%s_%d" % (args_dict.get('seed', 'NA'), os.getpid())
    run_name = os.path.splitext(os.path.basename(args_dict['cfg']))[0]
    cfg.cfg_dir = '%s/%s' % (cfg.base_dir, run_name + '_' + args_dict['mode'] + formatted_time)

    os.makedirs(cfg.cfg_dir, exist_ok=True)
    cfg.model_dir = '%s/models' % cfg.cfg_dir
    cfg.result_dir = '%s/results' % cfg.cfg_dir
    cfg.log_dir = '%s/log' % cfg.cfg_dir
    cfg.tb_dir = '%s/tb' % cfg.cfg_dir
    cfg.gif_dir = '%s/out' % cfg.cfg_dir
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.tb_dir, exist_ok=True)
    os.makedirs(cfg.gif_dir, exist_ok=True)
    cfg.model_path = os.path.join(cfg.model_dir)

    if cfg.save_metrics_interval == -1:
        cfg.save_metrics_interval = cfg.num_epoch - 1

    return cfg


class Config:
    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = resolve_cfg_path(cfg_id)
        if not os.path.exists(cfg_name):
            raise FileNotFoundError("Config file doesn't exist: %s" % cfg_name)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = 'inference' if test else 'results'
        os.makedirs(self.base_dir, exist_ok=True)

        # data
        self.dataset = cfg['dataset']
        if self.dataset not in JOINT_NUM:
            raise ValueError(f"unsupported dataset: {self.dataset}, supported: {list(JOINT_NUM)}")
        self.joint_num = JOINT_NUM[self.dataset]
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']
        self.t_total = self.t_his + self.t_pred
        self.padding = cfg.get('padding', 'LastFrame')
        self.idx_pad, self.zero_index = generate_pad(self.padding, self.t_his, self.t_pred)
        # optional custom multimodal-GT / candidate files (paths relative to the repository root)
        self.multimodal_path = cfg.get('multimodal_path', None)
        self.data_candi_path = cfg.get('data_candi_path', None)

        # training
        self.batch_size = cfg['batch_size']
        self.num_epoch = cfg['num_epoch']
        self.num_data_sample = cfg['num_data_sample']
        self.num_val_data_sample = cfg['num_val_data_sample']
        self.lr = cfg['lr']
        self.mod_train = cfg['mod_train']  # probability of keeping the condition (classifier-free guidance)
        self.mod_test = cfg['mod_test']

        # temporal representation: first n_pre coefficients of the DCT ('dct') or n_pre keyframes ('keyframe');
        # use_dct: False models all t_his + t_pred frames directly
        self.use_dct = cfg['use_dct']
        self.temporal_basis = cfg.get('temporal_basis', 'dct')
        self.n_pre = cfg['n_pre']

        # network
        self.model_name = cfg.get('model_name', 'AF')
        self.num_layers = cfg['num_layers']
        self.latent_dims = cfg['latent_dims']
        self.dropout = cfg['dropout']
        self.num_heads = cfg['num_heads']

        # AthleticFlow ablation switches (models/athletic_flow.py)
        self.use_ila = cfg.get('use_ila', True)
        self.use_anchor = cfg.get('use_anchor', True)
        self.bias_mode = cfg.get('bias_mode', 'both')
        self.n_chain_heads = cfg.get('n_chain_heads', 2)

        # sampling
        self.cfg_scale = cfg['cfg_scale']

        # evaluation: frames in front of the prediction used for the jitter at the observation/prediction seam;
        # 'observed' = last observed ground-truth frames, 'predicted' = the model's reconstruction of them
        self.jitter_seam = cfg.get('jitter_seam', 'observed')
        if self.jitter_seam not in ('observed', 'predicted'):
            raise ValueError(f"unknown jitter_seam: {self.jitter_seam}")
