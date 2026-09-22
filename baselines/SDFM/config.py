import yaml
import os
import time
import torch
from utils import util, generate_pad
from models.bone_kinematics import RootBoneKinematics


def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = '_' + str(len(dirs))

    return log_dir_index


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

    cfg.dct_m, cfg.idct_m = util.get_dct_matrix(cfg.t_pred + cfg.t_his)
    cfg.dct_m_all = cfg.dct_m.float().to(cfg.device)
    cfg.idct_m_all = cfg.idct_m.float().to(cfg.device)

    # index = get_log_dir_index(cfg.base_dir)
    # seed + PID suffix avoids collisions when multiple array tasks (e.g. different
    # seeds of the same cfg) start within the same wall-clock second -- this has
    # already happened in practice (see e.g. results/ap3d_af_train_20260910-094130,
    # which silently merged seeds 1/2/3/4 into one directory and corrupted their
    # checkpoints).
    formatted_time = "_" + time.strftime('%Y%m%d-%H%M%S', time.localtime()) + \
        "_seed%s_%d" % (args_dict.get('seed', 'NA'), os.getpid())
    if args_dict['mode'] == 'train' or  args_dict['mode'] == 'pred' or args_dict['mode'] == 'eval' or args_dict['mode'] == 'fine_tune':
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['cfg'] + '_' +args_dict['mode'] + formatted_time)
    else:
        cfg.cfg_dir = '%s/%s' % (cfg.base_dir, args_dict['mode'] + formatted_time)

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

    # flow matching
    cfg.generator = args_dict['generator']
    if cfg.generator == 'flow_matching':
        cfg.ode_method = args_dict['ode_method']
        cfg.ode_options = args_dict['ode_options']
        cfg.skewed_timesteps = args_dict['skewed_timesteps']
        cfg.edm_schedule = args_dict['edm_schedule']

    if cfg.save_metrics_interval == -1:
        cfg.save_metrics_interval = cfg.num_epoch - 1

    cfg.hdct = util.HierarchicalDCTTruncation(cfg.dct_m_all, k_max=cfg.n_pre, k_torso=cfg.n_pre-5, k_mid=cfg.n_pre-3, k_end=cfg.n_pre,
                                               num_joints=cfg.joint_num).to(cfg.device)
    # cfg.kin = RootBoneKinematics(num_joints=16).to(cfg.device)
    cfg.kin = None

    return cfg


class Config:
    def __init__(self, cfg_id, test=False):
        self.id = cfg_id
        cfg_name = './cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.base_dir = 'inference' if test else 'results'
        os.makedirs(self.base_dir, exist_ok=True)

        # common
        self.dataset = cfg.get('dataset', 'h36m')
        self.batch_size = cfg['batch_size']
        self.normalize_data = cfg.get('normalize_data', False)
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']

        self.num_epoch = cfg['num_epoch']
        self.num_data_sample = cfg['num_data_sample']
        self.num_val_data_sample = cfg['num_val_data_sample']
        self.lr = cfg['lr']

        self.n_pre = cfg['n_pre']
        self.multimodal_path = cfg['multimodal_path']
        self.data_candi_path = cfg['data_candi_path']

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

        self.use_dct = cfg['use_dct']
        self.dct_norm_enable = cfg['dct_norm_enable']

        self.resume = cfg['resume']
        self.ckpt_path = cfg['ckpt_path']
        self.refiner_ckpt_path = cfg.get('refiner_ckpt_path', None)
        self.frame_mask = cfg['frame_mask']
        self.joint_mask = cfg['joint_mask']
        self.residual_data = cfg['residual_data']
        self.random_sample = cfg['random_sample']

        self.clip_his = cfg['clip_his']
        self.clip_pred = cfg['clip_pred']
        self.autoregression = cfg['autoregression']
        self.iTransformer = cfg['iTransformer']
        self.cross_attention = cfg['cross_attention']
        self.stylization_block = cfg['stylization_block']
        self.flash_attention = cfg['flash_attention']
        self.se_layer = cfg['se_layer']
        self.milestone = cfg['milestone']
        self.cfg_scale = cfg['cfg_scale']
        self.skip_type = cfg['skip_type']
        self.model_name = cfg.get('model_name', 'MotionTransformerTransfusion')

        self.parallel = cfg.get('parallel', False)

        self.use_dwt = cfg.get('use_dwt', False)
        self.dwt_wave = cfg.get('dwt_wave', 'haar')
        self.dwt_level = cfg.get('dwt_level', 2)
        self.dwt_transpose = cfg.get('dwt_transpose', False)
        self.dwt_lens = []

        self.use_fft = cfg.get('use_fft', False)

        self.knowledge_distill = cfg.get('knowledge_distill', False)
        self.teacher_pt_paths = cfg.get('teacher_pt_paths', [])
        self.reg_kd_scale = cfg.get('reg_kd_scale', 0.0)
        self.fea_kd_scale = cfg.get('fea_kd_scale', 0.0)
        self.use_casual_mask = cfg.get('use_casual_mask', False)

        t_cfg_name = './cfg/%s.yml' % cfg.get('teacher_cfg', None)
        if os.path.exists(t_cfg_name):
            t_cfg = yaml.safe_load(open(t_cfg_name, 'r'))
            self.t_num_layers = t_cfg['num_layers']
            self.t_latent_dims = t_cfg['latent_dims']
            self.t_num_heads = t_cfg['num_heads']
        else:
            print("Teacher config file doesn't exist: %s" % t_cfg_name)

        # indirect variable
        if self.dataset == 'h36m' or self.dataset == 'ap3d' or self.dataset == 'ap' or self.dataset == 'sp':
            self.joint_num = 16
        elif self.dataset == 'humaneva':
            self.joint_num = 14
        elif self.dataset == 'assemble':
            self.joint_num = 13
        elif self.dataset == 'wp':
            self.joint_num = 23

        self.remove_root = cfg.get('remove_root', True)
        self.res_fm = cfg.get('res_fm', False)

        # AthleticFlow (models/athletic_flow.py) ablation switches.
        self.use_ila = cfg.get('use_ila', True)
        self.use_anchor = cfg.get('use_anchor', True)
        self.bias_mode = cfg.get('bias_mode', 'both')

        if not self.remove_root:
            self.joint_num += 1

        self.idx_pad, self.zero_index = generate_pad(self.padding, self.t_his, self.t_pred)
        self.clip_total = self.clip_his + self.clip_pred
        self.t_total = self.t_his + self.t_pred
        self.b_frequency_transform = self.use_dct or self.use_dwt or self.use_fft

