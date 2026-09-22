"""Fast smoke test: load checkpoint + one small-batch forward pass. No full eval."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import json
import sys
import time
sys.path.append(os.getcwd())
import torch
import utils
from utils import create_logger, seed_set
from utils.script import create_model_and_diffusion, dataset_split, sample_preprocessing
from config import Config, update_config

t0 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--ckpt', required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--generator', default='flow_matching')
parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
parser.add_argument('--save_metrics_interval', type=int, default=-1)
parser.add_argument('--ode_method', default='euler')
parser.add_argument('--ode_options', default='{"step_size": 0.1}', type=json.loads)
parser.add_argument('--skewed_timesteps', action='store_true')
parser.add_argument('--edm_schedule', default=False, action='store_true')
args = parser.parse_args()

seed_set(args.seed)
cfg = Config(args.cfg, test=True)
args_dict = vars(args)
args_dict['mode'] = 'eval'
cfg = update_config(cfg, args_dict)

dataset, dataset_multi_test = dataset_split(cfg)
logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
temporal_graph = utils.get_temporal_graph(cfg.n_pre)

model, generator, encoder, t_model = create_model_and_diffusion(
    cfg, dataset["train"].skeleton.multiscale_filters(), temporal_graph)

ckpt = torch.load(args.ckpt, map_location=cfg.device)
load_result = model.load_state_dict(ckpt, strict=False)
print(f"[LOAD] missing_keys={load_result.missing_keys} unexpected_keys={load_result.unexpected_keys}")
model.eval()

gen = dataset['test'].sampling_generator(num_samples=8, batch_size=8, aug=False)
data, _ = next(gen)
traj_np = data[..., 1:, :].reshape([data.shape[0], cfg.t_his + cfg.t_pred, -1]) if cfg.remove_root else \
    data.reshape([data.shape[0], cfg.t_his + cfg.t_pred, -1])
traj = torch.tensor(traj_np, device=cfg.device, dtype=cfg.dtype)

mode_dict, traj_dct, traj_dct_cond, vel_acc_pad, traj_pad = sample_preprocessing(traj, cfg, mode='metrics')
with torch.no_grad():
    if not cfg.res_fm:
        sampled_motion = generator.sample_fm(mode_dict, traj_dct, traj_dct_cond, traj_pad, pre_inject=True)
    else:
        sampled_motion = generator.sample_res_fm(mode_dict, traj_dct, traj_dct_cond, traj_pad)

if cfg.b_frequency_transform and cfg.use_dct:
    traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
else:
    traj_est = sampled_motion

print(f"[FORWARD] output_shape={tuple(traj_est.shape)}")
print(f"[TIME] elapsed={time.time()-t0:.1f}s")
print("[SUCCESS]")
