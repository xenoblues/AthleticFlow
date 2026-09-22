"""Fast smoke test: load checkpoint + one small-batch forward pass. No full eval."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import sys
import time
sys.path.append(os.getcwd())
import torch
from tensorboardX import SummaryWriter
from utils import create_logger, seed_set
from utils.script import create_model_and_diffusion, dataset_split, sample_preprocessing
from config import Config, update_config

t0 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--ckpt', required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
args = parser.parse_args()

seed_set(args.seed)
cfg = Config(args.cfg, test=True)
args_dict = vars(args)
args_dict['mode'] = 'eval'
cfg = update_config(cfg, args_dict)

dataset, dataset_multi_test = dataset_split(cfg)
logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

model, diffusion = create_model_and_diffusion(cfg)

ckpt = torch.load(args.ckpt, map_location=cfg.device)
load_result = model.load_state_dict(ckpt, strict=False)
print(f"[LOAD] missing_keys={load_result.missing_keys} unexpected_keys={load_result.unexpected_keys}")
model.eval()

# one small batch, real test data
gen = dataset['test'].sampling_generator(num_samples=8, batch_size=8, aug=False)
data = next(gen)
# SMRNet's own get_prediction() (utils/evaluation.py) always drops the root
# joint (index 0) regardless of cfg.remove_root, and reshapes via a
# transpose-based [B, T, C] layout rather than a plain flatten -- mirror it
# exactly here.
traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
traj = torch.tensor(traj_np, device=cfg.device, dtype=cfg.dtype)
traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)

mode_dict, traj_dct, traj_dct_mod = sample_preprocessing(traj, cfg, mode='metrics')
with torch.no_grad():
    sampled_motion = diffusion.sample_ddim(model, traj_dct, traj_dct_mod, mode_dict)
traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)

print(f"[FORWARD] output_shape={tuple(traj_est.shape)}")
print(f"[TIME] elapsed={time.time()-t0:.1f}s")
print("[SUCCESS]")
