"""Fast smoke test: load checkpoint + one small-batch forward pass. No full eval,
no TrainerCustom (its __init__ unconditionally computes the full multimodal GT set)."""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import sys
import time
sys.path.append(os.getcwd())
import torch

from utils import Config
from models.load_models import get_model
from models.GaussianDiffusion import GaussianDiffusion
from data_utils.dataset_ap3d import DatasetAP3D
from data_utils.dataset_athleticspose import DatasetAthleticsPose
from data_utils.dataset_wp import DatasetWP

t0 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--ckpt', required=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu_index', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
cfg = Config(args.cfg, test=True, run_dir='/tmp/comusion_quickcheck_%s' % args.cfg)
dtype = torch.float32 if cfg.dtype == 'float32' else torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

dataset_cls = {'ap3d': DatasetAP3D, 'ap': DatasetAthleticsPose, 'wp': DatasetWP}[cfg.dataset]
eval_dataset = dataset_cls(mode='test', t_his=cfg.t_his, t_pred=cfg.t_pred)

model = get_model(cfg).to(dtype).to(device)
diffuser = GaussianDiffusion(
    model=model,
    cfg=cfg,
    future_motion_size=(cfg.t_pred, cfg.node_n),
    timesteps=cfg.diffuse_steps,
    loss_type=cfg.loss_type,
    objective=cfg.objective,
    beta_schedule=cfg.beta_schedule,
    history_weight=cfg.history_weight,
    future_weight=cfg.future_weight,
    st_loss_weight=torch.zeros(cfg.t_his + cfg.t_pred, cfg.node_n * 3),
).to(dtype).to(device)

ckpt = torch.load(args.ckpt, map_location=device)
load_result = diffuser.load_state_dict(ckpt['model'], strict=False)
print(f"[LOAD] missing_keys={load_result.missing_keys} unexpected_keys={load_result.unexpected_keys}")
diffuser.eval()

# one small batch, real test data
gen = eval_dataset.sampling_generator(num_samples=8, batch_size=8, aug=False)
data = next(gen)
if isinstance(data, tuple):
    data = data[0]
data_t = torch.from_numpy(data).to(device).to(dtype)
traj = data_t[..., 1:, :].reshape(data_t.shape[0], data_t.shape[1], -1)
x_0_history = traj[:, :-cfg.t_pred, :]

with torch.no_grad():
    Y = diffuser.sample(x_0_history, None, batch_size=x_0_history.shape[0], clip_denoised=False, uncond=True)

print(f"[FORWARD] output_shape={tuple(Y.shape)}")
print(f"[TIME] elapsed={time.time()-t0:.1f}s")
print("[SUCCESS]")
