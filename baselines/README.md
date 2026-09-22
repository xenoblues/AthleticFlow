# Baseline code + weights

Consolidated copies of the 5 baseline methods compared against AthleticFlow in
Tables 1-3 of the paper. Each subfolder is a self-contained copy of that
method's source code (models/, data_loader/, utils/, config.py, main entry
script, relevant cfg files) plus the one seed-0 (default, non-ablation)
checkpoint used for each of the 3 datasets under `weights/{ap3d,ap,wp}/`.

Everything here was COPIED (not moved) from the working directories below;
those originals are untouched and still hold the full multi-seed/ablation
experiment history (results/, inference/) that was intentionally excluded
here to keep this repo small.

**Note on git:** this repo's `.gitignore` excludes `*.pt`/`*.pth.tar`, so the
`weights/` checkpoints below are present on disk but are NOT tracked by git
and will not be pushed to GitHub, matching this project's existing convention
of distributing large binary artifacts (see the root `README.md`'s dataset
download instructions) outside of git rather than through it. Only the source
code under each method's folder is version-controlled.

## Provenance

| Method | Dataset | Source checkpoint | Seed |
|---|---|---|---|
| TransFusion | ap3d | `/work7/y_zhou/TransFusion-main/results/ap3d_tf_train_20260611-153603/models/ckpt_ema_1000.pt` | 0 |
| TransFusion | ap | `/work7/y_zhou/TransFusion-main/results/ap_tf_train_20260612-091707/models/ckpt_ema_500.pt` | 0 |
| TransFusion | wp | `/work7/y_zhou/TransFusion-main/results/wp_tf_train_20260613-144522/models/ckpt_ema_1000.pt` | 0 |
| HumanMAC | ap3d | `/work7/y_zhou/TransFusion-main/results/ap3d_hmac_train_20260611-194037/models/ckpt_ema_1000.pt` | 0 |
| HumanMAC | ap | `/work7/y_zhou/TransFusion-main/results/ap_hmac_train_20260612-111115/models/ckpt_ema_500.pt` | 0 |
| HumanMAC | wp | `/work7/y_zhou/TransFusion-main/results/wp_hmac_train_20260613-144522/models/ckpt_ema_1000.pt` | 0 |
| SMRNet | ap3d | `/work7/y_zhou/SMRNet-main/results/ap3d_seed0_20260919-001545_2964328/models/ckpt_ema_1000.pt` | 0 |
| SMRNet | ap | `/work7/y_zhou/SMRNet-main/results/ap_5/models/ckpt_ema_500.pt` | 0 |
| SMRNet | wp | `/work7/y_zhou/SMRNet-main/results/wp_5/models/ckpt_ema_1000.pt` | 0 |
| CoMusion | ap3d | `/work7/y_zhou/CoMusion-main/results/ap3d/models/ckpt_ap3d.pth.tar` | 0 (see note) |
| CoMusion | ap | `/work7/y_zhou/CoMusion-main/results/ap/models/ckpt_ap.pth.tar` | 0 (see note) |
| CoMusion | wp | `/work7/y_zhou/CoMusion-main/results/wp/models/ckpt_wp.pth.tar` | 0 (see note) |
| SDFM | ap3d | `/work7/y_zhou/HumanMAC/results/ap3d_sdfm_train_20260918-211634_seed0_2865231/models/ckpt_ema_1000.pt` | 0 |
| SDFM | ap | `/work7/y_zhou/HumanMAC/results/ap_sdfm_train_20260612-103613/models/ckpt_ema_501.pt` | 0 |
| SDFM | wp | `/work7/y_zhou/HumanMAC/results/wp_sdfm_train_20260612-142717/models/ckpt_ema_1000.pt` | 0 |

Seed was read from each run's `log/log.txt` header (`'cfg': ..., 'seed': ...`),
except CoMusion, whose logs are CSV-based and don't record a `seed` field
directly — the `ap3d`/`ap`/`wp` (bare-name) result directories were inferred
to be the seed-0 runs by directory-naming symmetry with the other 4 repos
(explicit alternates exist as `ap3d_seed3_retrain`, `ap3d_seed4_retrain`,
etc., which were NOT used here). Lower confidence than the other 4 methods;
worth double-checking against training logs if exact seed provenance matters.

## Notes / caveats

- **HumanMAC baseline code source**: the dedicated `/work7/y_zhou/HumanMAC`
  dev repo only ever trained a HumanMAC-baseline (`model_name:
  'MotionTransformerOrigin'`, cfg `ap3d_humanmac`) for the **ap3d** dataset
  (`results/ap3d_humanmac_train_20260611-150050`, seed 0, final ckpt
  `ckpt_ema_1000.pt`). No `ap`/`wp` HumanMAC-baseline run exists anywhere
  under that repo. The actual `ap`/`wp` (and a redundant `ap3d`) HumanMAC
  baseline was instead trained inside the `TransFusion-main` repo's shared
  multi-model framework, cfg id `{dataset}_hmac` (`model_name: 'hmac'`) — a
  separate re-implementation from the dev-repo's `MotionTransformerOrigin`.
  For consistency across all 3 datasets, this `baselines/HumanMAC/` folder
  uses the TransFusion-main `hmac` implementation and its 3 checkpoints
  throughout, NOT the dev-repo's `MotionTransformerOrigin` ap3d checkpoint.
  If the paper's reported HumanMAC numbers were produced with the dev-repo's
  ap3d implementation instead, swap in
  `/work7/y_zhou/HumanMAC/results/ap3d_humanmac_train_20260611-150050/models/ckpt_ema_1000.pt`
  for the ap3d slot and note the code-source mismatch with ap/wp.
- **`baselines/HumanMAC/` and `baselines/TransFusion/` contain duplicate
  code** (both are copies of the `TransFusion-main` codebase, since that's
  where both `tf` and `hmac` model variants are implemented side by side).
  This mirrors the same intentional duplication in `baselines/SDFM/`, which
  is a copy of the shared HumanMAC/AthleticFlow infrastructure code plus
  SDFM's own cfg/weights.
- **SDFM has no separate repo**: `baselines/SDFM/` is a copy of
  `/work7/y_zhou/HumanMAC`'s shared infrastructure (`models/`, `data_loader/`,
  `utils/`, `config.py`, `main_fm.py`) — the same codebase AthleticFlow's own
  code lives in — plus SDFM's own `cfg/{ap3d,ap,wp}_sdfm.yml` and weights.
  AthleticFlow's own model code (`athletic_flow.py`, `flow_macthing.py`'s
  AF-specific paths, `cfg/*_af*.yml`) is not part of this copy. Do not
  confuse `*_sdfm.yml` (model_name `MotioniTransformer2`, the actual SDFM
  baseline) with `*_rtldfm*.yml` (model_name `RTLDFM`) — the latter is an
  unrelated model, not SDFM, and was excluded.
- **Excluded from every method**: `results/`, `inference/`, `slurm_logs/`,
  `slurm-*.out`, `.git*`, `__pycache__/`, and the large raw-dataset `data/`
  directories (TransFusion-main: 5.4G, HumanMAC: 4.2G) — those still live
  only in the original working directories. CoMusion's own `data/` (8K,
  metadata only) was small enough to include via its code copy.
- **No git operations were run** in this repo — nothing has been staged,
  committed, or pushed. Review before committing, especially given this
  repo has git-lfs configured (`weights/*.pt` / `*.pth.tar` files should
  likely be tracked via LFS rather than committed as plain blobs).
