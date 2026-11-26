## Code for "AthleticFlow: Flow Matching with Guidance of Human Kinematics for 3D Athletic Motion Prediction"

AthleticFlow is a few-step stochastic human motion prediction model for athletic motions. It learns a spectral
(truncated-DCT) velocity field by conditional flow matching and uses a limb-to-body hierarchical Transformer
(LBHTrans) with torso-anchored internal-limb attention, global attention, and a motion-conditioned kinematic
chain-relative bias.

> **Second-revision update.** This version of the repository corresponds to the second revision of the paper.
> It fixes the evaluation protocol on AthletePose3D and AthleticsPose (test windows are now extracted within each
> test sequence), makes the dynamic-plus-static chain-relative bias the default configuration, and reports results
> over five training seeds. See [Changes in the second revision](#changes-in-the-second-revision).

### Requirements

```
pip install -r requirements.txt
```

`torch` is pinned to the version used in development but should be installed from the
[official PyTorch index](https://pytorch.org/get-started/locally/) for the CUDA build matching your GPU/driver
(see the comment at the top of `requirements.txt`). The Adan optimizer is installed from its
[repository](https://github.com/sail-sg/Adan); see its README if the fused CUDA kernel cannot be built.

### Data

Download all files from [Google Drive](https://drive.google.com/drive/folders/1hTTGkzFtvehHheMAltARRHRfy1ZLqQO3?usp=drive_link)
and put the `data` directory at the root of the project.

The final `./data` directory is shown below. The current code reads only the files marked with `*`.

```
data
├─athlete_pose_3d_v3
│  │  train_box_fps.npy
│  │  train_center_fps.npy
│  │  train_joint_3d_camera.npy
│  │  train_joint_3d_camera_fps.npy                                   *
│  │  train_joint_3d_image_fps.npy
│  │  train_meta.joblib
│  │  train_meta_fps.joblib                                           *
│  │  train_scale_fps.npy
│  │  valid_box_fps.npy
│  │  valid_center_fps.npy
│  │  valid_joint_3d_camera.npy
│  │  valid_joint_3d_camera_fps.npy                                   *
│  │  valid_joint_3d_image_fps.npy
│  │  valid_meta.joblib
│  │  valid_meta_fps.joblib                                           *
│  │  valid_scale_fps.npy
│  │
│  └─multimodal
│          data_candi_t_his15_t_pred60_skiprate15.npz                 *
│          t_his15_top50_t_pred60_thre0.100_filtered_dlow.npz          (first revision, unused)
│          t_his15_top50_t_pred60_thre0.100_filtered_dlow_perseq.npz   *  (second revision)
│
├─AthleticsPose
│  │  test.npz                                                        *
│  │  train.npz                                                       *
│  │
│  └─multimodal
│          data_candi_t_his15_t_pred60_skiprate15.npz                 *
│          t_his15_1_thre0.500_t_pred60_thre0.100_index_filterd.npz   *
│
└─worldpose
    │  wp_data_py3.npz                                                *
    │
    └─multimodal
            data_candi_t_his25_t_pred100_skiprate25.npz               *
            t_his25_1_thre0.500_t_pred100_thre0.010_index_filterd.npz
            t_his25_1_thre0.500_t_pred100_thre0.100_index_filterd.npz *
```

The AthletePose3D multimodal file of the second revision (`*_perseq.npz`) can also be generated from the training
and test data (this also rewrites `data_candi_t_his15_t_pred60_skiprate15.npz` with identical content):

```
python -m data_loader.ap3d_multimodal_data_generation
```

### Evaluation protocol

- **Windows.** For each test sequence, windows of `t_his + t_pred` frames are extracted with a stride of `t_his`
  frames; sequences shorter than one window are skipped. This gives 13,517 windows (1,151 sequences) on
  AthletePose3D, 1,228 windows (190 sequences) on AthleticsPose, and 6,907 windows on WorldPose. Observation /
  prediction lengths are 15 / 60 frames (0.25 s / 1.0 s at 60 fps) on AthletePose3D and AthleticsPose and 25 / 100
  frames (0.5 s / 2.0 s at 50 fps) on WorldPose.
- **Samples and metrics.** 50 samples per window; APD, ADE, FDE, MMADE, MMFDE, AEE (end-effector error), ABE
  (bone-length error), and jitter, as defined in the paper. Multimodal references (MMADE/MMFDE) are taken from the
  multimodal files in `data/*/multimodal/`, whose clip order is checked against the evaluation windows.
- **Seeds.** Every method is trained with seeds 0–4; the tables report mean (standard deviation) over seeds.

### Pretrained models

Checkpoints are not tracked by git. Download `athleticflow_weights_rev2.zip` (link will be added) and extract
it at the root of the project. It contains the seed-0 checkpoints of AthleticFlow and SDFM and of the TransFusion
and HumanMAC baselines:

```
checkpoints
│  ap3d_af.pt      ap_af.pt      wp_af.pt        AthleticFlow
│  ap3d_sdfm.pt    ap_sdfm.pt    wp_sdfm.pt      SDFM
baselines
└─TransFusion
    └─weights
        ├─ap3d   tf.pt  hmac.pt                  TransFusion, HumanMAC
        ├─ap     tf.pt  hmac.pt
        └─wp     tf.pt  hmac.pt
```

### Training

```
python main_fm.py --cfg ap3d_af --mode train   # AthletePose3D
python main_fm.py --cfg ap_af   --mode train   # AthleticsPose
python main_fm.py --cfg wp_af   --mode train   # WorldPose
```

Use `--seed <k>` to train the other seeds.

### Evaluation

Run from the root of the project:

```
python main_fm.py --cfg ap3d_af --mode eval --ckpt ./checkpoints/ap3d_af.pt   # AthletePose3D
python main_fm.py --cfg ap_af   --mode eval --ckpt ./checkpoints/ap_af.pt     # AthleticsPose
python main_fm.py --cfg wp_af   --mode eval --ckpt ./checkpoints/wp_af.pt     # WorldPose
```

SDFM is evaluated in the same way with `--cfg {ap3d,ap,wp}_sdfm --ckpt ./checkpoints/{ap3d,ap,wp}_sdfm.pt`.
Sampling uses 10 Euler steps (`--ode_options '{"step_size": 0.1}'`, the default) and the default `--seed 0`; with
these settings the released seed-0 checkpoints reproduce the seed-0 results used in the paper. Results are written
to `inference/<cfg>_eval_<time>_seed<seed>_<pid>/`: `log/log.txt` lists the summary metrics followed by per-joint,
per-bone, and per-frame errors, and `results/stats.csv` holds the summary. `--save_per_window` additionally saves
per-window metrics (`results/per_window.npz`) and `--save_pred` all samples (large).

### Ablations

The ablation configurations for AthletePose3D (`ap3d_af_*`) and AthleticsPose (`ap_af_*`) are in `cfg/ablation/`.
Each file differs from the default configuration in one switch, noted in its first line:

| Configuration | Change | Paper |
|---|---|---|
| `*_dct5`, `*_dct10`, `*_dct20`, `*_dct_full` | 5 / 10 / 20 / all DCT coefficients (`n_pre`; default 15) | Table 4 |
| `*_wo_dct` | no DCT, all frames in the time domain (`use_dct: False`) | Table 4 |
| `*_wo_ila` | without internal-limb attention (and hence without chain-relative bias) | Table 5 |
| `*_wo_anchor` | without torso anchors | Table 5 |
| `*_bias_none` | without chain-relative bias | Tables 5, 6 |
| `*_bias_static`, `*_bias_dynamic` | static or dynamic chain-relative bias only (default: both) | Table 6 |
| `*_chead4`, `*_chead6`, `*_chead8` | 4 / 6 / 8 heads with chain-relative bias (default 2) | Supplementary |
| `*_keyframe15` | 15 time-domain keyframes with linear interpolation instead of 15 DCT coefficients | Supplementary |

Train and evaluate them like the default models, e.g.

```
python main_fm.py --cfg ablation/ap3d_af_wo_ila --mode train --seed 0
python main_fm.py --cfg ablation/ap3d_af_wo_ila --mode eval --ckpt <path/to/ckpt_ema_1001.pt>
```

Training writes EMA checkpoints to `results/<cfg>_train_<time>_seed<seed>_<pid>/models/ckpt_ema_<epoch>.pt`. The
released checkpoints are the final epochs (`ckpt_ema_1001.pt` for AthletePose3D, `ckpt_ema_501.pt` for
AthleticsPose, `ckpt_ema_1000.pt` for WorldPose).

### Results

Mean (standard deviation) over five training seeds; lower is better except APD. Bold: best mean.

**AthletePose3D**

| Method | APD ↑ | ADE ↓ | FDE ↓ | MMADE ↓ | MMFDE ↓ | AEE ↓ | ABE ↓ | Jitter ↓ |
|---|---|---|---|---|---|---|---|---|
| HumanMAC (ICCV'23) | 2.751 (0.210) | 0.238 (0.004) | 0.290 (0.003) | 0.406 (0.013) | 0.413 (0.014) | 0.0843 (0.0015) | 0.0124 (0.0003) | 246.6 (5.5) |
| TransFusion (RAL'24) | 2.571 (0.061) | **0.224 (0.001)** | 0.275 (0.001) | 0.399 (0.005) | 0.401 (0.006) | **0.0795 (0.0002)** | 0.0108 (0.0000) | 248.3 (1.4) |
| CoMusion (ECCV'24) | 2.831 (0.163) | 0.261 (0.008) | 0.337 (0.012) | 0.448 (0.008) | 0.463 (0.012) | 0.0902 (0.0031) | 0.0163 (0.0007) | 2562.5 (230.1) |
| SMRNet (PR'26) | 2.856 (0.162) | 0.227 (0.002) | 0.271 (0.002) | 0.393 (0.009) | 0.387 (0.012) | 0.0802 (0.0008) | 0.0102 (0.0002) | **227.6 (5.1)** |
| SDFM (ICASSP'26) | **3.738 (0.034)** | 0.260 (0.001) | 0.296 (0.001) | 0.382 (0.000) | 0.389 (0.001) | 0.0923 (0.0005) | 0.0108 (0.0001) | 408.7 (1.5) |
| AthleticFlow | 3.133 (0.006) | 0.228 (0.001) | **0.263 (0.002)** | **0.370 (0.001)** | **0.373 (0.001)** | 0.0805 (0.0003) | **0.0081 (0.0001)** | 372.1 (1.3) |

**AthleticsPose**

| Method | APD ↑ | ADE ↓ | FDE ↓ | MMADE ↓ | MMFDE ↓ | AEE ↓ | ABE ↓ | Jitter ↓ |
|---|---|---|---|---|---|---|---|---|
| HumanMAC (ICCV'23) | 3.834 (0.065) | 0.3898 (0.0035) | 0.4563 (0.0064) | 0.4163 (0.0027) | 0.3942 (0.0027) | 0.1314 (0.0016) | 0.0251 (0.0007) | 514.26 (6.04) |
| TransFusion (RAL'24) | **4.874 (0.088)** | 0.3849 (0.0048) | 0.4273 (0.0055) | 0.3734 (0.0018) | 0.3561 (0.0025) | 0.1293 (0.0018) | 0.0245 (0.0003) | 543.36 (1.72) |
| CoMusion (ECCV'24) | 4.310 (0.101) | **0.3500 (0.0064)** | **0.3946 (0.0030)** | 0.4190 (0.0099) | 0.3655 (0.0067) | **0.1153 (0.0026)** | 0.0204 (0.0011) | 2854.88 (166.20) |
| SMRNet (PR'26) | 3.896 (0.225) | 0.3862 (0.0024) | 0.4339 (0.0023) | 0.3841 (0.0105) | 0.3564 (0.0081) | 0.1284 (0.0009) | 0.0242 (0.0006) | **502.34 (4.94)** |
| SDFM (ICASSP'26) | 4.283 (0.045) | 0.3795 (0.0013) | 0.4290 (0.0031) | **0.3523 (0.0025)** | **0.3316 (0.0047)** | 0.1257 (0.0008) | 0.0222 (0.0002) | 808.92 (4.54) |
| AthleticFlow | 4.085 (0.034) | 0.3576 (0.0015) | 0.4170 (0.0044) | 0.3542 (0.0027) | 0.3350 (0.0036) | 0.1180 (0.0004) | **0.0189 (0.0002)** | 726.03 (3.08) |

**WorldPose**

| Method | APD ↑ | ADE ↓ | FDE ↓ | MMADE ↓ | MMFDE ↓ | AEE ↓ | ABE ↓ | Jitter ↓ |
|---|---|---|---|---|---|---|---|---|
| HumanMAC (ICCV'23) | 3.064 (1.464) | 0.1840 (0.0165) | 0.1743 (0.0127) | 0.2024 (0.0084) | 0.1850 (0.0071) | 0.0330 (0.0032) | 0.0209 (0.0019) | 97.24 (10.85) |
| TransFusion (RAL'24) | **16.184 (1.447)** | 0.2154 (0.0086) | 0.2269 (0.0081) | 0.2303 (0.0051) | 0.2339 (0.0054) | 0.0412 (0.0024) | 0.0233 (0.0009) | 305.12 (26.65) |
| CoMusion (ECCV'24) | 3.460 (6.087) | 0.9593 (0.2712) | 1.0488 (0.2508) | 0.9683 (0.2668) | 1.0534 (0.2486) | 0.2809 (0.0765) | 0.0620 (0.0141) | 2343.05 (1246.36) |
| SMRNet (PR'26) | 3.431 (0.431)† | 0.1703 (0.0027) | 0.1664 (0.0026) | 0.2021 (0.0021) | 0.1883 (0.0028) | 0.0314 (0.0005) | 0.0188 (0.0003) | 93.18 (5.48) |
| SDFM (ICASSP'26) | 2.277 (0.018) | 0.1688 (0.0006) | 0.1563 (0.0004) | 0.1877 (0.0005) | 0.1660 (0.0004) | 0.0302 (0.0001) | 0.0181 (0.0000) | 84.63 (0.71) |
| AthleticFlow | 2.354 (0.012) | **0.1626 (0.0007)** | **0.1502 (0.0005)** | **0.1855 (0.0003)** | **0.1620 (0.0003)** | **0.0289 (0.0002)** | **0.0173 (0.0001)** | **73.21 (0.66)** |

On WorldPose, TransFusion and CoMusion are averaged over four seeds, and † excludes one diverged SMRNet seed from APD.
Metric magnitudes are comparable only within each dataset.

### Baselines

The paper compares AthleticFlow with HumanMAC, TransFusion, CoMusion, SMRNet, and SDFM. All baselines were trained
with seeds 0–4 on the three datasets and evaluated with the same protocol and metric implementation as AthleticFlow.

| Method | Code in this repository | Weights in the download | Upstream code |
|---|---|---|---|
| SDFM | yes (`cfg/*_sdfm.yml`, shares the AthleticFlow code base) | yes | — (authors' previous work) |
| TransFusion | yes (`baselines/TransFusion`, MIT license) | yes | https://github.com/sibotian96/TransFusion |
| HumanMAC | yes (HumanMAC variant `hmac` in `baselines/TransFusion`) | yes | https://github.com/LinghaoChan/HumanMAC |
| SMRNet | no | no | see the SMRNet paper |
| CoMusion | no | no | https://github.com/jsun57/CoMusion |

**Why some baselines are not included.** The license of CoMusion permits use for non-commercial research only and
does not allow distributing the software or any modified version of it, and the SMRNet code was obtained without a
license that permits redistribution. We therefore do not distribute our modified versions of these two code bases or
the checkpoints trained with them. To reproduce their results, start from the official code and apply the changes
we made to the other baselines: (i) data loaders for AthletePose3D, AthleticsPose, and WorldPose that read the
files in `data/`; (ii) the per-sequence test windows and aligned multimodal references described in
[Evaluation protocol](#evaluation-protocol); (iii) the shared metric implementation (`utils/metrics.py`); and
(iv) for SMRNet on AthletePose3D, the common normalization scale used by all other methods (the original SMRNet
pipeline normalizes AthletePose3D by 1.858 instead of 2.10; metrics are reported in the common scale).

TransFusion and HumanMAC are run from `baselines/TransFusion` (see `baselines/README.md` and
`baselines/TransFusion/README.md`), e.g.

```
cd baselines/TransFusion
python main.py --cfg ap3d_tf --mode eval --seed 0     # loads weights/ap3d/tf.pt
python main.py --cfg ap3d_hmac --mode eval --seed 0   # loads weights/ap3d/hmac.pt
```

### Visualization

```
python main_fm.py --cfg ap3d_af --mode pred --ckpt ./checkpoints/ap3d_af.pt
```

This renders 30 randomly selected test windows to `inference/<cfg>_pred_<time>_seed<seed>_<pid>/out/pred_<i>.gif`,
each showing the observation, the ground truth, and `--vis_col` (default 5) predicted samples.

### Changes in the second revision

- **Chain-relative bias configuration.** In the first-revision code, a configuration default disabled the dynamic
  branch of the chain-relative bias, so the reported multi-seed models used the static bias only. The default is now
  the dynamic-plus-static bias (`bias_mode: both`).
- **Evaluation protocol on AthletePose3D and AthleticsPose.** The previous test loaders slid windows over the
  concatenated test sequences, so part of the windows spanned two sequences, and the AthleticsPose multimodal
  references were not aligned with the windows. Windows are now extracted within each sequence and the multimodal
  references are aligned with them; the AthletePose3D multimodal file was regenerated accordingly.
- **Metrics and statistics.** All methods are evaluated with the same metric implementation; results are reported
  over five training seeds.
- Removed code for datasets and models that are not part of the paper, and the outdated first-revision results.

### Acknowledgments

Part of the code is borrowed from the [HumanMAC](https://github.com/LinghaoChan/HumanMAC) repository. The
`baselines/TransFusion` folder is adapted from the official [TransFusion](https://github.com/sibotian96/TransFusion)
code (see its `NOTICE.md`). Please cite the respective papers if you use the baselines.

### License

This code is distributed under an [MIT LICENSE](https://github.com/LinghaoChan/HumanMAC/blob/main/LICENSE). The
`baselines/TransFusion` folder keeps the MIT licenses of TransFusion and HumanMAC (`LICENSE`, `LICENSE.HumanMAC`).
Note that our code depends on other libraries and datasets which each have their own respective licenses that must
also be followed.
