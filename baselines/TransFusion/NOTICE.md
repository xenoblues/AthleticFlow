# Notice

This folder contains modified third-party code used to produce the TransFusion and
HumanMAC baseline results reported in the AthleticFlow paper.

- **TransFusion**: S. Tian, M. Zheng, X. Liang, "TransFusion: A Practical and Effective
  Transformer-based Diffusion Model for 3D Human Motion Prediction", IEEE RA-L 2024.
  Upstream code: https://github.com/sibotian96/TransFusion.
  MIT License, Copyright (c) 2024 Sibo Tian. See `LICENSE`.
- **HumanMAC**: L.-H. Chen et al., "HumanMAC: Masked Motion Completion for Human Motion
  Prediction", ICCV 2023. Upstream code: https://github.com/LinghaoChan/HumanMAC.
  MIT License, Copyright (c) 2023 Ling-Hao CHEN. See `LICENSE.HumanMAC`.
  `models/transformer_hmac.py` follows the HumanMAC denoiser. It is used inside the
  TransFusion framework (same DCT/DDIM pipeline) as the HumanMAC baseline.

## Changes with respect to upstream TransFusion

- Removed the Human3.6M, HumanEva-I and AMASS loaders, configs and checkpoints, the
  visualisation/prediction mode and other unused code. Added loaders and configs for
  AthletePose3D (`ap3d`), AthleticsPose (`ap`) and WorldPose (`wp`).
- Evaluation windows are taken per test sequence (stride `t_his`) and never cross a
  sequence boundary. Their order is asserted against `test_clip_meta` of the multimodal
  ground-truth file.
- Fixed the memory-saving (`sliced`) evaluation branch, which sampled only the last slice.
- Evaluation reports the metric set shared by all methods in the paper: APD, ADE, FDE,
  MMADE, MMFDE, AEE, ABE and jitter (`utils/metrics.py`, `utils/evaluation.py`).
- Added the HumanMAC denoiser variant (`model_name: 'hmac'`).
