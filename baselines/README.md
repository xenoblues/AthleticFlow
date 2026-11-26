# Baselines

The paper compares AthleticFlow with HumanMAC, TransFusion, CoMusion, SMRNet, and SDFM on AthletePose3D (`ap3d`),
AthleticsPose (`ap`), and WorldPose (`wp`). All baselines were trained with seeds 0–4 and evaluated with the
per-sequence test windows, the aligned multimodal references, and the metric set described in the main
[README](../README.md#evaluation-protocol).

| Method | Code | Seed-0 weights in `athleticflow_weights_rev2.zip` |
|---|---|---|
| TransFusion | [`TransFusion/`](TransFusion) (cfg `{ap3d,ap,wp}_tf`) | `baselines/TransFusion/weights/{ap3d,ap,wp}/tf.pt` |
| HumanMAC | [`TransFusion/`](TransFusion) (cfg `{ap3d,ap,wp}_hmac`) | `baselines/TransFusion/weights/{ap3d,ap,wp}/hmac.pt` |
| SDFM | main code base (`cfg/{ap3d,ap,wp}_sdfm.yml`, `models/sdfm.py`) | `checkpoints/{ap3d,ap,wp}_sdfm.pt` |
| CoMusion | not included; official code: https://github.com/jsun57/CoMusion | not included |
| SMRNet | not included; see the SMRNet paper | not included |

- **TransFusion and HumanMAC.** `TransFusion/` is a modified copy of the official TransFusion code (MIT license).
  The HumanMAC baseline is the HumanMAC denoiser (`models/transformer_hmac.py`) run inside the same TransFusion
  diffusion pipeline, so one folder covers both methods. See [`TransFusion/README.md`](TransFusion/README.md) for
  usage and [`TransFusion/NOTICE.md`](TransFusion/NOTICE.md) for attribution and the list of changes.
- **SDFM** shares the AthleticFlow code base and is run with `main_fm.py` (see the main README).
- **CoMusion and SMRNet.** Our modified versions of these code bases and the checkpoints trained with them are not
  distributed (see the main README for the reason). Their results in the paper were obtained with the same data,
  per-sequence test windows, multimodal references, and metric implementation as the other methods.
