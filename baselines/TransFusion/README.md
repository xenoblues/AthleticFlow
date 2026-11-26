# TransFusion and HumanMAC baselines

Code used for the TransFusion (`tf`) and HumanMAC (`hmac`) baseline results of the
AthleticFlow paper on AthletePose3D (`ap3d`), AthleticsPose (`ap`) and WorldPose (`wp`).
It is a modified copy of [TransFusion](https://github.com/sibotian96/TransFusion)
(MIT License). Both variants share the same diffusion pipeline and differ only in the
denoiser (`models/transformer.py` vs `models/transformer_hmac.py`).
See `NOTICE.md` for attribution and the list of changes.

## Setup

```
pip install -r requirements.txt
```

Data are read from the `data/` folder of the AthleticFlow repository (see the main README),
i.e. `../../data` relative to this folder. Use `--data_root` to point elsewhere.
The files used are

```
data/
├── athlete_pose_3d_v3/{train,valid}_joint_3d_camera_fps.npy, {train,valid}_meta_fps.joblib
│   └── multimodal/t_his15_top50_t_pred60_thre0.100_filtered_dlow_perseq.npz
│       multimodal/data_candi_t_his15_t_pred60_skiprate15.npz
├── AthleticsPose/{train,test}.npz
│   └── multimodal/t_his15_1_thre0.500_t_pred60_thre0.100_index_filterd.npz
│       multimodal/data_candi_t_his15_t_pred60_skiprate15.npz
└── worldpose/wp_data_py3.npz
    └── multimodal/t_his25_1_thre0.500_t_pred100_thre0.100_index_filterd.npz
        multimodal/data_candi_t_his25_t_pred100_skiprate25.npz
```

## Pretrained weights

The seed-0 checkpoints used in the paper are distributed separately (Google Drive link in
the main README) and are not tracked by git. Place them as

```
weights/ap3d/{tf,hmac}.pt
weights/ap/{tf,hmac}.pt
weights/wp/{tf,hmac}.pt
```

## Evaluation

Run from this folder (a GPU is required):

```
python main.py --cfg ap_tf --mode eval --seed 0      # loads weights/ap/tf.pt
python main.py --cfg ap3d_hmac --mode eval --seed 0  # loads weights/ap3d/hmac.pt
python main.py --cfg wp_tf --mode eval --seed 0 --ckpt path/to/ckpt.pt
```

Each test window gets K = 50 samples (100 DDIM steps). The log
(`inference/<cfg>_eval_seed0_<time>/log/log.txt`) lists APD, ADE, FDE, MMADE, MMFDE,
jitter, AEE and ABE, followed by per-joint, per-bone and per-frame errors.
`results/stats.csv` holds the summary and `results/per_window.npz` the per-window metrics.

- ADE/FDE and MMADE/MMFDE are best-of-K errors against the ground truth and the
  multimodal ground truth. APD is the average pairwise distance of the K samples.
- AEE is the mean ADE of the end effectors (LWrist, RWrist, LFoot, RFoot) and ABE the mean
  bone-length error, both for the best-of-K (minimum-ADE) sample.
- Jitter is the mean jerk magnitude (third finite difference, unit/s^3) of the best-of-K
  sample. The last three frames before the prediction are prepended. For `ap3d` and `ap`
  these are the observed frames (`jitter_seam: observed`). The WorldPose configs keep the
  model's reconstruction of these frames (`jitter_seam: predicted`), as used for the
  WorldPose table.
- Errors are in the normalised, root-centred coordinates of each dataset (see `data_loader/`).
- Test windows start every `t_his` frames inside each test sequence. The loader checks
  this order against `test_clip_meta` of the multimodal file (AthletePose3D, AthleticsPose).
- With at least 12 GB of free GPU memory the whole test set is sampled in one batch
  (`--sample_mode auto`). The paper numbers were produced in this mode. `--sample_mode sliced`
  samples ten slices in sequence. It gives a different random stream and slightly different numbers.

## Training

```
python main.py --cfg ap3d_tf --mode train --seed 0
```

Checkpoints of the EMA model are written every 100 epochs to
`results/<cfg>_train_seed<seed>_<time>/models/ckpt_ema_<epoch>.pt`, and the final epoch
is evaluated. The released weights are `ckpt_ema_1000.pt` (`ap3d`, `wp`) and
`ckpt_ema_500.pt` (`ap`).
