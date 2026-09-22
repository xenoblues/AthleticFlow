from __future__ import absolute_import
import logging
import os
import torch
import pandas as pd
import numpy as np

class AverageMeterTorch(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_val = 0 
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        """
        val: tensor, elementwise-shaped statistic for this update. This can be
        a per-sample vector (e.g. [eval_batch_size] for APD/ADE/FDE/MMADE/MMFDE)
        or an already batch-averaged vector/matrix (e.g. [num_joints] for
        joint_ade, [T, num_joints] for joint_time_error). NaN entries (e.g.
        MMADE/MMFDE for samples that only have one multimodal GT) are excluded
        per-element from the running average; unlike the previous
        nan_to_num()+count_nonzero() approach, this does NOT conflate a
        legitimate zero-valued metric with an excluded NaN, and it preserves
        the shape of `val` instead of collapsing it to a single scalar (which
        used to break vector-valued stats such as joint_ade/joint_time_error).
        """
        if not torch.is_tensor(val):
            val = torch.as_tensor(val)
        self.raw_val = val
        valid = ~torch.isnan(val)
        val_clean = torch.where(valid, val, torch.zeros_like(val))
        self.val = val_clean

        if not torch.is_tensor(self.sum):
            self.sum = torch.zeros_like(val_clean)
            self.count = torch.zeros_like(val_clean)
        self.sum = self.sum + val_clean
        self.count = self.count + valid.to(self.count.dtype)

        avg = torch.zeros_like(self.sum)
        has_count = self.count > 0
        avg[has_count] = self.sum[has_count] / self.count[has_count]
        self.avg = avg.item() if avg.numel() == 1 else avg

    def direct_set_avg(self, val):
        self.raw_val = val 
        self.val = val
        self.avg = val
        self.sum = val
        self.count = 1


def save_csv_log(cfg, head, value, is_create=False, file_name='test'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = cfg.log_dir + '/{}.csv'.format(file_name)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)


def save_ckpt(cfg, trainer, file_name='ckpt_CoMusion.pth.tar'):
    file_path = os.path.join(cfg.model_dir, file_name)
    trainer.save(file_path)

