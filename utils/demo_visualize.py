import os
from utils.pose_gen import pose_generator
from utils.visualization import render_animation


def demo_visualize(mode, cfg, model, generator, dataset, num_gifs=30):
    """
    draw gifs of randomly selected test sequences: observation, ground truth and cfg.vis_col predicted futures
    """
    if mode != 'pred':
        raise NotImplementedError(f"unknown visualization mode: {mode}")
    for i in range(num_gifs):
        pose_gen = pose_generator(dataset['test'], model, generator, cfg, mode='pred', nrow=cfg.vis_row)
        render_animation(dataset['test'].skeleton, pose_gen, ['AF'], cfg.t_his, ncol=cfg.vis_col + 2,
                         output=os.path.join(cfg.gif_dir, f'pred_{i}.gif'), mode=mode, dataset_name=cfg.dataset)
