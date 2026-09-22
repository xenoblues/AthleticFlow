import os
import numpy as np
from utils.pose_gen import pose_generator
from utils.visualization import render_animation


def demo_visualize(mode, cfg, model, diffusion, dataset):
    """
    script for drawing gifs in different modes
    """
    if mode == 'pred':
        if cfg.dataset == 'amass':
            action_list = ['Transitions', 'SSM', 'DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA']
        else:    
            action_list = dataset['test'].prepare_iter_action(cfg.dataset)
        # action_list = ['Walking 1 chunk0', 'Box 1 chunk4', 'Gestures 1 chunk1', 'Gestures 1 chunk5', 'ThrowCatch 1 chunk10',
        #  'Jog 1 chunk0', 'ThrowCatch 1 chunk4']
        action_list = ['Sitting 1', 'Smoking', 'Greeting', 'WalkTogether', 'Discussion 2', 'Photo', 'WalkDog 1',
                       'Purchases 1', 'SittingDown 1', 'Greeting 1', 'Smoking 1', 'Waiting 1', 'Directions 1',
                       'WalkTogether 1', 'Eating 1', 'Eating', 'Posing', 'Purchases', 'Walking 1', 'Posing 1',
                       'Directions', 'Walking', 'Photo 1', 'Waiting', 'Discussion 1', 'Phoning 1', 'SittingDown',
                       'Phoning', 'WalkDog', 'Sitting']
        prediciton_results = {}
        for i in range(0, len(action_list)):
            pose_gen = pose_generator(dataset['test'], model, diffusion, cfg,
                                      mode='pred', action=action_list[i], nrow=cfg.vis_row)
            suffix = action_list[i]
            # render_animation(dataset['test'].skeleton, pose_gen, ['TransFusion'], cfg.t_his, ncol=cfg.vis_col + 2,
            #                  output=os.path.join(cfg.gif_dir, f'pred_{suffix}.gif'), mode=mode)
            algos = ['TransFusion']
            all_poses = next(pose_gen)
            algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
            poses = dict(
                filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                       all_poses.items()))
            prediciton_results[action_list[i]] = poses
        np.save(os.path.join(cfg.cfg_dir, f'{cfg.dataset}.npy'), prediciton_results)


    else:
        raise NotImplementedError(f"sorry, {mode} is not only available.")  
