import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from vpython import *
import time

from datetime import datetime


def render_pictures(dataset_name, skeleton, t_hist, fix_0=True, azim=0.0, output=None, mode='pred', size=2,
                     ncol=5, bitrate=3000, fix_index=None):
    if mode == 'switch':
        fix_0 = False
    if fix_index is not None:
        fix_list = [
            [1, 2, 3],  #
            [4, 5, 6],
            [7, 8, 9, 10],
            [11, 12, 13],
            [14, 15, 16],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        fix_i = fix_list[fix_index]
        fix_col = 'darkblue'
    else:
        fix_i = None

    if dataset_name == 'h36m':
        interval = 25
        # path = "./inference/h36m_dct30l8_pred_2025-03-31-16-01/h36m.npy"
        path = "./inference/h36m_dct25_itrans2l5_wosty_pred_2025-09-03-14-57/h36m.npy"
        path1 = "./draw_data/h36m_tf.npy"
    elif dataset_name == 'humaneva':
        interval = 15
        path = "./inference/he_bs256_dct15_pred_2025-03-31-13-17/humaneva.npy"
        path1 = "./draw_data/humaneva_tf.npy"

    data = np.load(path, allow_pickle=True)
    fm_data = data.item()
    data1 = np.load(path1, allow_pickle=True)
    tf_data = data1.item()

    nrow = 6
    ncol = 12
    matplotlib.rcParams['font.sans-serif'] = ['Calibri']

    for key, value in fm_data.items():
    # for i in range(1):
    #     key = list(fm_data.keys())[i]
    #     value = list(fm_data.values())[i]
        fm_pose = value
        tf_pose = tf_data[key]
        t_total = fm_pose['gt'].shape[0]
        plt.ioff()

        fig, axs = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow), subplot_kw={'projection': '3d'})
        ax_3d = []
        lines_3d = []
        radius = 1.5
        index = 0
        for r in range(nrow):
            for c in range(ncol):
                # ax = fig.add_subplot(nrow, ncol, index + 1, projection='3d')
                ax = axs[r, c]
                ax.view_init(elev=15., azim=azim)
                ax.set_xlim3d([-radius / 2, radius / 2])
                ax.set_zlim3d([0, radius])
                ax.set_ylim3d([-radius / 2, radius / 2])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                ax.dist = 4.0
                if mode == 'switch':
                    if index == 0:
                        ax.set_title('target', y=1.0, fontsize=12)
                if mode == 'pred' or 'fix' in mode or mode == 'control' or mode == 'zero_shot':
                    if r == 0:
                        if c == 0:
                            # title = list(fm_data.keys())[index]
                            title = "Observation"
                            ax.set_title(title, y=0.85, fontsize=24)
                        elif c == 1:
                            title = "GT"
                            ax.set_title(title, y=0.85, fontsize=24)
                        elif c == 4:
                            title = 'Ours'
                            ax.set_title(title, y=0.85, fontsize=24)
                        elif c == 9:
                            title = 'TransFusion'
                            ax.set_title(title, y=0.85, fontsize=24)

                ax.set_axis_off()

                ax.patch.set_alpha(0.0)
                ax_3d.append(ax)
                lines_3d.append([])
                index += 1
        fig.tight_layout(h_pad=0.0, w_pad=0.0)
        # fig.tight_layout()
        fig.subplots_adjust(wspace=-0.9, hspace=-0.55)

        hist_lcol, hist_mcol, hist_rcol = 'gray', 'black', 'red'
        pred_lcol, pred_mcol, pred_rcol = 'purple', 'black', 'green'
        tran_lcol, tran_mcol, tran_rcol = 'orange', 'black', 'blue'
        parents = skeleton.parents()

        for r in range(nrow):
            for c in range(ncol):
                if r != nrow - 1:
                    f_num = r * interval
                else:
                    f_num = t_total - 1
                n = r * ncol + c
                if c == 0:  # 第1列Observation
                    f_num = np.clip(f_num, 0, t_hist - 1)
                    k = list(fm_pose.keys())[c]
                    pose = fm_pose[k][f_num, :, :]
                elif 0 < c < 7:
                    k = list(fm_pose.keys())[c]
                    pose = fm_pose[k][f_num, :, :]
                else:
                    k = list(tf_pose.keys())[c - 5]

                    pose = tf_pose[k][f_num, :, :]

                # if fix_0 and n == 0 and f_num >= t_hist:
                #     continue
                # if fix_0 and n % ncol == 0 and f_num >= t_hist:
                #     continue

                # ax = ax_3d[r * ncol + c]
                ax = axs[r, c]
                ax.set_xlim3d([-radius / 2 + pose[0, 0], radius / 2 + pose[0, 0]])
                ax.set_ylim3d([-radius / 2 + pose[0, 1], radius / 2 + pose[0, 1]])
                ax.set_zlim3d([-radius / 2 + pose[0, 2], radius / 2 + pose[0, 2]])

                if mode == 'switch':
                    if f_num < t_hist:
                        lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
                    elif f_num > 75:
                        lcol, mcol, rcol = tran_lcol, pred_mcol, tran_rcol
                    else:
                        lcol, mcol, rcol = pred_lcol, tran_mcol, pred_rcol
                else:
                    if f_num < t_hist or k == 'context' or k == 'gt':
                        lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
                    else:
                        lcol, mcol, rcol = pred_lcol, pred_mcol, pred_rcol

                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue

                    if j in skeleton.joints_right():
                        col = rcol
                    elif j in skeleton.joints_left():
                        col = lcol
                    else:
                        col = mcol

                    if fix_i is not None and j in fix_i:
                        col = fix_col

                    lines_3d[n].append(ax.plot([pose[j, 0], pose[j_parent, 0]], [pose[j, 1], pose[j_parent, 1]], [pose[j, 2], pose[j_parent, 2]], zdir='z', c=col, linewidth=3.0))


        plt.savefig(f"./save_figures3/{dataset_name}_{key}.tiff", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()


def draw_diversity_comparison(dataset_name, skeleton, t_hist, fix_0=True, azim=0.0, output=None, mode='pred', size=2,
                     ncol=5, bitrate=3000, fix_index=None, model_name="gt"):

    if mode == 'switch':
        fix_0 = False
    if fix_index is not None:
        fix_list = [
            [1, 2, 3],  #
            [4, 5, 6],
            [7, 8, 9, 10],
            [11, 12, 13],
            [14, 15, 16],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        fix_i = fix_list[fix_index]
        fix_col = 'darkblue'
    else:
        fix_i = None

    if dataset_name == 'h36m':
        interval = 25
        # path = "./inference/h36m_dct30l8_pred_2025-03-31-16-01/h36m.npy"
        path = "./draw_data/pred_all/pred_all_humanmac.npy"
        path1 = "./inference/h36m_dct30l8_eval_2025-10-26-15-39/results/pred_all.npy"
        path2 = "./draw_data/pred_all/data_all.npy"
        path3 = "./draw_data/pred_all/pred_all_tf.npy"
    elif dataset_name == 'humaneva':
        interval = 15
        path = "./inference/he_bs256_dct15_pred_2025-03-31-13-17/humaneva.npy"
        path1 = "./draw_data/humaneva_tf.npy"


    if model_name == 'fm':
        data = np.load(path, allow_pickle=True)
        frame_index = [i for i in range(50, 125, 25)]
        frame_index.append(124)
    elif model_name == 'humanmac':
        data = np.load(path1, allow_pickle=True)
        frame_index = [i for i in range(50, 125, 25)]
        frame_index.append(124)
    elif model_name == 'gt':
        data = np.load(path2, allow_pickle=True)
        frame_index = [i for i in range(0, 125, 25)]
        frame_index.append(124)
    elif model_name == 'tf':
        data = np.load(path3, allow_pickle=True)
        frame_index = [i for i in range(50, 125, 25)]
        frame_index.append(124)


    nrow = 1
    ncol = len(frame_index)
    sample_num = 10
    matplotlib.rcParams['font.sans-serif'] = ['Calibri']

    hist_lcol, hist_mcol, hist_rcol = 'gray', 'black', 'red'
    pred_lcol, pred_mcol, pred_rcol = 'purple', 'black', 'green'
    tran_lcol, tran_mcol, tran_rcol = 'orange', 'black', 'blue'
    parents = skeleton.parents()

    seq_indices = [i for i in range(0, 5000, 500)]

    for seq_id in seq_indices:

        fig, axs = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow), subplot_kw={'projection': '3d'})
        ax_3d = []
        lines_3d = []
        radius = 1.5
        index = 0

        for c in range(ncol):
            # ax = fig.add_subplot(nrow, ncol, index + 1, projection='3d')
            ax = axs[c]
            ax.view_init(elev=15., azim=azim)
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_zlim3d([0, radius])
            ax.set_ylim3d([-radius / 2, radius / 2])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.dist = 4.0

            ax.set_axis_off()

            ax.patch.set_alpha(0.0)
            ax_3d.append(ax)
            lines_3d.append([])
            index += 1
        fig.tight_layout(h_pad=0.0, w_pad=0.0)
        # fig.tight_layout()
        fig.subplots_adjust(wspace=-0.8, hspace=-0.55)


        root = np.array([[0, 0, 0]])

        for c in range(ncol):
            if model_name != 'gt':
                sample_index = np.random.choice(np.arange(50), size=sample_num, replace=False)
            else:
                sample_index = [0]
            for i in sample_index:
                if model_name != 'gt':
                    pose = data[i, seq_id, frame_index[c], :]
                    pose = np.reshape(pose, (16, 3))
                    pose = np.concatenate((root, pose), axis=0)
                else:
                    pose = data[seq_id, frame_index[c], :, :]
                    pose[0, :] = 0.0

                ax = axs[c]
                ax.set_xlim3d([-radius / 2 + pose[0, 0], radius / 2 + pose[0, 0]])
                ax.set_ylim3d([-radius / 2 + pose[0, 1], radius / 2 + pose[0, 1]])
                ax.set_zlim3d([-radius / 2 + pose[0, 2], radius / 2 + pose[0, 2]])


                if frame_index[c] < t_hist or model_name=='gt':
                    lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
                else:
                    lcol, mcol, rcol = pred_lcol, pred_mcol, pred_rcol

                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue

                    if j in skeleton.joints_right():
                        col = rcol
                    elif j in skeleton.joints_left():
                        col = lcol
                    else:
                        col = mcol

                    if fix_i is not None and j in fix_i:
                        col = fix_col

                    lines_3d[c].append(ax.plot([pose[j, 0], pose[j_parent, 0]], [pose[j, 1], pose[j_parent, 1]], [pose[j, 2], pose[j_parent, 2]], zdir='z', c=col, linewidth=1.0))

        plt.savefig(f"./diversity_figure/{dataset_name}_{model_name}_{seq_id}.tiff", dpi=600, bbox_inches='tight', pad_inches=-0.6)
        plt.show()
        plt.close()
