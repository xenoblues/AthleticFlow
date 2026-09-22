# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
SMPL_MODEL_PATH  = r"data/worldpose/models/SMPL_NEUTRAL.pkl"
SMPL_MODEL_PATH  = os.path.join(root_path, SMPL_MODEL_PATH )
SAVE_PARAMS = r"models/worldpose/smpl_standard_params.npz"
SAVE_PARAMS = os.path.join(root_path, SAVE_PARAMS)
WORLDPOSE_DIR = r"data/worldpose/poses"
WORLDPOSE_DIR = os.path.join(root_path, WORLDPOSE_DIR)
OUTPUT_PATH = r"data/worldpose/wp_data.npz"
OUTPUT_PATH = os.path.join(root_path, OUTPUT_PATH)
TEST_CLIPS = [
    "ARG_FRA_182345", "ARG_FRA_201902",
    "BRA_KOR_231503", "CRO_MOR_1800400",
    "FRA_MOR_231753", "NET_ARG_231259"
]


def load_smpl_params(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    params = {
        "J": np.array(data["J"]),
        "parents": np.array(data["kintree_table"])[1],
        "J_regressor": np.array(data["J_regressor"].todense()),
        "v_template": np.array(data["v_template"]),
        "shapedirs": np.array(data["shapedirs"]),
        "posedirs": np.array(data["posedirs"])
    }
    return params


# ================== 向量化轴角→旋转矩阵（修复维度版）==================
def batch_aa2rotmat(aa):
    angle = np.linalg.norm(aa + 1e-8, axis=-1, keepdims=True)
    axis = aa / (angle + 1e-8)
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    c, s = np.cos(angle), np.sin(angle)
    ones = np.ones_like(c)
    rot = np.concatenate([
        c + x * x * (ones - c), x * y * (ones - c) - z * s, x * z * (ones - c) + y * s,
        y * x * (ones - c) + z * s, c + y * y * (ones - c), y * z * (ones - c) - x * s,
        z * x * (ones - c) - y * s, z * y * (ones - c) + x * s, c + z * z * (ones - c)
    ], axis=-1)
    return rot.reshape(aa.shape[:-1] + (3, 3))


# ================== SMPL正向运动学 ==================
def forward_kinematics(J_rest, rot_mats, parents):
    T, num_joints = rot_mats.shape[:2]
    trans = np.zeros((T, num_joints, 4, 4), dtype=np.float32)
    trans[..., :3, :3] = rot_mats
    trans[..., :3, 3] = J_rest
    trans[..., 3, 3] = 1.0
    for i in range(1, num_joints):
        trans[:, i] = np.matmul(trans[:, parents[i]], trans[:, i])
    return trans[..., :3, 3]


# ================== 学术标准：NaN时间轴线性插值 ==================
def interpolate_motion_nan(joints):
    T, J, D = joints.shape
    for j in range(J):
        for d in range(D):
            seq = joints[:, j, d]
            valid_idx = np.where(~np.isnan(seq))[0]
            if len(valid_idx) == 0:
                continue
            seq_interp = np.interp(np.arange(T), valid_idx, seq[valid_idx])
            joints[:, j, d] = seq_interp
    return joints


# ================== 主程序（修复维度+提速+插值）==================
if __name__ == "__main__":
    smpl_params = load_smpl_params(SMPL_MODEL_PATH)
    J_rest = smpl_params["J"].reshape(1, 24, 3)
    parents = smpl_params["parents"]

    train_seqs, test_seqs = [], []

    for fname in os.listdir(WORLDPOSE_DIR):
        print(fname)
        if not fname.endswith(".npz"):
            continue
        is_test = any(clip in fname for clip in TEST_CLIPS)
        data = np.load(os.path.join(WORLDPOSE_DIR, fname), allow_pickle=True)

        # 读取数据 (N, T, ...)
        global_orient = data["global_orient"].astype(np.float32)
        body_pose = data["body_pose"].astype(np.float32)
        transl = data["transl"].astype(np.float32)
        N, T = global_orient.shape[:2]

        # 拼接完整姿态: (N, T, 24, 3) ✅ 标准SMPL姿态维度
        full_pose = np.concatenate([global_orient, body_pose], axis=2)

        # ✅ 修复核心：正确重塑维度计算旋转矩阵
        # 展平为 (N*T*24, 3) → 批量计算 → 恢复为 (N,T,24,3,3)
        rot_mat_flat = batch_aa2rotmat(full_pose.reshape(-1, 3))
        rot_mats = rot_mat_flat.reshape(N, T, 24, 3, 3)

        # 单个人物序列处理（保留必要循环，速度最大化）
        for i in range(N):
            if np.all(np.isnan(global_orient[i])):
                continue
            # 正向运动学计算3D关节
            joints_3d = forward_kinematics(J_rest, rot_mats[i], parents)
            joints_3d += transl[i].reshape(T, 1, 3)
            # 缺失值插值
            joints_3d = interpolate_motion_nan(joints_3d)
            test_seqs.append(joints_3d) if is_test else train_seqs.append(joints_3d)

    # 保存最终运动数据
    np.savez_compressed(OUTPUT_PATH, train=train_seqs, test=test_seqs)
