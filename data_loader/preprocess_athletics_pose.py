import math
import os
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d

from utils import mocap_to_h36m

# ===================== 配置参数（完全匹配你的数据） =====================
ROOT_RAW_DIR = r"E:\MyPyProjects\HumanMAC\data\AthleticsPose\raw_markers_in_world"
SAVE_DIR = r"E:\MyPyProjects\HumanMAC\data\AthleticsPose"
ACTIONS = ["discus", "hurdle", "javelin", "racewalk", "running", "sd", "shotput", "sprint"]
TEST_SUBJECTS = ["S00", "S05", "S11", "S12", "S13", "S16", "S17", "S20", "S21", "S22", "S23"]

# FPS映射表（完全按照你提供的数据）
FPS_MAPPING = {
    ("discus", "S00", "20250125"): 60,
    ("discus", "S02", "20250125"): 60,
    ("hurdle", "S01", "20250125"): 60,
    ("hurdle", "S01", "20250215"): 60,
    ("hurdle", "S17", "20250215"): 60,
    ("hurdle", "S18", "20250215"): 60,
    ("hurdle", "S20", "20250215"): 60,
    ("javelin", "S03", "20250125"): 60,
    ("racewalk", "S05", "20250126"): 30,
    ("racewalk", "S10", "20250126"): 30,
    ("running", "S07", "20250126"): 30,
    ("running", "S09", "20250126"): 30,
    ("running", "S11", "20250126"): 30,
    ("running", "S13", "20250126"): 30,
    ("running", "S16", "20250215"): 60,
    ("running", "S19", "20250215"): 60,
    ("sd", "S04", "20250126"): 60,
    ("sd", "S04", "20250216"): 60,
    ("sd", "S06", "20250126"): 60,
    ("sd", "S08", "20250126"): 30,
    ("sd", "S08", "20250216"): 60,
    ("sd", "S12", "20250126"): 30,
    ("sd", "S14", "20250126"): 60,
    ("sd", "S14", "20250216"): 60,
    ("sd", "S15", "20250126"): 60,
    ("sd", "S15", "20250216"): 60,
    ("sd", "S21", "20250216"): 60,
    ("sd", "S22", "20250216"): 60,
    ("sd", "S23", "20250216"): 60,
    ("shotput", "S00", "20250125"): 60,
    ("shotput", "S02", "20250125"): 60,
    ("shotput", "S03", "20250125"): 60,
    ("sprint", "S04", "20250126"): 60,
    ("sprint", "S04", "20250216"): 60,
    ("sprint", "S06", "20250126"): 60,
    ("sprint", "S08", "20250126"): 60,
    ("sprint", "S08", "20250216"): 60,
    ("sprint", "S12", "20250126"): 60,
    ("sprint", "S14", "20250126"): 30,
    ("sprint", "S14", "20250216"): 60,
    ("sprint", "S15", "20250126"): 60,
    ("sprint", "S15", "20250216"): 60,
    ("sprint", "S21", "20250216"): 60,
    ("sprint", "S22", "20250216"): 60,
    ("sprint", "S23", "20250216"): 60,
}

# 运动预测参数（保持不变）
WINDOW_SIZE = 75  # 15历史 + 60未来（60fps对应1.25秒）
JOINT_NUM = 17  # 17关节拓扑

# ========================================================================

def cubic_spline_interpolate_30_to_60(seq):
    """三次样条插值：将30fps序列平滑插值到60fps"""
    T, J, C = seq.shape
    seq_60fps = np.zeros((2 * T, J, C), dtype=np.float32)

    t_30 = np.arange(T) / 30.0
    t_60 = np.arange(2 * T) / 60.0

    for j in range(J):
        for c in range(C):
            f = interp1d(t_30, seq[:, j, c], kind='cubic', fill_value="extrapolate")
            seq_60fps[:, j, c] = f(t_60)

    return np.clip(seq_60fps, -1000.0, 1000.0)


def get_file_fps(action, subject, file_name):
    """从文件名提取日期，查询FPS映射表"""
    date = file_name.split('_')[0]
    key = (action, subject, date)
    return FPS_MAPPING.get(key, 60)


def load_all_raw_sequences():
    """加载所有原始序列，自动将30fps插值到60fps，过滤过短序列"""
    all_sequences = []
    all_subjects = []
    all_actions = []
    all_lengths = []
    fps_stats = {30: 0, 60: 0}

    for action in tqdm(ACTIONS, desc="加载动作数据"):
        action_dir = os.path.join(ROOT_RAW_DIR, action)
        if not os.path.exists(action_dir):
            continue

        for subject in os.listdir(action_dir):
            subject_dir = os.path.join(action_dir, subject)
            if not os.path.isdir(subject_dir):
                continue

            for file in os.listdir(subject_dir):
                if not file.endswith(".npy"):
                    continue

                file_path = os.path.join(subject_dir, file)
                seq = np.load(file_path)
                seq = mocap_to_h36m(seq)
                print(seq.shape)
                # 自动修正形状
                if len(seq.shape) == 2 and seq.shape[1] == JOINT_NUM * 3:
                    seq = seq.reshape(-1, JOINT_NUM, 3)

                if seq.shape[1] != JOINT_NUM or seq.shape[2] != 3:
                    print(f"跳过格式错误文件：{file_path}")
                    continue

                # FPS插值
                fps = get_file_fps(action, subject, file)
                if fps == 30:
                    seq = cubic_spline_interpolate_30_to_60(seq)
                    fps_stats[30] += 1
                else:
                    fps_stats[60] += 1

                # 过滤过短序列（无法截取75帧窗口）
                if seq.shape[0] < WINDOW_SIZE:
                    print(f"跳过过短序列：{file_path}，长度：{seq.shape[0]}")
                    continue

                all_sequences.append(seq.astype(np.float32))
                all_subjects.append(subject)
                all_actions.append(action)
                all_lengths.append(seq.shape[0])

    print(f"\nFPS统计：30fps序列 {fps_stats[30]} 个，60fps序列 {fps_stats[60]} 个")
    print(f"序列长度范围：{min(all_lengths)} ~ {max(all_lengths)} 帧")
    print(f"所有序列已统一为60fps")

    return all_sequences, all_subjects, all_actions, all_lengths


def root_joint_normalize_per_sequence(sequences):
    """对每个完整序列单独做根节点归一化（减去序列第0帧的根关节）"""
    normalized_sequences = []
    for seq in sequences:
        root_joint = seq[:, 0:1, :]  # 每个序列第0帧的根关节
        normalized_seq = seq - root_joint
        normalized_sequences.append(normalized_seq)
    return normalized_sequences


def compute_global_max(sequences):
    """计算训练集所有坐标的全局最大值"""
    all_coords = []
    for seq in sequences:
        all_coords.append(np.abs(seq))
    return np.max(np.concatenate(all_coords)).astype(np.float32)


def max_normalize_sequences(sequences, global_max):
    """对所有序列做全局max归一化"""
    return [seq / global_max for seq in sequences]


def split_train_test(sequences, subjects, actions, lengths):
    """按照原论文划分训练/测试集"""
    train_mask = ~np.isin(subjects, TEST_SUBJECTS)
    test_mask = np.isin(subjects, TEST_SUBJECTS)

    train_data = [sequences[i] for i in range(len(sequences)) if train_mask[i]]
    test_data = [sequences[i] for i in range(len(sequences)) if test_mask[i]]
    train_subs = [subjects[i] for i in range(len(subjects)) if train_mask[i]]
    test_subs = [subjects[i] for i in range(len(subjects)) if test_mask[i]]
    train_acts = [actions[i] for i in range(len(actions)) if train_mask[i]]
    test_acts = [actions[i] for i in range(len(actions)) if test_mask[i]]
    train_lens = [lengths[i] for i in range(len(lengths)) if train_mask[i]]
    test_lens = [lengths[i] for i in range(len(lengths)) if test_mask[i]]

    return (train_data, test_data,
            train_subs, test_subs,
            train_acts, test_acts,
            train_lens, test_lens)


def main():
    # 1. 加载所有原始数据并统一FPS到60
    all_seqs, all_subs, all_acts, all_lens = load_all_raw_sequences()
    print(f"\n加载完成：共 {len(all_seqs)} 个完整序列")

    # 2. 每个序列单独做根节点归一化
    seqs_root_norm = root_joint_normalize_per_sequence(all_seqs)
    print("根节点归一化完成")

    # 3. 划分训练/测试集
    (train_seqs, test_seqs,
     train_subs, test_subs,
     train_acts, test_acts,
     train_lens, test_lens) = split_train_test(seqs_root_norm, all_subs, all_acts, all_lens)
    print(f"数据集划分：训练集 {len(train_seqs)} 个序列，测试集 {len(test_seqs)} 个序列")

    # 4. 全局max归一化（仅用训练集统计量）
    global_max_train = math.ceil(compute_global_max(train_seqs))
    global_max_test = math.ceil(compute_global_max(test_seqs))
    print(global_max_train, global_max_test)
    global_max = max(global_max_train, global_max_test)
    train_seqs_norm = max_normalize_sequences(train_seqs, global_max)
    test_seqs_norm = max_normalize_sequences(test_seqs, global_max)
    print(f"全局max归一化完成，全局最大值：{global_max:.4f}")

    # 5. 关键修复：转换为对象数组保存非均匀长度序列
    train_seqs_obj = np.empty(len(train_seqs_norm), dtype=object)
    for i, seq in enumerate(train_seqs_norm):
        train_seqs_obj[i] = seq

    test_seqs_obj = np.empty(len(test_seqs_norm), dtype=object)
    for i, seq in enumerate(test_seqs_norm):
        test_seqs_obj[i] = seq

    # 6. 保存为npz文件
    os.makedirs(SAVE_DIR, exist_ok=True)

    np.savez_compressed(
        os.path.join(SAVE_DIR, "train.npz"),
        trajectories=train_seqs_obj,  # 对象数组，支持非均匀长度
        subjects=np.array(train_subs),
        actions=np.array(train_acts),
        lengths=np.array(train_lens),
        global_max=global_max,
        window_size=WINDOW_SIZE,
        joint_num=JOINT_NUM,
        fps=60
    )

    np.savez_compressed(
        os.path.join(SAVE_DIR, "test.npz"),
        trajectories=test_seqs_obj,
        subjects=np.array(test_subs),
        actions=np.array(test_acts),
        lengths=np.array(test_lens),
        global_max=global_max,
        window_size=WINDOW_SIZE,
        joint_num=JOINT_NUM,
        fps=60
    )

    print(f"\n数据保存成功：{SAVE_DIR}")
    print(f"  - train.npz: {os.path.getsize(os.path.join(SAVE_DIR, 'train.npz')) / 1024 / 1024:.2f} MB")
    print(f"  - test.npz: {os.path.getsize(os.path.join(SAVE_DIR, 'test.npz')) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()