import math
import os
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d

from utils.joints import mocap_to_h36m

# Raw AthleticsPose marker data (<action>/<subject>/<date>_*.npy) and output directory of train.npz / test.npz.
# Run from the repository root:  python -m data_loader.preprocess_athletics_pose
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_RAW_DIR = os.path.join(ROOT_DIR, "data", "AthleticsPose", "raw_markers_in_world")
SAVE_DIR = os.path.join(ROOT_DIR, "data", "AthleticsPose")
ACTIONS = ["discus", "hurdle", "javelin", "racewalk", "running", "sd", "shotput", "sprint"]
TEST_SUBJECTS = ["S00", "S05", "S11", "S12", "S13", "S16", "S17", "S20", "S21", "S22", "S23"]

# FPS of Each Video
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

# Prediction Parameters
WINDOW_SIZE = 75  # 15 observed + 60 predicted frames (1.25 s at 60 fps)
JOINT_NUM = 17  # H36M-17 joints

# ========================================================================

def cubic_spline_interpolate_30_to_60(seq):
    """cubic-spline interpolation of a 30 fps sequence to 60 fps"""
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
    """frame rate of a recording (date taken from the file name)"""
    date = file_name.split('_')[0]
    key = (action, subject, date)
    return FPS_MAPPING.get(key, 60)


def load_all_raw_sequences():
    """load all raw sequences, interpolate 30 fps recordings to 60 fps, drop too short sequences"""
    all_sequences = []
    all_subjects = []
    all_actions = []
    all_lengths = []
    fps_stats = {30: 0, 60: 0}

    for action in tqdm(ACTIONS, desc="loading actions"):
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
                # (T, 51) -> (T, 17, 3)
                if len(seq.shape) == 2 and seq.shape[1] == JOINT_NUM * 3:
                    seq = seq.reshape(-1, JOINT_NUM, 3)

                if seq.shape[1] != JOINT_NUM or seq.shape[2] != 3:
                    print(f"skipping file with unexpected shape: {file_path}")
                    continue

                # frame-rate interpolation
                fps = get_file_fps(action, subject, file)
                if fps == 30:
                    seq = cubic_spline_interpolate_30_to_60(seq)
                    fps_stats[30] += 1
                else:
                    fps_stats[60] += 1

                # drop sequences shorter than one window
                if seq.shape[0] < WINDOW_SIZE:
                    print(f"skipping short sequence: {file_path}, length {seq.shape[0]}")
                    continue

                all_sequences.append(seq.astype(np.float32))
                all_subjects.append(subject)
                all_actions.append(action)
                all_lengths.append(seq.shape[0])

    print(f"\n30 fps sequences: {fps_stats[30]}, 60 fps sequences: {fps_stats[60]}")
    print(f"sequence length: {min(all_lengths)} - {max(all_lengths)} frames")
    print("all sequences are at 60 fps")

    return all_sequences, all_subjects, all_actions, all_lengths


def root_joint_normalize_per_sequence(sequences):
    """root-relative coordinates (root joint of every frame subtracted)"""
    normalized_sequences = []
    for seq in sequences:
        root_joint = seq[:, 0:1, :]
        normalized_seq = seq - root_joint
        normalized_sequences.append(normalized_seq)
    return normalized_sequences


def compute_global_max(sequences):
    """maximum absolute coordinate"""
    all_coords = []
    for seq in sequences:
        all_coords.append(np.abs(seq))
    return np.max(np.concatenate(all_coords)).astype(np.float32)


def max_normalize_sequences(sequences, global_max):
    """divide all sequences by the global maximum"""
    return [seq / global_max for seq in sequences]


def split_train_test(sequences, subjects, actions, lengths):
    """train / test split by subject as in the AthleticsPose paper"""
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
    # 1. load all raw sequences at 60 fps
    all_seqs, all_subs, all_acts, all_lens = load_all_raw_sequences()
    print(f"\nloaded {len(all_seqs)} sequences")

    # 2. root-relative coordinates
    seqs_root_norm = root_joint_normalize_per_sequence(all_seqs)
    print("root normalization done")

    # 3. train / test split
    (train_seqs, test_seqs,
     train_subs, test_subs,
     train_acts, test_acts,
     train_lens, test_lens) = split_train_test(seqs_root_norm, all_subs, all_acts, all_lens)
    print(f"train: {len(train_seqs)} sequences, test: {len(test_seqs)} sequences")

    # 4. global max normalization (ceiled maximum over the train and test sets)
    global_max_train = math.ceil(compute_global_max(train_seqs))
    global_max_test = math.ceil(compute_global_max(test_seqs))
    print(global_max_train, global_max_test)
    global_max = max(global_max_train, global_max_test)
    train_seqs_norm = max_normalize_sequences(train_seqs, global_max)
    test_seqs_norm = max_normalize_sequences(test_seqs, global_max)
    print(f"global max normalization done, scale {global_max:.4f}")

    # 5. object arrays for sequences of different lengths
    train_seqs_obj = np.empty(len(train_seqs_norm), dtype=object)
    for i, seq in enumerate(train_seqs_norm):
        train_seqs_obj[i] = seq

    test_seqs_obj = np.empty(len(test_seqs_norm), dtype=object)
    for i, seq in enumerate(test_seqs_norm):
        test_seqs_obj[i] = seq

    # 6. save
    os.makedirs(SAVE_DIR, exist_ok=True)

    np.savez_compressed(
        os.path.join(SAVE_DIR, "train.npz"),
        trajectories=train_seqs_obj,
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

    print(f"\nsaved to {SAVE_DIR}")
    print(f"  - train.npz: {os.path.getsize(os.path.join(SAVE_DIR, 'train.npz')) / 1024 / 1024:.2f} MB")
    print(f"  - test.npz: {os.path.getsize(os.path.join(SAVE_DIR, 'test.npz')) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()