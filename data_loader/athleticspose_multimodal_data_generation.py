import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

T_HIS = 15
T_PRED = 60
T_TOTAL = T_HIS + T_PRED
STEP = T_HIS
K_NEIGHBORS = 50

THRE_HIS = 0.5
THRE_PRED = 0.1

DATA_PATH = r"D:\MyPyProjects\HumanMAC\data\AthleticsPose"
SAVE_DIR = r"D:\MyPyProjects\HumanMAC\data\AthleticsPose\multimodal"

os.makedirs(SAVE_DIR, exist_ok=True)


def to_float32_sequence(seq):
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 2:
        if seq.shape[-1] % 3 != 0:
            raise ValueError(f"2D sequence last dim must be divisible by 3, got {seq.shape}")
        seq = seq.reshape(seq.shape[0], seq.shape[-1] // 3, 3)
    if seq.ndim != 3:
        raise ValueError(f"sequence must be [T,J,3] or [T,J*3], got {seq.shape}")
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    return seq.astype(np.float32)


def load_trajectories(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "trajectories" not in data.files:
        raise KeyError(f"'trajectories' not found in {npz_path}. Available keys: {data.files}")
    seqs = [to_float32_sequence(x) for x in list(data["trajectories"])]
    return seqs


def build_sliding_clips(seqs, desc):
    clips = []
    meta = []
    print(f"\n{desc}...")
    for seq_id, seq in enumerate(tqdm(seqs)):
        if len(seq) < T_TOTAL:
            continue
        max_start = len(seq) - T_TOTAL
        for start in range(0, max_start + 1, STEP):
            clips.append(seq[start:start + T_TOTAL])
            meta.append((seq_id, start))
    if len(clips) == 0:
        raise RuntimeError(f"No clips generated in {desc}. Check sequence length and T_TOTAL={T_TOTAL}.")
    clips = np.asarray(clips, dtype=np.float32)
    meta = np.asarray(meta, dtype=np.int32)
    print(f"{desc} Shape: {clips.shape}")
    return clips, meta


def extract_last_history_features(clips):
    return clips[:, T_HIS - 1].reshape(len(clips), -1).astype(np.float32)


def greedy_future_diversity_filter(candi_data, valid_idx, thre_pred):
    if len(valid_idx) <= 1:
        return valid_idx.astype(np.int32)

    candidates = candi_data[valid_idx]
    future = candidates[:, T_HIS:].reshape(len(candidates), -1).astype(np.float32)

    dist_matrix = np.linalg.norm(future[:, None] - future[None], axis=2)

    keep = [0]
    for j in range(1, len(candidates)):
        min_dist = np.min(dist_matrix[j, keep])
        if min_dist > thre_pred:
            keep.append(j)

    final_idx = valid_idx[keep].astype(np.int32)

    if len(final_idx) == 0:
        final_idx = np.array([valid_idx[0]], dtype=np.int32)

    return final_idx


def generate_athleticspose_multimodal():
    print("=" * 80)
    print("AthleticsPose Multimodal Ground Truth Generation")
    print(f"T_HIS={T_HIS}, T_PRED={T_PRED}, T_TOTAL={T_TOTAL}, STEP={STEP}")
    print(f"THRE_HIS={THRE_HIS}, THRE_PRED={THRE_PRED}, K_NEIGHBORS={K_NEIGHBORS}")
    print("=" * 80)

    train_path = os.path.join(DATA_PATH, "train.npz")
    test_path = os.path.join(DATA_PATH, "test.npz")

    print("\nLoading data...")
    train_seqs = load_trajectories(train_path)
    test_seqs = load_trajectories(test_path)

    print(f"Train Sequences: {len(train_seqs)}")
    print(f"Test Sequences : {len(test_seqs)}")

    max_abs = max(np.max(np.abs(seq)) for seq in train_seqs + test_seqs)
    print(f"Max absolute coordinate value: {max_abs:.6f}")

    candi_data, candi_meta = build_sliding_clips(train_seqs, "Building candidate pool")
    test_clips, test_meta = build_sliding_clips(test_seqs, "Building test clips")

    total_clips = len(test_clips)
    n_neighbors = min(K_NEIGHBORS, len(candi_data))

    if n_neighbors <= 0:
        raise RuntimeError("Candidate pool is empty.")

    print("\nExtracting KNN features...")
    candi_features = extract_last_history_features(candi_data)
    test_features = extract_last_history_features(test_clips)

    print("\nRunning KNN...")
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="euclidean", n_jobs=-1)
    knn.fit(candi_features)
    distances_his, indices_his = knn.kneighbors(test_features)

    print("\nHistory Distance Percentiles")
    print("Nearest neighbor:", np.percentile(distances_his[:, 0], [50, 75, 90, 95]))
    print("All retrieved   :", np.percentile(distances_his.reshape(-1), [25, 50, 75, 95]))

    print("\nFuture diversity filtering...")
    multi_indices = []
    stats_before = []
    stats_after = []

    for i in tqdm(range(total_clips)):
        valid_mask = distances_his[i] < THRE_HIS
        valid_idx = indices_his[i][valid_mask].astype(np.int64)

        if len(valid_idx) == 0:
            valid_idx = np.array([indices_his[i][0]], dtype=np.int64)

        stats_before.append(len(valid_idx))

        final_idx = greedy_future_diversity_filter(candi_data, valid_idx, THRE_PRED)

        if len(final_idx) == 0:
            final_idx = np.array([indices_his[i][0]], dtype=np.int32)

        multi_indices.append(final_idx.astype(np.int32))
        stats_after.append(len(final_idx))

    stats_before = np.asarray(stats_before, dtype=np.float32)
    stats_after = np.asarray(stats_after, dtype=np.float32)

    print("\nStatistics")
    print(f"Average Candidates Before Filtering: {stats_before.mean():.2f}")
    print(f"Average Modes: {stats_after.mean():.2f}")
    print(f"Median Modes : {np.median(stats_after):.2f}")
    print(f"Max Modes    : {int(stats_after.max())}")
    print(f"Min Modes    : {int(stats_after.min())}")
    print(f"Diversity Retention: {100.0 * stats_after.mean() / max(stats_before.mean(), 1e-8):.2f}%")

    max_saved_idx = max(int(np.max(x)) for x in multi_indices if len(x) > 0)
    if max_saved_idx >= len(candi_data):
        raise RuntimeError(f"Invalid multimodal index: max index {max_saved_idx}, candidate size {len(candi_data)}")

    if int(stats_after.min()) < 1:
        raise RuntimeError("Min Modes is 0. This should never happen after nearest-neighbor fallback.")

    data_multimodal = {"all_subjects": {"all_actions": {np.int64(i): multi_indices[i] for i in range(total_clips)}}}

    candi_file = f"data_candi_t_his{T_HIS}_t_pred{T_PRED}_skiprate{STEP}.npz"
    index_file = f"t_his{T_HIS}_1_thre{THRE_HIS:.3f}_t_pred{T_PRED}_thre{THRE_PRED:.3f}_index_filterd.npz"

    candi_path = os.path.join(SAVE_DIR, candi_file)
    index_path = os.path.join(SAVE_DIR, index_file)

    print("\nSaving candidate pool...")
    np.savez_compressed(candi_path, **{"data_candidate.npy": candi_data, "candidate_meta.npy": candi_meta})

    print("Saving multimodal index...")
    np.savez_compressed(index_path, data_multimodal=data_multimodal, test_clip_meta=test_meta, stats_before=stats_before, stats_after=stats_after)

    candi_size = os.path.getsize(candi_path) / 1024 / 1024
    index_size = os.path.getsize(index_path) / 1024 / 1024

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Candidate file: {candi_path}")
    print(f"Index file    : {index_path}")
    print(f"Candidate size: {candi_size:.2f} MB")
    print(f"Index size    : {index_size:.2f} MB")
    print(f"Total size    : {candi_size + index_size:.2f} MB")
    print("Protocol      : per-sequence sliding clips, global test clip index, Min Modes >= 1")
    print("=" * 80)


if __name__ == "__main__":
    generate_athleticspose_multimodal()