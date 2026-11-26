"""
AthletePose3D multimodal ground truth (Filtered-DLow style) for the per-sequence evaluation protocol.

Candidate pool: all training windows (t_his + t_pred frames, stride t_his) of every training sequence.
Test windows:   windows WITHIN every test sequence (stride t_his; sequences shorter than t_his + t_pred are skipped),
                in the order expected by data_loader/dataset_ap3d_multimodal.py (saved as 'test_clip_meta').
Multimodal GT:  the K nearest candidates of the observed history (15 frames, all joints), greedily filtered so that
                every kept future differs from all previously kept futures by more than THRE_PRED (mean joint distance).

Run from the repository root:  python -m data_loader.ap3d_multimodal_data_generation
"""
import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from data_loader.datareader_ap3d import DataReaderAP3D

T_HIS = 15
T_PRED = 60
T_TOTAL = T_HIS + T_PRED
STEP = T_HIS
K_NEIGHBORS = 50
THRE_PRED = 0.1

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(ROOT_DIR, 'data', 'athlete_pose_3d_v3', 'multimodal')


def sliding_windows(seqs):
    """(sequence id, start frame) of all windows within the sequences"""
    return [(sid, s) for sid, seq in enumerate(seqs) if len(seq) >= T_TOTAL
            for s in range(0, len(seq) - T_TOTAL + 1, STEP)]


def generate_ap3d_multimodal():
    reader = DataReaderAP3D(normalize_to_neg1_pos1=True)
    train_seqs = [seq.astype(np.float32) for seq in reader.get_train_sequences()]
    test_seqs = [seq.astype(np.float32) for seq in reader.get_test_sequences()]

    # candidate pool
    candi_data = np.asarray([train_seqs[sid][s:s + T_TOTAL] for sid, s in sliding_windows(train_seqs)], np.float32)
    print(f"Candidate pool: {candi_data.shape}")

    # test windows
    test_clip_meta = np.asarray(sliding_windows(test_seqs), dtype=np.int32)
    test_clips = np.asarray([test_seqs[sid][s:s + T_TOTAL] for sid, s in test_clip_meta], np.float32)
    print(f"Test windows: {test_clips.shape} from {sum(len(s) >= T_TOTAL for s in test_seqs)} sequences")

    # nearest neighbours of the observed history
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree', n_jobs=-1)
    knn.fit(candi_data[:, :T_HIS].reshape(len(candi_data), -1))
    _, indices_his = knn.kneighbors(test_clips[:, :T_HIS].reshape(len(test_clips), -1))

    # greedy future-diversity filter
    future = candi_data[:, T_HIS:]
    multi_indices = []
    for i in tqdm(range(len(test_clips))):
        cand = indices_his[i]
        keep = [0]
        for j in range(1, len(cand)):
            d = np.linalg.norm(future[cand[j]][None] - future[cand[keep]], axis=-1).mean(axis=(1, 2)).min()
            if d > THRE_PRED:
                keep.append(j)
        multi_indices.append(cand[keep].astype(np.int32))
    num_modes = np.array([len(m) for m in multi_indices])
    print(f"Modes per window: mean {num_modes.mean():.2f}, median {np.median(num_modes):.0f}, "
          f"min {num_modes.min()}, max {num_modes.max()}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    candi_file = os.path.join(SAVE_DIR, f"data_candi_t_his{T_HIS}_t_pred{T_PRED}_skiprate{STEP}.npz")
    np.savez_compressed(candi_file, **{"data_candidate.npy": candi_data})

    data_multimodal = {"all_subjects": {"all_actions": {np.int64(i): m for i, m in enumerate(multi_indices)}}}
    multimodal_file = os.path.join(
        SAVE_DIR, f"t_his{T_HIS}_top{K_NEIGHBORS}_t_pred{T_PRED}_thre{THRE_PRED:.3f}_filtered_dlow_perseq.npz")
    np.savez_compressed(multimodal_file, data_multimodal=data_multimodal, test_clip_meta=test_clip_meta)
    print(f"Saved {candi_file}\nSaved {multimodal_file}")


if __name__ == "__main__":
    generate_ap3d_multimodal()
