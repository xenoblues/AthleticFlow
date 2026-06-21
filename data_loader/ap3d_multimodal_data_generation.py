import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from data_loader.datareader_ap3d import DataReaderAP3D


# =====================================================
# Config
# =====================================================

T_HIS = 15
T_PRED = 60
T_TOTAL = T_HIS + T_PRED

STEP = T_HIS

K_NEIGHBORS = 50

# 只保留Future Filtering
THRE_PRED = 0.1

SAVE_DIR = "data/athlete_pose_3d_v3/multimodal"

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)

SAVE_DIR = os.path.join(root_path, SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# Future Distance
# =====================================================

def future_distance(a, b):
    """
    MPJPE-style future distance

    a,b:
    [T_PRED*17*3]
    """

    a = a.reshape(T_PRED, 17, 3)
    b = b.reshape(T_PRED, 17, 3)

    return np.mean(
        np.linalg.norm(
            a - b,
            axis=-1
        )
    )


# =====================================================
# Main
# =====================================================

def generate_ap3d_multimodal():

    print("=" * 80)
    print("AthletePose3D Filtered-DLow Generator")
    print("=" * 80)

    reader = DataReaderAP3D(
        normalize_to_neg1_pos1=True
    )

    train_seqs = reader.get_train_sequences()
    test_seqs = reader.get_test_sequences()

    train_seqs = [
        seq.astype(np.float32)
        for seq in train_seqs
    ]

    test_seqs = [
        seq.astype(np.float32)
        for seq in test_seqs
    ]

    # =================================================
    # Candidate Pool
    # =================================================

    print("\nBuilding candidate pool...")

    candi_list = []

    for seq in tqdm(train_seqs):

        if len(seq) < T_TOTAL:
            continue

        for start in range(
            0,
            len(seq) - T_TOTAL + 1,
            STEP
        ):

            candi_list.append(
                seq[start:start + T_TOTAL]
            )

    candi_data = np.asarray(
        candi_list,
        dtype=np.float32
    )

    print(
        f"Candidate Pool Shape: {candi_data.shape}"
    )

    # =================================================
    # Test Clips
    # IMPORTANT:
    # Must follow DatasetAP3D_multi order
    # =================================================

    print("\nBuilding test clips...")

    test_global = np.concatenate(
        test_seqs,
        axis=0
    )

    test_clips = []

    max_start = len(test_global) - T_TOTAL

    for start in range(
        0,
        max_start + 1,
        STEP
    ):

        test_clips.append(
            test_global[
                start:start + T_TOTAL
            ]
        )

    test_clips = np.asarray(
        test_clips,
        dtype=np.float32
    )

    print(
        f"Test Clips Shape: {test_clips.shape}"
    )

    # =================================================
    # History Features
    # =================================================

    candi_features = candi_data[
        :, :T_HIS
    ].reshape(
        len(candi_data),
        -1
    )

    test_features = test_clips[
        :, :T_HIS
    ].reshape(
        len(test_clips),
        -1
    )

    print("\nRunning KNN...")

    knn = NearestNeighbors(
        n_neighbors=K_NEIGHBORS,
        algorithm="ball_tree",
        n_jobs=-1
    )

    knn.fit(candi_features)

    distances_his, indices_his = knn.kneighbors(
        test_features
    )

    print(
        "History Distance Percentiles:"
    )

    print(
        np.percentile(
            distances_his,
            [50, 75, 90, 95]
        )
    )

    # =================================================
    # Future Filtering
    # =================================================

    print("\nFuture filtering...")

    multi_indices = []

    num_modes = []

    for i in tqdm(
        range(len(test_clips))
    ):

        valid_idx = indices_his[i]

        candidates = candi_data[
            valid_idx
        ]

        future_candidates = candidates[
            :,
            T_HIS:
        ].reshape(
            len(candidates),
            -1
        )

        keep_idx = [0]

        for j in range(
            1,
            len(future_candidates)
        ):

            min_dist = np.inf

            for k in keep_idx:

                dist = future_distance(
                    future_candidates[j],
                    future_candidates[k]
                )

                if dist < min_dist:
                    min_dist = dist

            if min_dist > THRE_PRED:
                keep_idx.append(j)

        final_idx = valid_idx[
            keep_idx
        ].astype(np.int32)

        multi_indices.append(
            final_idx
        )

        num_modes.append(
            len(final_idx)
        )

    # =================================================
    # Statistics
    # =================================================

    print("\nStatistics")

    print(
        f"Average Modes: {np.mean(num_modes):.2f}"
    )

    print(
        f"Median Modes: {np.median(num_modes):.2f}"
    )

    print(
        f"Max Modes: {np.max(num_modes)}"
    )

    print(
        f"Min Modes: {np.min(num_modes)}"
    )

    # =================================================
    # Save Candidate Pool
    # =================================================

    candi_file = os.path.join(
        SAVE_DIR,
        f"data_candi_t_his{T_HIS}_t_pred{T_PRED}_skiprate{STEP}.npz"
    )

    np.savez_compressed(
        candi_file,
        **{
            "data_candidate.npy":
                candi_data
        }
    )

    # =================================================
    # Save Multimodal Index
    # =================================================

    multimodal_dict = {
        np.int64(i):
            multi_indices[i]
        for i in range(
            len(multi_indices)
        )
    }

    data_multimodal = {
        "all_subjects": {
            "all_actions":
                multimodal_dict
        }
    }

    multimodal_file = os.path.join(
        SAVE_DIR,
        f"t_his{T_HIS}"
        f"_top{K_NEIGHBORS}"
        f"_t_pred{T_PRED}"
        f"_thre{THRE_PRED:.3f}"
        f"_filtered_dlow.npz"
    )

    np.savez_compressed(
        multimodal_file,
        data_multimodal=data_multimodal
    )

    print("\nSaved")

    print(candi_file)
    print(multimodal_file)

    print("=" * 80)


if __name__ == "__main__":
    generate_ap3d_multimodal()