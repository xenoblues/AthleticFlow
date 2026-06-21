import os
import math
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# =====================================================
# Config
# =====================================================

T_HIS = 25
T_PRED = 100
T_TOTAL = T_HIS + T_PRED

STEP = T_HIS

K_NEIGHBORS = 50

# DLow-style future diversity threshold
THRE_PRED = 0.1

current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
root_path = current_path

SAVE_DIR = os.path.join(root_path, "data/worldpose/multimodal")
RAW_MOTION_PATH = os.path.join(root_path, "data/worldpose/wp_data.npz")
PY3_MOTION_PATH = os.path.join(root_path, "data/worldpose/wp_data_py3.npz")

os.makedirs(SAVE_DIR, exist_ok=True)

NUM_JOINTS = 24

# =====================================================
# Convert Py2 -> Py3
# =====================================================

def convert_py2_to_py3():

    if os.path.exists(PY3_MOTION_PATH):
        print("✅ Python3 data already exists.")
        return

    print("Converting Python2 dataset...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        raw_data = np.load(
            RAW_MOTION_PATH,
            allow_pickle=True,
            encoding="latin1"
        )

        train_seqs = list(raw_data["train"])
        test_seqs = list(raw_data["test"])

    def clean_sequence(seq):

        if len(seq) < T_TOTAL:
            return None

        if np.all(np.isnan(seq)):
            return None

        seq = seq.astype(np.float32)

        T, J, D = seq.shape

        for j in range(J):
            for d in range(D):

                valid = ~np.isnan(seq[:, j, d])

                if np.sum(valid) == 0:
                    continue

                t = np.arange(T)

                seq[:, j, d] = np.interp(
                    t,
                    t[valid],
                    seq[valid, j, d]
                )

        return seq

    train_clean = []

    for seq in tqdm(train_seqs):

        seq = clean_sequence(seq)

        if seq is not None:
            train_clean.append(seq)

    test_clean = []

    for seq in tqdm(test_seqs):

        seq = clean_sequence(seq)

        if seq is not None:
            test_clean.append(seq)

    np.savez_compressed(
        PY3_MOTION_PATH,
        train=np.array(train_clean, dtype=object),
        test=np.array(test_clean, dtype=object)
    )

    print("Conversion Finished")

# =====================================================
# Data Reader
# =====================================================

class DataReaderWP:

    def __init__(self, normalize_to_neg1_pos1=True):

        self.normalize = normalize_to_neg1_pos1

        self._load_data()

    def _load_data(self):

        raw_data = np.load(
            PY3_MOTION_PATH,
            allow_pickle=True
        )

        self.train_seqs = list(raw_data["train"])
        self.test_seqs = list(raw_data["test"])

        print(
            f"Train:{len(self.train_seqs)} "
            f"Test:{len(self.test_seqs)}"
        )

        if self.normalize:

            all_values = []

            for seq in self.train_seqs + self.test_seqs:

                root = seq[:, 0:1]

                centered = seq - root

                all_values.append(
                    centered.reshape(-1)
                )

            all_values = np.concatenate(all_values)

            self.max_abs = np.nanmax(
                np.abs(all_values)
            )

            if self.max_abs < 1e-8:
                self.max_abs = 1.0

            self.max_abs = 2.08

            print(
                f"max_abs={self.max_abs:.4f}"
            )

    def _normalize(self, seq):

        if not self.normalize:
            return seq.astype(np.float32)

        root = seq[:, 0:1]

        return (
            (seq - root) / self.max_abs
        ).astype(np.float32)

    def get_train_sequences(self):

        return [
            self._normalize(x)
            for x in self.train_seqs
        ]

    def get_test_sequences(self):

        return [
            self._normalize(x)
            for x in self.test_seqs
        ]

# =====================================================
# MPJPE Future Distance
# =====================================================

def future_distance(a, b):

    a = a.reshape(
        T_PRED,
        NUM_JOINTS,
        3
    )

    b = b.reshape(
        T_PRED,
        NUM_JOINTS,
        3
    )

    return np.mean(
        np.linalg.norm(
            a - b,
            axis=-1
        )
    )

# =====================================================
# Main
# =====================================================

def generate_worldpose_multimodal():

    convert_py2_to_py3()

    print("=" * 80)
    print("WorldPose Filtered-DLow Generator")
    print("=" * 80)

    reader = DataReaderWP(
        normalize_to_neg1_pos1=True
    )

    train_seqs = reader.get_train_sequences()
    test_seqs = reader.get_test_sequences()

    # =================================================
    # Candidate Pool
    # =================================================

    print("\nBuilding candidate pool...")

    candi_list = []

    for seq in tqdm(train_seqs):

        if len(seq) < T_TOTAL:
            continue

        max_start = len(seq) - T_TOTAL

        for start in range(
            0,
            max_start + 1,
            STEP
        ):

            candi_list.append(
                seq[
                    start:
                    start + T_TOTAL
                ]
            )

    candi_data = np.asarray(
        candi_list,
        dtype=np.float32
    )

    print(
        "Candidate Pool:",
        candi_data.shape
    )

    # =================================================
    # Test Clips
    # Must match DatasetWorldPose
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
                start:
                start + T_TOTAL
            ]
        )

    test_clips = np.asarray(
        test_clips,
        dtype=np.float32
    )

    print(
        "Test Clips:",
        test_clips.shape
    )

    # =================================================
    # History Feature
    # =================================================

    candi_feat = candi_data[
        :, :T_HIS
    ].reshape(
        len(candi_data),
        -1
    )

    test_feat = test_clips[
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

    knn.fit(candi_feat)

    _, indices_his = knn.kneighbors(
        test_feat
    )

    # =================================================
    # Future Filtering
    # =================================================

    print("\nFuture Filtering...")

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

                min_dist = min(
                    min_dist,
                    dist
                )

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

    # np.savez_compressed(
    #     candi_file,
    #     **{
    #         "data_candidate.npy":
    #             candi_data
    #     }
    # )

    # =================================================
    # Save Multi GT
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
    #
    # np.savez_compressed(
    #     multimodal_file,
    #     data_multimodal=data_multimodal
    # )

    print("\nSaved")

    print(candi_file)
    print(multimodal_file)

    print("=" * 80)

if __name__ == "__main__":
    generate_worldpose_multimodal()