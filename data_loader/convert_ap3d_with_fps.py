import os
import joblib
import pickle
import numpy as np
from tqdm import tqdm

# AthletePose3D: pickled samples -> per-split arrays + metadata, all cameras at 60 Hz
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
dt_root = os.path.join(root_path, 'data', 'athlete_pose_3d_v3')


def convert_with_fps(split):
    pkl_path = os.path.join(dt_root, f'{split}.pkl')
    meta_path = os.path.join(dt_root, f'{split}_meta_fps.joblib')

    print(f"\n=== processing {split} ===")

    print("1. loading the raw data...")
    data = joblib.load(pkl_path, mmap_mode='r')
    print(f"   raw samples: {len(data)}")

    print("2. splitting arrays and metadata...")
    n = len(data)

    temp_arrays = {
        'joint_3d_image': [],
        'joint_3d_camera': [],
        'box': [],
        'center': [],
        'scale': []
    }
    temp_meta = []

    for i, item in enumerate(tqdm(data, desc="   splitting")):
        # large arrays
        for key in temp_arrays.keys():
            if key in item:
                temp_arrays[key].append(item[key])

        # metadata
        meta_item = {}
        for k, v in item.items():
            if k not in temp_arrays:
                meta_item[k] = v
        temp_meta.append(meta_item)

    print("3. converting to NumPy arrays...")
    arrays_dict = {}
    for key in temp_arrays.keys():
        if temp_arrays[key]:
            arrays_dict[key] = np.array(temp_arrays[key], dtype=np.float32)
            print(f"   {key}: {arrays_dict[key].shape}")

    print("4. extracting metadata...")
    meta_dict = {
        'cameraid': np.array([m['cameraid'] for m in temp_meta]),
        'videoid': np.array([m['videoid'] for m in temp_meta]),
        'video_width': np.array([m['video_width'] for m in temp_meta]),
        'video_height': np.array([m['video_height'] for m in temp_meta]),
        'action': np.array([m.get('action', '') for m in temp_meta]),
        'ratio': np.array([m.get('ratio', 1.0) for m in temp_meta])
    }

    # 5. common frame rate for every split
    print(f"5. resampling {split} to 60 Hz (120 Hz -> 60 Hz)...")
    sources = meta_dict['videoid']
    cam_names = meta_dict['cameraid']

    # the rm cameras (running) record at 120 Hz
    rm_cam_mask = np.array([name.startswith('rm') for name in cam_names])
    all_indices = np.arange(len(sources))
    unique_sources = np.unique(sources)

    kept_indices = []
    for src in tqdm(unique_sources, desc="   resampling"):
        src_mask = (sources == src)
        src_indices = all_indices[src_mask]
        src_cam_rm = rm_cam_mask[src_indices[0]]
        if src_cam_rm:
            kept_indices.append(src_indices[::2])  # 120 Hz -> 60 Hz
        else:
            kept_indices.append(src_indices)  # 60 Hz: keep all frames
    kept_indices = np.concatenate(kept_indices)

    print(f"   kept frames: {len(kept_indices)} / {len(sources)}")
    for key in arrays_dict.keys():
        arrays_dict[key] = arrays_dict[key][kept_indices]
    for key in meta_dict.keys():
        meta_dict[key] = meta_dict[key][kept_indices]

    print("6. saving arrays...")
    for key in arrays_dict.keys():
        npy_path = os.path.join(dt_root, f'{split}_{key}_fps.npy')
        np.save(npy_path, arrays_dict[key])
        print(f"   saved {key}: {arrays_dict[key].shape}")

    print("7. saving metadata...")
    joblib.dump(meta_dict, meta_path, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"{split} done")


if __name__ == "__main__":
    # convert both train and valid
    convert_with_fps('train')
    convert_with_fps('valid')
    print("\nDone: train and valid (test) splits are at 60 Hz")