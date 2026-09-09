import os
import joblib
import pickle
import numpy as np
from tqdm import tqdm

# 路径
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
dt_root = os.path.join(current_path, 'data/athlete_pose_3d_v3')


def convert_with_fps(split):
    pkl_path = os.path.join(dt_root, f'{split}.pkl')
    meta_path = os.path.join(dt_root, f'{split}_meta_fps.joblib')

    print(f"\n=== 正在处理 {split} ===")

    # 1. 加载原始数据
    print("1. 加载原始数据...")
    data = joblib.load(pkl_path, mmap_mode='r')
    print(f"   原始数据长度: {len(data)}")

    # 2. 分离数据
    print("2. 分离数据...")
    n = len(data)

    # 初始化临时列表
    temp_arrays = {
        'joint_3d_image': [],
        'joint_3d_camera': [],
        'box': [],
        'center': [],
        'scale': []
    }
    temp_meta = []

    for i, item in enumerate(tqdm(data, desc="   分离进度")):
        # 提取大数组
        for key in temp_arrays.keys():
            if key in item:
                temp_arrays[key].append(item[key])

        # 保存元数据
        meta_item = {}
        for k, v in item.items():
            if k not in temp_arrays:
                meta_item[k] = v
        temp_meta.append(meta_item)

    # 3. 转换为 NumPy 数组
    print("3. 转换为 NumPy 数组...")
    arrays_dict = {}
    for key in temp_arrays.keys():
        if temp_arrays[key]:
            arrays_dict[key] = np.array(temp_arrays[key], dtype=np.float32)
            print(f"   {key}: {arrays_dict[key].shape}")

    # 4. 向量化提取元数据
    print("4. 提取元数据...")
    meta_dict = {
        'cameraid': np.array([m['cameraid'] for m in temp_meta]),
        'videoid': np.array([m['videoid'] for m in temp_meta]),
        'video_width': np.array([m['video_width'] for m in temp_meta]),
        'video_height': np.array([m['video_height'] for m in temp_meta]),
        'action': np.array([m.get('action', '') for m in temp_meta]),
        'ratio': np.array([m.get('ratio', 1.0) for m in temp_meta])
    }

    # 5. 统一帧率（✅ 所有 split 都处理，包括 valid/test）
    print(f"5. 统一 {split} 集帧率（120Hz→60Hz）...")
    sources = meta_dict['videoid']
    cam_names = meta_dict['cameraid']

    # 向量化筛选 rm 相机（running 动作专用）
    rm_cam_mask = np.array([name.startswith('rm') for name in cam_names])
    all_indices = np.arange(len(sources))
    unique_sources = np.unique(sources)

    kept_indices = []
    for src in tqdm(unique_sources, desc="   帧率统一进度"):
        src_mask = (sources == src)
        src_indices = all_indices[src_mask]
        src_cam_rm = rm_cam_mask[src_indices[0]]
        if src_cam_rm:
            kept_indices.append(src_indices[::2])  # 120Hz→60Hz 下采样
        else:
            kept_indices.append(src_indices)  # 60Hz 保留全部
    kept_indices = np.concatenate(kept_indices)

    # 同步更新所有数据
    print(f"   保留帧数: {len(kept_indices)} / {len(sources)}")
    for key in arrays_dict.keys():
        arrays_dict[key] = arrays_dict[key][kept_indices]
    for key in meta_dict.keys():
        meta_dict[key] = meta_dict[key][kept_indices]

    # 6. 保存为单独的 .npy 文件
    print("6. 保存大数组...")
    for key in arrays_dict.keys():
        npy_path = os.path.join(dt_root, f'{split}_{key}_fps.npy')
        np.save(npy_path, arrays_dict[key])
        print(f"   已保存 {key}: {arrays_dict[key].shape}")

    # 7. 保存元数据
    print("7. 保存元数据...")
    joblib.dump(meta_dict, meta_path, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ {split} 处理完成！")


if __name__ == "__main__":
    # 执行转换（✅ 同时处理 train 和 valid）
    convert_with_fps('train')
    convert_with_fps('valid')
    print("\n🎉 全部转换完成！Train 和 Test 集均已统一至 60Hz")