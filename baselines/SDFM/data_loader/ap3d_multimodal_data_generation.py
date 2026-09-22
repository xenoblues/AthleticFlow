import os
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from data_loader.datareader_ap3d import DataReaderAP3D

# ===================== 配置参数（和你的要求完全一致） =====================
T_HIS = 15  # 历史帧长度
T_PRED = 60  # 预测帧长度
T_TOTAL = T_HIS + T_PRED  # 总序列长度 75
SKIP_RATE = T_HIS
STEP = T_HIS
K_NEIGHBORS = 50

SAVE_DIR = "data/athlete_pose_3d_v3/multimodal"

THRE_HIS = 0.15  # 历史帧距离阈值（米）
THRE_PRED = 0.05  # 未来帧多样性阈值（米）
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
SAVE_DIR = os.path.join(root_path, SAVE_DIR)


def generate_ap3d_multimodal():
    print("=" * 80)
    print("🔥 AP3D 极致优化多模态生成 | 索引存储 | 体积<5MB")
    print(f"参数：t_his={T_HIS}, t_pred={T_PRED}, step={STEP}")
    print(f"阈值：历史={THRE_HIS}m, 未来={THRE_PRED}m")
    print("=" * 80)

    # 1. 加载数据（强制float32，内存减半）
    reader = DataReaderAP3D(normalize_to_neg1_pos1=True)
    train_seqs = reader.get_train_sequences()
    test_seqs = reader.get_test_sequences()

    # 转换为float32
    train_seqs = [seq.astype(np.float32) for seq in train_seqs]
    test_seqs = [seq.astype(np.float32) for seq in test_seqs]

    # 2. 全局拼接
    train_global = np.concatenate(train_seqs, axis=0)
    test_global = np.concatenate(test_seqs, axis=0)
    print(f"训练集总帧数：{len(train_global)} | 测试集总帧数：{len(test_global)}")

    # 3. 生成候选库（float32存储）
    print("\n📦 生成候选库...")
    candi_list = []
    max_start_train = len(train_global) - T_TOTAL
    for i in range(0, max_start_train + 1, STEP):
        candi_list.append(train_global[i:i + T_TOTAL])
    candi_data = np.stack(candi_list, dtype=np.float32)
    print(f"候选库大小：{candi_data.shape} | 内存占用：{candi_data.nbytes / 1024 / 1024:.1f}MB")

    # 4. 提取测试特征
    print("\n🔍 提取测试特征...")
    test_features = []
    max_start_test = len(test_global) - T_TOTAL
    for i in range(0, max_start_test + 1, STEP):
        clip = test_global[i:i + T_TOTAL]
        test_features.append(clip[T_HIS - 1].reshape(-1))
    test_features = np.array(test_features, dtype=np.float32)
    total_clips = len(test_features)
    print(f"测试Clip总数：{total_clips}")

    # 5. KNN检索
    print("\n🚀 历史帧相似性检索...")
    candi_features = candi_data[:, T_HIS - 1].reshape(len(candi_data), -1).astype(np.float32)
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree', n_jobs=-1)
    knn.fit(candi_features)
    distances_his, indices_his = knn.kneighbors(test_features)

    # 6. 双阈值过滤（只保存最终整数索引）
    print("\n🧹 双阈值过滤...")
    multi_indices = []
    stats_before = []
    stats_after = []

    for i in tqdm(range(total_clips)):
        # 历史帧过滤
        valid_mask_his = distances_his[i] < THRE_HIS
        valid_idx_his = indices_his[i][valid_mask_his]
        stats_before.append(len(valid_idx_his))

        if len(valid_idx_his) <= 1:
            multi_indices.append(valid_idx_his.astype(np.int32))
            stats_after.append(len(valid_idx_his))
            continue

        # 未来帧多样性过滤
        candidates = candi_data[valid_idx_his]
        future_candidates = candidates[:, T_HIS:].reshape(len(candidates), -1)
        dist_matrix = np.linalg.norm(future_candidates[:, None] - future_candidates, axis=2)

        keep_idx = [0]
        for j in range(1, len(candidates)):
            min_dist = np.min(dist_matrix[j, keep_idx])
            if min_dist > THRE_PRED:
                keep_idx.append(j)

        # ✅ 只保存整数索引（核心优化）
        final_indices = valid_idx_his[keep_idx].astype(np.int32)
        multi_indices.append(final_indices)
        stats_after.append(len(final_indices))

    # 统计
    avg_before = np.mean(stats_before)
    avg_after = np.mean(stats_after)
    print(f"\n📊 过滤统计：")
    print(f"   过滤前平均候选数：{avg_before:.1f}")
    print(f"   过滤后平均候选数：{avg_after:.1f}")
    print(f"   多样性保留率：{avg_after / avg_before * 100:.1f}%")

    # 7. 保存（只保存索引，体积<5MB）
    print("\n💾 保存多模态数据...")
    data_multimodal = {
        "all_subjects": {
            "all_actions": {np.int64(i): multi_indices[i] for i in range(total_clips)}
        }
    }

    # 候选库用float32压缩保存
    np.savez_compressed(
        os.path.join(SAVE_DIR, f"data_candi_t_his{T_HIS}_t_pred{T_PRED}_skiprate{STEP}.npz"),
        **{"data_candidate.npy": candi_data}
    )

    # 多模态索引用int32保存，体积极小
    np.savez_compressed(
        os.path.join(SAVE_DIR,
                     f"t_his{T_HIS}_1_thre{THRE_HIS:.3f}_t_pred{T_PRED}_thre{THRE_PRED:.3f}_index_filterd.npz"),
        data_multimodal=data_multimodal
    )

    # 打印最终文件大小
    candi_size = os.path.getsize(
        os.path.join(SAVE_DIR, f"data_candi_t_his{T_HIS}_t_pred{T_PRED}_skiprate{STEP}.npz")) / 1024 / 1024
    index_size = os.path.getsize(os.path.join(SAVE_DIR,
                                              f"t_his{T_HIS}_1_thre{THRE_HIS:.3f}_t_pred{T_PRED}_thre{THRE_PRED:.3f}_index_filterd.npz")) / 1024 / 1024

    print("=" * 80)
    print("🎉 极致优化完成！")
    print(f"✅ 候选库大小：{candi_size:.2f}MB")
    print(f"✅ 多模态索引大小：{index_size:.2f}MB")
    print(f"✅ 总大小：{candi_size + index_size:.2f}MB（与Human3.6M相当）")
    print("=" * 80)

if __name__ == "__main__":
    generate_ap3d_multimodal()


