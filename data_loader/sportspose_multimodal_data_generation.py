import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from dataset_sportspose import DatasetSP

from data_loader.dataset_sportspose import DatasetSP

# ===================== 配置参数（与你的训练流程完全一致） =====================
T_HIS = 45  # 历史帧长度（和训练完全一致）
T_PRED = 180  # 预测帧长度（和训练完全一致）
T_TOTAL = T_HIS + T_PRED  # 总序列长度 75
THRE_HIS = 0.15
THRE_PRED = 0.05
SKIP_RATE = T_HIS  # 训练集候选库采样步长
STEP = T_HIS  # 测试集采样步长（和评估一致）
K_NEIGHBORS = 50  # 单样本最大真值数量（避免显存溢出）

# 路径配置（和你的项目结构一致）
DATA_DIR = "data/sportspose/data"
SCALE_FACTOR_PATH = "data/sportspose_scale_factor.npy"
SAVE_DIR = "data/sportspose/multimodal"
os.makedirs(SAVE_DIR, exist_ok=True)


# ========================================================================
def generate_sportspose_multimodal():
    print("="*80)
    print("🔥 SportsPose 极致优化多模态生成 | 索引存储 | 总大小<50MB")
    print(f"参数：t_his={T_HIS}, t_pred={T_PRED}, step={STEP}")
    print(f"阈值：历史={THRE_HIS}m, 未来={THRE_PRED}m")
    print("="*80)

    # 1. 加载原始归一化数据（强制float32）
    print("\n📥 加载SportsPose原始数据...")
    train_dataset = DatasetSP(mode='train', t_his=T_HIS, t_pred=T_PRED, normalization=False)
    test_dataset = DatasetSP(mode='test', t_his=T_HIS, t_pred=T_PRED, normalization=False)

    # 转换为float32，内存减半
    train_seqs = [seq.astype(np.float32) for seq in train_dataset.data]
    test_seqs = [seq.astype(np.float32) for seq in test_dataset.data]

    # 2. 全局拼接（无视Subject/Action）
    train_global = np.concatenate(train_seqs, axis=0)
    test_global = np.concatenate(test_seqs, axis=0)
    print(f"训练集总帧数：{len(train_global)} | 测试集总帧数：{len(test_global)}")

    # 3. 生成训练集候选库（float32存储）
    print("\n📦 生成全局候选库...")
    candi_list = []
    max_start_train = len(train_global) - T_TOTAL
    for i in range(0, max_start_train + 1, STEP):
        candi_list.append(train_global[i:i+T_TOTAL])
    candi_data = np.stack(candi_list, dtype=np.float32)
    print(f"候选库大小：{candi_data.shape} | 内存占用：{candi_data.nbytes/1024/1024:.1f}MB")

    # 4. 提取测试集检索特征
    print("\n🔍 提取测试集特征...")
    test_features = []
    max_start_test = len(test_global) - T_TOTAL
    for i in range(0, max_start_test + 1, STEP):
        clip = test_global[i:i+T_TOTAL]
        test_features.append(clip[T_HIS-1].reshape(-1))
    test_features = np.array(test_features, dtype=np.float32)
    total_clips = len(test_features)
    print(f"测试Clip总数：{total_clips}")

    # 5. 批量KNN检索（历史帧相似性）
    print("\n🚀 历史帧相似性检索...")
    candi_features = candi_data[:, T_HIS-1].reshape(len(candi_data), -1).astype(np.float32)
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree', n_jobs=-1)
    knn.fit(candi_features)
    distances_his, indices_his = knn.kneighbors(test_features)

    # 6. 双阈值过滤（只保存整数索引）
    print("\n🧹 双阈值过滤...")
    multi_indices = []
    stats_before = []
    stats_after = []

    for i in tqdm(range(total_clips)):
        # 第一步：历史帧相似性过滤
        valid_mask_his = distances_his[i] < THRE_HIS
        valid_idx_his = indices_his[i][valid_mask_his]
        stats_before.append(len(valid_idx_his))

        if len(valid_idx_his) <= 1:
            multi_indices.append(valid_idx_his.astype(np.int32))
            stats_after.append(len(valid_idx_his))
            continue

        # 第二步：未来帧多样性过滤（NMS）
        candidates = candi_data[valid_idx_his]
        future_candidates = candidates[:, T_HIS:].reshape(len(candidates), -1)
        dist_matrix = np.linalg.norm(future_candidates[:, None] - future_candidates, axis=2)

        keep_idx = [0]
        for j in range(1, len(candidates)):
            min_dist = np.min(dist_matrix[j, keep_idx])
            if min_dist > THRE_PRED:
                keep_idx.append(j)

        # ✅ 只保存最终整数索引（核心优化）
        final_indices = valid_idx_his[keep_idx].astype(np.int32)
        multi_indices.append(final_indices)
        stats_after.append(len(final_indices))

    # 统计信息
    avg_before = np.mean(stats_before)
    avg_after = np.mean(stats_after)
    print(f"\n📊 过滤统计：")
    print(f"   过滤前平均候选数：{avg_before:.1f}")
    print(f"   过滤后平均候选数：{avg_after:.1f}")
    print(f"   多样性保留率：{avg_after/avg_before*100:.1f}%")

    # 7. 保存文件（与AP3D/HumanEva格式完全一致）
    print("\n💾 保存多模态数据...")
    data_multimodal = {
        "all_subjects": {
            "all_actions": {np.int64(i): multi_indices[i] for i in range(total_clips)}
        }
    }

    # 候选库float32压缩保存
    np.savez_compressed(
        os.path.join(SAVE_DIR, f"data_candi_t_his{T_HIS}_t_pred{T_PRED}_skiprate{STEP}.npz"),
        **{"data_candidate.npy": candi_data}
    )

    # 多模态索引int32保存（体积极小）
    np.savez_compressed(
        os.path.join(SAVE_DIR, f"t_his{T_HIS}_1_thre{THRE_HIS:.3f}_t_pred{T_PRED}_thre{THRE_PRED:.3f}_index_filterd.npz"),
        data_multimodal=data_multimodal
    )

    # 打印最终文件大小
    candi_size = os.path.getsize(os.path.join(SAVE_DIR, f"data_candi_t_his{T_HIS}_t_pred{T_PRED}_skiprate{STEP}.npz"))/1024/1024
    index_size = os.path.getsize(os.path.join(SAVE_DIR, f"t_his{T_HIS}_1_thre{THRE_HIS:.3f}_t_pred{T_PRED}_thre{THRE_PRED:.3f}_index_filterd.npz"))/1024/1024

    print("="*80)
    print("🎉 SportsPose 极致优化完成！")
    print(f"✅ 候选库大小：{candi_size:.2f}MB")
    print(f"✅ 多模态索引大小：{index_size:.2f}MB")
    print(f"✅ 总大小：{candi_size + index_size:.2f}MB")
    print("="*80)

if __name__ == "__main__":
    generate_sportspose_multimodal()
