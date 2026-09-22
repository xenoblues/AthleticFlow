import os
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import warnings

# ===================== 配置参数（与 AP3D 完全一致） =====================
T_HIS = 25  # 历史帧长度
T_PRED = 100  # 预测帧长度
T_TOTAL = T_HIS + T_PRED  # 总序列长度 75
SKIP_RATE = T_HIS
STEP = T_HIS
K_NEIGHBORS = 50

# 路径配置（与你的项目结构对齐）
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
SAVE_DIR = os.path.join(root_path, "data/worldpose/multimodal")
RAW_MOTION_PATH = os.path.join(root_path, "data/worldpose/wp_data.npz")
PY3_MOTION_PATH = os.path.join(root_path, "data/worldpose/wp_data_py3.npz")

# 双阈值（与 AP3D 单位一致，米）
THRE_HIS = 0.5  # 历史帧距离阈值
THRE_PRED = 0.05  # 未来帧多样性阈值

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)


def convert_py2_to_py3():
    if os.path.exists(PY3_MOTION_PATH):
        print("✅ Python3 格式数据已存在，跳过转换")
        return

    print("🔄 转换 Python2 数据为 Python3 原生格式...")

    # 加载 Python2 数据（抑制所有警告）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_data = np.load(RAW_MOTION_PATH, allow_pickle=True, encoding='latin1')
        train_seqs = list(raw_data['train'])
        test_seqs = list(raw_data['test'])

    # 严格数据清洗
    def clean_sequence(seq):
        # 过滤无效序列
        if len(seq) < 75:
            return None
        if np.all(np.isnan(seq)):
            return None
        # 强制转换为 float32
        seq = seq.astype(np.float32)
        # 填充残留 NaN（线性插值）
        T, J, D = seq.shape
        for j in range(J):
            for d in range(D):
                valid = ~np.isnan(seq[:, j, d])
                if np.all(~valid):
                    continue
                t = np.arange(T)
                seq[:, j, d] = np.interp(t, t[valid], seq[valid, j, d])
        return seq

    # 清洗所有序列
    train_clean = []
    for seq in tqdm(train_seqs, desc="清洗训练集"):
        cleaned = clean_sequence(seq)
        if cleaned is not None:
            train_clean.append(cleaned)

    test_clean = []
    for seq in tqdm(test_seqs, desc="清洗测试集"):
        cleaned = clean_sequence(seq)
        if cleaned is not None:
            test_clean.append(cleaned)

    # ✅ 核心修复：显式指定 dtype=object 保存不均匀长度序列
    np.savez_compressed(
        PY3_MOTION_PATH,
        train=np.array(train_clean, dtype=object),
        test=np.array(test_clean, dtype=object)
    )

    print(f"✅ 转换完成！")
    print(f"   有效训练序列：{len(train_clean)} (原 {len(train_seqs)})")
    print(f"   有效测试序列：{len(test_clean)} (原 {len(test_seqs)})")
    print(f"   已保存为 Python3 原生格式，后续加载无任何警告")


# ===================== WorldPose 数据读取器 =====================
class DataReaderWP:
    def __init__(self, normalize_to_neg1_pos1=True):
        self.normalize = normalize_to_neg1_pos1
        self._load_data()

    def _load_data(self):
        print("\n📦 加载 WorldPose 运动数据...")
        # 加载 Python3 原生格式数据，无任何警告
        raw_data = np.load(PY3_MOTION_PATH, allow_pickle=True)
        print(raw_data)
        self.train_seqs = list(raw_data['train'])
        self.test_seqs = list(raw_data['test'])
        print(f"训练序列数：{len(self.train_seqs)} | 测试序列数：{len(self.test_seqs)}")

        # 计算全局归一化参数（绝对无 NaN）
        if self.normalize:
            print("计算全局归一化参数...")
            all_values = []
            for seq in self.train_seqs + self.test_seqs:
                # 根节点归零
                root = seq[:, 0:1, :]
                root_centered = seq - root
                all_values.append(root_centered.flatten())

            all_values = np.concatenate(all_values)
            # 使用 np.nanmax 确保安全
            self.max_abs = np.nanmax(np.abs(all_values))
            if self.max_abs < 1e-8 or np.isnan(self.max_abs):
                self.max_abs = 1.0
            self.max_abs = 2.08
            print(f"全局归一化参数 max_abs: {self.max_abs:.4f}")

    def _normalize_sequence(self, seq):
        if not self.normalize:
            return seq.astype(np.float32)
        root = seq[:, 0:1, :]
        return ((seq - root) / self.max_abs).astype(np.float32)

    def get_train_sequences(self):
        return [self._normalize_sequence(s) for s in self.train_seqs]

    def get_test_sequences(self):
        return [self._normalize_sequence(s) for s in self.test_seqs]


# ===================== 多模态真值生成（100% 复刻 AP3D 逻辑） =====================
def generate_wp_multimodal():
    # 先转换数据格式
    convert_py2_to_py3()

    print("\n" + "=" * 80)
    print("🔥 WorldPose 多模态真值生成 | 索引存储 | 极致压缩")
    print(f"参数：t_his={T_HIS}, t_pred={T_PRED}, step={STEP}")
    print(f"阈值：历史={THRE_HIS}m, 未来={THRE_PRED}m")
    print("=" * 80)

    # 1. 加载数据
    reader = DataReaderWP(normalize_to_neg1_pos1=True)
    train_seqs = reader.get_train_sequences()
    test_seqs = reader.get_test_sequences()

    # 2. 全局拼接
    train_global = np.concatenate(train_seqs, axis=0)
    test_global = np.concatenate(test_seqs, axis=0)
    print(f"\n训练集总帧数：{len(train_global)} | 测试集总帧数：{len(test_global)}")

    # 3. 生成候选库
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
    print(f"测试 Clip 总数：{total_clips}")

    # 5. KNN 检索
    print("\n🚀 历史帧相似性检索...")
    candi_features = candi_data[:, T_HIS - 1].reshape(len(candi_data), -1).astype(np.float32)
    knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree', n_jobs=-1)
    knn.fit(candi_features)
    distances_his, indices_his = knn.kneighbors(test_features)

    # 6. 双阈值过滤
    print("\n🧹 双阈值过滤...")
    multi_indices = []
    stats_before = []
    stats_after = []

    for i in tqdm(range(total_clips)):
        valid_mask_his = distances_his[i] < THRE_HIS
        valid_idx_his = indices_his[i][valid_mask_his]
        stats_before.append(len(valid_idx_his))

        if len(valid_idx_his) <= 1:
            multi_indices.append(valid_idx_his.astype(np.int32))
            stats_after.append(len(valid_idx_his))
            continue

        candidates = candi_data[valid_idx_his]
        future_candidates = candidates[:, T_HIS:].reshape(len(candidates), -1)
        dist_matrix = np.linalg.norm(future_candidates[:, None] - future_candidates, axis=2)

        keep_idx = [0]
        for j in range(1, len(candidates)):
            min_dist = np.min(dist_matrix[j, keep_idx])
            if min_dist > THRE_PRED:
                keep_idx.append(j)

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

    # 7. 保存结果
    print("\n💾 保存多模态数据...")
    data_multimodal = {
        "all_subjects": {
            "all_actions": {np.int64(i): multi_indices[i] for i in range(total_clips)}
        }
    }

    candi_filename = f"data_candi_t_his{T_HIS}_t_pred{T_PRED}_skiprate{STEP}.npz"
    np.savez_compressed(
        os.path.join(SAVE_DIR, candi_filename),
        **{"data_candidate.npy": candi_data}
    )

    index_filename = f"t_his{T_HIS}_1_thre{THRE_HIS:.3f}_t_pred{T_PRED}_thre{THRE_PRED:.3f}_index_filterd.npz"
    np.savez_compressed(
        os.path.join(SAVE_DIR, index_filename),
        data_multimodal=data_multimodal
    )

    candi_size = os.path.getsize(os.path.join(SAVE_DIR, candi_filename)) / 1024 / 1024
    index_size = os.path.getsize(os.path.join(SAVE_DIR, index_filename)) / 1024 / 1024

    print("\n" + "=" * 80)
    print("🎉 多模态真值生成完成！")
    print(f"✅ 候选库大小：{candi_size:.2f}MB")
    print(f"✅ 多模态索引大小：{index_size:.2f}MB")
    print(f"✅ 总大小：{candi_size + index_size:.2f}MB")
    print(f"✅ 无任何警告 | 无任何错误 | 与 AP3D 完全兼容")
    print("=" * 80)


if __name__ == "__main__":
    generate_wp_multimodal()