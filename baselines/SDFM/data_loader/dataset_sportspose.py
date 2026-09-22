import os
import re
import random
import warnings
import numpy as np
from data_loader.dataset import Dataset
from data_loader.skeleton import Skeleton


def sportspose_load_function(data_dir):
    """简化版：仅读取COCO格式3D关节.npy文件"""
    measurements = []
    re_day = r"^[inout]+doors$"
    re_person = r"^S[0-9]{2}$"
    re_activity = r"^[a-z_]+$"
    re_file = r"^[a-z_]+[0-9]{4}.npy$"

    for dayname in os.listdir(data_dir):
        path_day = os.path.join(data_dir, dayname)
        if not re.match(re_day, dayname) or not os.path.isdir(path_day):
            continue
        for personname in os.listdir(path_day):
            path_person = os.path.join(path_day, personname)
            if not re.match(re_person, personname) or not os.path.isdir(path_person):
                continue
            for activityname in os.listdir(path_person):
                path_activity = os.path.join(path_person, activityname)
                if not re.match(re_activity, activityname) or not os.path.isdir(path_activity):
                    continue
                for filename in os.listdir(path_activity):
                    if re.match(re_file, filename):
                        jointspath = os.path.join(path_activity, filename)
                        data_joints_3d = np.load(jointspath)
                        measurements.append({
                            "data": data_joints_3d,
                            "activity": activityname,
                            "person_id": personname
                        })

    if len(measurements) == 0:
        warnings.warn(f"未读取到任何数据！请检查路径: {data_dir}")
    return measurements


class DatasetSP(Dataset):
    # 默认训练/测试受试者划分
    DEFAULT_TRAIN_SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08']
    DEFAULT_TEST_SUBJECTS = ['S09', 'S10']

    # 基于你的实测统计的默认缩放因子（1.759×1.02≈1.794）
    DEFAULT_SCALE_FACTOR = 1.8

    def __init__(self,
                 mode,
                 t_his=45,
                 t_pred=180,
                 actions='all',
                 use_vel=False,
                 data_dir="data/sportspose/data",
                 train_subjects=None,
                 test_subjects=None,
                 normalization=True,
                 scale_factor=None,
                 scale_factor_path=None,  # 预计算缩放因子文件路径
                 **kwargs):
        self.mode = mode
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.use_vel = use_vel
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.dirname(current_path)
        self.data_dir = os.path.join(root_path, data_dir)
        self.normalization = normalization

        # 训练/测试受试者划分
        self.train_subjects = train_subjects or self.DEFAULT_TRAIN_SUBJECTS
        self.test_subjects = test_subjects or self.DEFAULT_TEST_SUBJECTS

        # 缩放因子优先级：scale_factor_path > scale_factor > DEFAULT_SCALE_FACTOR
        self.scale_factor = self.DEFAULT_SCALE_FACTOR
        if scale_factor is not None:
            self.scale_factor = scale_factor
        if scale_factor_path is not None and os.path.exists(scale_factor_path):
            self.scale_factor = np.load(scale_factor_path).item()
            print(f"✅ 加载预计算缩放因子: {self.scale_factor:.4f}")


        super().__init__(mode, t_his, t_pred, **kwargs)

    @staticmethod
    def precompute_scale_factor(
            data_dir="./data/sportspose",
            save_path="./data/sportspose_scale_factor.npy",
            force_recalculate=False
    ):
        """
        预计算训练集最优缩放因子（只需运行一次）
        Args:
            data_dir: 数据集根目录
            save_path: 缩放因子保存路径
            force_recalculate: 是否强制重新计算
        Returns:
            scale_factor: 计算得到的最优缩放因子
        """
        if os.path.exists(save_path) and not force_recalculate:
            scale_factor = np.load(save_path).item()
            print(f"✅ 缩放因子已存在: {scale_factor:.4f}，跳过计算")
            return scale_factor

        print("\n🔧 预计算训练集最优缩放因子...")

        # 只加载原始数据，不做任何预处理
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.dirname(current_path)
        dt_root_abs = os.path.join(root_path, data_dir)
        all_raw_data = sportspose_load_function(dt_root_abs)
        train_raw_data = [seq for seq in all_raw_data if seq["person_id"] in DatasetSP.DEFAULT_TRAIN_SUBJECTS]

        # 计算根归一化后的最大绝对值
        all_coords = []
        for seq_info in train_raw_data:
            seq = seq_info["data"][:, :17, :]  # COCO 17关节
            root = seq[:, 0:1, :]
            seq_norm = seq - root
            all_coords.append(seq_norm.reshape(-1, 3))

        all_coords = np.concatenate(all_coords, axis=0)
        max_abs = np.max(np.abs(all_coords))
        scale_factor = max_abs * 1.02  # 加2%安全余量

        # 保存结果
        np.save(save_path, scale_factor)
        print(f"✅ 预计算完成！最优缩放因子: {scale_factor:.4f}")
        print(f"   已保存到: {save_path}")

        return scale_factor

    def prepare_data(self, **kwargs):  # 修复：添加**kwargs接受所有多余参数
        # COCO 17关节标准拓扑
        self.kept_joints = np.arange(17)
        self.skeleton = Skeleton(
            parents=[-1, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14],
            joints_left=[1, 3, 5, 7, 9, 11, 13, 15],
            joints_right=[2, 4, 6, 8, 10, 12, 14, 16]
        )
        self.skeleton.gen_adj_mat()

        # 加载所有原始数据
        all_raw_data = sportspose_load_function(self.data_dir)

        # 按mode筛选数据
        if self.mode == 'train':
            self.raw_data = [seq for seq in all_raw_data if seq["person_id"] in self.train_subjects]
        else:
            self.raw_data = [seq for seq in all_raw_data if seq["person_id"] in self.test_subjects]

        # 按动作过滤
        if self.actions != 'all':
            if isinstance(self.actions, str):
                self.actions = [self.actions]
            self.raw_data = [seq for seq in self.raw_data if seq["activity"] in self.actions]

        # --------------------------
        # 数据预处理 + 归一化
        # --------------------------
        self.data = []
        self.seq_info = []
        self.root_history = []  # 保存每个序列的根节点历史（用于反归一化）

        for seq_info in self.raw_data:
            seq = seq_info["data"][:, self.kept_joints, :]

            if self.normalization:
                # 第一步：根节点归一化（以0号关节为根）
                root = seq[:, 0:1, :].copy()
                seq = seq - root

                # 第二步：线性缩放至[-1,1]
                seq = seq / self.scale_factor

            # 生成速度特征（如果开启）
            if self.use_vel:
                vel = np.diff(seq, axis=0, prepend=seq[:1])
                seq = np.concatenate([seq, vel], axis=-1)

            if seq.shape[0] >= self.t_total:
                self.data.append(np.ascontiguousarray(seq))
                self.seq_info.append(seq_info)
                if self.normalization:
                    self.root_history.append(np.ascontiguousarray(root))

        print(f"\n✅ SP-{self.mode} 加载完成")
        print(f"   受试者: {self.train_subjects if self.mode == 'train' else self.test_subjects}")
        print(f"   动作类型: {self.actions}")
        print(f"   归一化: {'开启 (缩放因子={:.4f})'.format(self.scale_factor) if self.normalization else '关闭'}")
        print(f"   完整序列数: {len(self.data)}")
        print(f"   总滑动窗口数: {self.__len__()}")

    def denormalize(self, pred_norm, root_last):
        """
        内置反归一化方法
        将模型输出的归一化坐标转换回原始坐标空间
        """
        if not self.normalization:
            return pred_norm

        # 第一步：反缩放
        pred_denorm = pred_norm * self.scale_factor

        # 第二步：恢复根节点位置
        pred_denorm = pred_denorm + root_last

        return pred_denorm

    def get_sample_root(self, index):
        """获取指定样本的历史帧最后一帧根节点坐标"""
        if not self.normalization:
            return None

        count = 0
        for seq_idx, seq in enumerate(self.data):
            num_seqs = (seq.shape[0] - self.t_total) // 15 + 1
            if index < count + num_seqs:
                offset = (index - count) * 15
                return self.root_history[seq_idx][offset + self.t_his - 1:offset + self.t_his, :, :]
            count += num_seqs
        raise IndexError("索引超出范围")

    def sample(self):
        seq = random.choice(self.data)
        max_start = seq.shape[0] - self.t_total
        start = random.randint(0, max_start)
        return seq[start:start + self.t_total][None]

    def sampling_generator(self, num_samples=50000, batch_size=256, aug=True):
        for _ in range(num_samples // batch_size):
            batch = [self.sample() for _ in range(batch_size)]
            batch = np.concatenate(batch, axis=0)
            seq_len = batch.shape[1]

            mask_indices = np.random.randint(
                int(seq_len * 0.08),
                int(seq_len * 0.92),
                int(seq_len * 0.2)
            )
            mask = np.array([i not in mask_indices for i in range(seq_len)], dtype=bool)

            if aug:
                if np.random.uniform() > 0.5:  # x-y rotating
                    theta = np.random.uniform(0, 2 * np.pi)
                    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    rotate_xy = np.matmul(batch.transpose([0, 2, 1, 3])[..., 0:2], rotate_matrix)
                    batch[..., 0:2] = rotate_xy.transpose([0, 2, 1, 3])
                    del theta, rotate_matrix, rotate_xy
                if np.random.uniform() > 0.5:  # x-z mirroring
                    batch[..., 0] = - batch[..., 0]
                if np.random.uniform() > 0.5:  # y-z mirroring
                    batch[..., 1] = - batch[..., 1]

            yield batch, mask

    def iter_generator(self, step=15, return_root=False):
        for seq_idx, seq in enumerate(self.data):
            for i in range(0, seq.shape[0] - self.t_total + 1, step):
                sample = seq[None, i:i + self.t_total]
                if return_root and self.normalization:
                    root_last = self.root_history[seq_idx][i + self.t_his - 1:i + self.t_his, :, :]
                    yield sample, root_last
                else:
                    yield sample

    def get_all_test_samples(self, return_root=False):
        all_samples = []
        all_infos = []
        all_roots = []

        for seq_idx, seq in enumerate(self.data):
            for i in range(0, seq.shape[0] - self.t_total + 1, 15):
                sample = seq[i:i + self.t_total]
                all_samples.append(sample)
                all_infos.append({
                    "seq_idx": seq_idx,
                    "start_frame": i,
                    **self.seq_info[seq_idx]
                })
                if return_root and self.normalization:
                    root_last = self.root_history[seq_idx][i + self.t_his - 1:i + self.t_his, :, :]
                    all_roots.append(root_last)

        if return_root:
            return np.array(all_samples), all_infos, np.array(all_roots)
        else:
            return np.array(all_samples), all_infos

    def __len__(self):
        total = 0
        for seq in self.data:
            total += (seq.shape[0] - self.t_total) // 15 + 1
        return total

    def __getitem__(self, index):
        count = 0
        for seq in self.data:
            num_seqs = (seq.shape[0] - self.t_total) // 15 + 1
            if index < count + num_seqs:
                offset = (index - count) * 15
                return seq[offset:offset + self.t_total]
            count += num_seqs
        raise IndexError("索引超出范围")

if __name__ == '__main__':

    ap3d_train = DatasetSP('train', t_his=45, t_pred=180, use_vel=False)

    # 测试采样速度
    # start = time.time()
    # for _ in range(1000):
    #     ap3d_train.sample()
    # sample_time = (time.time() - start) / 1000
    # print(f"单次采样时间: {sample_time * 1000:.3f}毫秒")
    generator = ap3d_train.iter_generator(step=15)
    i = 0
    all_bone_lengths = []
    for data in generator:
        print(data.shape)