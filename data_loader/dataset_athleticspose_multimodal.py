import os
import numpy as np
from data_loader.dataset import PerSequenceMultimodalDataset, ROOT_DIR

DATA_DIR = os.path.join('data', 'AthleticsPose')


class DatasetAthleticsPose_multi(PerSequenceMultimodalDataset):
    """AthleticsPose test windows with multimodal ground truth (see PerSequenceMultimodalDataset)."""
    default_multimodal_path = os.path.join(DATA_DIR, 'multimodal', 't_his15_1_thre0.500_t_pred60_thre0.100_index_filterd.npz')
    default_candi_path = os.path.join(DATA_DIR, 'multimodal', 'data_candi_t_his15_t_pred60_skiprate15.npz')

    def load_test_sequences(self):
        data = np.load(os.path.join(ROOT_DIR, DATA_DIR, 'test.npz'), allow_pickle=True)
        return list(data['trajectories'])
