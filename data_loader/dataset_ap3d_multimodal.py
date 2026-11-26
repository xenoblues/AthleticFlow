import os
from data_loader.dataset import PerSequenceMultimodalDataset
from data_loader.datareader_ap3d import DataReaderAP3D

MM_DIR = os.path.join('data', 'athlete_pose_3d_v3', 'multimodal')


class DatasetAP3D_multi(PerSequenceMultimodalDataset):
    """AthletePose3D test windows with multimodal ground truth (see PerSequenceMultimodalDataset)."""
    default_multimodal_path = os.path.join(MM_DIR, 't_his15_top50_t_pred60_thre0.100_filtered_dlow_perseq.npz')
    default_candi_path = os.path.join(MM_DIR, 'data_candi_t_his15_t_pred60_skiprate15.npz')

    def load_test_sequences(self):
        return DataReaderAP3D(normalize_to_neg1_pos1=True).get_test_sequences()
