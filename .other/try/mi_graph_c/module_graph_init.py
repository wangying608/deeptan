import os
import time
import numpy as np
import h5py
from typing import Optional
from itertools import combinations
from joblib import Parallel, delayed
import multiprocessing as mp
from util_mi import pcc_optimal, slide_win_1d_sd, iterate_feature_pairs_mt_mi
# from util_mi_c import bins_fd, histogram2d_fast, mi_fd, gen_windows_and_slides, std_each_window, slide_win_1d_sd, slide_win_2d_pcc, pcc_optimal, slide_win_2d_mutualinfo, mi_optimal, iterate_feature_pairs_mt_mi


class mi_graph_init:
    '''
    Initialize a mutual information graph using the matrix of features.
    '''
    def __init__(
            self,
            matrix_np: np.ndarray,
            feature_names: list[str],
            std_thresh: float = 0.01,
            corr_thresh: float = 0.95,
            save_dir: Optional[str] = None,
            n_processes: int = 3,
        ):
        self.matrix_np = matrix_np
        self.feature_names = feature_names
        self.std_thresh = std_thresh
        self.corr_thresh = corr_thresh
        self.n_processes = n_processes
        time_init = time.strftime("%Y%m%d%H%M%S", time.localtime())
        if save_dir is not None:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = os.getcwd()
        # Def the output file path
        self.output_file_path = os.path.join(self.save_dir, f'graph_init_{time_init}.h5')

    def generate_feat_combinations(
            self,
            n: int,
        ):
        '''
        Generates all possible combinations of 2 features from n features.
        '''
        n_combinations = n * (n - 1) // 2
        features_combinations = np.zeros((n_combinations, 2), dtype=int)
        features = np.arange(n)
        feat_combinations = combinations(features, 2)
        i = 0
        for ft1, ft2 in feat_combinations:
            features_combinations[i, 0] = ft1
            features_combinations[i, 1] = ft2
            i += 1
        return features_combinations

    def minmax_scale_matrix(self, mat_in: np.ndarray) -> np.ndarray:
        '''
        Min-max scale the matrix to [0,1]
        '''
        min_vals = np.min(mat_in, axis=0)
        max_vals = np.max(mat_in, axis=0)
        minmaxed = (mat_in - min_vals) / (max_vals - min_vals)
        return minmaxed

    def pcc_iter_feature_pairs(self, mat_in: np.ndarray) -> np.ndarray:
        '''
        Compute PCC for all feature pairs
        '''
        features = np.arange(mat_in.shape[1])
        feat_combns = combinations(features, 2)
        vec_out = Parallel(n_jobs=self.n_processes)(delayed(pcc_optimal)(mat_in[:,feat1], mat_in[:,feat2]) for feat1, feat2 in feat_combns)
        return np.array(vec_out)
    
    def rm_low_std_feats(
            self,
            mat_in: np.ndarray,
            feature_names: list[str],
            std_thresh: float,
        ) -> tuple[np.ndarray, list[str], list[str]]:
        '''
        Remove low-std features using slide_win_1d_sd
        '''
        cols2save_bool = np.apply_along_axis(slide_win_1d_sd, 0, mat_in, threshold=std_thresh)
        cols2rm = np.where(cols2save_bool == False)[0]
        matx = np.delete(mat_in, cols2rm, axis=1)
        feat_names = np.delete(feature_names, cols2rm).tolist()
        feat_low_std_names = np.array(feature_names)[cols2rm].tolist()
        return matx, feat_names, feat_low_std_names
    
    def rm_high_corr_feats(
            self,
            mat_in: np.ndarray,
            feature_names: list[str],
            corr_thresh: float,
        ) -> tuple[np.ndarray, list[str], list[list[str]]]:
        '''
        Remove high-corr features using pcc_iter_feature_pairs with 2D sliding windows.
        '''
        pccs = self.pcc_iter_feature_pairs(mat_in)
        combs_ind2rm = np.where(pccs > corr_thresh)[0]
        simi_feature_pairs: list[tuple[int, int]] = []
        feat2rm: list[int] = []
        simi_feat_names: list[list[str]] = []
        n_feat = mat_in.shape[1]
        if combs_ind2rm.shape[0] > 0:
            for i in range(combs_ind2rm.shape[0]):
                feat1, feat2 = combs_ind2rm[0] // n_feat, combs_ind2rm[0] % n_feat
                '''
                Sort the features in ascending order.
                '''
                if feat1 > feat2:
                    feat1, feat2 = feat2, feat1
                simi_feature_pairs.append((feat1, feat2))
                simi_feat_names.append([feature_names[feat1], feature_names[feat2]])
                feat2rm.append(feat1)
            feat2rm = np.unique(np.array(feat2rm)).tolist()
            matx = np.delete(mat_in, feat2rm, axis=1)
            feat_names = np.delete(feature_names, feat2rm).tolist()
        else:
            matx = mat_in
            feat_names = feature_names
        return matx, feat_names, simi_feat_names

    def save_after_feat_select(
            self,
            mat_in: np.ndarray,
            feature_names: list[str],
            simi_feat_names: list[list[str]],
            low_std_feat_names: list[str],
        ) -> None:
        '''
        Save [ matx, feat_names, simi_feat_names ] to h5 file.
        '''
        h5file = h5py.File(self.output_file_path, 'a')
        h5file.create_dataset('matin', data=mat_in)
        h5file.create_dataset('featnames', data=feature_names)
        h5file.create_dataset('simifeatpairs', data=simi_feat_names)
        h5file.create_dataset('lowstdfeatnames', data=low_std_feat_names)
        h5file.close()

    def generate_chuncks(
            self,
            feat_pairs: np.ndarray,
            n_processes: int = 3,
        ) -> list[np.ndarray]:
        '''
        Generate chuncks of feature pairs for multi-processing.
        '''
        n_pairs = feat_pairs.shape[0]
        # n_chuncks = n_processes
        n_chuncks = min(n_processes, n_pairs)
        n_pairs_per_chunck = int(n_pairs / n_chuncks)
        feat_pairs_chuncks = [feat_pairs[i:i+n_pairs_per_chunck] for i in range(0, n_pairs, n_pairs_per_chunck)]
        # Check if there is any remaining pairs
        if n_pairs % n_chuncks != 0:
            feat_pairs_chuncks.append(feat_pairs[n_pairs_per_chunck*n_chuncks:])
            n_chuncks += 1
        return feat_pairs_chuncks

    def get_mis_from_chunck(
            self,
            chunck_result,
        ):
        mis_chunck = chunck_result.get()
        return mis_chunck[0], mis_chunck[1]

    def mi_foreach_chunck(
            self,
            mat_in: np.ndarray,
            feat_pairs_chuncks: list[np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Calculate mutual information for each chunck of feature pairs.
        '''
        # Calculate mutual information for each chunck
        n_chuncks = len(feat_pairs_chuncks)
        # Initialize multiprocessing pool
        pool = mp.Pool(processes=n_chuncks)
        # Calculate MI values for each chunck
        mis_chuncks = []
        for i in range(n_chuncks):
            mis_chuncks.append(
                pool.apply_async(
                    iterate_feature_pairs_mt_mi,
                    args=(mat_in, feat_pairs_chuncks[i]),
                )
            )
        # Close multiprocessing pool
        pool.close()
        '''
        Get MI values from each chunck
        The output of each chunck is a tuple of two ndarrays, one for MI values and one for feature pairs.
        e.g. (np.array([1.1,2.1]), np.array([[5,6],[7,8]]))
        We need to concatenate these ndarrays to get the final MI values and feature pairs.
        '''
        mis_chuncks_sorted = [self.get_mis_from_chunck(mis_chunck) for mis_chunck in mis_chuncks]
        # Concatenate MI values and feature pairs from each chunck
        mis = np.concatenate([mis_chunck[0] for mis_chunck in mis_chuncks_sorted], axis=0)
        feat_pairs_mi = np.concatenate([mis_chunck[1] for mis_chunck in mis_chuncks_sorted], axis=0)
        # Sort MI values in descending order
        mis_sorted_indices = np.argsort(-mis)
        mis_sorted = mis[mis_sorted_indices]
        feat_pairs_mi_sorted = feat_pairs_mi[mis_sorted_indices]
        return mis_sorted, feat_pairs_mi_sorted
    
    def save_mi(
            self,
            mi_values_sorted: np.ndarray,
            feat_pairs_mi_sorted: np.ndarray,
        ) -> None:
        '''
        Save MI values and feature pairs in a h5 file.
        '''
        h5file = h5py.File(self.output_file_path, 'a')
        h5file.create_dataset('sortedmi', data=mi_values_sorted)
        h5file.create_dataset('sortedfeatpairs', data=feat_pairs_mi_sorted)
        h5file.close()

    def mi_matrix_init(
            self,
            mat_in: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Calculate mutual information between each pair of features(columns) through multiple processes with feature pairs chuncks.
        '''
        # Generate feature pairs
        feat_pairs = self.generate_feat_combinations(mat_in.shape[1])
        # Generate chuncks of feature pairs for parallel processing
        feat_pairs_chuncks = self.generate_chuncks(feat_pairs, self.n_processes)
        mi_values_sorted, feat_pairs_mi_sorted = self.mi_foreach_chunck(mat_in, feat_pairs_chuncks)
        # Save output files
        self.save_mi(mi_values_sorted, feat_pairs_mi_sorted)
        return mi_values_sorted, feat_pairs_mi_sorted

    def mi_matrix_init_with_feat_select(
            self,
        ):
        '''
        Run feature selection and MI calculation.
        1. Min-max scale the matrix to [0,1]
        2. Remove low-std features using `slide_win_1d_sd`.
        3. Remove high-corr features using `pcc_iter_feature_pairs` with 2D sliding windows.
        4. Save processed matrix and feature names.
        5. Calculate mutual information between each pair of features(columns) through multiple processes with feature pairs chuncks.
        6. Save sorted MI values and feature pairs.
        '''
        matx = self.minmax_scale_matrix(self.matrix_np)
        matx, feat_names, feat_names_lowstd = self.rm_low_std_feats(matx, self.feature_names, std_thresh=self.std_thresh)
        matx, feat_names, simi_feat_names = self.rm_high_corr_feats(matx, feat_names, corr_thresh=self.corr_thresh)
        self.save_after_feat_select(matx, feat_names, simi_feat_names, feat_names_lowstd)
        mi_values_sorted, feat_pairs_mi_sorted = self.mi_matrix_init(matx)
        
        return self.output_file_path
