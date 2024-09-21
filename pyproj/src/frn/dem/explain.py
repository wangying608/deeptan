import os
from typing import Optional, Union, List, Dict, Any
from frn.dem.pipeline import DEMPredict


class DEMFeatureRanking:
    r"""
    DEM model feature ranking.
    """
    def __init__(
            self,
            model_path: str,
            batch_size: int = 32,
            map_location: Optional[str] = None,
        ):
        """
        Initialize DEM model feature ranking.
        - map_location: `cuda:0` for GPU, `cpu` for CPU.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.map_location = map_location

    def rank_features(
            self,
            litdata_dir: str,
            output_dir: str,
            random_states: list[int],
        ):
        """
        Rank features by their importance in predicting phenotypes using a trained DEM model.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        path_tmp_dir = os.path.join(output_dir, '.tmp')
        if not os.path.exists(path_tmp_dir):
            os.makedirs(path_tmp_dir, exist_ok=True)
        
        # Get original prediction loss

        # Shuffle features and get shuffled prediction loss
        importance_scores = []
        feat_names_all = []
        for i in range(len(omics_paths)):
            # n_features = pd.read_csv(omics_paths[i], index_col=0).shape[1]
            feat_names = pd.read_csv(omics_paths[i], index_col=0).columns
            feat_names_all.append(feat_names)
            n_features = len(feat_names)
            for j in range(n_features):
                importance_scores_tmp = []
                for random_state in random_states:
                    shuffled_omics_paths, shuffled_omics_path = self.shuffle_features(omics_paths, result_dir, i, j, random_state)
                    data_module_shuffled = DEMLTNDataModule(
                        batch_size=self.batch_size,
                        trait_name=trait_name,
                        n_label_classes=n_label_class,
                        path_label_tst=pheno_path,
                        paths_omics_tst=shuffled_omics_paths,
                    )
                    data_module_shuffled.setup()
                    os.remove(shuffled_omics_path)
                    trainer_shuffled = ltn.Trainer(default_root_dir=result_dir, logger=False)
                    losses_shuffled = trainer_shuffled.test(self.dem_model, datamodule=data_module_shuffled)
                    loss_shuffled = np.mean(np.concatenate(losses_shuffled, axis=0))
                    importance_scores_tmp.append(loss_orig - loss_shuffled)

                importance_scores.append(np.mean(importance_scores_tmp))

        # Rank features by their importance scores
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
        random_str = random_string()
        filename_suffix = f'_{time_str}_rs{random_str}.csv'

        feat_importance = pd.DataFrame(importance_scores, index=feat_names_all, columns=['importance_score'])
        feat_importance.to_csv(os.path.join(result_dir, 'feature_importance' + filename_suffix))
        feat_importance_ranked = feat_importance.sort_values(by='importance_score', ascending=False)
        feat_importance_ranked.to_csv(os.path.join(result_dir, 'feature_importance_ranked' + filename_suffix))
    
    def shuffle_features(
            self,
            # omics_paths: list[str],
            result_dir: str,
            which_omics2shuffle: int,
            which_feature2shuffle: int,
            random_state: int = 47,
        ):
        """
        Shuffle one feature in one omics data and save the shuffled data.
        """
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        tmp_omics = pd.read_csv(omics_paths[which_omics2shuffle], index_col=0)

        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Shuffle one feature
        tmp_omics.iloc[:, which_feature2shuffle] = np.random.permutation(tmp_omics.iloc[:, which_feature2shuffle])
        # Save shuffled omics data
        time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
        random_str = random_string()
        shuffled_omics_path = os.path.join(result_dir, f'shuffled_omics_{which_omics2shuffle}_{which_feature2shuffle}_{time_str}_rs{random_str}.csv')
        tmp_omics.to_csv(shuffled_omics_path)

        shuffled_omics_paths = omics_paths.copy()
        shuffled_omics_paths[which_omics2shuffle] = shuffled_omics_path
        return shuffled_omics_paths, shuffled_omics_path


def rank_feat(
        path_model_i: str,
        paths_omics_i: list[str],
        path_pheno_i: str,
        trait_name: str,
        n_label_class: int,
        result_dir: str,
        random_seeds: list[int],
        batch_size: int,
    ):
    """
    Rank features.
    """
    forranking = DEMFeatureRanking(path_model_i, batch_size)
    forranking.rank_features(paths_omics_i, path_pheno_i, trait_name, n_label_class, result_dir, random_seeds)
