import os
import sys
from frn.s2g_sparse.pipeline import SNP2GBTrainPipe#, SNP2GBTransPipe
from frn.utils.uni import CollectFitLog


list_ncv = [[int(sys.argv[1]), int(sys.argv[2])]]
work_dir_home = '/mnt/hdd2/homext/wuch/xn2p'
litdata_dir = os.path.join(work_dir_home, 'data/_optimized/SNP_zsLabel_FT16_except_external_val')
is_regression = True
# dense_layers_hidden_dims = [1024, 256]
log_dir = os.path.join(work_dir_home, 's2g_models')
n_jobs = 5
n_trials = 15


if __name__ == '__main__':    
    pipe_fit = SNP2GBTrainPipe(
        litdata_dir=litdata_dir,
        list_ncv=list_ncv,
        log_dir=log_dir,
        regression=is_regression,
        # dense_layers_hidden_dims=dense_layers_hidden_dims,
        n_jobs=n_jobs,
        n_trials=n_trials,
    )
    pipe_fit.train_pipeline()

    # Remove inferior models
    ckpt_collector = CollectFitLog(log_dir)
    ckpt_collector.remove_inferior_models()
