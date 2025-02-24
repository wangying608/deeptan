mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
seed=$1

dirlitdata=$2
logdir=/mnt/hdd1/wuch/logs/scmulti_GSE235510
onehotclass=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/Ath_scMultiome_WT_celltypes_onehot.parquet

$mypython run_05_fit.py --litdata $dirlitdata --bs 8 --lr 1e-4 --log_dir $logdir/seed_$seed --acc_grad_batch 4 --onehot_class $onehotclass --output_dim_g_emb 512 --chunk_size 1024
