mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
seed=$1

dirlitdata=$2
logdir=/mnt/hdd1/wuch/logs/scmulti_GSE235510
onehotclass=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/Ath_scMultiome_WT_celltypes_onehot.parquet

$mypython run_05_fit_tune.py --litdata $dirlitdata --bs 8 --log_dir $logdir/seed_$seed --onehot_class $onehotclass
