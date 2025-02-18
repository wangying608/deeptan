mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
seed=$1

# dirlitdata="/mnt/hdd1/wuch/optimized_data/GSE155304_SRP273996/seed_${seed}"
dirlitdata=$2
logdir=/mnt/hdd1/wuch/logs/GSE155304_SRP273996
onehotclass=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scRNA/SRP273996_celltypes_onehot.parquet

$mypython run_fit.py --litdata $dirlitdata --bs 4 --lr 5e-4 --log_dir $logdir/seed_$seed --heads_node_emb 2 --heads_pooling 2 --acc_grad_batch 16 --onehot_class $onehotclass
