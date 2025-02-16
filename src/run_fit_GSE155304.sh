mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
seed=$1

# dirlitdata="/mnt/hdd1/wuch/optimized_data/GSE155304_SRP273996/seed_${seed}"
dirlitdata=$2
logdir=/mnt/hdd1/wuch/logs/GSE155304_SRP273996

$mypython run_fit.py --litdata $dirlitdata --bs 8 --lr 1e-4 --log_dir $logdir/seed_$seed --heads_node_emb 2 --heads_pooling 2
