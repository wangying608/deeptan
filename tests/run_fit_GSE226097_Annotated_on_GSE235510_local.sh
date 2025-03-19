#!/bin/bash

mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
# mypython=/home/wuch/miniforge3/envs/sc/bin/python
DATA_HOME=/mnt/hdd2/homext/wuch/xn2p

myscript=run_05_fit_tune.py
SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif

# seed=$1
# optdata=$2
# ntrial=$3
# njob=$4
# bsize=$5
# agd=$6
seed=42
optdata=GSE226097_Annotated_split_strata
ntrial=20
njob=1
bsize=32
agd=1
lr=0.0001

path_ckpt=$DATA_HOME/run/logs/GSE235510_WT_strata/seed_42/DeepTAN_20250314171648_lzA6T/best-model-epoch=0018-val_loss=0.0000.ckpt

dirlitdata=$DATA_HOME/data/optimized_data/$optdata/seed_$seed

dirlogs=$DATA_HOME/run/logs/$optdata/seed_$seed
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$mypython $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --em $path_ckpt --lr $lr
# singularity exec --nv -B $storedir:$storedir $SIF python $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --atune
