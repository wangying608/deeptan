#!/bin/bash

mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
# mypython=/home/wuch/miniforge3/envs/sc/bin/python
myscript=run_05_fit_tune.py
storedir=/mnt/hdd1/wuch
SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif

# seed=$1
# optdata=$2
# ntrial=$3
# njob=$4
# bsize=$5
# agd=$6
seed=42
optdata=bulk_exp
ntrial=20
njob=1
bsize=32
agd=1

path_ckpt=/mnt/hdd1/wuch/logs/GSE226097_Annotated_split_strata/seed_42/DeepTAN_20250316021630_lkDu8/best-model-epoch=0005-val_loss=0.0000.ckpt

dirlitdata=$storedir/optimized_data/$optdata/seed_$seed

dirlogs=$storedir/logs/$optdata/seed_$seed
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$mypython $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --em $path_ckpt
# singularity exec --nv -B $storedir:$storedir $SIF python $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --atune
