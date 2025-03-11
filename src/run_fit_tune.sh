#!/bin/bash

mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
# mypython=/home/wuch/miniforge3/envs/sc/bin/python

myscript=run_05_fit_tune.py
storedir=/mnt/hdd1/wuch
SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif

seed=$1
optdata=$2
ntrial=$3
njob=$4
bsize=$5
agd=$6

dirlitdata=$storedir/optimized_data/$optdata/seed_$seed

dirlogs=$storedir/logs/$optdata/seed_$seed
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$mypython $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd
# singularity exec --nv -B $storedir:$storedir $SIF python $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd
