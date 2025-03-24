#!/bin/bash

mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
# SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif
# mypython=/home/wuch/miniforge3/envs/sc/bin/python
storedir=/mnt/hdd2/homext/wuch/xn2p
myscript=run_05_fit_tune.py

optdata=sc_multiome
folder=seed_42
ntrial=20
njob=1
bsize=4
agd=8

dirlitdata=$storedir/data/optimized_data/$optdata/$folder
dirlogs=$storedir/run/logs/$optdata/$folder
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$mypython $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd
