#!/bin/bash

mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
# SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif
# mypython=/home/wuch/miniforge3/envs/sc/bin/python
storedir=/mnt/hdd2/homext/wuch/xn2p
myscript=run_05_fit_tune.py

# optdata=bulk_exp_meth_nmic_g_mincv2.0_minmi0.6_0.72_log1p_ft16
optdata=bulk_exp_meth
folder=seed_42
ntrial=20
njob=1
bsize=16
agd=2

path_ckpt=$storedir/run/logs/GSE226097_Annotated_split_strata/seed_42/DeepTAN_20250318162452_U0Jku/best-model-epoch=0006-val_loss=0.0000.ckpt

dirlitdata=$storedir/data/optimized_data/$optdata/$folder
dirlogs=$storedir/run/logs/$optdata/$folder
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$mypython $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --em $path_ckpt --focus label --ir
