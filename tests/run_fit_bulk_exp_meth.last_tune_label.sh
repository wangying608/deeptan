#!/bin/bash

mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
# SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif
# mypython=/home/wuch/miniforge3/envs/sc/bin/python
storedir=/mnt/hdd2/homext/wuch/xn2p
myscript=run_05_fit_tune.py

data=bulk_exp_meth
folder=seed_42
ntrial=20
njob=1
bsize=16
agd=4
lr=0.00001

path_ckpt=$storedir/run/logs/bulk_exp_meth_nmic_g_mincv2.0_minmi0.6_0.72_log1p_ft16/seed_42/DeepTAN_20250321024731_NdCYE/best-model-epoch=0015-val_loss=0.0000.ckpt

dirlitdata=$storedir/data/optimized_data/$data/$folder
dirlogs=$storedir/run/logs/$data/$folder
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$mypython $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --em $path_ckpt --lr $lr --focus label --ir
