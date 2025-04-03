#!/bin/bash

mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
# SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif
# mypython=/home/wuch/miniforge3/envs/sc/bin/python
storedir=/mnt/hdd2/homext/wuch/xn2p
myscript=run_05_fit_tune.py

optdata=sc_rna_annotated
folder=seed_42
ntrial=20
njob=1
bsize=32
agd=1
lr=0.0001

path_ckpt=$storedir/run/logs/GSE235510_WT_strata/seed_42/DeepTAN_20250314171648_lzA6T/best-model-epoch=0018-val_loss=0.0000.ckpt

dirlitdata=$storedir/data/optimized_data/$optdata/$folder
dirlogs=$storedir/run/logs/$optdata/$folder
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$mypython $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --em $path_ckpt --lr $lr
# singularity exec --nv -B $storedir:$storedir $SIF which python
# singularity exec --nv -B $storedir:$storedir $SIF python -m pip list
# singularity exec --nv -B $storedir:$storedir $SIF python $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --atune
