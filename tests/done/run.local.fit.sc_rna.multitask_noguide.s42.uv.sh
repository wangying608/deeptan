#!/bin/bash

DEEPTAN_HOME=/mnt/hdd2/homext/wuch/xn2p
mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
myscript=run_05_fit_tune.py

optdata=sc_rna_annotated_minmi0.0_top2000
folder=seed_42
task_name=multitask_noguide
ntrial=20
njob=1
bs=16
agd=2
ck=384

dirlitdata=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlogs=$DEEPTAN_HOME/run/logs/$optdata/$folder/$task_name
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$mypython $myscript --data $dirlitdata --bs $bs --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --ck $ck --nog
