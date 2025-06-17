#!/bin/bash

DEEPTAN_HOME=/mnt/hdd2/homext/wuch/xn2p
mypython=python
myscript=run_05_fit_tune.opt.py

optdata=sc_multiome_minmi0.35_top2000
folder=seed_42
task_name=multitask_opt
ntrial=20
njob=1
bs=32
agd=1
ck=512

dirlitdata=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlogs=$DEEPTAN_HOME/run/logs/$optdata/$folder/$task_name
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOTAL_VRAM=32

$mypython $myscript --data $dirlitdata --bs $bs --ck $ck --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --dev 1
