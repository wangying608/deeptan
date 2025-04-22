#!/bin/bash

DEEPTAN_HOME=/mnt/hdd2/homext/wuch/xn2p
mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
myscript=run_05_fit_tune.py

optdata=sc_multiome_minmi0.35_top2000
folder=seed_43
task_name=focus_recon
ntrial=20
njob=1
bs=16
agd=2
ck=384

dirlitdata=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlogs=$DEEPTAN_HOME/run/logs/$optdata/$folder/$task_name
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$mypython $myscript --data $dirlitdata --bs $bs --ck $ck --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --focus recon # --atune
