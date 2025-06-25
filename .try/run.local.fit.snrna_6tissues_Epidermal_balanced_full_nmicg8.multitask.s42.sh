#!/bin/bash

DEEPTAN_HOME=/mnt/hdd2/homext/wuch/xn2p
mypython=python
myscript=run_05_fit_tune.py

optdata=snrna_6tissues_Epidermal.balanced.full.nmicg8
folder=seed_42
task_name=multitask
ntrial=20
njob=1
bs=16
agd=2
ck=2048
devices="[0]"

dirlitdata=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlogs=$DEEPTAN_HOME/run/logs/$optdata/$folder/$task_name
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOTAL_VRAM=32

$mypython $myscript --data $dirlitdata --bs $bs --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --ck $ck --dev $devices --atune
