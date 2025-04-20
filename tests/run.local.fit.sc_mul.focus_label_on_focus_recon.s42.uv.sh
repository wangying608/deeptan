#!/bin/bash

DEEPTAN_HOME=/mnt/hdd2/homext/wuch/xn2p
mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
myscript=run_05_fit_tune.py

optdata=sc_multiome_minmi0.35_top2000
folder=seed_42
task_name=focus_label_on_focus_recon
ntrial=20
njob=1
bs=16
agd=2
ck=384

path_ckpt=$DEEPTAN_HOME/run/logs/sc_multiome_minmi0.35_top2000/seed_42/focus_recon/DeepTAN_20250405143337_g2qQL/best-model-epoch=0023-val_loss=0.0000.ckpt

dirlitdata=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlogs=$DEEPTAN_HOME/run/logs/$optdata/$folder/$task_name
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$mypython $myscript --data $dirlitdata --bs $bs --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --ck $ck --focus label --em $path_ckpt
