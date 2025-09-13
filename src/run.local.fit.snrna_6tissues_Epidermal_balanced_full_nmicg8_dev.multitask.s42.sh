#!/bin/bash

DEEPTAN_HOME=/mnt/hdd2/homext/wuch/xn2p
mypython=python
myscript=run_05_fit_tune.py

optdata=snrna_6tissues_Epidermal.balanced.full.nmicg8_dev
folder=seed_42
task_name=multitask
ntrial=20
njob=1
bs=1
agd=32
ck=32
devices="[0]"

dirlitdata=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlogs=$DEEPTAN_HOME/run/logs/$optdata/$folder/$task_name
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOTAL_VRAM=32

# Additional memory optimization flags
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCHINDUCTOR_FREEZING=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True,garbage_collection_threshold:0.6"

$mypython $myscript --data $dirlitdata --bs $bs --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --ck $ck --dev $devices --nog # --atune
