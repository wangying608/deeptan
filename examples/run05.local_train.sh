#!/bin/bash

DATA_HOME=/path/to/DATA_HOME 
mypython=python
myscript=run_05_fit_tune.py

optdata=ath_pretrain.full.nmicg
task_name=multitask
ntrial=20
njob=1
bs=16
agd=1
ck=32
devices="[0]"

dirlitdata=$DATA_HOME/data/optimized_data/$optdata # Directory path for storing optimized litdata
dirlogs=$DATA_HOME/run/logs/$optdata/$task_name  # Directory path for storing log files
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOTAL_VRAM=32

# Additional memory optimization flags
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCHINDUCTOR_FREEZING=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,expandable_segments:True,garbage_collection_threshold:0.6"

$mypython $myscript --data $dirlitdata --bs $bs --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --ck $ck --dev $devices --nog # Add --atune for Optuna hyperparameter tuning; Add --ir for phenotype regression
