#!/bin/bash

mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
# SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif
# mypython=/home/wuch/miniforge3/envs/sc/bin/python
myscript=run_06_predict.py

storedir=/mnt/hdd2/homext/wuch/xn2p

model_path=$storedir/run/logs/bulk_exp_meth_nmic_g_mincv2.0_minmi0.6_0.72_log1p_ft16/seed_42/DeepTAN_20250321024731_NdCYE_bac/best-model-epoch=0015-val_loss=0.0000.ckpt
litdata_dir=$storedir/data/optimized_data/bulk_exp_meth_nmic_g_mincv2.0_minmi0.6_0.72_log1p_ft16/seed_42/tst

$mypython $myscript --em $model_path --data $litdata_dir
