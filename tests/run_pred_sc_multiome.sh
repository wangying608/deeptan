#!/bin/bash

mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
# SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif
# mypython=/home/wuch/miniforge3/envs/sc/bin/python
myscript=run_06_predict.py

storedir=/mnt/hdd2/homext/wuch/xn2p

model_path=$storedir/run/logs/GSE235510_WT_strata/seed_42/DeepTAN_20250314171648_lzA6T/best-model-epoch=0018-val_loss=0.0000.ckpt
litdata_dir=$storedir/data/optimized_data/sc_multiome/seed_42/tst

$mypython $myscript --em $model_path --data $litdata_dir

# OUTPUT:
# dict_keys(['g_embedding', 'node_recon', 'node_recon_all', 'labels'])
# Key: g_embedding, Shape: (564, 256)
# Key: node_recon, Shape: (564, 5064, 128)
# Key: node_recon_all, Shape: (564, 5064, 1)
# Key: labels, Shape: (564, 16)
# Saving results to /mnt/hdd2/homext/wuch/xn2p/run/logs/GSE235510_WT_strata/seed_42/predicted_DeepTAN_20250314171648_lzA6T/seed_42_tst_numpy.pkl
