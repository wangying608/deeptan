#!/bin/bash

mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
# SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif
# mypython=/home/wuch/miniforge3/envs/sc/bin/python
myscript=run_06_predict.py

storedir=/mnt/hdd2/homext/wuch/xn2p

model_path=$storedir/run/logs/GSE226097_Annotated_split_strata/seed_42/DeepTAN_20250318162452_U0Jku/best-model-epoch=0006-val_loss=0.0000.ckpt
litdata_dir=$storedir/data/optimized_data/sc_rna_annotated/seed_42/tst

$mypython $myscript --em $model_path --data $litdata_dir

# OUTPUT:
# dict_keys(['g_embedding', 'node_recon', 'node_recon_all', 'labels'])
# Key: g_embedding, Shape: (679, 256)
# Key: node_recon, Shape: (679, 7338, 128)
# Key: node_recon_all, Shape: (679, 7338, 1)
# Key: labels, Shape: (679, 5)
# Saving results to /mnt/hdd2/homext/wuch/xn2p/run/logs/GSE226097_Annotated_split_strata/seed_42/predicted_DeepTAN_20250318162452_U0Jku/seed_42_tst_numpy.pkl
