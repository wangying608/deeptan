#!/bin/bash
DEEPTAN_HOME=/mnt/hdd2/homext/wuch/xn2p
mypython=/home/wuch/miniforge3/envs/pt28/bin/python
myscript=run_07_perturb.py

optdata=bulk_exp_meth
folder=seed_45
path_ckpt=${DEEPTAN_HOME}/run/logs/${optdata}/${folder}/multitask/DeepTAN_20250507213559_omqYu/trial_0/best-model-epoch=0049-val_loss=0.0000.ckpt

dirlitdata=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
diroutputs=$DEEPTAN_HOME/run/perturb/$optdata
mkdir -p $diroutputs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOTAL_VRAM=32

# for _split in trn val tst
# do
#     path_data_=$dirlitdata/$_split
#     path_output=$diroutputs/${folder}+${_split}
#     $mypython $myscript --em $path_ckpt --litdata $path_data_ --output $path_output
# done

_split=tst
path_data_=$dirlitdata/$_split
path_output=$diroutputs/${folder}+${_split}
$mypython $myscript --em $path_ckpt --litdata $path_data_ --output $path_output
