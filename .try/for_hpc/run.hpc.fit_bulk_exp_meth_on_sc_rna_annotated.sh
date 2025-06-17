#!/bin/bash
#JSUB -J deeptan
#JSUB -q gpu
#JSUB -n 1
#JSUB -gpgpu '1 mig=1'
#JSUB -o log_out.%J
#JSUB -e log_err.%J

module purge
module load singularity-4.2.1

MY_HOME=/storage/public/home/2022051346
DEEPTAN_HOME=$MY_HOME/prj/deeptan
SIF=$DEEPTAN_HOME/deeptan.sif
myscript=run_05_fit_tune.py

### !!! Editing
optdata=bulk_exp_meth
folder=seed_42_nmic_g_mincv2.0_minmi0.65
ntrial=20
njob=1
bs=2
agd=16
ck=512

dirlitdata_t=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlitdata_v=/mnt/litdata

dirlogs_t=$DEEPTAN_HOME/run/logs
dirlogs_v=/mnt/litlogs
dirlogs=$dirlogs_v/$optdata/$folder

path_ckpt=$dirlogs_v/sc_rna/seed_42/DeepTAN_20250318162452_U0Jku/best-model-epoch=0006-val_loss=0.0000.ckpt

singularity exec \
--bind $dirlitdata_t:$dirlitdata_v,$dirlogs_t:$dirlogs_v \
--nv $SIF sh -c "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && mkdir -p $dirlogs && python $myscript --data $dirlitdata_v --bs $bs --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --ck $ck --em $path_ckpt --ir --atune"
