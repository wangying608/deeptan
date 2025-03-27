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

optdata=bulk_exp_meth
folder=seed_42_nmic_g_mincv2.0_minmi0.65
ntrial=20
njob=1
bsize=2
agd=16

path_ckpt=$DEEPTAN_HOME/logs/GSE226097_Annotated_split_strata/seed_42/DeepTAN_20250318162452_U0Jku/best-model-epoch=0006-val_loss=0.0000.ckpt

dirlitdata=$DEEPTAN_HOME/optimized_data/$optdata/$folder
dirlogs=$DEEPTAN_HOME/logs/$optdata/$folder
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

singularity exec --nv -B $DEEPTAN_HOME:$DEEPTAN_HOME $SIF python $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --em $path_ckpt --ir
