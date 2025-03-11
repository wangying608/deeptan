#!/bin/bash
#JSUB -J deeptan
#JSUB -q gpu
#JSUB -n 4
#JSUB -gpgpu '1 mig=4'
#JSUB -o log_out.%J
#JSUB -e log_err.%J

module purge
module load singularity-4.2.1

MY_HOME=/storage/public/home/2022051346

DEEPTAN_HOME=$MY_HOME/prj/deeptan

myscript=run_05_fit_tune.py

# seed=$1
# optdata=$2
# ntrial=$3
# njob=$4
# bsize=$5
# agd=$6
seed=42
optdata=GSE235510_WT_strata
ntrial=30
njob=1
bsize=8
agd=8

dirlitdata=$DEEPTAN_HOME/optimized_data/$optdata/seed_$seed
dirlogs=$DEEPTAN_HOME/logs/$optdata/seed_$seed
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

singularity exec --nv $DEEPTAN_HOME/deeptan.sif python $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd
