#!/bin/bash
#JSUB -J deeptan_sc_multiome
#JSUB -q gpu
#JSUB -n 4
#JSUB -gpgpu '1 mig=4'
#JSUB -o log_out.%J
#JSUB -e log_err.%J

module purge
module load singularity-4.2.1

MY_HOME=/storage/public/home/2022051346

DEEPTAN_HOME=$MY_HOME/prj/deeptan
SIF=$DEEPTAN_HOME/deeptan.sif

myscript=run_05_fit_tune.py

# seed=$1
folder=$1
# optdata=$2
# ntrial=$3
# njob=$4
# bsize=$5
# agd=$6

optdata=sc_multiome
# folder=seed_42
ntrial=30
njob=1
bsize=8
agd=4

dirlitdata=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlogs=$DEEPTAN_HOME/run/logs/$optdata/$folder
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

singularity exec --nv -B $DEEPTAN_HOME:$DEEPTAN_HOME $SIF python $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --atune
