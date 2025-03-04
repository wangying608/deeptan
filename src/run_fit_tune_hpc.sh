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
# log_err=$myhome/mylogs/error.%J
# log_out=$myhome/mylogs/output.%J

DEEPTAN_HOME=$MY_HOME/prj/deeptan

myscript=run_05_fit_tune.py

seed=$1
optdata=$2
ntrial=$3
njob=$4
bsize=$5
agd=$6
ck=$7

dirlitdata=$DEEPTAN_HOME/optimized_data/$optdata/seed_$seed
dirlogs=$DEEPTAN_HOME/logs/$optdata/seed_$seed
mkdir -p $dirlogs

singularity exec --nv $DEEPTAN_HOME/deeptan.sif python $myscript --litdata $dirlitdata --bs $bsize --log_dir $dirlogs --ntrials $ntrial --njobs $njob --chunk_size $ck --acc_grad_batch $agd
