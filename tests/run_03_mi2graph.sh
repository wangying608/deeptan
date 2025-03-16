#!/bin/bash
dataset_name=$1
nthreads=$2
minmi=$3

MY_HOME=/storage/public/home/2022051346
log_err=$MY_HOME/mylogs/error.%J
log_out=$MY_HOME/mylogs/output.%J
exebin=$MY_HOME/mi2graph

DEEPTAN_HOME=$MY_HOME/prj/deeptan

for seed in $(seq 42 46)
do
# seed=$3

fname=split_${seed}_0.parquet
xi=$DEEPTAN_HOME/raw_df/$dataset_name/$fname
xo=$DEEPTAN_HOME/raw_df/$dataset_name/nmic_g_thremi${minmi}/$fname

# jsub -n $nthreads -e $log_err -o $log_out -J mic_${dataset_name}_$seed "$exebin -i $xi -o $xo -t $nthreads --threcv 0.05 --thremi 0.05 --minwin 0.01"
jsub -n $nthreads -e $log_err -o $log_out -J mic_${dataset_name}_${seed}_thremi${minmi} "$exebin -i $xi -o $xo -t $nthreads --threcv 0.1 --thremi $minmi --minwin 0.05"

sleep 3
done
