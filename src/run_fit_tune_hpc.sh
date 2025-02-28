#!/bin/bash
myhome=/storage/public/home/2022051346
log_err=$myhome/mylogs/error.%J
log_out=$myhome/mylogs/output.%J

mypython=$myhome/prj/deeptan/.venv/bin/python
myscript=run_05_fit_tune.py

seed=$1
optdata=$2
ntrial=$3
njob=$4
bsize=$5
ck=$6
# 8192 or 4096

# for seed in $(seq 42 51)
# do

dirlitdata=$myhome/prj/deeptan/optimized_data/$optdata/seed_$seed
dirlogs=$myhome/prj/deeptan/logs/$optdata/seed_$seed
mkdir -p $dirlogs

jsub -q gpu -n 5 -gpgpu "1 mig=5" -e $log_err -o $log_out -J deeptan_$2+$1 "$mypython $myscript --litdata $dirlitdata --bs $bsize --log_dir $dirlogs --ntrials $ntrial --njobs $njob --chunk_size $ck"

# sleep 8
# done
