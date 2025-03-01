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
agd=$6
ck=$7

# for seed in $(seq 42 51)
# do

dirlitdata=$myhome/prj/deeptan/optimized_data/$optdata/seed_$seed
dirlogs=$myhome/prj/deeptan/logs/$optdata/seed_$seed
mkdir -p $dirlogs

jsub -q gpu -n 5 -gpgpu "1 mig=4" -e $log_err -o $log_out -J deeptan_$2+$1 "$mypython $myscript --litdata $dirlitdata --bs $bsize --log_dir $dirlogs --ntrials $ntrial --njobs $njob --chunk_size $ck --acc_grad_batch $agd"

# sleep 8
# done
