#!/bin/bash
myhome=/storage/public/home/2022051346
log_err=$myhome/mylogs/error.%J
log_out=$myhome/mylogs/output.%J

mypython=$myhome/prj/deeptan/.venv/bin/python
myscript=pipeline.py

# seed=$1
for seed in $(seq 42 51)
do

dirlitdata=$myhome/prj/deeptan/optimized_data/seed_${seed}

jsub -q gpu -n 4 -gpgpu "1 mig=4" -e $log_err -o $log_out -J deeptan_$seed "$mypython $myscript --litdata $dirlitdata --bs 2 --lr 1e-3 --log_dir logs/seed_${seed} --heads_node_emb 2"
sleep 8

done
