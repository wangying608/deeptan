#!/bin/bash
#JSUB -J deeptan_sc_rna_annotated
#JSUB -q gpu
#JSUB -n 1
#JSUB -gpgpu '1 mig=1'
#JSUB -o log_out.%J
#JSUB -e log_err.%J

module purge
module load singularity-4.2.1

MY_HOME=/storage/public/home/2022051346
DEEPTAN_HOME=$MY_HOME/prj/deeptan
SIF=$DEEPTAN_HOME/deeptan_20250408.sif
myscript=run_05_fit_tune.py

optdata=sc_rna_annotated_minmi0.1_top2000_quarter
folder=seed_43
task_name=multitask
ntrial=1
njob=1
bs=16
agd=2
ck=384

dirlitdata_t=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlitdata_v=/mnt/litdata

dirlogs_t=$DEEPTAN_HOME/run/logs
dirlogs_v=/mnt/litlogs
dirlogs=$dirlogs_v/$optdata/$folder/$task_name

singularity exec \
--bind $dirlitdata_t:$dirlitdata_v,$dirlogs_t:$dirlogs_v \
--nv $SIF sh -c "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && mkdir -p $dirlogs && python $myscript --data $dirlitdata_v --bs $bs --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --ck $ck"
