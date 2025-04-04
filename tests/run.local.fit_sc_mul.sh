#!/bin/bash

DEEPTAN_HOME=/mnt/hdd2/homext/wuch/xn2p
SIF=$DEEPTAN_HOME/deeptan.sif
myscript=run_05_fit_tune.py

optdata=sc_multiome_minmi0.35_top2000
folder=seed_42
ntrial=20
njob=1
bs=16
agd=2
ck=256

dirlitdata_t=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlitdata_v=/mnt/litdata

dirlogs_t=$DEEPTAN_HOME/run/logs
dirlogs_v=/mnt/litlogs
dirlogs=$dirlogs_v/$optdata/$folder

singularity exec \
--bind $dirlitdata_t:$dirlitdata_v,$dirlogs_t:$dirlogs_v \
--nv $SIF sh -c "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && mkdir -p $dirlogs && python $myscript --data $dirlitdata_v --bs $bs --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --ck $ck"
