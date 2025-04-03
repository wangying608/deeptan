#!/bin/bash

DEEPTAN_HOME=/mnt/hdd2/homext/wuch/xn2p
SIF=$DEEPTAN_HOME/deeptan.sif

PACPATH=$DEEPTAN_HOME/DeepTAN/src
myscript=$DEEPTAN_HOME/DeepTAN/tests/run_05_fit_tune.py

optdata=sc_rna_annotated
folder=seed_42
ntrial=30
njob=1
bsize=8
agd=4

dirlitdata=$DEEPTAN_HOME/data/optimized_data/$optdata/$folder
dirlogs=$DEEPTAN_HOME/run/logs/$optdata/$folder
mkdir -p $dirlogs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# singularity exec --nv -B $DEEPTAN_HOME:$DEEPTAN_HOME $SIF pwd
# singularity exec --nv -B $DEEPTAN_HOME:$DEEPTAN_HOME $SIF ls
# singularity exec --nv -B $DEEPTAN_HOME:$DEEPTAN_HOME $SIF ls /usr/lib/python3/dist-packages
# singularity exec --nv -B $DEEPTAN_HOME:$DEEPTAN_HOME $SIF ls /usr/lib/python3.10
# singularity exec --nv -B $DEEPTAN_HOME:$DEEPTAN_HOME $SIF python -m pip list
# singularity exec --nv -B $DEEPTAN_HOME:$DEEPTAN_HOME $SIF python -c "import sys; print(sys.path)"
singularity exec --nv -B $DEEPTAN_HOME:$DEEPTAN_HOME $SIF sh -c "export PYTHONPATH=$PACPATH && python $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --agb $agd --atune"
