mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
myscript=run_05_fit_tune.py
storedir=/mnt/hdd1/wuch
SIF=/home/wuch/prjs/git_nwafu/DeepTAN/deeptan.sif

seed=$1
optdata=$2
ntrial=$3
njob=$4
bsize=$5
agd=$6
ck=$7

dirlitdata=$storedir/optimized_data/$optdata/seed_$seed

dirlogs=$storedir/logs/$optdata/seed_$seed
mkdir -p $dirlogs

$mypython $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --ck $ck --agb $agd --atune
# singularity exec --nv -B $storedir:$storedir $SIF python $myscript --data $dirlitdata --bs $bsize --logdir $dirlogs --nt $ntrial --nj $njob --ck $ck --agb $agd --atune
