mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
myscript=run_05_fit_tune.py
storedir=/mnt/hdd1/wuch

seed=$1
optdata=$2
labelonehot=$3
ntrial=$4
njob=$5
bsize=$6
ck=$7
# 8192 or 4096

dirlitdata=$storedir/optimized_data/$optdata/seed_$seed
onehotclass=$storedir/optimized_data/$optdata/$labelonehot
dirlogs=$storedir/logs/$optdata/seed_$seed
mkdir -p $dirlogs

$mypython $myscript --litdata $dirlitdata --bs $bsize --log_dir $dirlogs --onehot_class $onehotclass --ntrials $ntrial --njobs $njob --chunk_size $ck
