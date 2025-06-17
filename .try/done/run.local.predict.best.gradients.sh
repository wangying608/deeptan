mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python

for seed in $(seq 42 46)
do
$mypython run.local.predict.best.gradients.py $seed
done
