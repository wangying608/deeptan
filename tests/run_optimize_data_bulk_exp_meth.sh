mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
DATA_HOME=/mnt/hdd2/homext/wuch/xn2p/data

datadir=$DATA_HOME/raw_df/bulk/exp_meth_split
label_df=$DATA_HOME/raw_df/bulk/exp_meth.pheno_ft16_log1p.parquet

# seed=42
minmi=0.72
nmic_g=nmic_g_mincv2.0_minmi0.6

for seed in $(seq 42 46)
do

ds1_trn="$datadir/$nmic_g/split_${seed}_0.parquet.npz"
ds1_val="$datadir/split_${seed}_1.parquet"
ds1_tst="$datadir/split_${seed}_2.parquet"
outputdir=$DATA_HOME/optimized_data/bulk_exp_meth_${nmic_g}_${minmi}_log1p_ft16/seed_${seed}

echo "Running optimization for seed $seed $nmic_g $minmi"
$mypython run_04_optimize_data.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --output_dir $outputdir --thre_mi $minmi
# --n_workers 1

done
