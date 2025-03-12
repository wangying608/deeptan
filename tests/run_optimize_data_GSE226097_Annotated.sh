mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python

datadir=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scRNA/GSE226097_Annotated_split_strata
label_df=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scRNA/GSE226097_Annotated_celltypes_onehot.parquet
outputdir=/mnt/hdd1/wuch/optimized_data/GSE226097_Annotated_split_strata

# seed=$1
for seed in $(seq 42 46)
do

ds1_trn="$datadir/nmic_g/split_${seed}_0.parquet.npz"
ds1_val="$datadir/split_${seed}_1.parquet"
ds1_tst="$datadir/split_${seed}_2.parquet"

echo "Running optimization for seed $seed"
$mypython run_04_optimize_data.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --bs 2 --output_dir $outputdir/seed_$seed

done
