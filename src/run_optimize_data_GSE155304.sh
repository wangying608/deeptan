mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python

datadir=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scRNA/SRP273996.h5ad_split_resampled
label_df=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scRNA/SRP273996_celltypes_onehot.parquet
outputdir=/mnt/hdd1/wuch/optimized_data/GSE155304_SRP273996_resampled

seed=$1
# for seed in $(seq 42 51)
# do

ds1_trn="$datadir/nmic_g/split_${seed}_0.parquet.npz"
ds1_val="$datadir/split_${seed}_1.parquet"
ds1_tst="$datadir/split_${seed}_2.parquet"

echo "Running optimization for seed $seed"
$mypython run_optimize_data.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --bs 2 --output_dir $outputdir/seed_$seed

# done
