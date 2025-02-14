mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
datadir=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control_split
label_df=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/cell_type_annotations_one_hot.parquet
outputdir=/mnt/hdd1/wuch/optimized_data

# seed=$1
for seed in $(seq 42 51)
do

ds1_trn="$datadir/nmic_g/split_seed_${seed}_0.parquet.npz"
ds1_val="$datadir/split_seed_${seed}_1.parquet"
ds1_tst="$datadir/split_seed_${seed}_2.parquet"

echo "Running optimization for seed $seed"
$mypython run_optimize_data.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --bs 2 --output_dir $outputdir/seed_$seed

done
