mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
datadir=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control_split
seed=47

ds1_trn="$datadir/nmic_g/split_seed_${seed}_0.parquet.npz"
ds1_val="$datadir/split_seed_${seed}_1.parquet"
ds1_tst="$datadir/split_seed_${seed}_2.parquet"

$mypython pipeline.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --bs 2 --lr 1e-5 --log_dir logs/seed_$seed
