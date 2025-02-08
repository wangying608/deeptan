mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
seed=$1

# datadir=/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control_split
# ds1_trn="$datadir/nmic_g/split_seed_${seed}_0.parquet.npz"
# ds1_val="$datadir/split_seed_${seed}_1.parquet"
# ds1_tst="$datadir/split_seed_${seed}_2.parquet"
# label_df="/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/cell_type_annotations_one_hot.parquet"

dirlitdata="/home/wuch/prjs/git_nwafu/DeepTAN/src/optimized_data/seed_${seed}"

# Test:
# $mypython pipeline.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --bs 2 --lr 1e-5 --log_dir logs/seed_$seed
$mypython pipeline.py --litdata $dirlitdata --bs 2 --lr 1e-3 --log_dir logs/seed_$seed
# $mypython pipeline.py --litdata $dirlitdata --bs 2 --lr 1e-4 --log_dir logs/seed_$seed
# $mypython pipeline.py --litdata $dirlitdata --bs 1 --lr 1e-5 --log_dir logs/seed_$seed
