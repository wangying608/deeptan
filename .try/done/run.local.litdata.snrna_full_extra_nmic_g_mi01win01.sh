mypython=python
DATA_HOME=/mnt/hdd2/homext/wuch/xn2p/data

nmic_g=nmic_g_mi01win01
suffix_trn=full_extra
datadir_trn=$DATA_HOME/raw_df/snRNA/ath_snrna_balanced_flower_seedling_rosette_split_${suffix_trn}
datadir_valtst=$DATA_HOME/raw_df/snRNA/ath_snrna_balanced_flower_seedling_rosette_split_${suffix_trn}
label_df=$datadir_trn/celltypes_onehot.parquet

for seed in $(seq 42 46)
do

specify_obs=${datadir_trn}/split_${seed}_0.parquet
ds1_trn=${datadir_valtst}/${nmic_g}/split_${seed}_0.npz
ds1_val=${datadir_valtst}/split_${seed}_1.parquet
ds1_tst=${datadir_valtst}/split_${seed}_2.parquet
outputdir=$DATA_HOME/optimized_data/snrna_${suffix_trn}_${nmic_g}/seed_$seed

echo "Running optimization for seed $seed"
$mypython run_04_optimize_data.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --output_dir $outputdir --in_obs $specify_obs

done
