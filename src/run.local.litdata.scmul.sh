mypython=python
DATA_HOME=/mnt/hdd2/homext/wuch/xn2p/data

nmic_g=nmic_g
suffix_trn=mi035_top2000
datadir=$DATA_HOME/raw_df/scMultiome/Ath_scMultiome_WT_split_strata
label_df=$datadir/celltypes_onehot.parquet

for seed in $(seq 42 46)
do

specify_obs=${datadir}/split_${seed}_0.parquet
ds1_trn=${datadir}/${nmic_g}/split_${seed}_0.npz
ds1_val=${datadir}/split_${seed}_1.parquet
ds1_tst=${datadir}/split_${seed}_2.parquet
outputdir=$DATA_HOME/optimized_data/scmul_${suffix_trn}/seed_$seed

echo "Running optimization for seed $seed"
$mypython run_04_optimize_data.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --output_dir $outputdir --in_obs $specify_obs

done
