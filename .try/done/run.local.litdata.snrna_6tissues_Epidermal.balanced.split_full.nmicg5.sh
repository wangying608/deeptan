mypython=python
DATA_HOME=/mnt/hdd2/homext/wuch/xn2p/data

nmic_g=nmicg5
litdata_name=snrna_6tissues_Epidermal.balanced.full.${nmic_g}
datadir=$DATA_HOME/raw_df/snRNA/ath_snrna_6tissues_Epidermal.balanced.split_full
label_df=$DATA_HOME/raw_df/snRNA/ath_snrna_6tissues_Epidermal.celltypes_onehot.parquet

# for seed in $(seq 42 46)
# do
seed=42

specify_obs=${datadir}/${nmic_g}/split_${seed}_0.parquet
ds1_trn=${datadir}/${nmic_g}/split_${seed}_0.npz
ds1_val=${datadir}/split_${seed}_1.parquet
ds1_tst=${datadir}/split_${seed}_2.parquet
outputdir=$DATA_HOME/optimized_data/${litdata_name}/seed_$seed

echo "Running optimization for seed $seed"
$mypython run_04_optimize_data.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --output_dir $outputdir --in_obs $specify_obs

# done
