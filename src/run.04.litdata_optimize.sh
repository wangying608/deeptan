mypython=python
DATA_HOME=/mnt/hdd2/homext/xn2p/data
nmic_g=nmicg8
litdata_name=ath_pretrain.full.${nmic_g}
datadir=$DATA_HOME/raw_df/snRNA_GSE226097/ath_pretrain.split_full
label_df=$DATA_HOME/raw_df/snRNA_GSE226097/ath_pretrain.split_full/celltypes_onehot.parquet

# export OMP_NUM_THREADS=1
for seed in $(seq 42 46)
do
specify_obs=${datadir}/${nmic_g}/split_${seed}_0.parquet
ds1_trn=${datadir}/${nmic_g}/split_${seed}_0.npz
ds1_val=${datadir}/split_${seed}_1.parquet
ds1_tst=${datadir}/split_${seed}_2.parquet
outputdir=$DATA_HOME/optimized_new/${litdata_name}/seed_$seed

echo "Running optimization for seed $seed"
$mypython run_04_litdata.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --output_dir $outputdir --in_obs $specify_obs --n_workers 20

done
