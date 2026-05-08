mypython=python
DATA_HOME=/path/to/DATA_HOME 
nmic_g=nmicg
litdata_name=ath_pretrain.full.${nmic_g}  # You can name it according to your dataset name to store the generated litdata
datadir=$DATA_HOME/xxx/xxx/ath_pretrain.split_full  # Directory containing data files
label_df=$DATA_HOME/xxx/xxx/ath_pretrain.split_full/celltypes_onehot.parquet  # Path to your label parquet file (one-hot encoded cell types)

# export OMP_NUM_THREADS=1
specify_obs=${datadir}/${nmic_g}/split_${seed}_0.parquet  #  # Path to your training target observation parquet file
ds1_trn=${datadir}/${nmic_g}/split_42_0.npz  # Path to your training guide graph npz file
ds1_val=${datadir}/split_42_1.parquet  # Path to your validation expression parquet file
ds1_tst=${datadir}/split_42_2.parquet  # Path to your test expression parquet file
outputdir=$DATA_HOME/optimized_new/${litdata_name}  # Path to save the generated litdata

echo "Running optimization"
$mypython run_04_litdata.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --output_dir $outputdir --in_obs $specify_obs --n_workers 10
