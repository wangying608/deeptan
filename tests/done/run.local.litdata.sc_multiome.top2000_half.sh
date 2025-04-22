mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
DATA_HOME=/mnt/hdd2/homext/wuch/xn2p/data

suffix_trn=half
datadir_trn=$DATA_HOME/raw_df/scMultiome/Ath_scMultiome_WT_split_${suffix_trn}
datadir_valtst=$DATA_HOME/raw_df/scMultiome/Ath_scMultiome_WT_split_strata
label_df=$DATA_HOME/raw_df/scMultiome/Ath_scMultiome_WT_celltypes_onehot.parquet

# seed=$1
minmi=0.35
# minmi=0.1
top_x=top2000

for seed in $(seq 42 46)
do
specify_obs=${datadir_trn}/split_${seed}_0.parquet
specify_features=${datadir_valtst}/$top_x/feat_train${seed}.csv
ds1_trn=${datadir_valtst}/nmic_g/split_${seed}_0.parquet.npz
ds1_val=${datadir_valtst}/split_${seed}_1.parquet
ds1_tst=${datadir_valtst}/split_${seed}_2.parquet
outputdir=$DATA_HOME/optimized_data/sc_multiome_minmi${minmi}_${top_x}_${suffix_trn}/seed_$seed

echo "Running optimization for seed $seed"
$mypython run_04_optimize_data.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --output_dir $outputdir --thre_mi $minmi --in_feat $specify_features --in_obs $specify_obs

done
