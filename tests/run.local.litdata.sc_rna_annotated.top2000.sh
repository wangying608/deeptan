mypython=/home/wuch/prjs/git_nwafu/DeepTAN/.venv/bin/python
DATA_HOME=/mnt/hdd2/homext/wuch/xn2p/data

datadir=$DATA_HOME/raw_df/scRNA/GSE226097_Annotated_split_strata
label_df=$DATA_HOME/raw_df/scRNA/GSE226097_Annotated_celltypes_onehot.parquet

# seed=$1
minmi=0.0
top_x=top2000

for seed in $(seq 42 46)
do

specify_features=${datadir}/$top_x/top2000_train_${seed}_rna.csv
ds1_trn="$datadir/nmic_g/split_${seed}_0.parquet.npz"
ds1_val="$datadir/split_${seed}_1.parquet"
ds1_tst="$datadir/split_${seed}_2.parquet"
outputdir=$DATA_HOME/optimized_data/sc_rna_annotated_minmi${minmi}_${top_x}/seed_$seed

echo "Running optimization for seed $seed"
$mypython run_04_optimize_data.py --trn_npz $ds1_trn --val_parquet $ds1_val --tst_parquet $ds1_tst --labels $label_df --output_dir $outputdir --thre_mi $minmi --in_feat $specify_features

done
