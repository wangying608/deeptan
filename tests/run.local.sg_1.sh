for seed in $(seq 42 46)
do
mi2graph -i split_${seed}_0.parquet -o nmic_g_mi02win005/split_${seed}_0 --thremi 0.2 --minwin 0.05 -t 16
done
