for seed in $(seq 42 46)
do
mi2graph -i split_${seed}_0.parquet -o nmic_g_mi01win001/split_${seed}_0 --thremi 0.1 --minwin 0.01 -t 25
done
