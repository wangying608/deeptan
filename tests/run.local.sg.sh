for seed in $(seq 42 46)
do
mi2graph -i split_${seed}_0.parquet -o nmic_g/split_${seed}_0 --thremi 0.1 --minwin 0.1
done
