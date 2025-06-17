for seed in $(seq 42 46)
do
mi2graph -i split_${seed}_0.parquet -o nmicg1/split_${seed}_0 --threcv 0.05 --thremi 0.2 --minwin 0.01 --maxwin 0.2 --stepwin 0.02 --stepsli 0.01 -t 30
done
