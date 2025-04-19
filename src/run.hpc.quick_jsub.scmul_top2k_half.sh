# for rep in {1..20}; do
for seed in {42..46}; do
    jsub < "run.hpc.fit.sc_mul_half.s${seed}.sh"
    sleep 1
done
# done
