# for rep in {1..20}; do
for seed in {42..46}; do
    jsub < "run.hpc.fit.sc_rna_eighth.s${seed}.sh"
    sleep 1
done
# done
