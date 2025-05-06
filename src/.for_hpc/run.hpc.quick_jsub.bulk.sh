for rep in {1..20}; do
    for seed in {43..46}; do
        jsub < "run.hpc.fit.bulk_exp_meth.multitask.s${seed}.sh"
        sleep 1
    done
done
