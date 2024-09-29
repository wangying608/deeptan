py=/home/wuch/miniforge3/envs/xn/bin/python
run_litdata=.run_litdata.py


for xo in $(seq 0 9)
do
    for xi in $(seq 0 1)
    do
        $py $run_litdata $xo $xi &
        sleep 2
    done
    wait

    for xi in $(seq 2 3)
    do
        $py $run_litdata $xo $xi &
        sleep 2
    done
    wait

    $py $run_litdata $xo 4
done
