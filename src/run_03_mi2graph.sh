#!/bin/bash
myhome=/storage/public/home/2022051346
log_err=$myhome/mylogs/error.%J
log_out=$myhome/mylogs/output.%J
exebin=$myhome/mi2graph

xi=$myhome/prj/deeptan/splits/$1
xo=$myhome/prj/deeptan/splits/nmic_g/$1

jsub -n $2 -e $log_err -o $log_out -J mic_$1 "$exebin -i $xi -o $xo -t $2 --threcv 0.05 --thremi 0.05 --minwin 0.01"
