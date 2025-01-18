#!/bin/bash
myhome=/storage/public/home/2022051346
log_err=$myhome/mylogs/error.%J
log_out=$myhome/mylogs/output.%J
exebin=$myhome/mi2graph

xi=$myhome/prj/xn/data_prep/ath/sc/$1.h5ad.parquet
xo=$myhome/prj/xn/data_prep/ath/sc/mic_g_init/$1

jsub -n $2 -e $log_err -o $log_out -J mic_$1 "$exebin -i $xi -o $xo"
