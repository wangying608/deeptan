f_i=/mnt/bank/scPlantDB/ath/mic_g_init/randpart_1007.parquet
f_o=/mnt/bank/scPlantDB/ath/mic_g_init/.mic.randpart_1007.parquet
stackcollapse=/home/wuch/.local/FlameGraph/stackcollapse-perf.pl
flamegraph=/home/wuch/.local/FlameGraph/flamegraph.pl

# perf record --call-graph=dwarf mi2graphdev -i $f_i -o $f_o -s
# perf report --stdio
perf record -g mi2graphdev -i $f_i -o $f_o
perf script | $stackcollapse > out.perf_folded
$flamegraph out.perf_folded > perf.svg

rm perf.data
rm out.perf_folded
