#

ls 00rawdata | awk -F '.' '{print "  "$1":"}' > samplelist.txt
cat run.yaml samplelist.txt > run2.yaml
mv run2.yaml run.yaml
rm samplelist.txt

snakemake -s ~/Documents/atac_test/workflow/RNA-Seq.smk --configfile run.yaml -j3 --rerun-incomplete
