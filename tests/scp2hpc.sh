DEEPTAN_HPC=2022051346@172.16.115.106:/storage/public/home/2022051346/prj/deeptan
DEEPTAN_MALAB13=/mnt/hdd2/homext/wuch/xn2p

LITDATA_MALAB13=$DEEPTAN_MALAB13/data/optimized_data
LITDATA_HPC=$DEEPTAN_HPC/data/optimized_data

scp -r $LITDATA_MALAB13/sc_rna_annotated.zip $LITDATA_HPC/
scp -r $LITDATA_MALAB13/bulk_exp_meth.zip $LITDATA_HPC/
scp -r $LITDATA_MALAB13/sc_multiome.zip $LITDATA_HPC/
