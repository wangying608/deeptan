f_gff=/mnt/bank/ref/zea_mays/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3
f_gff_dict=/mnt/bank/ref/zea_mays/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3.bin.gz

f_vcf=
f_pkl=

# pregv gff2bin -g $f_gff -o $f_gff_dict
pregv vcf2enc -v $f_vcf -d $f_gff_dict -o $f_pkl
