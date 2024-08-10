f_gff=/mnt/bank/ref/zea_mays/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3
f_gff_dict=/mnt/bank/ref/zea_mays/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3.bin.gz

f_vcf=/mnt/bank/CropGS-Hub/maize_1404/vcf/maize_385_filtered_maf0.05_B73v5.vcf
f_pkl=/mnt/bank/CropGS-Hub/maize_1404/vcf/maize_385_filtered_maf0.05_B73v5.vcf.pkl.gz

pregv gff2bin -g $f_gff -o $f_gff_dict
pregv vcf2enc -v $f_vcf -d $f_gff_dict -o $f_pkl
