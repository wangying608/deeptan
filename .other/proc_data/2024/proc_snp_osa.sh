f_gff=/mnt/bank/ref/Oryza_sativa/MSU_v.7.0/all.gff3
f_gff_dict=/mnt/bank/ref/Oryza_sativa/MSU_v.7.0/all.gff3.bin.gz

f_vcf=/mnt/bank/CropGS-Hub/rice_378/bed/filtered.vcf
f_pkl=/mnt/bank/CropGS-Hub/rice_378/bed/filtered.vcf.pkl.gz

pregv gff2bin -g $f_gff -o $f_gff_dict
pregv vcf2enc -v $f_vcf -d $f_gff_dict -o $f_pkl
