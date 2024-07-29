f_gff=/mnt/bank/ref/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff
f_gff_dict=/mnt/bank/ref/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff.bin.gz

f_vcf=/mnt/bank/1001/gmi_mpi/std_merged_group_vcf/filtered.vcf
f_pkl=/mnt/bank/1001/gmi_mpi/std_merged_group_vcf/filtered.vcf.pkl

pregv gff2bin -g $f_gff -o $f_gff_dict
pregv vcf2enc -v $f_vcf -d $f_gff_dict -o $f_pkl
