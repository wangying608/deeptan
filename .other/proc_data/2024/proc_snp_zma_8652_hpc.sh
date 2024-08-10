f_gff_dict=Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3.bin.gz

f_vcf=converted_8652_Hybrid_nsn_8153_f_B73v5.vcf
f_pkl=converted_8652_Hybrid_nsn_8153_f_B73v5.vcf.pkl.gz

mypregv=/storage/public/home/2022051346/prj/xn/test/bin/pregv

jsub -q normal -n 60 -e /storage/public/home/2022051346/mylogs/error.%J -o /storage/public/home/2022051346/mylogs/output.%J -J pregv_zma8153 "$mypregv vcf2enc -v $f_vcf -d $f_gff_dict -o $f_pkl"
