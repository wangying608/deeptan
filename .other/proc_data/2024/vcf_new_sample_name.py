import pandas as pd


path_pheno = "/mnt/bank/CropGS-Hub/maize_8652/filtered/GSTP001_filtered_nonna_0.9_.pheno"
path_new_sample_name = "/mnt/bank/CropGS-Hub/maize_8652/GSTP001.bed/new_sample_name.txt"
path_correct_name = "/mnt/bank/CropGS-Hub/maize_8652/GSTP001.bed/correct_name.txt"

pheno = pd.read_table(path_pheno, sep='\t', header=0)
correct_name = pheno['LINE']

'''
correct name is x, name in vcf is x_x
'''
vcf_sample_name = pheno['LINE'] + '_' + pheno['LINE']

df_o = pd.concat([vcf_sample_name, correct_name], axis=1)
df_o.to_csv(path_new_sample_name, sep='\t', index=False, header=False)
correct_name.to_csv(path_correct_name, sep='\t', index=False, header=False)
