from utils import read_pkl_gv


path_gtype_pkl = "/mnt/bank/1001/gmi_mpi/std_merged_group_vcf/filtered.vcf.pkl.gz"
snp_data_dict = read_pkl_gv(path_gtype_pkl)
print(snp_data_dict.keys())
print(snp_data_dict['gt_mat'].shape)
