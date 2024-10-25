import os
import sys
from frn.utils.data_ncv import OptimizeLitdataNCV


which_o = int(sys.argv[1])
which_i = int(sys.argv[2])


if __name__ == "__main__":
    k_outer = 10
    k_inner = 5
    dir_home = "/mnt/hdd2/homext/wuch/xn2p"
    output_dir = os.path.join(dir_home, "data", "litdata", "s2g", "ath", "ft16")
    path_gt = os.path.join("/mnt/bank/1001/gmi_mpi/std_merged_group_vcf/filtered.vcf.pkl.gz")
    path_labels = os.path.join("/mnt/bank/1001/_suit01_Phenotypes+SNP/tmp_Ath_label_data/original_FT16_except_external_val.csv")

    _opt = OptimizeLitdataNCV(
        paths_omics={"gv": path_gt},
        path_label=path_labels,
        output_dir=output_dir,
        k_outer=k_outer,
        k_inner=k_inner,
        which_outer_inner=[which_o, which_i],
        col2use_in_labels=["FT16"],
    )
    _opt.run_optimization()


# k_outer = 10
# k_inner = 5


# if __name__ == '__main__':
#     path_gt = '/mnt/bank/1001/gmi_mpi/std_merged_group_vcf/filtered.vcf.pkl.gz'

#     # For FT16 NCV
#     output_dir = '/mnt/hdd2/homext/wuch/xn2p/data/_optimized/SNP_zsLabel_FT16_except_external_val'
#     path_label = '/mnt/bank/1001/_suit01_Phenotypes+SNP/tmp_Ath_label_data/original_FT16_except_external_val.csv'
#     traits_name = 'FT16'
#     snp_data_opt_ncv(
#         output_dir = output_dir,
#         k_outer = k_outer,
#         k_inner = k_inner,
#         path_gtype_pkl = path_gt,
#         path_label = path_label,
#         col2use = [traits_name],
#         std_labels = True,
#         fragment_elem_ids=None,
#         compression="zstd",
#         n_workers=3,
#     )

    # # For RL NCV
    # output_dir = '/mnt/hdd2/homext/wuch/xn2p/data/_optimized/SNP_originLabel_RL_except_external_val'
    # path_label = '/mnt/bank/1001/_suit01_Phenotypes+SNP/tmp_Ath_label_data/original_RL_except_external_val.csv'
    # traits_name = 'RL'
    # snp_data_opt_ncv(
    #     output_dir = output_dir,
    #     k_outer = k_outer,
    #     k_inner = k_inner,
    #     path_gtype_pkl = path_gt,
    #     path_label = path_label,
    #     col2use = [traits_name],
    #     standardize_label = True,
    # )

'''
    # For FT16 external validation
    output_dir = '/mnt/bank/1001/_suit01_Phenotypes+SNP/optimized_onehotSNP_originLabel_FT16_external_val'
    path_label = 
    traits_name = 'FT16'
    output_dim = 1
    snp_data_opt_external(
        output_dir = output_dir,
        path_gtype_pkl = path_gt,
        path_label = path_label,
        traits_name = traits_name,
        output_dim = output_dim,
    )

    # For FT16 external validation
    output_dir = '/mnt/bank/1001/_suit01_Phenotypes+SNP/optimized_onehotSNP_originLabel_RL_external_val'
    path_label = 
    traits_name = 'RL'
    output_dim = 1
    snp_data_opt_external(
        output_dir = output_dir,
        path_gtype_pkl = path_gt,
        path_label = path_label,
        traits_name = traits_name,
        output_dim = output_dim,
    )
'''
