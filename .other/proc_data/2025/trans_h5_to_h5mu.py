import sys
import muon as mu
import mudata


def trans_10xh5_to_h5mu(input_file, output_file):
    # Load the H5 file using muon
    mdata = mu.read_10x_h5(input_file)
    print(mdata)

    # Unique the data if necessary
    # mdata.obs_names_make_unique()
    mdata.var_names_make_unique()

    # Save the MuData object to an H5MU file
    print(mdata)
    mdata.write(output_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python trans_10xh5_to_h5mu.py <input_file> <output_file>")
        sys.exit(1)
    trans_10xh5_to_h5mu(sys.argv[1], sys.argv[2])
    print(f"Conversion complete. Output saved to {sys.argv[2]}")

    # path_rep1 = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control_rep1_filtered_feature_bc_matrix.h5"
    # path_rep2 = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control_rep2_filtered_feature_bc_matrix.h5"

    # mdata_rep1 = mu.read_10x_h5(path_rep1)
    # mdata_rep2 = mu.read_10x_h5(path_rep2)

    # print(mdata_rep1)
    # print(mdata_rep2)
