import os


if __name__ == "__main__":
    path_sh = "run_mic_g.sh"
    exec_bin = "mi2graph"
    dir_data_parquet = "/mnt/bank/scPlantDB/ath/mic_g_init"
    dir_output = os.path.join(dir_data_parquet, "results")

    os.makedirs(dir_output, exist_ok=True)
    files_ = [f for f in os.listdir(dir_data_parquet) if f.endswith(".parquet")]

    # Generate a shell script
    with open(path_sh, "w") as f:
        for file_ in files_:
            f_i = os.path.join(dir_data_parquet, file_)
            f_o = os.path.join(dir_output, os.path.splitext(file_)[0])
            f.write(f"{exec_bin} -i {f_i} -o {f_o}\n")
