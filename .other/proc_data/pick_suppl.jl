include("md_0_io.jl")
using CSV, DataFrames


function GSE_soft2ftp(path_soft::String)
    lines = my_read_table(path_soft, String, '=')
    whLineIsSuppl = findall(x -> x == "!Sample_supplementary_file_1 ", lines[:, 1])
    suppl_links = lines[whLineIsSuppl, 2]
    suppl_links .= map(x -> x[2:end], suppl_links)
    df_suppl = DataFrame(ftp=suppl_links)
    return df_suppl
end


# fpath1 = "/mnt/hdd2/data/1001/methylomes/GSE43857/GSE43857_family.soft"
# CSV.write("/mnt/hdd2/data/1001/methylomes/GSE43857/links_suppl.txt", GSE_soft2ftp(fpath1), header=false)

# fpath2 = "/mnt/hdd2/data/1001/methylomes/GSE54292/GSE54292_family.soft"
# CSV.write("/mnt/hdd2/data/1001/methylomes/GSE54292/links_suppl.txt", GSE_soft2ftp(fpath2), header=false)

# fpath3 = "/mnt/hdd2/data/data_xrn2p/origin/data_08_y3G6Qkd9td/GSE60143/GSE60143_family.soft"
# CSV.write("/mnt/hdd2/data/data_xrn2p/origin/data_08_y3G6Qkd9td/GSE60143/links_suppl.txt", GSE_soft2ftp(fpath3), header=false)

# fpath4 = "/mnt/hdd2/data/GSE155304/GSE155304_family.soft"
# CSV.write("/mnt/hdd2/data/GSE155304/links_suppl.txt", GSE_soft2ftp(fpath4), header=false)
