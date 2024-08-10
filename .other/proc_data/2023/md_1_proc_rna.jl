# 
__precompile__(true)
using DataFrames, CSV
using Base.Threads: nthreads, @spawn, @threads
function tmapreduce(f, op, itr; tasks_per_thread::Int = 16, kwargs...)
    chunk_size = max(1, length(itr) ÷ (tasks_per_thread * nthreads()))
    tasks = map(Iterators.partition(itr, chunk_size)) do chunk
        @spawn mapreduce(f, op, chunk; kwargs...)
    end
    mapreduce(fetch, op, tasks; kwargs...)
end


#

@doc "Generate random tags"
function getRandomName(num::Int, path::String, pure::Bool=true, getbasename::Bool=true,
                       tagsExist::Vector{String}=Vector{String}(undef,0))
    #
    len_exist = length(tagsExist)
    tags = Vector{String}(undef, num)
    if getbasename
        for i in eachindex(tags)
            tags[i] = basename(tempname(dirname(path)))
        end
        if pure
            tags .= broadcast(x -> x[4:end], tags)
        end
        if length(unique(vcat(tagsExist, tags))) ≠ (num + len_exist)
            tags = getRandomName(num, path, pure, getbasename, tagsExist)
        end
    else
        for i in eachindex(tags)
            tags[i] = tempname(dirname(path))
        end
        tags_base = basename.(tags)
        if length(unique(vcat(basename.(tagsExist), tags_base))) ≠ (num + len_exist)
            tags = getRandomName(num, path, pure, getbasename, tagsExist)
        end
    end
    return tags
end


@doc "Merge fastq.gz files (only for Linux)"
function mergeFiles(inNames::Vector{String}, inDir::String, outPath::String)
    # Ref code:
    # io = open(path_sh, "w"); write(io, tmp_sh); close(io);
    # run(`sh $path_sh`)
    # rm(path_sh, force=true)
    # return nothing
    #
    inDir = dirname(inDir)
    outDir = dirname(outPath)
    if isfile(outPath)
        outPath = outPath * tempname(inDir)
    end
    if !isdir(outDir)
        mkdir(outDir)
    end
    fnames = ""
    for xn in eachindex(inNames)
        fnames = fnames * " " * inDir * "/" * inNames[xn]
    end
    sh1 = string("cat", fnames, " > ", outPath)
    return sh1
end
@doc """
Please input a DataFrame that columns are "group" and "filename".
"""
function getShMergeFiles(inNames2Groups::DataFrame, inDir::String, outDir::String, outShPath::String)
    outDir = dirname(outDir)
    # inDir = dirname(inDir)
    num_group = length(inNames2Groups[:,1])
    out_sh = Vector{String}(undef, num_group)
    for xgrp in 1:num_group
        # out_sh[xgrp] = mergeFiles(inNames2Groups[xgrp,2], inDir, string(outDir, "/", inNames2Groups[xgrp,1], ".", split(inNames2Groups[xgrp,2][1], ".")[end]))
        out_sh[xgrp] = mergeFiles(inNames2Groups[xgrp,2], inDir, string(outDir, "/", inNames2Groups[xgrp,1], ".fastq.gz"))
    end
    CSV.write(outShPath, Tables.table(out_sh); header=false)
    return nothing
end

function getShMergeFilesPipe(pathSraRunTable::String, inDir::String, outDir::String, path2saveSh::String)
    n2p = CSV.read(pathSraRunTable, DataFrame)
    in_names = n2p[:, 1] |> Vector{String}
    in_names .= in_names .* ".fastq.gz"
    in_groups = unique(n2p[:, "Experiment"]) |> Vector{String}
    g2n = Vector{Vector{String}}(undef, length(in_groups))
    @threads for xg in eachindex(in_groups)
        g2n[xg] = in_names[findall(x -> x == in_groups[xg], n2p[:, "Experiment"])]
    end
    inN2P = DataFrame(group=in_groups, filename=g2n)
    getShMergeFiles(inN2P, inDir, outDir, path2saveSh)
    return nothing
end


function cleanDir(dirpath::String)
    outs = dirpath
    if dirpath[end] == Char('/')
        outs = dirpath[1:end-1]
    end
    return outs
end


@doc """
Generate yaml configs to run RNA-Seq files separately for saving storage space.
Make soft links to fit pipeline's fold structure requirements.
"""
function genYaml(inDir::String, path2makeDir::String)
    inDir1 = cleanDir(inDir)
    path2makeDir1 = cleanDir(path2makeDir)
    inNames = readdir(inDir1)
    accessions = popfirst!.(split.(inNames, '.'))
    # make dir
    mkpath(path2makeDir1)
    # make yaml
    for x in eachindex(accessions)
        # make soft links
        mkpath(joinpath(path2makeDir1, accessions[x], "00rawdata"))
        symlink(joinpath(inDir1, inNames[x]), joinpath(path2makeDir1, accessions[x], "00rawdata", inNames[x]))
        # prepare yaml text for each RNA-Seq sample
        path2data = joinpath(path2makeDir1, accessions[x])
        text_yaml = string(
            "# config file", "\n",
            "data_dir: \"", path2data, "/\"", "\n",
            "species_dir: \"/home/hwy/Documents/atac_test\"", "\n",
            "species_name: \"Ath\"", "\n",
            "se_or_pe: \"SE\"", "\n",
            "adapter1: \"auto\"", "\n",
            "adapter2: \"auto\"", "\n",
            "intron_max: \"30000\"", "\n",
            "bamidxtype: \"large\" # wheat is large", "\n",
            "samples:", "\n",
            "  ", accessions[x], ":\n"
        )
        # write it to files
        path_yaml = joinpath(path2makeDir1, accessions[x], string(accessions[x], ".yaml"))
        xio = open(path_yaml, "w")
        write(xio, text_yaml)
        close(xio)
    end
    return nothing
end



# =============================================================================


function procOriginExpFile(filePath::AbstractString, acc::AbstractString)
    f0 = CSV.read(filePath, DataFrame)
    f1 = select(f0, "Gene ID" => "Gene_ID", "FPKM" => "FPKM_" * acc, "TPM" => "TPM_" * acc)
    return f1
end

function collectGeneID(dfs::Vector{DataFrame})
    gene_ids = Vector{String}()
    for df in dfs
        for i in 1:size(df, 1)
            push!(gene_ids, df[i, 1])
        end
    end
    sort!(unique!(gene_ids))
    return gene_ids
end

function fillFPKMTPM(idx::Int64, gene_ids::Vector{String}, df::DataFrame)
    find1 = findfirst(x -> x == gene_ids[idx], df.Gene_ID)
    if isnothing(find1)
        fpkm = 0.0
        tpm = 0.0
    else
        fpkm = df[find1, 2]
        tpm = df[find1, 3]
    end
    return fpkm, tpm
end
function pickExpByGeneIDs(gene_ids::Vector{String}, df::DataFrame)
    acc = split(names(df)[end], "_")[end]
    fpkm = zeros(length(gene_ids))
    tpm = zeros(length(gene_ids))
    @threads for idx in eachindex(gene_ids)
        fpkm[idx], tpm[idx] = fillFPKMTPM(idx, gene_ids, df)
    end
    dfout = DataFrame(Gene_ID = gene_ids, FPKM = fpkm, TPM = tpm)
    return select(dfout, "FPKM" => "FPKM_" * acc, "TPM" => "TPM_" * acc)
end

function concExpFiles(inDir::AbstractString, grepWord::AbstractString="ERR")::DataFrame
    files = readdir(inDir)
    files = filter(x -> occursin(grepWord, x), files)
    # accessions = map(x -> split(x, ".")[1], files)
    # Read all files
    dfs = map(x -> procOriginExpFile(joinpath(inDir, x), split(x, ".")[1]), files)
    # Collect all gene IDs
    gene_ids = collectGeneID(dfs)
    #
    # Format DataFrames
    # dfs1 = map(x -> pickExpByGeneIDs(gene_ids, x), dfs)
    # Join DataFrames
    # df_out0 = reduce(hcat, dfs1)
    #
    tpickExpByGeneIDs(xdf::DataFrame) = pickExpByGeneIDs(gene_ids, xdf)
    df_out0 = tmapreduce(tpickExpByGeneIDs, hcat, dfs)
    # Sort colnames
    df_out = df_out0[:, sort(names(df_out0))]
    df_out0 = nothing
    # Add a col of Gene_ID
    df_out = hcat(gene_ids, df_out)
    rename!(df_out, :x1 => "Gene_ID")
    return df_out
end


@doc """
read count to TPM
"""
function count2TPM(path_readcount::String, path_output::String="")
    df_readcount = CSV.read(path_readcount, DataFrame; delim='\t', header=2)
    # read_lengths = df_readcount[!, 6]
    # read_counts = df_readcount[!, 7]
    r_rate = df_readcount[!, 7] ./ df_readcount[!, 6]
    r_n = sum(r_rate)
    TPMs = r_rate ./ r_n * 1e6
    # Sort by Gene ID
    ssperm = sortperm(df_readcount[!, 1])
    gids = df_readcount[ssperm, 1]
    TPMs = TPMs[ssperm]
    df_out = DataFrame(Gene_ID=gids, TPM=TPMs)
    if length(path_output) > 0
        CSV.write(path_output, df_out; delim='\t')
        return nothing
    end
    return df_out
end

function count2TPMinDir(path_dir::String, recog::String="SR",
                        readcountFile::String=joinpath("final_output","readcount.txt"))
    dirs = readdir(path_dir)
    dirs = dirs[occursin.(recog, dirs)]
    files = joinpath.(path_dir, dirs, readcountFile)
    @threads for xf in eachindex(files)
        count2TPM(files[xf], joinpath(dirname(files[xf]), string(dirs[xf], ".tpm.txt")))
    end
    return nothing
end


function procOriginTPMFile(filePath::AbstractString, acc::AbstractString)
    f0 = CSV.read(filePath, DataFrame)
    colname_f0 = names(f0)
    if "Gene_ID" in colname_f0
        f1 = select(f0, "Gene_ID" => "Gene_ID", "TPM" => "TPM_" * acc)
    elseif "Gene ID" in colname_f0
        f1 = select(f0, "Gene ID" => "Gene_ID", "TPM" => "TPM_" * acc)
    else
        throw(error("Gene ID has not been detected."))
    end
    return f1
end
function fillTPM(idx::Int64, gene_ids::Vector{String}, df::DataFrame)
    find1 = findfirst(x -> x == gene_ids[idx], df.Gene_ID)
    if isnothing(find1)
        tpm = 0.0
    else
        tpm = df[find1, 2]
    end
    return tpm
end
function pickTPMByGeneIDs(gene_ids::Vector{String}, df::DataFrame)
    acc = split(names(df)[end], "_")[end]
    tpm = zeros(length(gene_ids))
    @threads for idx in eachindex(gene_ids)
        tpm[idx] = fillTPM(idx, gene_ids, df)
    end
    dfout = DataFrame(Gene_ID = gene_ids, TPM = tpm)
    return select(dfout, "TPM" => "TPM_" * acc)
end

function concTPMFiles(inDir::AbstractString, grepWord::AbstractString="SR")# SRX, SRR or CRR, etc.
    files = readdir(inDir)
    files = filter(x -> occursin(grepWord, x), files)
    # Read all files
    dfs = map(x -> procOriginTPMFile(joinpath(inDir, x), split(x, ".")[1]), files)
    # Collect all gene IDs
    gene_ids = collectGeneID(dfs)
    #
    # Format DataFrames
    # dfs1 = map(x -> pickTPMbyGeneIDs(gene_ids, x), dfs)
    # Join DataFrames
    # df_out0 = reduce(hcat, dfs1)
    #
    tpickTPMByGeneIDs(xdf::DataFrame) = pickTPMByGeneIDs(gene_ids, xdf)
    df_out0 = tmapreduce(tpickTPMByGeneIDs, hcat, dfs)
    #
    # Sort colnames
    df_out = df_out0[:, sort(names(df_out0))]
    df_out0 = nothing
    # Add a col of Gene_ID
    df_out = hcat(gene_ids, df_out)
    rename!(df_out, :x1 => "Gene_ID")
    return df_out
end


function discrSth(x::Union{Missing, Integer, Float64}, sth::Union{Missing, Integer, Float64})
    if ismissing(x)
        if ismissing(sth)
            return true
        else
            return false
        end
    elseif ismissing(sth)
        return false
    else
        if x == sth
            return true
        else
            return false
        end
    end
end
# Main
function delSthRowInDf(dfIn::DataFrame, Sth::Union{Missing, Integer, Float64}, skipCols::Vector{Int64}=Int64[], maxPercSth::Float64=1.0)
    rows2del = Int64[]
    availCols = ncol(dfIn) - length(skipCols)
    # Collect rows whose elements are all sth
    if ismissing(Sth)
        for xr in 1:nrow(dfIn)
            if all(x -> ismissing(x), dfIn[xr, Not(skipCols)])
                push!(rows2del, xr)
            end
        end
    else
        for xr in 1:nrow(dfIn)
            if all(x -> discrSth(x, Sth), dfIn[xr, Not(skipCols)])
                push!(rows2del, xr)
            end
        end
    end
    # Collect rows that has more than maxPercSth sth.
    if maxPercSth < 1.0
        if ismissing(Sth)
            for xr in 1:nrow(dfIn)
                if (sum(x -> ismissing(x), dfIn[xr, Not(skipCols)]) / availCols) > maxPercSth
                    push!(rows2del, xr)
                end
            end
        else
            for xr in 1:nrow(dfIn)
                if (sum(x -> discrSth(x, Sth), dfIn[xr, Not(skipCols)]) / availCols) > maxPercSth
                    push!(rows2del, xr)
                end
            end
        end
    end
    # Remove collected rows
    dfOut = dfIn[Not(unique(rows2del)), :]
    return dfOut
end

function delSmlRowInDf(dfIn::Union{AbstractDataFrame, AbstractMatrix}, ThresholdSml::Union{Integer, Float64}, skipCols::Vector{Int64}=Int64[])
    rows2del = Int64[]
    for xr in 1:nrow(dfIn)
        if all(x -> x < ThresholdSml, dfIn[xr, Not(skipCols)])
            push!(rows2del, xr)
        end
    end
    dfOut = dfIn[Not(rows2del), :]
    return dfOut
end

function convMissing2Sth(df::Union{AbstractDataFrame, AbstractMatrix}, Sth::Union{AbstractString, Int, Float64})
    df1 = df
    for xc in 1:ncol(df)
        for xr in 1:nrow(df)
            if ismissing(df[xr,xc])
                df1[xr,xc] = Sth
            end
        end
    end
    return df1
end

function rmDuplicatedRow(dfIn::Union{AbstractDataFrame, AbstractMatrix})
    
end
