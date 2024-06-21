__precompile__(true)

module MaskAnnData

using Muon
using Random
using DataFrames

export maskAnndata


function maskAnndata(path2rnaAnndata::String, path2atacAnndata::String, percentageMask::Float64=0.1, seedMask::Int64=1234)
    rngx = MersenneTwister(seedMask)
    ad_rna = readh5ad(path2rnaAnndata)
    ad_atac = readh5ad(path2atacAnndata)
    # Test subset
    #  part1 = ad_atac[1:3, 6:10]
    #  part2 = ad_atac[3:9190, 2:241757]

    # Sample intersection
    obs_rna = ad_rna.obs_names
    obs_atac = ad_atac.obs_names
    obs_inters = intersect(obs_atac, obs_rna)
    # length(unique(obs_inters))
    n_obs = length(obs_inters)
    
    n_region = length(ad_atac.var_names)
    n_gene = length(ad_rna.var_names)

    n_elem_rna = n_obs * n_gene
    n_elem_atac = n_obs * n_region

    # Random.seed!(rngx)
    rna2mask = rand(rngx, 1:n_elem_rna, ceil(Int64, percentageMask * n_elem_rna))
    atac2mask = rand(rngx, 1:n_elem_atac, ceil(Int64, percentageMask * n_elem_atac))
    # mask location
    mask_rna = DataFrame(obs_p = rem.(rna2mask, n_obs), var_p = cld.(rna2mask, n_obs))
    mask_atac = DataFrame(obs_p = rem.(atac2mask, n_obs), var_p = cld.(atac2mask, n_obs))
    # rem 0 to n_obs
    zeros_in_rna = findall(x -> rem(x, n_obs) == 0, rna2mask)
    zeros_in_atac = findall(x -> rem(x, n_obs) == 0, atac2mask)
    mask_rna[zeros_in_rna, 1] .= n_obs
    mask_atac[zeros_in_atac, 1] .= n_obs

    # Mask as zeros OR None ?
    Threads.@threads for xr in eachindex(rna2mask)
        ad_rna.X[mask_rna[xr, 1], mask_rna[xr, 2]] = 0.0f0
    end
    
    Threads.@threads for xr in eachindex(atac2mask)
        ad_atac.X[mask_atac[xr, 1], mask_atac[xr, 2]] = 0.0f0
    end
    
    return ad_rna, ad_atac
end


end
