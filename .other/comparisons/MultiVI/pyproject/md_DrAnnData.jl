__precompile__(true)

module DrAnnData

using Muon
using DataFrames

export drAnndata


function drAnndata(path2rnaAnndata::String, path2atacAnndata::String)
    ad_rna = readh5ad(path2rnaAnndata)
    ad_atac = readh5ad(path2atacAnndata)
    # Sample intersection
    obs_rna = ad_rna.obs_names
    obs_atac = ad_atac.obs_names
    obs_inters = intersect(obs_atac, obs_rna)
    # length(unique(obs_inters))
    n_obs = length(obs_inters)    
    n_region = length(ad_atac.var_names)
    n_gene = length(ad_rna.var_names)
    # 
    error("unfinished")
    return 0
end


end
