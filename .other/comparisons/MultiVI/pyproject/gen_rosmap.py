import anndata
import numpy as np
import pandas as pd
import scvi


path_to_rna = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/Chen-2019-RNA.h5ad"
path_to_atac = "/mnt/hdd2/data/GLUE/data/download/Chen-2019/Chen-2019-ATAC.h5ad"
templ_rna = anndata.read_h5ad(path_to_rna)
templ_atac = anndata.read_h5ad(path_to_atac)

# observations
path_rosmap_obs = "/mnt/hdd2/data/data_tmoia/formal/preprocessed_data/02_human/ROSMAP/ROSMAP_label_encoded.csv"
# mRNA
path_rosmap_om1 = "/mnt/hdd2/data/data_tmoia/formal/preprocessed_data/02_human/ROSMAP/ROSMAP_om1.csv"
# DNA meth
path_rosmap_om2 = "/mnt/hdd2/data/data_tmoia/formal/preprocessed_data/02_human/ROSMAP/ROSMAP_om2.csv"
# miRNA
path_rosmap_om3 = "/mnt/hdd2/data/data_tmoia/formal/preprocessed_data/02_human/ROSMAP/ROSMAP_om3.csv"

lab = pd.read_csv(path_rosmap_obs, index_col=0)
lab.index = [('sample_' + str(i)) for i in range(1, 1 + lab.shape[0])]
labs = list(lab.index)
om1 = pd.read_csv(path_rosmap_om1, index_col=0)
om2 = pd.read_csv(path_rosmap_om2, index_col=0)
om3 = pd.read_csv(path_rosmap_om3, index_col=0)

# =====================================================================================================================

om1 = (om1 * 100).astype(int)
om2 = (om2 * 100).astype(int)
om3 = (om3 * 100).astype(int)

om1.columns = [('om1_'+str(i)) for i in range(1, 1 + om1.shape[1])]
om2.columns = [('om2_'+str(i)) for i in range(1, 1 + om2.shape[1])]
om3.columns = [('om3_'+str(i)) for i in range(1, 1 + om3.shape[1])]
om1.index = labs
om2.index = labs
om3.index = labs

om1_1 = om1
om2_1 = om2
om3_1 = om3

om1 = pd.concat([om1_1, om2_1, om3_1], axis=1)
om2 = pd.concat([om1_1, om2_1, om3_1], axis=1)
om3 = pd.concat([om1_1, om2_1, om3_1], axis=1)
om1.columns = [('om1_'+str(i)) for i in range(1, 1 + om1.shape[1])]
om2.columns = [('om2_'+str(i)) for i in range(1, 1 + om2.shape[1])]
om3.columns = [('om3_'+str(i)) for i in range(1, 1 + om3.shape[1])]

# Again
om1_1 = om1
om2_1 = om2
om3_1 = om3

om1 = pd.concat([om1_1, om2_1, om3_1], axis=1)
om2 = pd.concat([om1_1, om2_1, om3_1], axis=1)
om3 = pd.concat([om1_1, om2_1, om3_1], axis=1)
om1.columns = [('om1_'+str(i)) for i in range(1, 1 + om1.shape[1])]
om2.columns = [('om2_'+str(i)) for i in range(1, 1 + om2.shape[1])]
om3.columns = [('om3_'+str(i)) for i in range(1, 1 + om3.shape[1])]

# =====================================================================================================================

var_om1 = pd.DataFrame(list(om1.columns), index=list(om1.columns), columns=['mRNA'])
var_om2 = pd.DataFrame(list(om2.columns), index=list(om2.columns), columns=['DNA methylation'])
var_om3 = pd.DataFrame(list(om3.columns), index=list(om3.columns), columns=['miRNA'])

rosmap_rna = anndata.AnnData(X=om1, obs=lab, var=var_om1)
rosmap_met = anndata.AnnData(X=om2, obs=lab, var=var_om2)
rosmap_mir = anndata.AnnData(X=om3, obs=lab, var=var_om3)

# scvi cannot process 3 modals
# rosmap_multi = anndata.concat([rosmap_rna, rosmap_met, rosmap_mir], axis=1)
rosmap_multi = anndata.concat([rosmap_rna, rosmap_met], axis=1)
rosmap_mvi = scvi.data.organize_multiome_anndatas(rosmap_multi, rosmap_rna, rosmap_met)


# Train
scvi.model.MULTIVI.setup_anndata(rosmap_mvi, batch_key="modality")
vae = scvi.model.MULTIVI(adata=rosmap_mvi, n_genes=200, n_regions=200, n_latent=100, n_hidden=200)
vae.train()

scvi.model.MULTIVI.save(vae, dir_path="model_multivi_rosmap", save_anndata=True)

latent_repr = vae.get_latent_representation()

# np.savetxt("latent_repr_rosmap.txt", latent_repr)

# pd.DataFrame(templ_rna.X.A).to_csv("chen2019_scRNA_9190.csv")
# pd.DataFrame(templ_atac.X.A).to_csv("chen2019_scATAC_9190.csv")
