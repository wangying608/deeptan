import sys
import scanpy as sc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_h5ad.py <path_to_h5ad_file>")
        sys.exit(1)
    h5ad_path = sys.argv[1]

    if h5ad_path.endswith(".h5ad"):
        adata = sc.read_h5ad(h5ad_path)
    elif h5ad_path.endswith(".h5"):
        adata = sc.read_10x_h5(h5ad_path)
    else:
        print("Unsupported file format. Please provide a .h5ad or .h5 file.")
        sys.exit(1)

    print(adata)

    # Check if adata has cell type annotations
    print("\nobs keys: ")
    print(adata.obs.keys())
    for _key in adata.obs.keys():
        print(f"\n{_key}:")
        print(adata.obs[_key])
    
    for _key in adata.var_keys():
        print(f"\n{_key}:")
        print(adata.var[_key])

    if "Celltype" in adata.obs.keys():
        print("\nCelltype:")
        print(type(adata.obs["Celltype"]))

    # The key "Orig.ident" denotes experiments (replicates)
    # See if these experiments has batch effects
    # Visualize the batch effects
    if "Orig.ident" in adata.obs.keys():
        sc.pl.umap(adata, color="Orig.ident", title="UMAP by Experiment")
