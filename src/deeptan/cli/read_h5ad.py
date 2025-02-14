import sys
import scanpy as sc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_h5ad.py <path_to_h5ad_file>")
        sys.exit(1)
    h5ad_path = sys.argv[1]
    adata = sc.read_h5ad(h5ad_path)
    print(adata)

    # Check if adata has cell type annotations
    print("\nobs keys: ")
    print(adata.obs.keys())
    for _key in adata.obs.keys():
        print(f"\n{_key}:")
        print(adata.obs[_key].value_counts())

    if "Celltype" in adata.obs.keys():
        print("\nCelltype:")
        print(type(adata.obs["Celltype"]))

    # The key "Orig.ident" denotes experiments (replicates)
    # See if these experiments has batch effects
    # Visualize the batch effects
    if "Orig.ident" in adata.obs.keys():
        sc.pl.umap(adata, color="Orig.ident", title="UMAP by Experiment")
