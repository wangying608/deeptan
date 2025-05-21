import argparse
import sys

import scanpy as sc


def print_sc_h5():
    parser = argparse.ArgumentParser(description="Read an H5AD or H5 file and print its contents.")
    parser.add_argument("-p", "--path", type=str, help="Path to the H5AD or H5 file.")
    args = parser.parse_args()
    if args.path.endswith(".h5ad"):
        adata = sc.read_h5ad(args.path)
    elif args.path.endswith(".h5"):
        adata = sc.read_10x_h5(args.path)
    else:
        print("Unsupported file format. Please provide a .h5ad or .h5 file.")
        sys.exit(1)
    print(adata)
    print("\nobs keys: ")
    print(adata.obs.keys())
    for _key in adata.obs.keys():
        print(f"\n{_key}:")
        print(adata.obs[_key])
    print("\nvar keys: ")
    for _key in adata.var_keys():
        print(f"\n{_key}:")
        print(adata.var[_key])
    if "Celltype" in adata.obs.keys():
        print("\nCelltype:")
        print(type(adata.obs["Celltype"]))
    if "Orig.ident" in adata.obs.keys():
        sc.pl.umap(adata, color="Orig.ident", title="UMAP by Experiment")
