import sys
import scanpy as sc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_h5ad.py <path_to_h5ad_file>")
        sys.exit(1)
    h5ad_path = sys.argv[1]
    adata = sc.read_h5ad(h5ad_path)
    print(adata)
