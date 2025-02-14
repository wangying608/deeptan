import sys
import polars as pl


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_csv.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    df = pl.read_csv(file_path)
    print(df)
