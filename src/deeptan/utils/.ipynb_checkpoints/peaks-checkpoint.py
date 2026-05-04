import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from scipy.sparse import csr_matrix, hstack


def read_peaks(adata: sc.AnnData) -> pl.DataFrame:
    """Read the peaks data from the AnnData object."""
    peaks: list[str] = adata.var[adata.var["feature_types"] == "Peaks"].copy().index.to_list()
    peak_chr = []
    peak_sta = []
    peak_end = []
    for peak in peaks:
        if ":" not in peak or "-" not in peak:
            raise ValueError(f"Invalid peak format: {peak}")
        chr, sta_end = peak.split(":")
        sta, end = sta_end.split("-")
        peak_chr.append(chr)
        peak_sta.append(int(sta))
        peak_end.append(int(end))
    peaks_df = pl.DataFrame(
        {
            "chr": peak_chr,
            "start": peak_sta,
            "end": peak_end,
            "peak": peaks,
        }
    ).sort(by=["chr", "start", "end"])
    return peaks_df


class MergePeaks:
    r"""
    Merge overlapping peaks across multiple AnnData objects.
    """

    def __init__(self, adata_list: list[sc.AnnData]):
        r"""
        Initialize the MergePeaks class with a list of AnnData objects.

        Args:
            adata_list (List[AnnData]): List of AnnData objects containing peak data.
        """
        # Collect all peaks and sort
        all_peaks = []
        for adata in adata_list:
            peaks_df = read_peaks(adata)
            all_peaks.append(peaks_df)

        # Combine and sort all peaks
        self.combined_peaks = pl.concat(all_peaks).sort(by=["chr", "start", "end"])
        # print(combined_peaks)

    def merge_peaks(self) -> pl.DataFrame:
        """Merge overlapping peaks across all AnnData objects."""
        merged_peaks = []

        # Process each chromosome separately
        for chr_name in self.combined_peaks["chr"].unique().to_list():
            chr_peaks = self.combined_peaks.filter(pl.col("chr") == chr_name)
            intervals = chr_peaks.select(["start", "end"]).rows()

            if not intervals:
                continue

            # Initialize with first interval
            current_start, current_end = intervals[0]

            for start, end in intervals[1:]:
                # if start <= current_end:
                if start < current_end:
                    # Merge overlapping intervals
                    current_end = max(current_end, end)
                else:
                    # Add non-overlapping interval
                    merged_peaks.append(
                        (
                            chr_name,
                            current_start,
                            current_end,
                            f"{chr_name}:{current_start}-{current_end}",
                        )
                    )
                    current_start, current_end = start, end

            # Add the last merged interval for the chromosome
            merged_peaks.append(
                (
                    chr_name,
                    current_start,
                    current_end,
                    f"{chr_name}:{current_start}-{current_end}",
                )
            )

        # Create final DataFrame with proper sorting
        self.merged_peaks = pl.DataFrame(merged_peaks, schema=["chr", "start", "end", "peak"], orient="row").sort(by=["chr", "start", "end"])

        return self.merged_peaks

    def check(self, merged_peaks: pl.DataFrame | None = None) -> bool:
        """Check if overlapping peaks exist."""
        if merged_peaks is None:
            merged_peaks = self.merged_peaks
        merged_peaks = merged_peaks.sort(by=["chr", "start", "end"])
        for i in range(1, len(merged_peaks)):
            if (
                merged_peaks["chr"][i] == merged_peaks["chr"][i - 1]
                # and merged_peaks["start"][i] <= merged_peaks["end"][i - 1]
                and merged_peaks["start"][i] < merged_peaks["end"][i - 1]
            ):
                return False
        return True


def align_peaks_to_ref(ref_peaks: pl.DataFrame, new_peaks: pl.DataFrame):
    r"""
    Align new peaks to the reference peaks.
    """
    # Prepare merged peaks with index column
    ref_peaks_ = ref_peaks.with_row_count("merged_index").sort(by=["chr", "start", "end"])

    # Join original peaks with merged peaks using interval overlap condition
    overlap_condition = (pl.col("start") <= pl.col("end_merged")) & (pl.col("end") >= pl.col("start_merged"))

    # Perform join and filter overlaps
    mapping_df = new_peaks.join(
        ref_peaks_.rename({"start": "start_merged", "end": "end_merged"}),
        on="chr",
        how="inner",
    ).filter(overlap_condition)
    return mapping_df, ref_peaks_


def map_peaks_to_ref(new_h5ad: str, merged_peaks: pl.DataFrame):
    """
    Map new peaks to the merged peaks and make a new anndata.

    Args:
        new_h5ad: Path to the new AnnData h5ad file.
        merged_peaks: DataFrame containing merged peaks from MergePeaks class. This DataFrame should have columns ['chr', 'start', 'end'].

    Returns:
        A new AnnData object with peaks mapped to the merged reference.
    """
    # Load new AnnData object
    adata = sc.read_h5ad(new_h5ad)

    # Separate peaks and RNA features based on feature_types
    if "feature_types" not in adata.var.columns:
        raise ValueError("The 'feature_types' column is missing in adata.var. Please ensure it exists to distinguish between Peaks and Gene Expression.")

    # Split the var into peaks and RNA features
    peaks_mask = adata.var["feature_types"] == "Peaks"
    rna_mask = adata.var["feature_types"] == "Gene Expression"

    peaks_adata = adata[:, peaks_mask]
    rna_adata = adata[:, rna_mask]
    # print("Peaks and RNA features separated successfully.")
    # print(f"Part of RNA:\n{rna_adata}")
    # print(f"Part of Peaks:\n{peaks_adata}\n")

    # Get original peaks and ensure sorted order
    original_peaks_df = read_peaks(peaks_adata)

    mapping_df, merged_peaks = align_peaks_to_ref(merged_peaks, original_peaks_df)

    # Check for unmapped peaks
    if mapping_df.shape[0] != original_peaks_df.shape[0]:
        missing = original_peaks_df.shape[0] - mapping_df.shape[0]
        raise ValueError(f"{missing} peaks could not be mapped to merged reference")

    # Create mapping from original column index to merged index
    var_indices = peaks_adata.var_names.get_indexer(original_peaks_df["peak"].to_list())
    mapping = dict(zip(var_indices, mapping_df["merged_index"].to_list()))

    # Build sparse transformation matrix
    rows = list(mapping.keys())
    cols = list(mapping.values())
    data = np.ones(len(rows))

    transform_matrix = csr_matrix((data, (rows, cols)), shape=(peaks_adata.shape[1], merged_peaks.shape[0]))

    # Apply matrix transformation
    new_X_peaks = peaks_adata.X @ transform_matrix

    # Create new AnnData object
    new_var_peaks = merged_peaks.select([pl.col("chr"), pl.col("start"), pl.col("end"), pl.col("peak")]).to_pandas().set_index("peak")

    new_var_peaks["feature_types"] = "Peaks"

    # Combine RNA data and transformed peaks data
    combined_X = hstack([new_X_peaks, rna_adata.X])
    # print(f"\nShape of combined data: {combined_X.shape}\n")
    combined_var = pd.concat([new_var_peaks, rna_adata.var])
    # print(f"\nShape of combined var: {combined_var.shape}\n")
    new_adata = sc.AnnData(
        X=combined_X,
        obs=adata.obs,
        var=combined_var,
        # uns=adata.uns,
        obsm=adata.obsm.copy(),
        # varm=adata.varm.copy(),
    )

    print(f"\nOriginal:\n {adata}\n")
    print(f"\nMerged peaks:\n{new_adata}\n")

    return new_adata
