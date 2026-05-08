#!/usr/bin/env python3
r"""
==============================================================================
Run01: Build Tissue-Type LitData for DeepTAN Fine-Tuning
==============================================================================

Purpose
-------
Convert user-provided tissue scRNA train/validation/test expression tables into
the LitData directory required by Run02 tissue-specific DeepTAN fine-tuning.

Pipeline position
-----------------
Run01: user-defined tissue scRNA files -> LitData
Run02: LitData -> tissue-specific fine-tuned DeepTAN checkpoint
Run03: LitData + fine-tuned checkpoint -> tissue-specific latent gene network

Core computation
----------------
This script keeps the original data-construction logic:

  1. Load pretrained DeepTAN graph resources:
       - pretrained_trn_npz
       - pretrained_pkl
  2. Read user-provided train/validation/test parquet files.
  3. Align user genes to the pretrained DeepTAN node vocabulary.
  4. Reuse the pretrained graph skeleton.
  5. Build temporary DeepTAN-compatible trn.npz and companion parquet files.
  6. Build or copy cell-type one-hot labels when labels are provided.
  7. Generate LitData directories:
       trn/
       val/
       tst/
  8. Save metadata and gene CV weights.

Input naming policy
-------------------
File names are fully user-defined. The input files do not need to follow
internal naming patterns such as split_42_0.parquet.

Recommended usage
-----------------
Use the wrapper:

    bash scripts/Run01_build_Tissue_LitData.sh

or run directly:

    python src/Run01_build_Tissue_LitData.py \
        --pretrained_trn_npz /path/to/pretrained_trn.npz \
        --pretrained_pkl /path/to/others2save.pkl \
        --trn_parquet /path/to/tissue_train.parquet \
        --val_parquet /path/to/tissue_valid.parquet \
        --tst_parquet /path/to/tissue_test.parquet \
        --labels_parquet /path/to/celltype_onehot.parquet \
        --tissue_name ExampleTissue \
        --output_dir /path/to/litdata

Notes
-----
For tissue-specific fine-tuning, class labels are recommended because Run02
usually optimizes both reconstruction and cell-type classification within each
tissue. Labels can be provided using one of:

  - --labels_parquet
  - --celltype_col
  - --celltype_csv

Author: DeepTAN Tissue-Specific Fine-Tuning Pipeline
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import litdata
import numpy as np
import polars as pl
from loguru import logger


# ── Optional local DeepTAN source checkout ───────────────────────────────────
# Set DEEPTAN_SRC in the shell wrapper when DeepTAN is not installed as a package.
# Example:
#   export DEEPTAN_SRC="/path/to/deeptan-dev/src"
_LOCAL_DEEPTAN = os.environ.get("DEEPTAN_SRC", "").strip()
if _LOCAL_DEEPTAN and _LOCAL_DEEPTAN not in sys.path:
    sys.path.insert(0, _LOCAL_DEEPTAN)
# ────────────────────────────────────────────────────────────────────────────

import deeptan.constants as const
from deeptan.utils.data import DeepTANDataModule, read_nmic_npz

# [BUGFIX-3] Do not print the DeepTAN module path at import time; print it once in main().


# ============================================================================
# 1. Command-line arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run01: build Tissue-Type DeepTAN LitData from user-defined scRNA parquet files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--pretrained_trn_npz", type=str, required=True,
        help="Pretraining-stage trn.npz used to reuse the pretrained graph skeleton.")
    parser.add_argument("--pretrained_pkl", type=str, required=True,
        help="Pretraining metadata pkl containing the pretrained node vocabulary.")
    parser.add_argument("--trn_parquet", type=str, required=True,
        help="User-provided tissue training expression parquet.")
    parser.add_argument("--val_parquet", type=str, required=True,
        help="User-provided tissue validation expression parquet.")
    parser.add_argument("--tst_parquet", type=str, required=True,
        help="User-provided tissue test expression parquet.")
    parser.add_argument("--celltype_col", type=str, default=None,
        help="Optional cell-type annotation column in the input parquet files.")
    parser.add_argument("--celltype_csv", type=str, default=None,
        help="Optional external CSV containing cell IDs and cell-type annotations.")
    parser.add_argument("--labels_parquet", type=str, default=None,
        help="Optional existing one-hot cell-type label parquet.")
    parser.add_argument("--tissue_name", type=str, default="Tissue",
        help="Human-readable tissue name used for metadata and logs.")
    parser.add_argument("--output_dir", type=str, default=".tmp_data_finetune",
        help="Output LitData directory consumed by Run02.")
    parser.add_argument("--bs", type=int, default=const.default.bs,
        help="Batch size")
    parser.add_argument("--thre_mi", type=float, default=const.default.threshold_nmic,
        help="Edge-weight threshold used during DeepTAN data construction.")
    parser.add_argument("--n_workers", type=int, default=const.default.n_threads,
        help="Number of parallel workers for LitData conversion.")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducibility. This is not a data split identifier in the single-run workflow.")
    parser.add_argument("--skip_cv", action="store_true",
        help="Skip calculation of gene_cv_weights.csv.")
    parser.add_argument("--validate_output", action="store_true", default=True,
        help="Validate the generated output directory after conversion.")

    return parser.parse_args()


# ============================================================================
# 2. Pretraining resource loading
# ============================================================================

def load_pretrained_meta(pkl_path: str) -> dict:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pretraining metadata file does not exist: {pkl_path}")
    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)
    for key in ["dict_node_names", "output_g_label_dim"]:
        if key not in meta:
            raise KeyError(f"Pretraining pkl is missing required key: {key}")
    logger.info(f"Loading pretraining metadata: {pkl_path}")
    logger.info(f"  dict_node_names:  {len(meta['dict_node_names'])} genes")
    logger.info(f"  output_g_label_dim: {meta['output_g_label_dim']}")
    return meta


def load_pretrained_graph(npz_path: str) -> Dict[str, Any]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Pretraining trn.npz does not exist: {npz_path}")
    logger.info(f"Extracting graph structure from pretraining trn.npz: {npz_path}")
    edge_attr, edge_index, mat, mat_feat_indices, obs_names, node_names = \
        read_nmic_npz(npz_path)
    n_genes = len(node_names)
    n_edges = edge_index.shape[1] if edge_index.ndim == 2 else len(edge_index) // 2
    logger.info(f"  Graph structure extracted:")
    logger.info(f"    Number of nodes / genes: {n_genes}")
    logger.info(f"    Number of edges:        {n_edges}")
    logger.info(f"    Edge-weight range:      [{edge_attr.min():.4f}, {edge_attr.max():.4f}]")
    logger.info(f"    Pretraining cells:      {mat.shape[0]} (reference only; not used for fine-tuning)")
    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_names": node_names,
        "mat_feat_indices": mat_feat_indices,
    }


# ============================================================================
# 3. Fine-tuning data loading (.parquet)
# ============================================================================

def load_parquet_expression(
    parquet_path: str,
    celltype_col: str = None,
    subset_name: str = "data",
) -> Dict[str, Any]:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")
    logger.info(f"  Reading [{subset_name}]: {parquet_path}")
    df = pl.read_parquet(parquet_path)

    obs_col = None
    for cand in ["obs_names", "obs_name", "barcode", "cell_id", "index"]:
        if cand in df.columns:
            obs_col = cand
            break
    if obs_col is None:
        for c in df.columns:
            if df[c].dtype == pl.Utf8 or df[c].dtype == pl.String:
                obs_col = c
                break
    if obs_col is None:
        obs_col = df.columns[0]
        logger.warning(f"    No standard obs column was found; using the first column: {obs_col!r}")
    obs_names = df[obs_col].to_list()

    cell_types = None
    ct_internal_col = None
    if celltype_col and celltype_col in df.columns:
        cell_types = df[celltype_col].cast(pl.Utf8).to_list()
        ct_internal_col = celltype_col
        logger.info(f"    Cell-type column {celltype_col!r}: {len(set(cell_types))} classes")

    exclude_cols = {obs_col}
    if ct_internal_col:
        exclude_cols.add(ct_internal_col)
    metadata_candidates = {
        "batch", "sample", "donor", "condition", "tissue",
        "n_genes", "n_counts", "pct_counts_mt",
    }
    for c in df.columns:
        if c.lower() in metadata_candidates:
            exclude_cols.add(c)
    gene_cols = [c for c in df.columns if c not in exclude_cols]
    non_numeric = [c for c in gene_cols if not df[c].dtype.is_numeric()]
    if non_numeric:
        logger.warning(f"    Skipping {len(non_numeric)} non-numeric columns: "
                       f"{non_numeric[:5]}{'...' if len(non_numeric) > 5 else ''}")
        gene_cols = [c for c in gene_cols if c not in set(non_numeric)]

    gene_names = gene_cols
    mat = df.select(gene_cols).to_numpy().astype(np.float32)
    logger.info(f"    [{subset_name}] cells={mat.shape[0]}, genes={mat.shape[1]}, "
                f"value_range=[{mat.min():.3f}, {mat.max():.3f}], "
                f"zero_fraction={((mat == 0).sum() / mat.size):.1%}")
    if mat.max() > 50:
        logger.warning(f"    Maximum expression value = {mat.max():.1f}; data may not be log-normalized")

    return {"mat": mat, "gene_names": gene_names,
            "obs_names": obs_names, "cell_types": cell_types}


def load_celltype_from_csv(csv_path: str, obs_names: List[str]) -> Optional[List[str]]:
    if not csv_path or not os.path.exists(csv_path):
        return None
    logger.info(f"  Reading cell-type annotations from external CSV: {csv_path}")
    ct_df = pl.read_csv(csv_path)
    obs_col_ct = None
    for cand in ["obs_names", "obs_name", "barcode", "cell_id"]:
        if cand in ct_df.columns:
            obs_col_ct = cand
            break
    ct_col = None
    for cand in ["celltype", "cell_type", "label", "annotation"]:
        if cand in ct_df.columns:
            ct_col = cand
            break
    if not obs_col_ct or not ct_col:
        logger.warning(f"  External CSV format is invalid; expected obs_names and celltype columns\n"
                       f"  Actual columns: {ct_df.columns}")
        return None
    ct_dict = dict(zip(ct_df[obs_col_ct].to_list(),
                       ct_df[ct_col].cast(pl.Utf8).to_list()))
    cell_types = [ct_dict.get(obs, "Unknown") for obs in obs_names]
    n_unknown = cell_types.count("Unknown")
    if n_unknown > 0:
        logger.warning(f"  {n_unknown}/{len(cell_types)} cells were not matched to a cell-type label")
    else:
        logger.info(f"  Cell-type matching completed: {len(set(cell_types))} classes")
    return cell_types


# ============================================================================
# 4. Gene alignment
# ============================================================================

def align_genes_to_pretrained(
    tissue_mat: np.ndarray,
    tissue_gene_names: List[str],
    pretrained_dict: Dict[str, int],
    subset_name: str = "data",
    tissue_name: str = "Tissue",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n_cells = tissue_mat.shape[0]
    n_pretrained = len(pretrained_dict)
    tissue_gene_to_idx = {g: i for i, g in enumerate(tissue_gene_names)}
    pretrained_genes = set(pretrained_dict.keys())
    tissue_genes = set(tissue_gene_names)
    common_genes = pretrained_genes & tissue_genes
    missing_genes = pretrained_genes - tissue_genes
    extra_genes = tissue_genes - pretrained_genes
    coverage = len(common_genes) / n_pretrained

    logger.info(f"    [{subset_name}] Gene alignment:")
    logger.info(f"      pretrained={n_pretrained}, fine_tuning={len(tissue_gene_names)}, "
                f"common={len(common_genes)}, "
                f"missing={len(missing_genes)}(filled with zeros), "
                f"extra={len(extra_genes)}(discarded)")
    logger.info(f"      Coverage: {coverage:.1%}")

    if coverage < 0.80:
        logger.error(f"  [{tissue_name}/{subset_name}] coverage {coverage:.1%} < 80%!\n"
                     f"  Please check whether the data and the pretrained model use the same gene annotation version.")
        sys.exit(1)
    elif coverage < 0.85:
        logger.warning(f"      Coverage is between 80% and 85%; acceptable but should be inspected.")
    else:
        logger.success(f"      Coverage >= 85%; alignment is considered safe.")

    aligned_mat = np.zeros((n_cells, n_pretrained), dtype=np.float32)
    common_list = sorted(common_genes)
    pre_idx = np.array([pretrained_dict[g] for g in common_list])
    tis_idx = np.array([tissue_gene_to_idx[g] for g in common_list])
    aligned_mat[:, pre_idx] = tissue_mat[:, tis_idx]

    logger.info(f"      After alignment: {aligned_mat.shape}, "
                f"nonzero_fraction: {(aligned_mat != 0).sum() / aligned_mat.size:.1%}")

    report = {
        "subset": subset_name, "n_pretrained": n_pretrained,
        "n_tissue": len(tissue_gene_names), "n_common": len(common_genes),
        "n_missing": len(missing_genes), "n_extra": len(extra_genes),
        "coverage": coverage, "missing_genes": sorted(missing_genes),
        "extra_genes": sorted(extra_genes),
    }
    return aligned_mat, report


# ============================================================================
# 5. Build graph data files: temporary trn.npz + trn/val/tst parquet
# ============================================================================

def build_finetune_npz(
    aligned_mat: np.ndarray, obs_names: List[str],
    graph: Dict[str, Any], pretrained_dict: Dict[str, int],
    output_path: str,
) -> str:
    """Build a temporary trn.npz file from the pretrained graph skeleton and tissue expression matrix.

The saved field names and array shapes must match read_nmic_npz(). The in-memory graph was already transposed by read_nmic_npz(), so this function writes arrays back in the storage orientation expected by the DeepTAN loader."""
    n_genes = len(pretrained_dict)
    n_edges = graph["edge_index"].shape[1] if graph["edge_index"].ndim == 2 \
        else len(graph["edge_index"])

    # [BUGFIX-2] Use the field names expected by read_nmic_npz() and reverse-transpose to storage format.
    np.savez(
        output_path,
        mi_values=graph["edge_attr"],                   # edge_attr -> mi_values
        feat_pairs=graph["edge_index"].T,               # [2,E] → [E,2]
        processed_mat=aligned_mat.T,                    # [cells, genes] -> [genes, cells]
        mat_feat_indices=graph["mat_feat_indices"],     # unchanged
    )

    logger.info(f"    Temporary npz saved: {output_path}")
    logger.info(f"      cells={len(obs_names)}, genes={n_genes}, edges={n_edges}")
    return output_path


def build_finetune_parquet(
    aligned_mat: np.ndarray, obs_names: List[str],
    pretrained_dict: Dict[str, int], output_path: str,
) -> str:
    idx_to_name = {v: k for k, v in pretrained_dict.items()}
    gene_names = [idx_to_name[i] for i in range(len(pretrained_dict))]
    data = {"obs_names": obs_names}
    for gi, gname in enumerate(gene_names):
        data[gname] = aligned_mat[:, gi].tolist()
    df = pl.DataFrame(data)
    df.write_parquet(output_path)
    logger.info(f"    Temporary parquet saved: {output_path} (cells={len(obs_names)})")
    return output_path


# ============================================================================
# 6. Cell-type labels
# ============================================================================

def build_celltype_onehot(
    cell_types: List[str], obs_names: List[str], output_path: str,
) -> Tuple[str, int]:
    unique_types = sorted(set(cell_types))
    n_classes = len(unique_types)
    logger.info(f"  Generating celltype_onehot.parquet: {n_classes} classes")
    data = {"obs_names": obs_names}
    for ct in unique_types:
        data[ct] = [1 if cell_types[i] == ct else 0 for i in range(len(cell_types))]
    df = pl.DataFrame(data)
    df.write_parquet(output_path)
    dist = Counter(cell_types)
    for ct in unique_types:
        logger.info(f"    {ct}: {dist[ct]} ({dist[ct]/len(cell_types):.1%})")
    return output_path, n_classes


# ============================================================================
# 7. CV-weight calculation
# ============================================================================

def compute_cv_weights(trn_npz_path: str, dict_node_names: dict, output_path: str):
    logger.info(f"Computing gene-level CV weights from the training split: {trn_npz_path}")
    _, _, mat, _, _, node_names = read_nmic_npz(trn_npz_path)
    mat_log = np.log1p(mat)
    mean_v = np.mean(mat_log, axis=0)
    std_v = np.std(mat_log, axis=0)
    cv = np.divide(std_v, mean_v, out=np.zeros_like(std_v), where=mean_v > 1e-8)
    cv = np.nan_to_num(cv, nan=0.0, posinf=0.0, neginf=0.0)
    cv_min, cv_max = cv.min(), cv.max()
    if cv_max > cv_min:
        cv_norm = (cv - cv_min) / (cv_max - cv_min)
    else:
        cv_norm = np.full_like(cv, 0.5)
    raw_w = 0.5 + 0.5 * cv_norm
    cv_w = np.clip(raw_w, 0.5, np.percentile(raw_w, 99))
    logger.info(f"  CV weights: min={cv_w.min():.4f}, max={cv_w.max():.4f}, "
                f"mean={cv_w.mean():.4f}")
    gene_cv = {name: float(cv_w[i]) if i < len(cv_w) else 0.75
               for i, name in enumerate(node_names)}
    rows = [{"gene": g, "cv_weight": gene_cv.get(g, 0.75)}
            for g, _ in sorted(dict_node_names.items(), key=lambda x: x[1])]
    pl.DataFrame(rows).write_csv(output_path)
    logger.success(f"  CV weights saved: {output_path} ({len(rows)} genes)")


# ============================================================================
# 8. Output validation
# ============================================================================

def validate_output(output_dir: str, tissue_name: str) -> bool:
    logger.info(f"  Validating generated output directory: {output_dir}")
    ok = True
    for subset in ["trn", "val", "tst"]:
        subset_dir = os.path.join(output_dir, subset)
        if not os.path.isdir(subset_dir) or len(os.listdir(subset_dir)) == 0:
            alt_key = getattr(const.dkey, f"abbr_{subset.replace('tst', 'test')}", subset)
            alt_dir = os.path.join(output_dir, alt_key)
            if not os.path.isdir(alt_dir) or len(os.listdir(alt_dir)) == 0:
                logger.error(f"    ✗ Missing or empty: {subset}/ (LitData)")
                ok = False
            else:
                logger.info(f"    ✓ {alt_key}/ ({len(os.listdir(alt_dir))} files)")
        else:
            logger.info(f"    ✓ {subset}/ ({len(os.listdir(subset_dir))} files)")
    for fname in [const.fname.litdata_others2save_pkl,
                  const.fname.litdata_others2save_json]:
        fpath = os.path.join(output_dir, fname)
        if not os.path.exists(fpath):
            logger.error(f"    ✗ Missing: {fname}")
            ok = False
        else:
            sz = os.path.getsize(fpath) / 1024
            logger.info(f"    ✓ {fname} ({sz:.1f} KB)")
    for fname in [const.fname.label_class_onehot, "gene_cv_weights.csv"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            sz = os.path.getsize(fpath) / 1024
            logger.info(f"    ✓ {fname} ({sz:.1f} KB)")
        else:
            logger.warning(f"    △ Recommended file is missing: {fname}")
    pkl_path = os.path.join(output_dir, const.fname.litdata_others2save_pkl)
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as f:
                meta = pickle.load(f)
            if "dict_node_names" not in meta or "output_g_label_dim" not in meta:
                logger.error(f"    ✗ Incomplete pkl content: required fields are missing")
                ok = False
            else:
                logger.info(f"    ✓ pkl content: {len(meta['dict_node_names'])} genes, "
                            f"label_dim={meta['output_g_label_dim']}")
        except Exception as e:
            logger.error(f"    ✗ Failed to read pkl file: {e}")
            ok = False
    if ok:
        logger.success(f"  [{tissue_name}] output validation passed ✓")
    else:
        logger.error(f"  [{tissue_name}] output validation failed ✗ — Run02 may not be able to load this directory correctly")
    return ok


# ============================================================================
# 9. Main workflow
# ============================================================================

def main():
    args = parse_args()
    tissue_name = args.tissue_name

    # [BUGFIX-3] Print the DeepTAN module path once in the main process only.
    logger.info(f"DeepTAN module path: {const.__file__}")

    logger.info("=" * 70)
    logger.info(f"  Tissue fine-tuning data preparation (reuse pretrained graph structure; no NMIC recomputation)")
    logger.info(f"  Tissue/run name: {tissue_name}")
    logger.info(f"  Input: trn={args.trn_parquet}")
    logger.info(f"        val={args.val_parquet}")
    logger.info(f"        tst={args.tst_parquet}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix=f"deeptan_finetune_{tissue_name}_")
    logger.info(f"  Temporary directory: {tmpdir}")

    try:
        _run_pipeline(args, tissue_name, tmpdir)
    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.info(f"  Temporary directory removed: {tmpdir}")


def _run_pipeline(args, tissue_name: str, tmpdir: str):

    # ====================================================================
    # Step 1-2: Load pretraining resources
    # ====================================================================
    pretrained_meta = load_pretrained_meta(args.pretrained_pkl)
    pretrained_dict = pretrained_meta["dict_node_names"]
    graph = load_pretrained_graph(args.pretrained_trn_npz)

    # ====================================================================
    # Step 3: Training split -> alignment -> temporary trn.npz + trn.parquet
    # ====================================================================
    logger.info(f"\n{'─' * 50}")
    logger.info(f"  Step 3: training split — load, align, and build temporary trn.npz / trn.parquet")
    logger.info(f"{'─' * 50}")

    trn_data = load_parquet_expression(
        args.trn_parquet, celltype_col=args.celltype_col, subset_name="trn",
    )
    trn_aligned, trn_report = align_genes_to_pretrained(
        trn_data["mat"], trn_data["gene_names"],
        pretrained_dict, subset_name="trn", tissue_name=tissue_name,
    )

    # [BUGFIX-2] Save npz fields and shapes expected by read_nmic_npz().
    trn_npz_path = os.path.join(tmpdir, "trn.npz")
    build_finetune_npz(
        trn_aligned, trn_data["obs_names"],
        graph, pretrained_dict, trn_npz_path,
    )

    # [BUGFIX-1] Also create trn.parquet as the companion file for read_nmic_npz().
    trn_pq_path = os.path.join(tmpdir, "trn.parquet")
    build_finetune_parquet(
        trn_aligned, trn_data["obs_names"],
        pretrained_dict, trn_pq_path,
    )
    logger.info(f"    [BUGFIX-1] ✓ trn.parquet created as the companion file for read_nmic_npz")

    # ====================================================================
    # Step 4: Validation split -> alignment -> temporary val.parquet
    # ====================================================================
    logger.info(f"\n{'─' * 50}")
    logger.info(f"  Step 4: validation split — load, align, and build temporary val.parquet")
    logger.info(f"{'─' * 50}")

    val_data = load_parquet_expression(
        args.val_parquet, celltype_col=args.celltype_col, subset_name="val",
    )
    val_aligned, val_report = align_genes_to_pretrained(
        val_data["mat"], val_data["gene_names"],
        pretrained_dict, subset_name="val", tissue_name=tissue_name,
    )

    val_pq_path = os.path.join(tmpdir, "val.parquet")
    build_finetune_parquet(
        val_aligned, val_data["obs_names"], pretrained_dict, val_pq_path,
    )

    # ====================================================================
    # Step 5: Test split -> alignment -> temporary tst.parquet
    # ====================================================================
    logger.info(f"\n{'─' * 50}")
    logger.info(f"  Step 5: test split — load, align, and build temporary tst.parquet")
    logger.info(f"{'─' * 50}")

    tst_data = load_parquet_expression(
        args.tst_parquet, celltype_col=args.celltype_col, subset_name="tst",
    )
    tst_aligned, tst_report = align_genes_to_pretrained(
        tst_data["mat"], tst_data["gene_names"],
        pretrained_dict, subset_name="tst", tissue_name=tissue_name,
    )

    tst_pq_path = os.path.join(tmpdir, "tst.parquet")
    build_finetune_parquet(
        tst_aligned, tst_data["obs_names"], pretrained_dict, tst_pq_path,
    )

    # ====================================================================
    # Save the gene-alignment report
    # ====================================================================
    rpt_path = os.path.join(args.output_dir, "gene_alignment_report.json")
    rpt_data = {}
    for label, report in [("trn", trn_report), ("val", val_report), ("tst", tst_report)]:
        rpt_data[label] = {k: v for k, v in report.items()
                           if k not in ("missing_genes", "extra_genes")}
    with open(rpt_path, "w") as f:
        json.dump(rpt_data, f, indent=2)

    if trn_report["missing_genes"]:
        miss_path = os.path.join(args.output_dir, "missing_genes.txt")
        with open(miss_path, "w") as f:
            f.write(f"# [{tissue_name}] Missing {len(trn_report['missing_genes'])} "
                    f"genes (coverage {trn_report['coverage']:.1%})\n\n")
            for g in trn_report["missing_genes"]:
                f.write(f"{g}\n")

    # ====================================================================
    # Step 6: Cell-type annotations -> celltype_onehot.parquet
    # ====================================================================
    logger.info(f"\n{'─' * 50}")
    logger.info(f"  Step 6: cell-type label processing")
    logger.info(f"{'─' * 50}")

    labels_path = None
    tissue_label_dim = None

    if args.labels_parquet and os.path.exists(args.labels_parquet):
        labels_path = os.path.join(args.output_dir, const.fname.label_class_onehot)
        shutil.copy(args.labels_parquet, labels_path)
        lbl_df = pl.read_parquet(labels_path)
        tissue_label_dim = len(lbl_df.columns) - 1
        logger.info(f"  Reusing existing label file: {args.labels_parquet}")
        logger.info(f"  Number of classes: {tissue_label_dim}")
    else:
        all_obs = trn_data["obs_names"] + val_data["obs_names"] + tst_data["obs_names"]
        all_ct = None
        if trn_data["cell_types"] is not None:
            all_ct = list(trn_data["cell_types"])
            if val_data["cell_types"] is not None:
                all_ct += list(val_data["cell_types"])
            if tst_data["cell_types"] is not None:
                all_ct += list(tst_data["cell_types"])
        elif args.celltype_csv:
            all_ct = load_celltype_from_csv(args.celltype_csv, all_obs)

        if all_ct is not None and len(all_ct) == len(all_obs):
            labels_path = os.path.join(args.output_dir, const.fname.label_class_onehot)
            labels_path, tissue_label_dim = build_celltype_onehot(
                all_ct, all_obs, labels_path,
            )
        else:
            logger.warning(
                f"  No cell-type annotation was found."
                f"If classification fine-tuning is needed, provide labels through --celltype_col / --celltype_csv / "
                f"--labels_parquet."
            )

    # ====================================================================
    # Step 7: Save litdata_others2save metadata
    # ====================================================================
    logger.info(f"\n{'─' * 50}")
    logger.info(f"  Step 7: save metadata to litdata_others2save")
    logger.info(f"{'─' * 50}")

    others = {
        "dict_node_names": pretrained_dict,
        "output_g_label_dim": (
            tissue_label_dim if tissue_label_dim
            else pretrained_meta["output_g_label_dim"]
        ),
    }

    pkl_out = os.path.join(args.output_dir, const.fname.litdata_others2save_pkl)
    json_out = os.path.join(args.output_dir, const.fname.litdata_others2save_json)
    with open(pkl_out, "wb") as f:
        pickle.dump(others, f)
    with open(json_out, "w") as f:
        json.dump(others, f)

    logger.info(f"  dict_node_names: {len(pretrained_dict)} genes from pretraining")
    logger.info(f"  output_g_label_dim: {others['output_g_label_dim']}")

    # ====================================================================
    # Step 8: DeepTANDataModule -> LitData conversion
    # ====================================================================
    logger.info(f"\n{'─' * 50}")
    logger.info(f"  Step 8: LitData conversion")
    logger.info(f"{'─' * 50}")

    files_fit = {
        const.dkey.abbr_train: trn_npz_path,
        const.dkey.abbr_val:   val_pq_path,
        const.dkey.abbr_test:  tst_pq_path,
    }

    dm = DeepTANDataModule(
        files_fit, labels_path,
        batch_size=args.bs,
        edge_attr_threshold=args.thre_mi,
    )
    dm.setup()

    nw = min(args.n_workers, const.default.n_threads)

    for subset_label, dataset_obj, abbr_key in [
        ("training split", dm.train, const.dkey.abbr_train),
        ("validation split", dm.val,   const.dkey.abbr_val),
        ("test split", dm.test,  const.dkey.abbr_test),
    ]:
        n_items = dataset_obj.len()
        logger.info(f"  {subset_label} → LitData ({n_items} samples, {nw} workers) ...")
        litdata.optimize(
            fn=dataset_obj.get,
            inputs=list(range(n_items)),
            output_dir=os.path.join(args.output_dir, abbr_key),
            chunk_bytes=const.default.lit_chunk_bytes,
            compression=const.default.lit_compression,
            num_workers=nw,
        )
        logger.success(f"  {subset_label} LitData conversion completed ✓ ({n_items} samples)")

    # ====================================================================
    # Step 9: CV weights
    # ====================================================================
    if not args.skip_cv:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"  Step 9: gene-level CV weights")
        logger.info(f"{'─' * 50}")
        compute_cv_weights(
            trn_npz_path, pretrained_dict,
            os.path.join(args.output_dir, "gene_cv_weights.csv"),
        )

    # ====================================================================
    # Step 10: Validate outputs
    # ====================================================================
    if args.validate_output:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"  Step 10: output validation")
        logger.info(f"{'─' * 50}")
        validate_output(args.output_dir, tissue_name)

    # ====================================================================
    # Final summary
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f"  [{tissue_name}] fine-tuning data preparation completed")
    logger.info("=" * 70)
    logger.info(f"  Output directory:        {args.output_dir}")
    logger.info(f"  Data splits:             trn + val + tst")
    logger.info(f"  dict_node_names:         {len(pretrained_dict)} genes from pretraining")
    logger.info(f"  Gene coverage:           trn={trn_report['coverage']:.1%}, "
                f"val={val_report['coverage']:.1%}, "
                f"tst={tst_report['coverage']:.1%}")
    logger.info(f"  Tissue label_dim:        {tissue_label_dim}")
    logger.info(f"  Pretraining label_dim:   {pretrained_meta['output_g_label_dim']}")

    if tissue_label_dim and \
       tissue_label_dim != pretrained_meta["output_g_label_dim"]:
        logger.info(
            f"  → Fine-tuning will trigger FIX-7b output-layer adaptation "
            f"({pretrained_meta['output_g_label_dim']} → {tissue_label_dim})"
        )

    logger.info(f"\n  Output files:")
    for item in sorted(os.listdir(args.output_dir)):
        p = os.path.join(args.output_dir, item)
        if os.path.isdir(p):
            n = len(os.listdir(p))
            logger.info(f"    📁 {item}/ ({n} files)")
        else:
            sz = os.path.getsize(p) / 1024
            logger.info(f"    📄 {item} ({sz:.1f} KB)")

    logger.info(f"\n  Next step — add the following entry to config.yaml:")
    logger.info(f"    tissues:")
    logger.info(f"      {tissue_name}:")
    logger.info(f"        litdata: \"{args.output_dir}\"")
    logger.info(f"        cv_weights: \"{os.path.join(args.output_dir, 'gene_cv_weights.csv')}\"")
    if labels_path:
        logger.info(f"        class_weights_parquet: \"{labels_path}\"")
    logger.info(f"\n    Then run: python script1_finetune.py --config config.yaml")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
