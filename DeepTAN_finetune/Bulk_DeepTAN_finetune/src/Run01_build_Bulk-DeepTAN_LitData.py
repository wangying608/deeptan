#!/usr/bin/env python3
r"""
==============================================================================
Run01: Build Trait-Aware Bulk LitData for BulkExpand-DeepTAN Fine-Tuning
==============================================================================

Purpose
-------
Convert user-provided bulk expression splits, a bulk NMIC guide graph, phenotype
labels, and scRNA-pretrained DeepTAN graph resources into model-ready LitData for
trait-aware BulkExpand-DeepTAN fine-tuning.

Pipeline position
-----------------
Run01: user-defined bulk files -> bulk-expanded LitData
Run02: LitData -> trait-aware bulk fine-tuned DeepTAN checkpoint
Run03: LitData + fine-tuned checkpoint -> trait-aware latent gene network
Run04: trait-aware network -> downstream module / hub / gene-set analysis

Core workflow
-------------
This script performs only data construction. It does not train a model.

Main steps:
  1. Use the user-specified train-only bulk NMIC .npz as the main bulk guide graph.
  2. Use the bulk NMIC companion parquet as the authoritative source of
     obs_names and feature names for the .npz file.
  3. Match continuous phenotype labels from the phenotype parquet.
  4. Build an expanded vocabulary from pretrained scRNA genes and bulk features.
  5. Build an expanded guide graph from the bulk train graph, optional scRNA
     pretrained old-old priors, and optional weak KNN edges.
  6. Store each MI/correlation relation once as a canonical undirected edge.
  7. Generate train/validation/test sample-level PyG guide-subgraph samples and
     convert them to LitData.
  8. Apply the configured expression transform before writing GData.x.
  9. Save expanded metadata, feature mappings, edge-source tables, and label maps.

Default graph policy
--------------------
The default mode follows a strict nonzero sample-subgraph strategy. The global
bulk-expanded guide graph defines candidate edges, while each sample keeps only
nodes with nonzero expression and induces edges among those nodes. This avoids
adding zero-expression nodes through coverage enhancement or fallback sampling
unless the user explicitly enables those options.

Input naming policy
-------------------
Input file names are fully user-defined. Users do not need to follow internal
batch naming patterns such as split_42_0.parquet.

Author: DeepTAN Bulk Fine-Tuning Pipeline
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
import sys

# # ── Prefer the local DeepTAN source tree when available ─────────────────────
# _LOCAL_DEEPTAN = "path/deeptan-dev/src"
# if _LOCAL_DEEPTAN not in sys.path:
#     sys.path.insert(0, _LOCAL_DEEPTAN)
# # ────────────────────────────────────────────────────────────────────────────
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import litdata
import numpy as np
import polars as pl
import torch
import yaml
from loguru import logger
from torch_geometric.data import Data as GData
from torch_geometric.utils import subgraph as pyg_subgraph


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


DEFAULT_CONFIG: Dict[str, Any] = {
    "local_deeptan_src": "",
    "pretrained": {
        "metadata_pkl": "",
        "checkpoint": "",
        "old_graph_npz": "",
        "old_graph_weight": 0.30,
        "old_graph_quantile": 0.995,
        "old_graph_mi_threshold": None,
    },
    "data": {
        "split_root": "",
        "nmic_root": "",
        "output_root": "",
        # Optional direct-CLI paths. If provided, they override split_root/nmic_root patterns.
        "direct_split_paths": {},
        "direct_nmic_path": "",
        "nmic_companion_parquet": "",
        # error: companion parquet is authoritative and mismatch stops; warn: record warnings and continue.
        "nmic_companion_check_action": "error",
        # Single-run interface. Run01 writes directly to output_root by default.
        # use_seed_subdir=True keeps the old output_root/seed_<seed> layout.
        "run_id": "bulk_run",
        "use_seed_subdir": False,
        # Legacy pattern-driven multi-seed mode. Leave empty for custom-path single-run usage.
        "seeds": [],
        "split_file_pattern": "split_{seed}_{split}.parquet",
        "nmic_file_pattern": "split_{seed}_0*.npz",
        "splits": {"0": "trn", "1": "val", "2": "tst"},
        "obs_col": "obs_names",
        "phenotype_parquet": "",
        "phenotype_col": "FT16",
        "phenotype_obs_col": "obs_names",
        "metadata_cols": [],
        "exclude_cols": [],
        "ecotype_regex": r"([^_]+)$",
        "drop_unlabeled": True,
        "label_standardize": True,
    },
    "graph": {
        "feat_pairs_index_space": "auto",  # auto | mat_feat_indices | local
        "bulk_mi_threshold": None,
        # By default, do NOT prune the bulk NMIC guide graph.
        # The bulk .npz has already defined the candidate graph; this builder
        # only maps it into the expanded vocabulary and optionally fuses priors.
        # Set these values explicitly only when a smaller graph is required.
        "bulk_edge_quantile": None,
        "bulk_topk_per_node": 0,
        "max_bulk_edges": 0,
        "bulk_edge_weight": 1.0,
        "include_pretrained_old_graph": True,
        "add_knn_for_isolated_new": True,
        "knn_k": 5,
        "knn_min_abs_corr": 0.15,
        "knn_edge_weight_scale": 0.25,
        # Canonical undirected storage is the recommended representation.
        # A biological MI/correlation relation A--B is stored once as
        # (min(A,B), max(A,B)); it is not duplicated as A→B and B→A.
        "edge_representation": "canonical_undirected",
        # Optional biological-priority edge preservation.
        # These pairs are forced into the final bulk guide graph only when optional
        # pruning is enabled. With default no-pruning, a pair already present in
        # the bulk NPZ is naturally retained.
        "force_keep_feature_pairs": [],
        "force_keep_missing_action": "warn",  # warn | error
        "force_keep_edge_source": "bulk_force_keep",
    },
    "dataset": {
        "value_source": "parquet",  # parquet is safest for all splits
        # Match the original DeepTAN Run04 data path: DeepTANDataModule uses
        # if_log1p=True by default, so raw expression values are converted by
        # np.log1p before they become GData.x.  The bulk builder keeps its
        # expanded-vocabulary / sample-subgraph logic, but applies the same
        # expression-scale transform explicitly.
        "x_transform": "log1p",  # none | log1p
        "x_log1p_negative_action": "error",  # error | clip_zero
        "x_clip_min": None,
        "x_clip_max": None,
        # Run04-style bulk fine-tuning mode: build a global bulk-expanded guide graph once,
        # but each sample only receives a sample-level active subgraph induced from that guide graph.
        # This avoids feeding the full 4k+ node / 100k+ edge guide graph into AMSGP for every sample.
        "graph_scope": "sample_subgraph",  # sample_subgraph | fixed_full
        "fixed_full_sort_by_expanded_idx": True,
        # The following options control sample-level subgraph construction.
        # For dense bulk expression, the max_nodes cap is essential; otherwise nonzero_top_abs
        # can degenerate into the full graph. 1200 roughly matches the largest observed Run04 tissue graphs.
        "value_threshold": 1e-8,
        "max_nodes_per_sample": 1200,
        "min_nodes_per_sample": 0,
        "node_selection": "nonzero_top_abs",  # all | top_abs | nonzero_top_abs
        # Strict mode: for nonzero_top_abs, keep only abs(x) > value_threshold nodes.
        # No min-node filling, coverage enhancement, force-include nodes, or top-abs fallback.
        "strict_nonzero_subgraph": True,
        # Coverage-aware enhancement keeps expression-driven sampling as the main
        # signal while adding a small number of rarely selected active nodes and
        # old-new bridge nodes. This improves new/low-frequency node training
        # coverage without reverting to the full fixed graph.
        "coverage_enhancement": False,
        "coverage_fraction": 0.0,
        "bridge_fraction": 0.0,
        "coverage_only_expressed": True,
        "coverage_random_seed": 20260429,
        # Force target genes into every sample-level subgraph when they exist in
        # the selected bulk feature space. Edges among these genes and from them
        # to other selected nodes are included automatically by guide-graph induction.
        "force_include_genes": [],
        "force_include_gene_pairs": [],
        "force_include_missing_action": "warn",  # warn | error
        "force_include_budget_exempt": False,
        "drop_isolated_nodes": False,
        "fallback_to_top_abs_if_empty": False,
        "lit_chunk_bytes": "256MB",
        "lit_compression": "zstd",
        "num_workers": 8,
    },
}


def load_config(path: str) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(cfg, user_cfg)


def setup_deeptan_path(local_src: str):
    if local_src and local_src not in sys.path:
        sys.path.insert(0, local_src)


def _read_optional_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """Return DEFAULT_CONFIG updated by an optional YAML file."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, user_cfg)
    return cfg


def apply_direct_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Map tissue-finetune-style direct CLI arguments into the internal config."""
    direct_args = [
        "pretrained_trn_npz",
        "pretrained_pkl",
        "bulk_trn_npz",
        "bulk_nmic_parquet",
        "nmic_companion_check_action",
        "trn_parquet",
        "val_parquet",
        "tst_parquet",
        "phenotype_parquet",
        "phenotype_col",
        "obs_col",
        "output_dir",
        "run_id",
        "use_seed_subdir",
        "bulk_mi_threshold",
        "bulk_edge_quantile",
        "bulk_topk_per_node",
        "max_bulk_edges",
        "graph_scope",
        "node_selection",
        "strict_nonzero_subgraph",
        "value_threshold",
        "max_nodes_per_sample",
        "min_nodes_per_sample",
        "drop_isolated_nodes",
        "coverage_enhancement",
        "coverage_fraction",
        "bridge_fraction",
        "coverage_only_expressed",
        "coverage_random_seed",
        "force_include_gene",
        "force_include_gene_pair",
        "force_include_missing_action",
        "force_include_budget_exempt",
        "force_keep_edge_pair",
        "x_transform",
        "x_log1p_negative_action",
        "x_clip_min",
        "x_clip_max",
    ]
    if not any(getattr(args, k, None) for k in direct_args):
        return cfg

    if args.pretrained_trn_npz:
        cfg["pretrained"]["old_graph_npz"] = args.pretrained_trn_npz
    if args.pretrained_pkl:
        cfg["pretrained"]["metadata_pkl"] = args.pretrained_pkl
    if args.bulk_trn_npz:
        cfg["data"]["direct_nmic_path"] = args.bulk_trn_npz
    if args.bulk_nmic_parquet:
        cfg["data"]["nmic_companion_parquet"] = args.bulk_nmic_parquet
    if getattr(args, "nmic_companion_check_action", None):
        cfg["data"]["nmic_companion_check_action"] = args.nmic_companion_check_action
    if args.phenotype_parquet:
        cfg["data"]["phenotype_parquet"] = args.phenotype_parquet
    if args.phenotype_col:
        cfg["data"]["phenotype_col"] = args.phenotype_col
    if args.obs_col:
        cfg["data"]["obs_col"] = args.obs_col
        cfg["data"]["phenotype_obs_col"] = args.obs_col
    if args.output_dir:
        cfg["data"]["output_root"] = args.output_dir
    if getattr(args, "run_id", None):
        cfg["data"]["run_id"] = str(args.run_id)
    if getattr(args, "use_seed_subdir", None) is not None:
        cfg["data"]["use_seed_subdir"] = bool(args.use_seed_subdir)

    # Optional graph-filter overrides for direct CLI runs.
    if getattr(args, "bulk_mi_threshold", None) is not None:
        cfg["graph"]["bulk_mi_threshold"] = float(args.bulk_mi_threshold)
    if getattr(args, "bulk_edge_quantile", None) is not None:
        cfg["graph"]["bulk_edge_quantile"] = float(args.bulk_edge_quantile)
        # If an explicit threshold was set in YAML/defaults, quantile should only
        # take effect when the user did not provide --bulk_mi_threshold.
        if getattr(args, "bulk_mi_threshold", None) is None:
            cfg["graph"]["bulk_mi_threshold"] = None
    if getattr(args, "bulk_topk_per_node", None) is not None:
        cfg["graph"]["bulk_topk_per_node"] = int(args.bulk_topk_per_node)
    if getattr(args, "max_bulk_edges", None) is not None:
        cfg["graph"]["max_bulk_edges"] = int(args.max_bulk_edges)
    if getattr(args, "graph_scope", None):
        cfg["dataset"]["graph_scope"] = str(args.graph_scope)
    if getattr(args, "node_selection", None):
        cfg["dataset"]["node_selection"] = str(args.node_selection)
    if getattr(args, "strict_nonzero_subgraph", None) is not None:
        cfg["dataset"]["strict_nonzero_subgraph"] = bool(args.strict_nonzero_subgraph)
    if getattr(args, "value_threshold", None) is not None:
        cfg["dataset"]["value_threshold"] = float(args.value_threshold)
    if getattr(args, "max_nodes_per_sample", None) is not None:
        cfg["dataset"]["max_nodes_per_sample"] = int(args.max_nodes_per_sample)
    if getattr(args, "min_nodes_per_sample", None) is not None:
        cfg["dataset"]["min_nodes_per_sample"] = int(args.min_nodes_per_sample)
    if getattr(args, "drop_isolated_nodes", None) is not None:
        cfg["dataset"]["drop_isolated_nodes"] = bool(args.drop_isolated_nodes)
    if getattr(args, "coverage_enhancement", None) is not None:
        cfg["dataset"]["coverage_enhancement"] = bool(args.coverage_enhancement)
    if getattr(args, "coverage_fraction", None) is not None:
        cfg["dataset"]["coverage_fraction"] = float(args.coverage_fraction)
    if getattr(args, "bridge_fraction", None) is not None:
        cfg["dataset"]["bridge_fraction"] = float(args.bridge_fraction)
    if getattr(args, "coverage_only_expressed", None) is not None:
        cfg["dataset"]["coverage_only_expressed"] = bool(args.coverage_only_expressed)
    if getattr(args, "coverage_random_seed", None) is not None:
        cfg["dataset"]["coverage_random_seed"] = int(args.coverage_random_seed)
    if getattr(args, "force_include_gene", None):
        cfg["dataset"]["force_include_genes"] = [str(x) for x in args.force_include_gene]
    if getattr(args, "force_include_gene_pair", None):
        cfg["dataset"]["force_include_gene_pairs"] = [list(pair) for pair in args.force_include_gene_pair]
        cfg["graph"]["force_keep_feature_pairs"] = [list(pair) for pair in args.force_include_gene_pair]
    if getattr(args, "force_include_missing_action", None):
        cfg["dataset"]["force_include_missing_action"] = str(args.force_include_missing_action)
    if getattr(args, "force_include_budget_exempt", None) is not None:
        cfg["dataset"]["force_include_budget_exempt"] = bool(args.force_include_budget_exempt)
    if getattr(args, "force_keep_edge_pair", None):
        cfg["graph"]["force_keep_feature_pairs"] = [list(pair) for pair in args.force_keep_edge_pair]
    if getattr(args, "x_transform", None):
        cfg["dataset"]["x_transform"] = str(args.x_transform)
    if getattr(args, "x_log1p_negative_action", None):
        cfg["dataset"]["x_log1p_negative_action"] = str(args.x_log1p_negative_action)
    if getattr(args, "x_clip_min", None) is not None:
        cfg["dataset"]["x_clip_min"] = float(args.x_clip_min)
    if getattr(args, "x_clip_max", None) is not None:
        cfg["dataset"]["x_clip_max"] = float(args.x_clip_max)

    split_paths = {}
    if args.trn_parquet:
        split_paths["trn"] = args.trn_parquet
    if args.val_parquet:
        split_paths["val"] = args.val_parquet
    if args.tst_parquet:
        split_paths["tst"] = args.tst_parquet
    if split_paths:
        required = {"trn", "val", "tst"}
        missing = sorted(required - set(split_paths))
        if missing:
            raise ValueError(f"Direct CLI split paths are incomplete. Missing: {missing}")
        cfg["data"]["direct_split_paths"] = split_paths

    if args.seed is not None:
        cfg["data"]["seeds"] = [int(args.seed)]
    return cfg


def validate_direct_cli_config(cfg: Dict[str, Any]) -> None:
    """Fail early with clear messages for direct-CLI runs."""
    direct_split_paths = cfg["data"].get("direct_split_paths") or {}
    if direct_split_paths:
        for split_name in ["trn", "val", "tst"]:
            path = direct_split_paths.get(split_name)
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"Missing direct {split_name} parquet: {path}")
    if cfg["data"].get("direct_nmic_path") and not os.path.exists(cfg["data"]["direct_nmic_path"]):
        raise FileNotFoundError(f"Missing direct bulk NMIC npz: {cfg['data']['direct_nmic_path']}")
    if cfg["data"].get("nmic_companion_parquet") and not os.path.exists(cfg["data"]["nmic_companion_parquet"]):
        raise FileNotFoundError(f"Missing bulk NMIC companion parquet: {cfg['data']['nmic_companion_parquet']}")
    if cfg["pretrained"].get("metadata_pkl") and not os.path.exists(cfg["pretrained"]["metadata_pkl"]):
        raise FileNotFoundError(f"Missing pretrained metadata pkl: {cfg['pretrained']['metadata_pkl']}")
    if cfg["pretrained"].get("old_graph_npz") and not os.path.exists(cfg["pretrained"]["old_graph_npz"]):
        raise FileNotFoundError(f"Missing pretrained train NMIC npz: {cfg['pretrained']['old_graph_npz']}")
    if cfg["data"].get("phenotype_parquet") and not os.path.exists(cfg["data"]["phenotype_parquet"]):
        raise FileNotFoundError(f"Missing phenotype parquet: {cfg['data']['phenotype_parquet']}")


# -----------------------------------------------------------------------------
# Low-level parsing / loading
# -----------------------------------------------------------------------------


def parse_ecotype(obs: Any, pattern: str) -> Optional[str]:
    s = str(obs)
    m = re.search(pattern, s)
    if m:
        return str(m.group(1))
    # conservative fallback: last underscore-delimited token
    if "_" in s:
        return s.split("_")[-1]
    return None


def list_numeric_feature_columns(
    df: pl.DataFrame,
    obs_col: str,
    metadata_cols: Sequence[str],
    exclude_cols: Sequence[str],
) -> List[str]:
    excluded = set([obs_col]) | set(metadata_cols or []) | set(exclude_cols or [])
    out = []
    for c in df.columns:
        if c in excluded:
            continue
        if df[c].dtype.is_numeric():
            out.append(c)
    return out


def resolve_single_file(pattern: str) -> str:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")
    if len(hits) > 1:
        logger.warning(f"Multiple files matched pattern; using latest lexicographic: {hits[-1]}")
    return hits[-1]


def read_bulk_npz(npz_path: str) -> Dict[str, np.ndarray]:
    logger.info(f"Reading bulk NMIC npz: {npz_path}")
    z = np.load(npz_path, allow_pickle=False)
    required = ["mi_values", "feat_pairs", "processed_mat", "mat_feat_indices"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"NPZ missing required keys: {missing}")
    return {k: z[k] for k in z.files}


def summarize_numeric_array(x: np.ndarray, max_sample: int = 2_000_000) -> Dict[str, Any]:
    """Return a compact JSON-serializable numeric summary for internal metadata."""
    arr = np.asarray(x)
    if arr.size == 0:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "n_total": 0,
            "n_finite": 0,
        }

    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)]
    out: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_total": int(flat.size),
        "n_finite": int(finite.size),
    }
    if finite.size == 0:
        return out

    if finite.size > max_sample:
        rng = np.random.default_rng(42)
        sample = rng.choice(finite, size=max_sample, replace=False)
    else:
        sample = finite

    q_probs = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    q_vals = np.quantile(sample.astype(np.float64, copy=False), q_probs)
    out.update(
        {
            "min": float(np.min(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "max": float(np.max(finite)),
            "zero_ratio": float(np.mean(finite == 0)),
            "negative_ratio": float(np.mean(finite < 0)),
            "quantiles": {str(p): float(v) for p, v in zip(q_probs, q_vals)},
        }
    )
    return out


def _format_summary_for_log(s: Dict[str, Any]) -> str:
    if not s or s.get("n_finite", 0) == 0:
        return "empty/non-finite"
    q = s.get("quantiles", {})
    return (
        f"shape={s.get('shape')}, min/mean/std/max="
        f"{s.get('min'):.6g}/{s.get('mean'):.6g}/{s.get('std'):.6g}/{s.get('max'):.6g}, "
        f"zero_ratio={s.get('zero_ratio'):.4f}, neg_ratio={s.get('negative_ratio'):.4f}, "
        f"q95={q.get('0.95', float('nan')):.6g}, q99={q.get('0.99', float('nan')):.6g}"
    )


def transform_expression_matrix(
    x: np.ndarray,
    dataset_cfg: Dict[str, Any],
    context: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the Run04-compatible expression transform before writing GData.x.

    The original DeepTAN Run04 path uses DeepTANDataModule with if_log1p=True,
    so NMICGraphDataset / NMICGraphDatasetRely apply np.log1p to expression
    values before constructing PyG graphs.  This bulk builder does not call
    those classes because it needs expanded-vocabulary and trait-specific
    sample-subgraph logic, so the equivalent transform is applied explicitly.
    """
    transform = str(dataset_cfg.get("x_transform", "log1p") or "none").lower()
    if transform in {"false", "no", "raw"}:
        transform = "none"
    if transform not in {"none", "log1p"}:
        raise ValueError("dataset.x_transform must be one of: none, log1p")

    x_arr = np.asarray(x, dtype=np.float32)
    report: Dict[str, Any] = {
        "context": context,
        "transform": transform,
        "log1p_negative_action": str(dataset_cfg.get("x_log1p_negative_action", "error")),
        "clip_min": dataset_cfg.get("x_clip_min", None),
        "clip_max": dataset_cfg.get("x_clip_max", None),
        "before": summarize_numeric_array(x_arr),
    }


    out = x_arr.copy()
    if transform == "log1p":
        n_negative = int(np.sum(out < 0))
        report["n_negative_before_log1p"] = n_negative
        if n_negative > 0:
            action = str(dataset_cfg.get("x_log1p_negative_action", "error") or "error").lower()
            if action == "clip_zero":
                logger.warning(
                    f"Expression matrix [{context}] has {n_negative} negative values before log1p; "
                    "clipping them to 0 because dataset.x_log1p_negative_action='clip_zero'."
                )
                out = np.maximum(out, 0.0)
            elif action == "error":
                raise ValueError(
                    f"Expression matrix [{context}] has {n_negative} negative values; "
                    "np.log1p expects non-negative expression values. Set "
                    "dataset.x_log1p_negative_action='clip_zero' only if this is intentional."
                )
            else:
                raise ValueError("dataset.x_log1p_negative_action must be 'error' or 'clip_zero'")
        out = np.log1p(out).astype(np.float32, copy=False)

    clip_min = dataset_cfg.get("x_clip_min", None)
    clip_max = dataset_cfg.get("x_clip_max", None)
    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else float(clip_min)
        hi = np.inf if clip_max is None else float(clip_max)
        if lo > hi:
            raise ValueError(f"Invalid x clipping range: x_clip_min={lo} > x_clip_max={hi}")
        out = np.clip(out, lo, hi).astype(np.float32, copy=False)

    report["after"] = summarize_numeric_array(out)
    return out, report


def load_pretrained_dict(metadata_pkl: str, checkpoint_path: str = "") -> Dict[str, int]:
    if metadata_pkl and os.path.exists(metadata_pkl):
        with open(metadata_pkl, "rb") as f:
            meta = pickle.load(f)
        if "dict_node_names" not in meta:
            raise KeyError(f"metadata_pkl lacks dict_node_names: {metadata_pkl}")
        return dict(meta["dict_node_names"])

    if checkpoint_path:
        # Fallback: load DeepTAN checkpoint to obtain dict_node_names.
        from deeptan.graph.model import DeepTAN  # type: ignore
        model_dir = os.path.dirname(checkpoint_path)
        hparams_path = os.path.join(model_dir, "version_0", "hparams.yaml")
        kwargs = {"map_location": "cpu"}
        if os.path.exists(hparams_path):
            kwargs["hparams_file"] = hparams_path
        model = DeepTAN.load_from_checkpoint(checkpoint_path, **kwargs)
        return dict(model.dict_node_names)

    raise ValueError("Provide pretrained.metadata_pkl or pretrained.checkpoint")


# -----------------------------------------------------------------------------
# Metadata binding and validation
# -----------------------------------------------------------------------------


def infer_processed_matrix_orientation(
    processed_mat: np.ndarray,
    n_rows_parquet: int,
    n_selected_features: int,
) -> str:
    if processed_mat.shape == (n_selected_features, n_rows_parquet):
        return "feature_by_sample"
    if processed_mat.shape == (n_rows_parquet, n_selected_features):
        return "sample_by_feature"
    raise ValueError(
        "Cannot infer processed_mat orientation. "
        f"processed_mat.shape={processed_mat.shape}, parquet rows={n_rows_parquet}, "
        f"n_selected_features={n_selected_features}"
    )


def build_feature_mapping(
    feature_cols: List[str],
    mat_feat_indices: np.ndarray,
    feature_index_mode: str = "full_index",
) -> pl.DataFrame:
    """
    Build mapping from processed/local NMIC feature ids to feature names.

    Parameters
    ----------
    feature_cols
        Numeric feature columns from the selected feature-name source.
    mat_feat_indices
        npz["mat_feat_indices"].  In DeepTAN/NMIC outputs this often stores
        the selected features' indices in the original full feature space.
    feature_index_mode
        full_index:
            feature_cols is a full feature list; feature_name = feature_cols[mat_feat_idx].
        selected_local:
            feature_cols is already the selected feature list in processed order;
            feature_name = feature_cols[processed_local_idx], while mat_feat_idx is retained
            as provenance to the original full feature index.
    """
    if mat_feat_indices.ndim != 1:
        raise ValueError("mat_feat_indices must be 1D")
    feature_index_mode = str(feature_index_mode or "full_index")
    if feature_index_mode not in {"full_index", "selected_local"}:
        raise ValueError("feature_index_mode must be 'full_index' or 'selected_local'")

    rows = []
    mat_idx_list = mat_feat_indices.astype(int).tolist()

    if feature_index_mode == "full_index":
        max_idx = int(mat_feat_indices.max(initial=-1))
        if max_idx >= len(feature_cols):
            raise IndexError(
                f"mat_feat_indices max={max_idx} >= number of numeric feature columns={len(feature_cols)}. "
                "This feature source cannot be used in full_index mode. "
                "If the companion parquet already contains selected features only, use selected_local mode."
            )
        for local_idx, original_idx in enumerate(mat_idx_list):
            rows.append(
                {
                    "processed_local_idx": local_idx,
                    "mat_feat_idx": original_idx,
                    "feature_name": feature_cols[original_idx],
                    "feature_index_mode": feature_index_mode,
                    "feature_source_col_idx": original_idx,
                    "original_parquet_feature_col_idx": original_idx,
                }
            )
        return pl.DataFrame(rows)

    # selected_local mode: companion parquet is already the selected matrix.
    # This is the common case when nmicg8/split_*_timestamp.parquet has
    # obs_names + len(mat_feat_indices) selected feature columns.
    if len(feature_cols) != len(mat_idx_list):
        raise ValueError(
            "selected_local mode requires number of numeric feature columns to equal len(mat_feat_indices): "
            f"n_feature_cols={len(feature_cols)}, len(mat_feat_indices)={len(mat_idx_list)}"
        )
    for local_idx, original_idx in enumerate(mat_idx_list):
        rows.append(
            {
                "processed_local_idx": local_idx,
                "mat_feat_idx": original_idx,
                "feature_name": feature_cols[local_idx],
                "feature_index_mode": feature_index_mode,
                "feature_source_col_idx": local_idx,
                "original_parquet_feature_col_idx": original_idx,
            }
        )
    return pl.DataFrame(rows)


def _handle_companion_check(ok: bool, msg: str, action: str, info: Dict[str, Any]) -> None:
    """Handle companion parquet checks with either error or warning semantics."""
    if ok:
        return
    info.setdefault("warnings", []).append(msg)
    if action == "error":
        raise ValueError(msg)
    logger.warning(msg)


def _check_selected_feature_presence(
    selected_feature_names: List[str],
    split_dfs: Dict[str, pl.DataFrame],
    check_action: str,
    info: Dict[str, Any],
    source_label: str,
) -> None:
    """Verify that selected features can be read by name from trn/val/tst parquets."""
    for split_name, df in split_dfs.items():
        missing = [c for c in selected_feature_names if c not in df.columns]
        info.setdefault("selected_feature_presence", {})[split_name] = {
            "n_selected": len(selected_feature_names),
            "n_missing": len(missing),
            "missing_examples": missing[:20],
        }
        _handle_companion_check(
            len(missing) == 0,
            f"{split_name} parquet is missing {len(missing)} selected features inferred from {source_label}. "
            f"Examples: {missing[:10]}",
            check_action,
            info,
        )


def validate_and_select_nmic_feature_source(
    npz: Dict[str, np.ndarray],
    train_df: pl.DataFrame,
    split_dfs: Dict[str, pl.DataFrame],
    companion_path: str,
    obs_col: str,
    metadata_cols: Sequence[str],
    exclude_cols: Sequence[str],
    check_action: str = "error",
) -> Tuple[pl.DataFrame, str, List[str], Dict[str, Any]]:
    """
    Resolve the correct feature-name source for a bulk NMIC .npz.

    The bulk NMIC .npz stores numeric arrays only. It usually does not store
    obs_names or feature names.  Two valid metadata layouts are supported:

    1. full_index companion / train parquet
       Numeric columns define the original full feature space, so
       feature_name = feature_cols[mat_feat_idx].

    2. selected_local companion parquet
       Numeric columns already contain exactly the selected features in
       processed_mat order, so feature_name = companion_feature_cols[local_idx]
       and mat_feat_idx is retained only as the original full-space index.

    This second case is what happens when nmicg8/split_*_timestamp.parquet has
    obs_names + len(npz["mat_feat_indices"]) selected feature columns.
    """
    check_action = str(check_action or "error").lower()
    if check_action not in {"error", "warn"}:
        raise ValueError("data.nmic_companion_check_action must be 'error' or 'warn'")

    mat_feat_indices = npz["mat_feat_indices"].astype(int)
    n_selected = int(mat_feat_indices.shape[0])
    max_idx = int(mat_feat_indices.max(initial=-1))

    info: Dict[str, Any] = {
        "provided": bool(companion_path),
        "path": companion_path or "",
        "check_action": check_action,
        "feature_source": "trn_parquet",
        "feature_index_mode": "full_index",
        "warnings": [],
        "n_selected_from_mat_feat_indices": n_selected,
        "max_mat_feat_index": max_idx,
    }

    train_feature_cols = list_numeric_feature_columns(train_df, obs_col, metadata_cols, exclude_cols)
    if not train_feature_cols:
        raise ValueError("No numeric feature columns identified in trn_parquet")
    info["train_numeric_feature_count"] = len(train_feature_cols)
    info["train_can_map_mat_feat_indices_as_full_index"] = bool(max_idx < len(train_feature_cols))

    if not companion_path:
        _handle_companion_check(
            max_idx < len(train_feature_cols),
            "No bulk_nmic_parquet was provided and trn_parquet cannot interpret npz['mat_feat_indices']: "
            f"max(mat_feat_indices)={max_idx}, train_numeric_feature_count={len(train_feature_cols)}.",
            check_action,
            info,
        )
        selected_feature_names = [train_feature_cols[int(i)] for i in mat_feat_indices.tolist()]
        _check_selected_feature_presence(selected_feature_names, split_dfs, check_action, info, "trn_parquet")
        info.update(
            {
                "used_as_feature_source": False,
                "reason": "No bulk_nmic_parquet was provided; trn_parquet is used in full_index mode.",
                "feature_source": "trn_parquet",
                "feature_index_mode": "full_index",
            }
        )
        return train_df, "trn_parquet", train_feature_cols, info

    if not os.path.exists(companion_path):
        raise FileNotFoundError(f"Missing bulk NMIC companion parquet: {companion_path}")

    companion_df = pl.read_parquet(companion_path)
    companion_feature_cols = list_numeric_feature_columns(companion_df, obs_col, metadata_cols, exclude_cols)
    info.update(
        {
            "companion_rows": companion_df.height,
            "train_rows": train_df.height,
            "companion_numeric_feature_count": len(companion_feature_cols),
        }
    )

    # The companion parquet should carry sample ids because the npz does not.
    has_obs = obs_col in companion_df.columns
    info["companion_has_obs_col"] = bool(has_obs)
    _handle_companion_check(
        has_obs,
        f"bulk_nmic_parquet lacks obs_col={obs_col!r}; it cannot serve as full npz metadata.",
        check_action,
        info,
    )

    # It should correspond to the same train split used to build the NMIC graph.
    same_n_rows = companion_df.height == train_df.height
    info["companion_train_n_rows_match"] = bool(same_n_rows)
    _handle_companion_check(
        same_n_rows,
        f"bulk_nmic_parquet row count ({companion_df.height}) != trn_parquet row count ({train_df.height}).",
        check_action,
        info,
    )

    if has_obs and obs_col in train_df.columns and same_n_rows:
        comp_obs = [str(x) for x in companion_df[obs_col].to_list()]
        train_obs = [str(x) for x in train_df[obs_col].to_list()]
        same_order = comp_obs == train_obs
        same_set = set(comp_obs) == set(train_obs)
        info["companion_train_obs_same_order"] = bool(same_order)
        info["companion_train_obs_same_set"] = bool(same_set)
        if not same_order:
            examples = []
            for i, (a, b) in enumerate(zip(comp_obs, train_obs)):
                if a != b:
                    examples.append({"idx": i, "companion": a, "train": b})
                if len(examples) >= 5:
                    break
            info["obs_order_mismatch_examples"] = examples
            _handle_companion_check(
                False,
                "bulk_nmic_parquet obs_names are not in the same order as trn_parquet obs_names. "
                f"Examples: {examples}",
                check_action,
                info,
            )

    companion_full_index = bool(companion_feature_cols) and max_idx < len(companion_feature_cols)
    companion_selected_local = bool(companion_feature_cols) and len(companion_feature_cols) == n_selected
    info["companion_can_map_mat_feat_indices_as_full_index"] = bool(companion_full_index)
    info["companion_is_selected_feature_table"] = bool(companion_selected_local)

    if companion_full_index:
        feature_index_mode = "full_index"
        selected_feature_names = [companion_feature_cols[int(i)] for i in mat_feat_indices.tolist()]
        source_explanation = (
            "bulk_nmic_parquet has enough numeric columns to interpret mat_feat_indices as original full-space indices."
        )
    elif companion_selected_local:
        feature_index_mode = "selected_local"
        selected_feature_names = companion_feature_cols
        source_explanation = (
            "bulk_nmic_parquet numeric columns equal len(mat_feat_indices), so it is treated as the selected-feature "
            "matrix in processed_mat/local order. feature_name is taken by processed_local_idx; mat_feat_idx is retained "
            "as original full-space provenance."
        )

        # Optional but useful: if train parquet can interpret mat_feat_indices, check whether the selected companion
        # feature names agree with train feature names indexed by mat_feat_indices.
        if max_idx < len(train_feature_cols):
            train_names_by_mat_idx = [train_feature_cols[int(i)] for i in mat_feat_indices.tolist()]
            same_as_train_by_mat_idx = train_names_by_mat_idx == selected_feature_names
            info["selected_companion_matches_train_by_mat_feat_indices"] = bool(same_as_train_by_mat_idx)
            if not same_as_train_by_mat_idx:
                examples = []
                for i, (a, b) in enumerate(zip(selected_feature_names, train_names_by_mat_idx)):
                    if a != b:
                        examples.append(
                            {
                                "processed_local_idx": i,
                                "mat_feat_idx": int(mat_feat_indices[i]),
                                "companion_feature": a,
                                "train_feature_by_mat_idx": b,
                            }
                        )
                    if len(examples) >= 10:
                        break
                info["selected_name_mismatch_examples"] = examples
                _handle_companion_check(
                    False,
                    "bulk_nmic_parquet appears to be a selected-feature table, but its feature names do not match "
                    "trn_parquet feature names indexed by mat_feat_indices. "
                    f"Examples: {examples}",
                    check_action,
                    info,
                )
    else:
        msg = (
            "bulk_nmic_parquet cannot be interpreted as either a full-index metadata table or a selected-feature table: "
            f"max(mat_feat_indices)={max_idx}, len(mat_feat_indices)={n_selected}, "
            f"companion_numeric_feature_count={len(companion_feature_cols)}. "
            "Expected either companion_numeric_feature_count > max(mat_feat_indices) for full_index mode, "
            "or companion_numeric_feature_count == len(mat_feat_indices) for selected_local mode."
        )
        _handle_companion_check(False, msg, check_action, info)
        # Fallback only for warn mode: use train full feature space if possible.
        _handle_companion_check(
            max_idx < len(train_feature_cols),
            "Fallback trn_parquet also cannot interpret mat_feat_indices as full_index: "
            f"max(mat_feat_indices)={max_idx}, train_numeric_feature_count={len(train_feature_cols)}.",
            check_action,
            info,
        )
        feature_index_mode = "full_index"
        selected_feature_names = [train_feature_cols[int(i)] for i in mat_feat_indices.tolist()]
        info["used_as_feature_source"] = False
        info["feature_source"] = "trn_parquet"
        info["feature_index_mode"] = feature_index_mode
        info["reason"] = "bulk_nmic_parquet incompatible; falling back to trn_parquet because check_action='warn'."
        _check_selected_feature_presence(selected_feature_names, split_dfs, check_action, info, "fallback trn_parquet")
        return train_df, "trn_parquet", train_feature_cols, info

    _check_selected_feature_presence(selected_feature_names, split_dfs, check_action, info, "bulk_nmic_parquet")

    full_feature_order_match = companion_feature_cols == train_feature_cols
    info["companion_train_full_numeric_feature_order_match"] = bool(full_feature_order_match)
    if not full_feature_order_match:
        logger.warning(
            "bulk_nmic_parquet numeric feature columns are not identical to trn_parquet numeric feature columns. "
            f"Using bulk_nmic_parquet as feature source in {feature_index_mode!r} mode; "
            "sample values will be read from trn/val/tst parquets by selected feature names."
        )

    info["used_as_feature_source"] = True
    info["feature_source"] = "bulk_nmic_parquet"
    info["feature_index_mode"] = feature_index_mode
    info["source_explanation"] = source_explanation
    return companion_df, "bulk_nmic_parquet", companion_feature_cols, info

def match_ft16_labels(
    split_dfs: Dict[str, pl.DataFrame],
    phenotype_df: pl.DataFrame,
    obs_col: str,
    phenotype_obs_col: str,
    phenotype_col: str,
    ecotype_regex: str,
) -> Tuple[Dict[str, pl.DataFrame], pl.DataFrame, Dict[str, Any]]:
    if phenotype_obs_col not in phenotype_df.columns or phenotype_col not in phenotype_df.columns:
        raise KeyError(f"Phenotype parquet must contain {phenotype_obs_col!r} and {phenotype_col!r}")

    ph = phenotype_df.select([phenotype_obs_col, phenotype_col]).with_columns(
        pl.col(phenotype_obs_col)
        .map_elements(lambda x: parse_ecotype(x, ecotype_regex), return_dtype=pl.Utf8)
        .alias("ecotype_id")
    )
    # If duplicate phenotype entries exist for an ecotype, use their mean and record counts.
    ph_agg = ph.group_by("ecotype_id").agg(
        pl.col(phenotype_col).cast(pl.Float64).mean().alias("FT16"),
        pl.len().alias("n_phenotype_records"),
    )

    out: Dict[str, pl.DataFrame] = {}
    label_tables = []
    info = {"splits": {}}
    for split_name, df in split_dfs.items():
        if obs_col not in df.columns:
            raise KeyError(f"{split_name} parquet lacks obs_col={obs_col!r}")
        lab = df.select([obs_col]).with_row_count("sample_idx").with_columns(
            pl.col(obs_col).map_elements(lambda x: parse_ecotype(x, ecotype_regex), return_dtype=pl.Utf8).alias("ecotype_id")
        )
        lab = lab.join(ph_agg, on="ecotype_id", how="left")
        n = lab.height
        n_missing = lab.filter(pl.col("FT16").is_null()).height
        info["splits"][split_name] = {
            "n_samples": n,
            "n_missing_ft16": n_missing,
            "ft16_match_rate": float((n - n_missing) / max(n, 1)),
            "n_unique_ecotypes": lab.select("ecotype_id").n_unique(),
        }
        out[split_name] = lab
        label_tables.append(lab.with_columns(pl.lit(split_name).alias("split")))

    labels_all = pl.concat(label_tables, how="vertical")
    return out, labels_all, info


# -----------------------------------------------------------------------------
# Expanded vocabulary and guide graph
# -----------------------------------------------------------------------------


def build_expanded_vocabulary(old_dict: Dict[str, int], bulk_feature_names: List[str]) -> Tuple[Dict[str, int], pl.DataFrame]:
    old_by_index = [None] * len(old_dict)
    for name, idx in old_dict.items():
        old_by_index[int(idx)] = name
    if any(x is None for x in old_by_index):
        raise ValueError("old_dict_node_names indices are not contiguous from 0 to n_old-1")

    expanded_names = list(old_by_index)
    seen = set(expanded_names)
    rows = []
    for name in expanded_names:
        rows.append({"feature_name": name, "expanded_idx": old_dict[name], "old_or_new": "old"})

    for name in bulk_feature_names:
        if name not in seen:
            seen.add(name)
            expanded_names.append(name)
            rows.append({"feature_name": name, "expanded_idx": len(expanded_names) - 1, "old_or_new": "new"})

    expanded_dict = {name: i for i, name in enumerate(expanded_names)}
    return expanded_dict, pl.DataFrame(rows)


def infer_feat_pair_index_space(
    feat_pairs: np.ndarray,
    mat_feat_indices: np.ndarray,
    cfg_value: str,
    sample_n: int = 200000,
) -> str:
    if cfg_value in {"mat_feat_indices", "local"}:
        return cfg_value
    if cfg_value != "auto":
        raise ValueError("graph.feat_pairs_index_space must be auto, mat_feat_indices, or local")

    flat = feat_pairs.reshape(-1)
    if flat.size > sample_n:
        rng = np.random.default_rng(17)
        flat = flat[rng.choice(flat.size, size=sample_n, replace=False)]
    mat_set = set(mat_feat_indices.astype(int).tolist())
    frac_in_mat = float(np.mean([int(x) in mat_set for x in flat])) if flat.size else 0.0
    max_pair = int(feat_pairs.max(initial=-1))
    if frac_in_mat > 0.99:
        return "mat_feat_indices"
    if max_pair < len(mat_feat_indices):
        return "local"
    raise ValueError(
        "Could not infer feat_pairs index space. "
        f"frac_in_mat_feat_indices={frac_in_mat:.4f}, max_pair={max_pair}, n_local={len(mat_feat_indices)}"
    )


def build_lookup_for_bulk_features(
    feature_mapping: pl.DataFrame,
    expanded_dict: Dict[str, int],
    index_space: str,
) -> np.ndarray:
    if index_space == "local":
        max_key = int(feature_mapping["processed_local_idx"].max())
        key_col = "processed_local_idx"
    else:
        max_key = int(feature_mapping["mat_feat_idx"].max())
        key_col = "mat_feat_idx"
    lookup = np.full(max_key + 1, -1, dtype=np.int64)
    for row in feature_mapping.iter_rows(named=True):
        lookup[int(row[key_col])] = int(expanded_dict[row["feature_name"]])
    return lookup


def reduce_edges_topk(
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility helper for directed edge tables.

    The bulk MI graph is biologically undirected.  New code should prefer
    ``reduce_undirected_edges_topk`` so that top-k pruning cannot keep only one
    direction of an undirected MI relationship.
    """
    if topk <= 0 or src.size == 0:
        return src, dst, w
    order = np.lexsort((-w, src))
    src_o, dst_o, w_o = src[order], dst[order], w[order]
    keep = np.zeros(src_o.size, dtype=bool)
    last = None
    c = 0
    for i, s in enumerate(src_o):
        if last is None or s != last:
            last = s
            c = 0
        if c < topk:
            keep[i] = True
            c += 1
    return src_o[keep], dst_o[keep], w_o[keep]


def canonicalize_edge_arrays(
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert edge arrays into single-copy undirected canonical pairs.

    Returns ``u, v, w`` where ``u < v``. Self-loops are removed.  If the same
    undirected pair occurs more than once within the same source, the maximum
    weight is retained.  This is appropriate for MI/correlation-derived guide
    graphs because duplicated orientations of the same relation should not
    double-count biological evidence.
    """
    src = src.astype(np.int64, copy=False)
    dst = dst.astype(np.int64, copy=False)
    w = w.astype(np.float32, copy=False)
    valid = src != dst
    if not valid.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    src = src[valid]
    dst = dst[valid]
    w = w[valid]
    u = np.minimum(src, dst)
    v = np.maximum(src, dst)

    order = np.lexsort((v, u))
    u_o, v_o, w_o = u[order], v[order], w[order]
    if u_o.size == 0:
        return u_o, v_o, w_o

    pair_change = np.ones(u_o.size, dtype=bool)
    pair_change[1:] = (u_o[1:] != u_o[:-1]) | (v_o[1:] != v_o[:-1])
    starts = np.flatnonzero(pair_change)
    w_max = np.maximum.reduceat(w_o, starts)
    return u_o[starts], v_o[starts], w_max.astype(np.float32, copy=False)


def reduce_undirected_edges_topk(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k pruning for undirected single-copy edge arrays.

    An undirected edge is retained if it is among the top-k strongest incident
    edges of either endpoint. The result remains a single-copy canonical edge
    table.
    """
    if topk <= 0 or u.size == 0:
        return u, v, w

    eidx = np.arange(u.size, dtype=np.int64)
    incident_node = np.concatenate([u, v])
    incident_edge = np.concatenate([eidx, eidx])
    incident_weight = np.concatenate([w, w])

    order = np.lexsort((-incident_weight, incident_node))
    keep_edge = np.zeros(u.size, dtype=bool)
    last_node = None
    count = 0
    for pos in order:
        node = int(incident_node[pos])
        if last_node is None or node != last_node:
            last_node = node
            count = 0
        if count < topk:
            keep_edge[int(incident_edge[pos])] = True
            count += 1

    return u[keep_edge], v[keep_edge], w[keep_edge]


def classify_edge_type(s: int, d: int, old_indices: set[int]) -> str:
    if s in old_indices and d in old_indices:
        return "old-old"
    if s in old_indices or d in old_indices:
        return "old-new"
    return "new-new"


def make_edge_table_from_undirected_arrays(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    expanded_dict: Dict[str, int],
    old_dict: Dict[str, int],
    edge_source: str,
) -> pl.DataFrame:
    """Create a canonical undirected edge table from ``u < v`` arrays."""
    if u.size == 0:
        return pl.DataFrame(
            {
                "src_expanded_idx": pl.Series([], dtype=pl.Int64),
                "dst_expanded_idx": pl.Series([], dtype=pl.Int64),
                "src": pl.Series([], dtype=pl.Utf8),
                "dst": pl.Series([], dtype=pl.Utf8),
                "edge_weight": pl.Series([], dtype=pl.Float32),
                "edge_source": pl.Series([], dtype=pl.Utf8),
                "edge_type": pl.Series([], dtype=pl.Utf8),
            }
        )

    idx_to_name = {v0: k for k, v0 in expanded_dict.items()}
    old_indices = set(int(v0) for v0 in old_dict.values())
    return pl.DataFrame(
        {
            "src_expanded_idx": u.astype(np.int64, copy=False),
            "dst_expanded_idx": v.astype(np.int64, copy=False),
            "src": [idx_to_name[int(x)] for x in u.tolist()],
            "dst": [idx_to_name[int(x)] for x in v.tolist()],
            "edge_weight": w.astype(np.float32, copy=False),
            "edge_source": [edge_source] * int(u.size),
            "edge_type": [classify_edge_type(int(s), int(d), old_indices) for s, d in zip(u.tolist(), v.tolist())],
        }
    )


def ensure_canonical_undirected_edge_table(df: pl.DataFrame) -> pl.DataFrame:
    """Return a normalized single-copy canonical undirected edge table.

    A MI/correlation-derived relation is biologically undirected, but it is stored
    once as ``src_expanded_idx < dst_expanded_idx`` to avoid duplicating evidence.
    This matches the canonical ``feat_pairs`` representation used by the DeepTAN
    NMIC npz files and by the original run_04 LitData pathway.
    """
    df = normalize_edge_table_schema(df).filter(pl.col("src_expanded_idx") != pl.col("dst_expanded_idx"))
    if df.height == 0:
        return df
    return df.with_columns(
        pl.min_horizontal("src_expanded_idx", "dst_expanded_idx").alias("_u"),
        pl.max_horizontal("src_expanded_idx", "dst_expanded_idx").alias("_v"),
        pl.when(pl.col("src_expanded_idx") <= pl.col("dst_expanded_idx"))
        .then(pl.col("src"))
        .otherwise(pl.col("dst"))
        .alias("_u_name"),
        pl.when(pl.col("src_expanded_idx") <= pl.col("dst_expanded_idx"))
        .then(pl.col("dst"))
        .otherwise(pl.col("src"))
        .alias("_v_name"),
    ).select(
        pl.col("_u").alias("src_expanded_idx"),
        pl.col("_v").alias("dst_expanded_idx"),
        pl.col("_u_name").alias("src"),
        pl.col("_v_name").alias("dst"),
        "edge_weight",
        "edge_source",
        "edge_type",
    ).sort(["src_expanded_idx", "dst_expanded_idx"])


def _normalize_force_keep_pairs(pairs: Any) -> List[Tuple[str, str]]:
    """Normalize graph.force_keep_feature_pairs into [(src_name, dst_name), ...]."""
    if not pairs:
        return []
    out: List[Tuple[str, str]] = []
    for item in pairs:
        if isinstance(item, str):
            # Accept "A,B", "A:B", or "A--B".
            if "--" in item:
                a, b = item.split("--", 1)
            elif "," in item:
                a, b = item.split(",", 1)
            elif ":" in item:
                a, b = item.split(":", 1)
            else:
                raise ValueError(
                    "force_keep_feature_pairs string entries must use 'A--B', 'A,B', or 'A:B'. "
                    f"Invalid entry: {item!r}"
                )
            out.append((a.strip(), b.strip()))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]), str(item[1])))
        else:
            raise ValueError(f"Invalid force_keep_feature_pairs entry: {item!r}")
    return [(a, b) for a, b in out if a and b]


def _handle_force_keep_check(ok: bool, msg: str, action: str) -> None:
    action = str(action or "warn").lower()
    if ok:
        return
    if action == "error":
        raise ValueError(msg)
    logger.warning(msg)


def _find_force_keep_mi_values(
    feat_pairs: np.ndarray,
    mi_values: np.ndarray,
    raw_pairs: List[Tuple[int, int]],
    make_undirected: bool,
    chunk: int = 5_000_000,
) -> Dict[Tuple[int, int], float]:
    """Find MI weights for a small set of required raw feature pairs.

    The bulk npz can contain >1e8 candidate pairs, so this scans in chunks.
    For undirected retention, either orientation is accepted and the maximum
    observed MI across orientations is used.
    """
    if not raw_pairs:
        return {}

    query_to_original: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for a, b in raw_pairs:
        query_to_original[(int(a), int(b))] = (int(a), int(b))
        if make_undirected:
            query_to_original[(int(b), int(a))] = (int(a), int(b))

    found: Dict[Tuple[int, int], float] = {}
    n_edges = int(feat_pairs.shape[0])
    for start in range(0, n_edges, chunk):
        end = min(start + chunk, n_edges)
        raw = feat_pairs[start:end]
        # Number of force-keep pairs is expected to be tiny, so this loop is cheap
        # compared with materializing a huge hash table for all edges.
        for query, original in query_to_original.items():
            a, b = query
            mask = (raw[:, 0] == a) & (raw[:, 1] == b)
            if mask.any():
                val = float(np.max(mi_values[start:end][mask]))
                if original not in found or val > found[original]:
                    found[original] = val
        if len(found) == len(raw_pairs):
            # All original pairs have been observed at least once.
            break
    return found


def _build_force_keep_edge_table(
    npz: Dict[str, np.ndarray],
    feature_mapping: pl.DataFrame,
    expanded_dict: Dict[str, int],
    old_dict: Dict[str, int],
    graph_cfg: Dict[str, Any],
    index_space: str,
) -> Optional[pl.DataFrame]:
    """Build canonical undirected rows for biologically prioritized bulk pairs.

    Force-kept pairs are stored as one canonical undirected relation. They are
    not duplicated into two PyG directions.
    """
    pairs = _normalize_force_keep_pairs(graph_cfg.get("force_keep_feature_pairs", []))
    if not pairs:
        return None

    action = str(graph_cfg.get("force_keep_missing_action", "warn")).lower()
    source = str(graph_cfg.get("force_keep_edge_source", "bulk_force_keep"))
    weight_scale = float(graph_cfg.get("bulk_edge_weight", 1.0))

    required_cols = {"feature_name", "processed_local_idx", "mat_feat_idx", "expanded_idx"}
    missing_cols = required_cols - set(feature_mapping.columns)
    if missing_cols:
        raise KeyError(f"feature_mapping lacks required columns for force_keep_feature_pairs: {sorted(missing_cols)}")

    name_to_row = {str(r["feature_name"]): r for r in feature_mapping.iter_rows(named=True)}

    raw_pairs: List[Tuple[int, int]] = []
    pair_records: List[Dict[str, Any]] = []
    for a_name, b_name in pairs:
        ok = True
        if a_name not in name_to_row:
            _handle_force_keep_check(False, f"force_keep feature {a_name!r} is not in selected bulk features.", action)
            ok = False
        if b_name not in name_to_row:
            _handle_force_keep_check(False, f"force_keep feature {b_name!r} is not in selected bulk features.", action)
            ok = False
        if not ok:
            continue

        a_row = name_to_row[a_name]
        b_row = name_to_row[b_name]
        if index_space == "local":
            a_raw = int(a_row["processed_local_idx"])
            b_raw = int(b_row["processed_local_idx"])
        else:
            a_raw = int(a_row["mat_feat_idx"])
            b_raw = int(b_row["mat_feat_idx"])

        raw_pairs.append((a_raw, b_raw))
        pair_records.append(
            {
                "a_name": a_name,
                "b_name": b_name,
                "a_raw": a_raw,
                "b_raw": b_raw,
                "a_expanded": int(a_row["expanded_idx"]),
                "b_expanded": int(b_row["expanded_idx"]),
            }
        )

    if not pair_records:
        return None

    mi_found = _find_force_keep_mi_values(
        npz["feat_pairs"],
        npz["mi_values"].astype(np.float32, copy=False),
        raw_pairs,
        make_undirected=True,
    )

    u_list: List[int] = []
    v_list: List[int] = []
    w_list: List[float] = []
    kept = 0
    for rec in pair_records:
        key = (rec["a_raw"], rec["b_raw"])
        mi = mi_found.get(key)
        if mi is None:
            _handle_force_keep_check(
                False,
                "force_keep pair was not found in npz['feat_pairs']: "
                f"{rec['a_name']}--{rec['b_name']} with raw indices {key}.",
                action,
            )
            continue

        s = int(rec["a_expanded"])
        d = int(rec["b_expanded"])
        if s == d:
            continue
        u_list.append(min(s, d))
        v_list.append(max(s, d))
        w_list.append(float(mi) * weight_scale)
        kept += 1
        logger.info(
            f"Force-kept undirected bulk edge {rec['a_name']}--{rec['b_name']}: "
            f"MI={float(mi):.6g}, index_space={index_space}, raw=({rec['a_raw']}, {rec['b_raw']})"
        )

    if not u_list:
        return None

    u, v, w = canonicalize_edge_arrays(
        np.asarray(u_list, dtype=np.int64),
        np.asarray(v_list, dtype=np.int64),
        np.asarray(w_list, dtype=np.float32),
    )
    logger.info(f"Force-kept {kept}/{len(pair_records)} configured bulk feature pairs ({u.size:,} undirected edges).")
    return make_edge_table_from_undirected_arrays(u, v, w, expanded_dict, old_dict, source)



def build_bulk_edge_table(
    npz: Dict[str, np.ndarray],
    feature_mapping: pl.DataFrame,
    expanded_dict: Dict[str, int],
    old_dict: Dict[str, int],
    graph_cfg: Dict[str, Any],
) -> pl.DataFrame:
    """Build the bulk NMIC edge table as canonical undirected pairs.

    The input bulk NMIC ``feat_pairs`` is expected to be an MI graph and often
    arrives as a single-copy canonical undirected table (i < j).  Therefore all
    optional thresholding, duplicate handling, optional top-k pruning, optional
    max-edge truncation, and force-keep merging are performed on canonical
    single-copy undirected pairs. No bidirectional edge expansion is applied.
    """
    feat_pairs = npz["feat_pairs"]
    mi_values = npz["mi_values"].astype(np.float32, copy=False)
    if feat_pairs.ndim != 2 or feat_pairs.shape[1] != 2:
        raise ValueError(f"feat_pairs must have shape [E, 2], got {feat_pairs.shape}")
    if mi_values.shape[0] != feat_pairs.shape[0]:
        raise ValueError("mi_values and feat_pairs length mismatch")

    index_space = infer_feat_pair_index_space(feat_pairs, npz["mat_feat_indices"], graph_cfg["feat_pairs_index_space"])
    logger.info(f"feat_pairs index space inferred/configured as: {index_space}")
    lookup = build_lookup_for_bulk_features(feature_mapping, expanded_dict, index_space)


    threshold = graph_cfg.get("bulk_mi_threshold")
    q = graph_cfg.get("bulk_edge_quantile")
    use_bulk_edge_filter = threshold is not None or q is not None
    if threshold is not None:
        threshold = float(threshold)
        logger.info(f"Using configured bulk MI threshold: {threshold:.6g}")
    elif q is not None:
        q = float(q)
        threshold = float(np.quantile(mi_values, q))
        logger.info(f"Using bulk MI quantile threshold q={q}: {threshold:.6g}")
    else:
        threshold = None
        logger.info("Bulk MI edge filtering disabled: keeping all valid mapped edges from bulk NPZ")

    chunk = 5_000_000
    src_parts: List[np.ndarray] = []
    dst_parts: List[np.ndarray] = []
    w_parts: List[np.ndarray] = []

    n_edges = feat_pairs.shape[0]
    for start in range(0, n_edges, chunk):
        end = min(start + chunk, n_edges)
        w = mi_values[start:end]
        if use_bulk_edge_filter:
            mask = w >= threshold
            if not mask.any():
                continue
            raw = feat_pairs[start:end][mask]
            ww_chunk = w[mask]
        else:
            raw = feat_pairs[start:end]
            ww_chunk = w
        raw_src = raw[:, 0].astype(np.int64, copy=False)
        raw_dst = raw[:, 1].astype(np.int64, copy=False)
        valid_raw = (raw_src >= 0) & (raw_dst >= 0) & (raw_src < lookup.size) & (raw_dst < lookup.size)
        if not valid_raw.any():
            continue

        raw_src, raw_dst, ww = raw_src[valid_raw], raw_dst[valid_raw], ww_chunk[valid_raw]
        src = lookup[raw_src]
        dst = lookup[raw_dst]
        valid = (src >= 0) & (dst >= 0) & (src != dst)
        if valid.any():
            src_parts.append(src[valid].astype(np.int64))
            dst_parts.append(dst[valid].astype(np.int64))
            w_parts.append(ww[valid].astype(np.float32))

    if not src_parts:
        raise ValueError("No bulk edges survived threshold/mapping. Lower threshold or check mapping.")

    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    w = np.concatenate(w_parts)

    # Canonicalize before pruning so top-k/max_edges cannot destroy symmetry.
    u, v, w = canonicalize_edge_arrays(src, dst, w)
    if u.size == 0:
        raise ValueError("No non-self bulk edges survived canonicalization.")

    topk = int(graph_cfg.get("bulk_topk_per_node", 0) or 0)
    n_before_topk = int(u.size)
    if topk > 0:
        u, v, w = reduce_undirected_edges_topk(u, v, w, topk=topk)
        logger.info(
            f"Bulk undirected edges after canonicalization/top-k: "
            f"{n_before_topk:,} → {u.size:,} pairs (topk={topk})"
        )
    else:
        logger.info(
            f"Bulk top-k pruning disabled: keeping {u.size:,} canonical undirected bulk pairs"
        )

    # max_bulk_edges is optional. 0/None disables this global edge-count cap.
    # Because final storage is canonical undirected, this cap refers directly to
    # the number of retained single-copy undirected pairs.
    max_edges_raw = graph_cfg.get("max_bulk_edges", 0)
    max_edges_cfg = int(max_edges_raw or 0)
    if max_edges_cfg > 0:
        if u.size > max_edges_cfg:
            logger.warning(
                f"Bulk undirected edges exceed cap: {u.size:,}; "
                f"keeping top {max_edges_cfg:,} canonical pairs by weight"
            )
            idx = np.argpartition(-w, max_edges_cfg - 1)[:max_edges_cfg]
            u, v, w = u[idx], v[idx], w[idx]
    else:
        logger.info(f"Bulk max-edge cap disabled: keeping {u.size:,} canonical undirected bulk pairs")

    w = w * float(graph_cfg.get("bulk_edge_weight", 1.0))
    df = make_edge_table_from_undirected_arrays(u, v, w, expanded_dict, old_dict, "bulk")

    force_df = _build_force_keep_edge_table(
        npz=npz,
        feature_mapping=feature_mapping,
        expanded_dict=expanded_dict,
        old_dict=old_dict,
        graph_cfg=graph_cfg,
        index_space=index_space,
    )
    if force_df is not None and force_df.height > 0:
        df = merge_edge_tables([df, force_df], make_undirected=False)
        logger.info(
            "Bulk edge table after force-keep merge: "
            f"{df.height:,} undirected pairs"
        )
    else:
        logger.info(f"Bulk edge table: {df.height:,} undirected pairs")
    return df



def build_pretrained_old_edge_table(
    old_graph_npz: str,
    expanded_dict: Dict[str, int],
    graph_cfg: Dict[str, Any],
    old_weight: float,
    local_src: str,
) -> Optional[pl.DataFrame]:
    """Build scRNA pretrained old-old prior as canonical undirected pairs."""
    if not old_graph_npz or not os.path.exists(old_graph_npz):
        logger.warning("No pretrained old graph npz provided; skipping pretrained old-old prior")
        return None
    try:
        from deeptan.utils.data import read_nmic_npz  # type: ignore
        edge_attr, edge_index, _mat, mat_feat_indices, _obs, node_names = read_nmic_npz(old_graph_npz)
    except Exception as e:
        logger.warning(f"Could not read pretrained graph with deeptan read_nmic_npz: {e}; skipping")
        return None

    # edge_index values are original mat feature indices. Map them to node_names through mat_feat_indices.
    raw_to_name = {int(raw): str(node_names[i]) for i, raw in enumerate(mat_feat_indices.astype(int).tolist())}
    mi = edge_attr.astype(np.float32, copy=False)
    threshold = graph_cfg.get("old_graph_mi_threshold")
    if threshold is None:
        q = float(graph_cfg.get("old_graph_quantile", 0.995))
        threshold = float(np.quantile(mi, q))
        logger.info(f"Using pretrained old graph MI quantile threshold q={q}: {threshold:.6g}")
    else:
        threshold = float(threshold)
        logger.info(f"Using configured pretrained old graph MI threshold: {threshold:.6g}")

    src_raw, dst_raw = edge_index[0], edge_index[1]
    src_list: List[int] = []
    dst_list: List[int] = []
    w_list: List[float] = []
    for s_raw, d_raw, w in zip(src_raw.tolist(), dst_raw.tolist(), mi.tolist()):
        if w < threshold:
            continue
        s_name = raw_to_name.get(int(s_raw))
        d_name = raw_to_name.get(int(d_raw))
        if s_name is None or d_name is None:
            continue
        if s_name not in expanded_dict or d_name not in expanded_dict:
            continue
        s = int(expanded_dict[s_name])
        d = int(expanded_dict[d_name])
        if s == d:
            continue
        src_list.append(s)
        dst_list.append(d)
        w_list.append(float(w) * old_weight)

    if not src_list:
        logger.warning("No pretrained old-old edges survived mapping/threshold")
        return None

    u, v, w = canonicalize_edge_arrays(
        np.asarray(src_list, dtype=np.int64),
        np.asarray(dst_list, dtype=np.int64),
        np.asarray(w_list, dtype=np.float32),
    )
    df = make_edge_table_from_undirected_arrays(u, v, w, expanded_dict, {k: v0 for k, v0 in expanded_dict.items() if v0 < len(raw_to_name)}, "pretrained_old")
    # The helper above needs old_dict only to assign edge_type. All mapped edges
    # from this source are old-old by construction, so enforce the type explicitly.
    df = df.with_columns(pl.lit("old-old").alias("edge_type"))
    logger.info(f"Pretrained old-old edge table: {df.height:,} undirected pairs")
    return df



def normalize_edge_table_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize edge-table dtypes before vertical concatenation.

    Polars requires identical schemas for ``pl.concat(..., how="vertical")``.
    Bulk NMIC edges usually carry ``edge_weight`` as Float32 because they are
    derived from ``mi_values.astype(np.float32)``, while pretrained old-old
    edges and expression-KNN edges may carry Python ``float`` values, which
    Polars infers as Float64.  This helper makes all edge tables share one
    canonical schema without changing graph semantics.
    """
    required_cols = [
        "src_expanded_idx",
        "dst_expanded_idx",
        "src",
        "dst",
        "edge_weight",
        "edge_source",
        "edge_type",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Edge table missing required columns: {missing}")

    return df.select(
        pl.col("src_expanded_idx").cast(pl.Int64),
        pl.col("dst_expanded_idx").cast(pl.Int64),
        pl.col("src").cast(pl.Utf8),
        pl.col("dst").cast(pl.Utf8),
        pl.col("edge_weight").cast(pl.Float32),
        pl.col("edge_source").cast(pl.Utf8),
        pl.col("edge_type").cast(pl.Utf8),
    )


def merge_edge_tables(edge_tables: List[pl.DataFrame], make_undirected: bool = False) -> pl.DataFrame:
    """Merge edge tables using undirected-pair semantics.

    Internally all rows are canonicalized to ``u=min(src,dst), v=max(src,dst)``.
    Rows from the same source/type for the same undirected pair are de-duplicated
    with ``max(edge_weight)`` to avoid double-counting A→B and B→A copies.  Then
    evidence from different sources is summed for the pair. The returned table
    remains canonical single-copy undirected; ``make_undirected`` is accepted
    only for backward-compatible call sites and is intentionally ignored.
    """
    valid_tables = [
        normalize_edge_table_schema(x)
        for x in edge_tables
        if x is not None and x.height > 0
    ]
    if not valid_tables:
        raise ValueError("No valid edge tables to merge.")

    df = pl.concat(valid_tables, how="vertical")
    df = df.filter(pl.col("src_expanded_idx") != pl.col("dst_expanded_idx"))
    if df.height == 0:
        raise ValueError("Only self-loop edges were provided; no valid undirected edges to merge.")

    df = df.with_columns(
        pl.min_horizontal("src_expanded_idx", "dst_expanded_idx").alias("_u"),
        pl.max_horizontal("src_expanded_idx", "dst_expanded_idx").alias("_v"),
        pl.when(pl.col("src_expanded_idx") <= pl.col("dst_expanded_idx"))
        .then(pl.col("src"))
        .otherwise(pl.col("dst"))
        .alias("_u_name"),
        pl.when(pl.col("src_expanded_idx") <= pl.col("dst_expanded_idx"))
        .then(pl.col("dst"))
        .otherwise(pl.col("src"))
        .alias("_v_name"),
    )

    # First remove duplicate orientations within the same evidence source.
    per_source = (
        df.group_by(["_u", "_v", "edge_source", "edge_type"])
        .agg(
            pl.first("_u_name").alias("_u_name"),
            pl.first("_v_name").alias("_v_name"),
            pl.col("edge_weight").max().cast(pl.Float32).alias("edge_weight"),
        )
    )

    # Then integrate evidence from different sources for the same undirected pair.
    merged = (
        per_source.group_by(["_u", "_v"])
        .agg(
            pl.first("_u_name").alias("src"),
            pl.first("_v_name").alias("dst"),
            pl.col("edge_weight").sum().cast(pl.Float32).alias("edge_weight"),
            pl.first("edge_source").alias("edge_source"),
            pl.first("edge_type").alias("edge_type"),
        )
        .rename({"_u": "src_expanded_idx", "_v": "dst_expanded_idx"})
        .select(
            "src_expanded_idx",
            "dst_expanded_idx",
            "src",
            "dst",
            "edge_weight",
            "edge_source",
            "edge_type",
        )
        .sort(["src_expanded_idx", "dst_expanded_idx"])
    )
    logger.info(f"Merged expanded edge table: {merged.height:,} undirected pairs")

    if make_undirected:
        logger.warning(
            "make_undirected=True requested, but canonical single-copy undirected "
            "storage is enforced to avoid duplicate MI/correlation evidence."
        )
    return ensure_canonical_undirected_edge_table(merged)



def add_knn_edges_for_isolated_new(
    edge_df: pl.DataFrame,
    x_train: np.ndarray,
    selected_feature_names: List[str],
    selected_expanded_indices: np.ndarray,
    old_dict: Dict[str, int],
    expanded_dict: Dict[str, int],
    cfg: Dict[str, Any],
) -> pl.DataFrame:
    """Add weak old-new KNN edges for new nodes lacking old-neighbor support.

    Previous behavior only supplemented degree-zero new nodes.  For transfer
    fine-tuning, the more relevant failure case is a bulk-only new node that has
    new-new edges but no old-new bridge into the pretrained scRNA embedding
    space.  This function therefore targets selected new nodes with no selected
    old neighbor in the current undirected guide graph.
    """
    if not cfg.get("add_knn_for_isolated_new", True):
        return edge_df

    edge_df = merge_edge_tables([edge_df], make_undirected=False)

    old_set = set(old_dict.keys())
    old_indices = set(int(v) for v in old_dict.values())
    selected_name_to_local = {n: i for i, n in enumerate(selected_feature_names)}
    new_selected = [n for n in selected_feature_names if n not in old_set]
    old_selected = [n for n in selected_feature_names if n in old_set]
    if not new_selected or not old_selected:
        logger.warning("KNN edges skipped: no selected new or selected old features")
        return edge_df

    has_old_neighbor: set[int] = set()
    for s, d in edge_df.select(["src_expanded_idx", "dst_expanded_idx"]).iter_rows():
        s = int(s)
        d = int(d)
        if s in old_indices and d not in old_indices:
            has_old_neighbor.add(d)
        elif d in old_indices and s not in old_indices:
            has_old_neighbor.add(s)

    target_new = [n for n in new_selected if int(expanded_dict[n]) not in has_old_neighbor]
    if not target_new:
        logger.info("All selected new nodes already have old-neighbor support; KNN edge-completion skipped")
        return edge_df

    k = int(cfg.get("knn_k", 5))
    min_abs_corr = float(cfg.get("knn_min_abs_corr", 0.15))
    scale = float(cfg.get("knn_edge_weight_scale", 0.25))
    old_locs = np.array([selected_name_to_local[n] for n in old_selected], dtype=np.int64)
    old_mat = x_train[:, old_locs].astype(np.float32, copy=False)
    old_mat = old_mat - old_mat.mean(axis=0, keepdims=True)
    old_std = old_mat.std(axis=0, keepdims=True) + 1e-8
    old_z = old_mat / old_std

    src_list: List[int] = []
    dst_list: List[int] = []
    w_list: List[float] = []
    for n in target_new:
        loc = selected_name_to_local[n]
        v = x_train[:, loc].astype(np.float32)
        v = (v - v.mean()) / (v.std() + 1e-8)
        corr = (old_z * v[:, None]).mean(axis=0)
        if corr.size == 0:
            continue
        idx = np.argsort(-np.abs(corr))[:k]
        for j in idx:
            c = float(corr[j])
            if abs(c) < min_abs_corr:
                continue
            old_name = old_selected[int(j)]
            s = int(expanded_dict[n])
            d = int(expanded_dict[old_name])
            if s == d:
                continue
            src_list.append(min(s, d))
            dst_list.append(max(s, d))
            w_list.append(abs(c) * scale)

    if src_list:
        u, v, w = canonicalize_edge_arrays(
            np.asarray(src_list, dtype=np.int64),
            np.asarray(dst_list, dtype=np.int64),
            np.asarray(w_list, dtype=np.float32),
        )
        knn_df = make_edge_table_from_undirected_arrays(u, v, w, expanded_dict, old_dict, "expression_knn")
        knn_df = knn_df.with_columns(pl.lit("old-new").alias("edge_type"))
        logger.info(
            f"Added {knn_df.height:,} undirected KNN weak old-new edges "
            f"for {len(target_new):,} selected new nodes without old-neighbor support"
        )
        return merge_edge_tables([edge_df, knn_df], make_undirected=False)

    logger.warning(f"{len(target_new)} selected new nodes still lack old-neighbor KNN edges after correlation filtering")
    return edge_df




# -----------------------------------------------------------------------------
# Graph sample dataset# -----------------------------------------------------------------------------
# Graph sample dataset
# -----------------------------------------------------------------------------


class BulkTraitGraphDataset:
    """PyG sample dataset for trait-aware bulk fine-tuning.

    Recommended mode
    ----------------
    graph_scope="sample_subgraph"
        Run04-style computation path.  A global bulk-expanded guide graph is
        still built from bulk NMIC, scRNA pretrained old-old prior, and optional
        expression KNN weak edges, but each sample only receives an active
        sample-level subgraph induced from its own expression profile.  This
        prevents AMSGP from seeing the global full graph for every bulk sample.

    Compatibility mode
    ------------------
    graph_scope="fixed_full"
        Every sample keeps the same full selected bulk feature node set and the
        same guide graph. This is kept for ablation/backward compatibility, but
        it can be too large for the original DeepTAN global pooling path.
    """

    def __init__(
        self,
        split_name: str,
        df: pl.DataFrame,
        labels: pl.DataFrame,
        selected_feature_names: List[str],
        selected_expanded_indices: np.ndarray,
        expanded_dict: Dict[str, int],
        old_dict: Dict[str, int],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        obs_col: str,
        feature_cols_all: List[str],
        dataset_cfg: Dict[str, Any],
        label_mean: float,
        label_std: float,
    ):
        self.split_name = split_name
        self.obs_col = obs_col
        self.expanded_dict = expanded_dict
        self.old_dict = dict(old_dict)
        self.old_names = set(self.old_dict.keys())
        self.old_indices = set(int(v) for v in self.old_dict.values())
        self.idx_to_name = {v: k for k, v in expanded_dict.items()}
        self.selected_feature_names = list(selected_feature_names)
        self.selected_expanded_indices = selected_expanded_indices.astype(np.int64)
        self.edge_index = edge_index.long()
        self.edge_attr = edge_attr.float() if edge_attr is not None else None
        self.feature_cols_all = list(feature_cols_all)
        self.cfg = dataset_cfg
        self.label_mean = float(label_mean)
        self.label_std = float(max(label_std, 1e-8))
        self.graph_scope = self._normalize_graph_scope(self.cfg.get("graph_scope", "sample_subgraph"))

        if len(self.selected_feature_names) != int(self.selected_expanded_indices.shape[0]):
            raise ValueError(
                "selected_feature_names and selected_expanded_indices must have the same length: "
                f"{len(self.selected_feature_names)} vs {self.selected_expanded_indices.shape[0]}"
            )
        if np.unique(self.selected_expanded_indices).size != self.selected_expanded_indices.size:
            raise ValueError("selected_expanded_indices contains duplicated expanded node ids")

        # Keep only labeled samples. This preserves the previous run_01 behavior.
        keep = labels.filter(pl.col("FT16").is_not_null())
        self.sample_indices = keep["sample_idx"].to_numpy().astype(np.int64)
        self.obs_names = keep[obs_col].to_list()
        self.ecotype_ids = keep["ecotype_id"].to_list()
        self.y_raw = keep["FT16"].to_numpy().astype(np.float32)
        self.y_norm = ((self.y_raw - self.label_mean) / self.label_std).astype(np.float32)

        # Read matrix in selected feature order from parquet. The selected
        # feature order is the authoritative bulk NMIC processed/local order.
        missing = [c for c in self.selected_feature_names if c not in df.columns]
        if missing:
            raise KeyError(f"{split_name} parquet missing selected feature columns, examples: {missing[:10]}")
        x_all_raw = df.select(self.selected_feature_names).to_numpy().astype(np.float32)
        x_all, self.x_transform_info = transform_expression_matrix(
            x_all_raw,
            self.cfg,
            context=f"{self.split_name}/selected_features_before_litdata",
        )
        self.x = x_all[self.sample_indices]

        # Precompute node-selection helpers for coverage-aware sample subgraphs.
        self._prepare_sample_selection_metadata()

        # Always precompute the full selected guide graph as a reference and as
        # the optional fixed_full path. In the default sample_subgraph mode, get()
        # uses this global guide graph only to induce a per-sample active subgraph.
        self._prepare_fixed_full_graph()

        logger.info(
            f"Dataset {split_name}: {len(self)} labeled samples, matrix={self.x.shape}, "
            f"graph_scope={self.graph_scope}, fixed_nodes={self.fixed_num_nodes}, "
            f"fixed_edges={self.fixed_num_edges}"
        )
        if self.graph_scope == "fixed_full" and str(self.cfg.get("node_selection", "all")) != "all":
            logger.warning(
                f"Dataset {split_name}: graph_scope='fixed_full' ignores node_selection="
                f"{self.cfg.get('node_selection')!r}; all selected bulk features are retained."
            )
        if self.graph_scope == "sample_subgraph" and str(self.cfg.get("node_selection", "nonzero_top_abs")) == "all":
            logger.warning(
                f"Dataset {split_name}: graph_scope='sample_subgraph' with node_selection='all' "
                "will still induce a full selected-feature subgraph. Use nonzero_top_abs or top_abs "
                "with max_nodes_per_sample to mimic Run04-style small sample graphs."
            )

    @staticmethod
    def _normalize_graph_scope(value: Any) -> str:
        v = str(value or "fixed_full").lower()
        if v in {"fixed", "full", "fixed_full", "full_graph", "fixed_full_graph"}:
            return "fixed_full"
        if v in {"sample", "sample_subgraph", "dynamic", "filtered", "per_sample"}:
            return "sample_subgraph"
        raise ValueError(
            f"Unsupported dataset.graph_scope={value!r}. Expected 'fixed_full' or 'sample_subgraph'."
        )

    def _prepare_fixed_full_graph(self) -> None:
        order = np.arange(self.selected_expanded_indices.size, dtype=np.int64)
        if bool(self.cfg.get("fixed_full_sort_by_expanded_idx", True)):
            order = order[np.argsort(self.selected_expanded_indices[order], kind="mergesort")]

        self.fixed_locs = order
        self.fixed_exp_ids = torch.tensor(self.selected_expanded_indices[order], dtype=torch.long)
        self.fixed_node_names = [self.idx_to_name[int(j)] for j in self.fixed_exp_ids.tolist()]

        edge_index_sub, edge_attr_sub = pyg_subgraph(
            self.fixed_exp_ids,
            self.edge_index,
            self.edge_attr,
            relabel_nodes=True,
            num_nodes=len(self.expanded_dict),
        )
        self.fixed_edge_index = edge_index_sub.long()
        self.fixed_edge_attr = edge_attr_sub.float() if edge_attr_sub is not None else None
        self.fixed_num_nodes = int(self.fixed_exp_ids.numel())
        self.fixed_num_edges = int(self.fixed_edge_index.shape[1]) if self.fixed_edge_index.ndim == 2 else 0
        if self.fixed_num_nodes == 0:
            raise ValueError(f"Dataset {self.split_name}: fixed_full graph has zero nodes")
        if self.fixed_num_edges == 0:
            logger.warning(f"Dataset {self.split_name}: fixed_full graph has zero edges after selected-node restriction")
        if self.fixed_num_edges > 0:
            max_idx = int(self.fixed_edge_index.max().item())
            if max_idx >= self.fixed_num_nodes:
                raise ValueError(
                    f"Dataset {self.split_name}: relabeled edge_index max={max_idx} >= n_nodes={self.fixed_num_nodes}"
                )

    def _prepare_sample_selection_metadata(self) -> None:
        """Precompute low-frequency, bridge, and forced-gene metadata."""
        thr = float(self.cfg.get("value_threshold", 1e-8))
        active = np.abs(self.x) > thr if self.x.size else np.zeros_like(self.x, dtype=bool)
        self.node_active_counts = active.sum(axis=0).astype(np.int64) if self.x.ndim == 2 else np.zeros(len(self.selected_feature_names), dtype=np.int64)
        self.node_rarity_scores = 1.0 / np.sqrt(self.node_active_counts.astype(np.float64) + 1.0)

        self.selected_loc_by_name = {n: i for i, n in enumerate(self.selected_feature_names)}
        self.selected_expanded_to_loc = {int(e): i for i, e in enumerate(self.selected_expanded_indices.tolist())}
        self.selected_is_old = np.asarray([n in self.old_names for n in self.selected_feature_names], dtype=bool)

        bridge_locs: set[int] = set()
        if self.edge_index is not None and self.edge_index.numel() > 0:
            ei = self.edge_index.cpu().numpy().astype(np.int64)
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                s_old = int(s) in self.old_indices
                d_old = int(d) in self.old_indices
                if s_old == d_old:
                    continue
                if int(s) in self.selected_expanded_to_loc:
                    bridge_locs.add(int(self.selected_expanded_to_loc[int(s)]))
                if int(d) in self.selected_expanded_to_loc:
                    bridge_locs.add(int(self.selected_expanded_to_loc[int(d)]))
        self.bridge_locs = np.asarray(sorted(bridge_locs), dtype=np.int64)

        force_names: List[str] = []
        for g in self.cfg.get("force_include_genes", []) or []:
            force_names.append(str(g))
        for pair in self.cfg.get("force_include_gene_pairs", []) or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                force_names.extend([str(pair[0]), str(pair[1])])
        seen = set()
        force_names = [g for g in force_names if not (g in seen or seen.add(g))]

        missing = [g for g in force_names if g not in self.selected_loc_by_name]
        action = str(self.cfg.get("force_include_missing_action", "warn")).lower()
        if missing:
            msg = (
                f"Dataset {self.split_name}: force_include genes are not in selected bulk features "
                f"and cannot be inserted into sample graphs: {missing}"
            )
            if action == "error":
                raise ValueError(msg)
            logger.warning(msg)
        self.force_include_gene_names = [g for g in force_names if g in self.selected_loc_by_name]
        self.force_include_locs = np.asarray([self.selected_loc_by_name[g] for g in self.force_include_gene_names], dtype=np.int64)
        min_count = int(self.node_active_counts.min()) if self.node_active_counts.size else 0
        max_count = int(self.node_active_counts.max()) if self.node_active_counts.size else 0
        logger.info(
            f"Dataset {self.split_name}: coverage-aware selection metadata prepared: "
            f"force_genes={self.force_include_gene_names}, bridge_locs={self.bridge_locs.size:,}, "
            f"active_count range=[{min_count}, {max_count}]"
        )

    def __len__(self) -> int:
        return len(self.y_raw)

    @staticmethod
    def _take_top_abs(values: np.ndarray, candidates: np.ndarray, n: int) -> np.ndarray:
        if n <= 0 or candidates.size == 0:
            return np.empty(0, dtype=np.int64)
        n = min(int(n), int(candidates.size))
        scores = np.abs(values[candidates])
        if n >= candidates.size:
            return candidates[np.argsort(-scores, kind="mergesort")]
        idx = np.argpartition(-scores, n - 1)[:n]
        chosen = candidates[idx]
        return chosen[np.argsort(-np.abs(values[chosen]), kind="mergesort")]

    def _weighted_coverage_sample(
        self,
        values: np.ndarray,
        candidates: np.ndarray,
        n: int,
        sample_i: int,
        already: set[int],
    ) -> np.ndarray:
        """Deterministically sample low-frequency active nodes for coverage."""
        if n <= 0 or candidates.size == 0:
            return np.empty(0, dtype=np.int64)
        candidates = np.asarray([int(c) for c in candidates.tolist() if int(c) not in already], dtype=np.int64)
        if candidates.size == 0:
            return np.empty(0, dtype=np.int64)
        n = min(int(n), int(candidates.size))
        expr = np.abs(values[candidates]).astype(np.float64)
        weights = self.node_rarity_scores[candidates].astype(np.float64) * np.power(expr + 1e-8, 0.25)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            return self._take_top_abs(values, candidates, n)
        weights = weights / weights.sum()
        seed = int(self.cfg.get("coverage_random_seed", 20260429)) + 1000003 * int(sample_i)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(candidates, size=n, replace=False, p=weights)
        exp_ids = self.selected_expanded_indices[chosen]
        return chosen[np.argsort(exp_ids, kind="mergesort")]

    def _select_nodes(self, values: np.ndarray, sample_i: int = 0) -> np.ndarray:
        """Select sample-level nodes.

        In strict_nonzero_subgraph mode with node_selection="nonzero_top_abs",
        the returned node set is exactly the sample's nonzero feature set under
        ``abs(values) > value_threshold``.  If max_nodes_per_sample is positive
        and the nonzero set is larger than that cap, only the strongest nonzero
        nodes by absolute expression are retained.  No 0-expression nodes are
        added by min_nodes_per_sample, coverage enhancement, force-include logic,
        or top-abs fallback.
        """
        thr = float(self.cfg.get("value_threshold", 1e-8))
        max_nodes_raw = int(self.cfg.get("max_nodes_per_sample", 1200) or 0)
        min_nodes = int(self.cfg.get("min_nodes_per_sample", 0) or 0)
        policy = str(self.cfg.get("node_selection", "nonzero_top_abs"))
        strict_nonzero = bool(self.cfg.get("strict_nonzero_subgraph", True))

        if max_nodes_raw <= 0:
            max_nodes = int(values.size)
        else:
            max_nodes = max(1, min(max_nodes_raw, values.size))
        min_nodes = max(0, min(min_nodes, values.size))

        # Recommended strict path: nodes are exactly the nonzero expressed
        # features of the current sample. Edges are induced later from the global
        # guide graph by _build_induced_subgraph_from_locs().
        if policy == "nonzero_top_abs" and strict_nonzero:
            locs = np.where(np.abs(values) > thr)[0].astype(np.int64)
            if locs.size == 0:
                raise ValueError(
                    f"Dataset {self.split_name}: sample {sample_i} ({self.obs_names[sample_i]}) has no "
                    f"nonzero nodes under value_threshold={thr}. Strict nonzero subgraph mode refuses "
                    "to add zero-expression fallback nodes. Lower value_threshold or remove this sample."
                )
            if locs.size > max_nodes:
                locs = self._take_top_abs(values, locs, max_nodes)
            exp_ids = self.selected_expanded_indices[locs]
            return locs[np.argsort(exp_ids, kind="mergesort")]

        force_locs = self.force_include_locs.copy() if hasattr(self, "force_include_locs") else np.empty(0, dtype=np.int64)
        force_set = set(int(x) for x in force_locs.tolist())
        budget_exempt = bool(self.cfg.get("force_include_budget_exempt", False))
        effective_budget = max_nodes if budget_exempt else max(1, max_nodes - len(force_set))

        if policy == "all":
            locs = np.arange(values.size, dtype=np.int64)
            if locs.size > max_nodes and max_nodes > 0:
                locs = self._take_top_abs(values, locs, max_nodes)
            locs = np.unique(np.concatenate([locs, force_locs])) if force_locs.size else np.unique(locs)
            exp_ids = self.selected_expanded_indices[locs]
            return locs[np.argsort(exp_ids, kind="mergesort")]

        if policy == "top_abs":
            candidate_locs = np.arange(values.size, dtype=np.int64)
        elif policy == "nonzero_top_abs":
            candidate_locs = np.where(np.abs(values) > thr)[0].astype(np.int64)
            if candidate_locs.size == 0:
                candidate_locs = np.arange(values.size, dtype=np.int64)
        else:
            raise ValueError(
                f"Unsupported dataset.node_selection={policy!r}. Expected all/top_abs/nonzero_top_abs."
            )

        coverage_on = bool(self.cfg.get("coverage_enhancement", False))
        coverage_fraction = max(0.0, min(float(self.cfg.get("coverage_fraction", 0.0)), 0.8)) if coverage_on else 0.0
        bridge_fraction = max(0.0, min(float(self.cfg.get("bridge_fraction", 0.0)), 0.8)) if coverage_on else 0.0
        if coverage_fraction + bridge_fraction > 0.8:
            scale = 0.8 / (coverage_fraction + bridge_fraction)
            coverage_fraction *= scale
            bridge_fraction *= scale

        coverage_n = int(round(effective_budget * coverage_fraction))
        bridge_n = int(round(effective_budget * bridge_fraction))
        base_n = max(0, effective_budget - coverage_n - bridge_n)
        base_n = max(base_n, min(min_nodes, effective_budget))
        if base_n + coverage_n + bridge_n > effective_budget:
            overflow = base_n + coverage_n + bridge_n - effective_budget
            reduce_cov = min(coverage_n, overflow)
            coverage_n -= reduce_cov
            overflow -= reduce_cov
            if overflow > 0:
                bridge_n = max(0, bridge_n - overflow)

        selected: List[int] = []
        selected_set: set[int] = set()

        base = self._take_top_abs(values, candidate_locs, base_n)
        for loc in base.tolist():
            selected.append(int(loc)); selected_set.add(int(loc))

        if coverage_on and coverage_n > 0:
            cov_candidates = candidate_locs if bool(self.cfg.get("coverage_only_expressed", True)) else np.arange(values.size, dtype=np.int64)
            cov = self._weighted_coverage_sample(values, cov_candidates, coverage_n, sample_i, selected_set | force_set)
            for loc in cov.tolist():
                if int(loc) not in selected_set:
                    selected.append(int(loc)); selected_set.add(int(loc))

        if coverage_on and bridge_n > 0 and getattr(self, "bridge_locs", np.empty(0)).size > 0:
            if bool(self.cfg.get("coverage_only_expressed", True)):
                bridge_candidates = self.bridge_locs[np.abs(values[self.bridge_locs]) > thr]
            else:
                bridge_candidates = self.bridge_locs
            bridge_candidates = np.asarray([int(c) for c in bridge_candidates.tolist() if int(c) not in selected_set and int(c) not in force_set], dtype=np.int64)
            bridge = self._take_top_abs(values, bridge_candidates, bridge_n)
            for loc in bridge.tolist():
                if int(loc) not in selected_set:
                    selected.append(int(loc)); selected_set.add(int(loc))

        for loc in force_locs.tolist():
            if int(loc) not in selected_set:
                selected.append(int(loc)); selected_set.add(int(loc))

        target_min = max(min_nodes, len(force_set))
        if len(selected) < target_min:
            remaining = np.asarray([j for j in range(values.size) if j not in selected_set], dtype=np.int64)
            fill = self._take_top_abs(values, remaining, target_min - len(selected))
            for loc in fill.tolist():
                if int(loc) not in selected_set:
                    selected.append(int(loc)); selected_set.add(int(loc))

        locs = np.asarray(selected, dtype=np.int64)
        if not budget_exempt and locs.size > max_nodes:
            non_force = np.asarray([j for j in locs.tolist() if int(j) not in force_set], dtype=np.int64)
            keep_n = max(0, max_nodes - len(force_set))
            kept_non_force = self._take_top_abs(values, non_force, keep_n)
            locs = np.unique(np.concatenate([force_locs, kept_non_force])) if force_locs.size else kept_non_force

        exp_ids = self.selected_expanded_indices[locs]
        return locs[np.argsort(exp_ids, kind="mergesort")]

    def _build_induced_subgraph_from_locs(
        self,
        i: int,
        values: np.ndarray,
        locs: np.ndarray,
    ) -> GData:
        """Build a PyG graph from selected feature locations.

        ``locs`` indexes ``selected_feature_names`` / ``selected_expanded_indices``.
        Edges are induced from the global bulk-expanded guide graph and relabeled
        to the local sample graph.  By default, isolated selected nodes are
        removed after edge induction, which makes the final graph closer to the
        Run04 tissue LitData behavior where the model sees a compact active
        sample subgraph rather than every expressed feature.
        """
        exp_ids = torch.tensor(self.selected_expanded_indices[locs], dtype=torch.long)
        x = torch.tensor(values[locs], dtype=torch.float32).unsqueeze(1)

        edge_index_sub, edge_attr_sub = pyg_subgraph(
            exp_ids,
            self.edge_index,
            self.edge_attr,
            relabel_nodes=True,
            num_nodes=len(self.expanded_dict),
        )

        drop_isolated = bool(self.cfg.get("drop_isolated_nodes", True))
        if drop_isolated and edge_index_sub.numel() > 0:
            used = torch.unique(edge_index_sub.reshape(-1)).long()
            # Forced genes are retained even if isolated; this guarantees that
            # ECT5/PRK1 remain present in every sample graph when available.
            if hasattr(self, "force_include_locs") and self.force_include_locs.size > 0:
                forced_exp = set(int(self.selected_expanded_indices[int(j)]) for j in self.force_include_locs.tolist())
                forced_local = [k for k, eid in enumerate(exp_ids.tolist()) if int(eid) in forced_exp]
                if forced_local:
                    used = torch.unique(torch.cat([used, torch.tensor(forced_local, dtype=torch.long)]))
            if used.numel() < exp_ids.numel():
                # Relabel the induced graph after removing isolated nodes.
                remap = torch.full((exp_ids.numel(),), -1, dtype=torch.long)
                remap[used] = torch.arange(used.numel(), dtype=torch.long)
                edge_index_sub = remap[edge_index_sub]
                exp_ids = exp_ids[used]
                x = x[used]

        node_names = [self.idx_to_name[int(j)] for j in exp_ids.tolist()]
        return GData(
            x=x,
            y=torch.tensor([self.y_norm[i]], dtype=torch.float32),
            y_raw=torch.tensor([self.y_raw[i]], dtype=torch.float32),
            edge_index=edge_index_sub.long(),
            edge_attr=edge_attr_sub.float() if edge_attr_sub is not None else None,
            node_names=node_names,
            node_global_ids=exp_ids,
            obs_name=str(self.obs_names[i]),
            ecotype_id=str(self.ecotype_ids[i]),
        )

    def _fallback_locs_with_seed_edge(self, values: np.ndarray) -> np.ndarray:
        """Return a conservative fallback node set that contains at least one guide edge.

        This protects DeepTAN's AMSGP forward path, which cannot process graphs
        with an empty ``edge_index``.  The fallback keeps the highest-expression
        nodes and, if possible, forces the endpoints of the strongest guide edge
        within the selected feature universe.
        """
        max_nodes_raw = int(self.cfg.get("max_nodes_per_sample", 1200) or 0)
        min_nodes = int(self.cfg.get("min_nodes_per_sample", 50))
        max_nodes = values.size if max_nodes_raw <= 0 else min(max_nodes_raw, values.size)
        n = max(2, min(max(max_nodes, min_nodes), values.size))
        locs = np.argpartition(-np.abs(values), n - 1)[:n]
        if hasattr(self, "force_include_locs") and self.force_include_locs.size > 0:
            locs = np.unique(np.concatenate([locs, self.force_include_locs.astype(np.int64)]))

        if self.fixed_edge_index.numel() > 0:
            if self.fixed_edge_attr is not None and self.fixed_edge_attr.numel() == self.fixed_edge_index.shape[1]:
                e = int(torch.argmax(self.fixed_edge_attr).item())
            else:
                e = 0
            # fixed_edge_index is local to fixed_locs.
            endpoints_in_fixed_order = self.fixed_edge_index[:, e].cpu().numpy().astype(np.int64)
            endpoint_locs = self.fixed_locs[endpoints_in_fixed_order]
            locs = np.unique(np.concatenate([locs, endpoint_locs.astype(np.int64)]))
            if locs.size > n:
                # Keep forced endpoints, then fill remaining slots by expression.
                forced = set(endpoint_locs.astype(np.int64).tolist())
                ranked = sorted(locs.tolist(), key=lambda j: -abs(float(values[j])))
                keep = []
                for j in endpoint_locs.tolist():
                    if int(j) not in keep:
                        keep.append(int(j))
                for j in ranked:
                    if len(keep) >= n:
                        break
                    if int(j) not in forced and int(j) not in keep:
                        keep.append(int(j))
                locs = np.asarray(keep, dtype=np.int64)

        exp_ids = self.selected_expanded_indices[locs]
        return locs[np.argsort(exp_ids, kind="mergesort")]

    def _get_sample_subgraph(self, i: int, values: np.ndarray) -> GData:
        locs = self._select_nodes(values, sample_i=i)
        g = self._build_induced_subgraph_from_locs(i, values, locs)

        # In strict_nonzero_subgraph mode, an empty induced edge set is a valid
        # sample graph and must not be repaired by adding zero-expression nodes.
        if (
            g.edge_index.numel() == 0
            and bool(self.cfg.get("fallback_to_top_abs_if_empty", False))
            and not bool(self.cfg.get("strict_nonzero_subgraph", True))
        ):
            locs = self._fallback_locs_with_seed_edge(values)
            g = self._build_induced_subgraph_from_locs(i, values, locs)

        return g

    def sample_graph_size(self, i: int) -> Dict[str, int]:
        """Return the graph size that would be written for sample ``i``."""
        g = self.get(i)
        return {"nodes": int(g.x.shape[0]), "edges": int(g.edge_index.shape[1])}

    def estimate_graph_size_stats(self, max_samples: int = 128) -> Dict[str, Any]:
        """Estimate sample-level graph-size distribution without scanning all samples."""
        n = len(self)
        if n == 0:
            return {"n_checked": 0}
        k = min(max_samples, n)
        idxs = np.linspace(0, n - 1, num=k, dtype=np.int64)
        nodes, edges = [], []
        for idx in idxs.tolist():
            sz = self.sample_graph_size(int(idx))
            nodes.append(sz["nodes"])
            edges.append(sz["edges"])
        nodes_a = np.asarray(nodes, dtype=np.int64)
        edges_a = np.asarray(edges, dtype=np.int64)

        def _stats(a: np.ndarray) -> Dict[str, float]:
            return {
                "min": int(a.min()),
                "median": float(np.percentile(a, 50)),
                "mean": float(a.mean()),
                "p90": float(np.percentile(a, 90)),
                "p95": float(np.percentile(a, 95)),
                "max": int(a.max()),
            }

        return {
            "n_checked": int(k),
            "nodes": _stats(nodes_a),
            "edges": _stats(edges_a),
        }

    def get(self, i: int) -> GData:
        values = self.x[i]
        if self.graph_scope == "sample_subgraph":
            return self._get_sample_subgraph(i, values)

        # fixed_full mode: no per-sample node filtering. All selected bulk
        # features are retained in one stable node order, and every sample uses
        # the same guide graph built upstream from bulk NMIC + pretrained prior.
        x = torch.tensor(values[self.fixed_locs], dtype=torch.float32).unsqueeze(1)
        return GData(
            x=x,
            y=torch.tensor([self.y_norm[i]], dtype=torch.float32),
            y_raw=torch.tensor([self.y_raw[i]], dtype=torch.float32),
            edge_index=self.fixed_edge_index,
            edge_attr=self.fixed_edge_attr,
            node_names=list(self.fixed_node_names),
            node_global_ids=self.fixed_exp_ids,
            obs_name=str(self.obs_names[i]),
            ecotype_id=str(self.ecotype_ids[i]),
        )
# -----------------------------------------------------------------------------
# Seed pipeline
# -----------------------------------------------------------------------------


def process_seed(seed: int, cfg: Dict[str, Any]) -> None:
    """Build one LitData directory.

    In the GitHub-facing single-run interface, output is written directly to
    data.output_root, producing:
        output_root/trn
        output_root/val
        output_root/tst
        output_root/expanded_metadata.pkl

    The old output_root/seed_<seed> layout remains available only when
    data.use_seed_subdir=True. The seed argument is then used for legacy
    pattern-based input discovery and metadata provenance.
    """
    data_cfg = cfg["data"]
    graph_cfg = cfg["graph"]
    ds_cfg = cfg["dataset"]
    run_id = str(data_cfg.get("run_id", "bulk_run"))
    use_seed_subdir = bool(data_cfg.get("use_seed_subdir", False))
    out_dir = (
        os.path.join(data_cfg["output_root"], f"seed_{seed}")
        if use_seed_subdir
        else data_cfg["output_root"]
    )
    os.makedirs(out_dir, exist_ok=True)

    split_map = {str(k): v for k, v in data_cfg["splits"].items()}
    split_dfs: Dict[str, pl.DataFrame] = {}
    split_paths: Dict[str, str] = {}
    direct_split_paths = data_cfg.get("direct_split_paths") or {}
    if direct_split_paths:
        logger.info("Using direct split parquet paths from CLI/config")
        for split_name in ["trn", "val", "tst"]:
            path = direct_split_paths.get(split_name)
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"Missing direct split parquet for {split_name}: {path}")
            split_paths[split_name] = path
            split_dfs[split_name] = pl.read_parquet(path)
            logger.info(f"seed={seed} {split_name}: {path} ({split_dfs[split_name].height} rows)")
    else:
        for split_id, split_name in split_map.items():
            path = os.path.join(data_cfg["split_root"], data_cfg["split_file_pattern"].format(seed=seed, split=split_id))
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            split_paths[split_name] = path
            split_dfs[split_name] = pl.read_parquet(path)
            logger.info(f"seed={seed} {split_name}: {path} ({split_dfs[split_name].height} rows)")

    train_df = split_dfs["trn"]
    obs_col = data_cfg["obs_col"]

    direct_nmic_path = data_cfg.get("direct_nmic_path") or ""
    if direct_nmic_path:
        nmic_path = direct_nmic_path
    else:
        nmic_pattern = os.path.join(data_cfg["nmic_root"], data_cfg["nmic_file_pattern"].format(seed=seed, split=0))
        nmic_path = resolve_single_file(nmic_pattern)
    npz = read_bulk_npz(nmic_path)

    feature_source_df, feature_source_name, feature_cols_train, companion_info = validate_and_select_nmic_feature_source(
        npz=npz,
        train_df=train_df,
        split_dfs=split_dfs,
        companion_path=data_cfg.get("nmic_companion_parquet") or "",
        obs_col=obs_col,
        metadata_cols=data_cfg.get("metadata_cols", []),
        exclude_cols=data_cfg.get("exclude_cols", []),
        check_action=data_cfg.get("nmic_companion_check_action", "error"),
    )

    feature_mapping = build_feature_mapping(
        feature_cols_train,
        npz["mat_feat_indices"],
        feature_index_mode=companion_info.get("feature_index_mode", "full_index"),
    )
    selected_feature_names = feature_mapping["feature_name"].to_list()
    orientation = infer_processed_matrix_orientation(npz["processed_mat"], feature_source_df.height, len(selected_feature_names))
    logger.info(f"processed_mat orientation: {orientation} (feature source: {feature_source_name})")

    phenotype_df = pl.read_parquet(data_cfg["phenotype_parquet"])
    label_tables, labels_all, _label_info = match_ft16_labels(
        split_dfs,
        phenotype_df,
        obs_col=obs_col,
        phenotype_obs_col=data_cfg["phenotype_obs_col"],
        phenotype_col=data_cfg["phenotype_col"],
        ecotype_regex=data_cfg["ecotype_regex"],
    )
    y_train = label_tables["trn"].filter(pl.col("FT16").is_not_null())["FT16"].to_numpy().astype(float)
    if y_train.size < 5:
        raise ValueError(f"Too few labeled train samples for seed={seed}: {y_train.size}")
    label_mean = float(y_train.mean()) if data_cfg.get("label_standardize", True) else 0.0
    label_std = float(y_train.std(ddof=0)) if data_cfg.get("label_standardize", True) else 1.0
    if label_std < 1e-8:
        label_std = 1.0

    old_dict = load_pretrained_dict(cfg["pretrained"].get("metadata_pkl", ""), cfg["pretrained"].get("checkpoint", ""))
    expanded_dict, vocab_df = build_expanded_vocabulary(old_dict, selected_feature_names)
    feature_mapping = feature_mapping.with_columns(
        pl.col("feature_name").map_elements(lambda x: int(expanded_dict[x]), return_dtype=pl.Int64).alias("expanded_idx"),
        pl.col("feature_name").map_elements(lambda x: "old" if x in old_dict else "new", return_dtype=pl.Utf8).alias("old_or_new"),
    )
    selected_expanded_indices = feature_mapping["expanded_idx"].to_numpy().astype(np.int64)

    bulk_edges = build_bulk_edge_table(npz, feature_mapping, expanded_dict, old_dict, graph_cfg)
    edge_tables = [bulk_edges]
    if graph_cfg.get("include_pretrained_old_graph", True):
        pre_edges = build_pretrained_old_edge_table(
            cfg["pretrained"].get("old_graph_npz", ""),
            expanded_dict,
            {**graph_cfg, **cfg["pretrained"]},
            old_weight=float(cfg["pretrained"].get("old_graph_weight", 0.30)),
            local_src=cfg.get("local_deeptan_src", ""),
        )
        if pre_edges is not None:
            edge_tables.append(pre_edges)
    # Merge all graph sources as canonical single-copy undirected pairs.
    edge_df_undirected = merge_edge_tables(edge_tables, make_undirected=False)

    # Matrix for expression-KNN: use train parquet in selected feature order and
    # apply the same x transform used for LitData GData.x.  This keeps the custom
    # bulk graph augmentation numerically consistent with the Run04-compatible
    # sample node features.
    x_train_selected_raw = train_df.select(selected_feature_names).to_numpy().astype(np.float32)
    x_train_selected, knn_x_transform_info = transform_expression_matrix(
        x_train_selected_raw,
        ds_cfg,
        context="trn/expression_knn_selected_features",
    )
    edge_df_undirected = add_knn_edges_for_isolated_new(
        edge_df_undirected,
        x_train_selected,
        selected_feature_names,
        selected_expanded_indices,
        old_dict,
        expanded_dict,
        graph_cfg,
    )

    # Final model input keeps one canonical edge per undirected MI/correlation
    # relation. This avoids duplicate evidence and matches the original DeepTAN
    # run_04 path where canonical feat_pairs are not expanded into reverse edges.
    edge_df = ensure_canonical_undirected_edge_table(edge_df_undirected)
    logger.info(f"Final guide graph for model input: {edge_df.height:,} canonical undirected edges")

    # Torch edge tensors for dataset conversion.
    edge_index = torch.tensor(
        np.vstack([edge_df["src_expanded_idx"].to_numpy(), edge_df["dst_expanded_idx"].to_numpy()]),
        dtype=torch.long,
    )
    edge_attr = torch.tensor(edge_df["edge_weight"].to_numpy().astype(np.float32), dtype=torch.float32)

    datasets: Dict[str, BulkTraitGraphDataset] = {}
    for split_name, df in split_dfs.items():
        datasets[split_name] = BulkTraitGraphDataset(
            split_name=split_name,
            df=df,
            labels=label_tables[split_name],
            selected_feature_names=selected_feature_names,
            selected_expanded_indices=selected_expanded_indices,
            expanded_dict=expanded_dict,
            old_dict=old_dict,
            edge_index=edge_index,
            edge_attr=edge_attr,
            obs_col=obs_col,
            feature_cols_all=list_numeric_feature_columns(df, obs_col, data_cfg.get("metadata_cols", []), data_cfg.get("exclude_cols", [])),
            dataset_cfg=ds_cfg,
            label_mean=label_mean,
            label_std=label_std,
        )

    # Save metadata before expensive conversion.
    n_old = len(old_dict)
    expanded_meta = {
        "run_id": run_id,
        "seed": seed,
        "dict_node_names": expanded_dict,
        "old_dict_node_names": old_dict,
        "num_old_nodes": n_old,
        "num_new_nodes": len(expanded_dict) - n_old,
        "old_node_indices": list(range(n_old)),
        "new_node_indices": list(range(n_old, len(expanded_dict))),
        "selected_feature_names": selected_feature_names,
        "selected_expanded_indices": selected_expanded_indices.tolist(),
        "x_transform": str(ds_cfg.get("x_transform", "log1p")),
        "x_log1p_negative_action": str(ds_cfg.get("x_log1p_negative_action", "error")),
        "x_clip_min": ds_cfg.get("x_clip_min", None),
        "x_clip_max": ds_cfg.get("x_clip_max", None),
        "input_dim": 1,
        "output_g_label_dim": 1,
        "is_regression": True,
        "label_name": data_cfg["phenotype_col"],
        "label_mean": label_mean,
        "label_std": label_std,
        "nmic_path": nmic_path,
        "nmic_companion_parquet": data_cfg.get("nmic_companion_parquet", ""),
        "feature_source": feature_source_name,
        "split_paths": split_paths,
        "processed_mat_orientation": orientation,
        "dataset_graph_scope": ds_cfg.get("graph_scope", "sample_subgraph"),
        "dataset_node_selection": ds_cfg.get("node_selection", "nonzero_top_abs"),
        "value_threshold": float(ds_cfg.get("value_threshold", 1e-8)),
        "min_nodes_per_sample": int(ds_cfg.get("min_nodes_per_sample", 50)),
        "max_nodes_per_sample": int(ds_cfg.get("max_nodes_per_sample", 1200)),
        "drop_isolated_nodes": bool(ds_cfg.get("drop_isolated_nodes", True)),
        "coverage_enhancement": bool(ds_cfg.get("coverage_enhancement", True)),
        "coverage_fraction": float(ds_cfg.get("coverage_fraction", 0.15)),
        "bridge_fraction": float(ds_cfg.get("bridge_fraction", 0.05)),
        "coverage_only_expressed": bool(ds_cfg.get("coverage_only_expressed", True)),
        "coverage_random_seed": int(ds_cfg.get("coverage_random_seed", 20260429)),
        "force_include_genes": list(ds_cfg.get("force_include_genes", [])),
        "force_include_gene_pairs": list(ds_cfg.get("force_include_gene_pairs", [])),
        "force_include_budget_exempt": bool(ds_cfg.get("force_include_budget_exempt", False)),
        "fixed_full_sort_by_expanded_idx": bool(ds_cfg.get("fixed_full_sort_by_expanded_idx", True)),
        "reference_full_guide_n_nodes": int(datasets["trn"].fixed_num_nodes),
        "reference_full_guide_n_edges": int(datasets["trn"].fixed_num_edges),
    }
    with open(os.path.join(out_dir, "expanded_metadata.pkl"), "wb") as f:
        pickle.dump(expanded_meta, f)
    with open(os.path.join(out_dir, "expanded_metadata.json"), "w") as f:
        json.dump({k: v for k, v in expanded_meta.items() if k != "dict_node_names" and k != "old_dict_node_names"}, f, indent=2)

    feature_mapping.write_parquet(os.path.join(out_dir, "feature_mapping.parquet"))
    edge_df.write_parquet(os.path.join(out_dir, "edge_source.parquet"))
    edge_df_undirected.write_parquet(os.path.join(out_dir, "edge_source_undirected.parquet"))
    labels_all.write_parquet(os.path.join(out_dir, "ecotype_ft16_labels.parquet"))
    vocab_df.write_parquet(os.path.join(out_dir, "expanded_vocabulary.parquet"))

    nw = int(ds_cfg.get("num_workers", 8))
    for split_name, dataset in datasets.items():
        lit_out = os.path.join(out_dir, split_name)
        logger.info(f"Writing LitData split={split_name}: {len(dataset)} samples → {lit_out}")
        litdata.optimize(
            fn=dataset.get,
            inputs=list(range(len(dataset))),
            output_dir=lit_out,
            chunk_bytes=ds_cfg.get("lit_chunk_bytes", "256MB"),
            compression=ds_cfg.get("lit_compression", "zstd"),
            num_workers=nw,
        )
    logger.success(f"LitData construction complete: {out_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build trait-aware BulkExpand-DeepTAN LitData datasets.")

    # Original YAML-driven interface. Kept for backward compatibility.
    parser.add_argument("--config", default=None, help="Optional YAML config path")

    # Tissue-finetune-style direct CLI interface. These arguments override YAML/default values.
    parser.add_argument("--pretrained_trn_npz", default=None, help="scRNA-seq pretraining train-only NMIC npz used as old-old graph prior")
    parser.add_argument("--pretrained_pkl", default=None, help="Pretraining others2save.pkl containing old dict_node_names")
    parser.add_argument("--bulk_trn_npz", default=None, help="bulk train-only NMIC npz used as the main guide graph")
    parser.add_argument("--bulk_nmic_parquet", default=None, help="Companion parquet used as authoritative metadata for the bulk NMIC npz")
    parser.add_argument("--nmic_companion_check_action", default=None, choices=["error", "warn"], help="error or warn on companion/train mismatch; default: error")
    parser.add_argument("--trn_parquet", default=None, help="bulk fine-tuning train parquet")
    parser.add_argument("--val_parquet", default=None, help="bulk fine-tuning validation parquet")
    parser.add_argument("--tst_parquet", default=None, help="bulk fine-tuning test parquet")
    parser.add_argument("--phenotype_parquet", default=None, help="phenotype parquet containing obs_names/ecotype and FT16")
    parser.add_argument("--phenotype_col", default=None, help="phenotype column name, e.g. FT16")
    parser.add_argument("--obs_col", default=None, help="sample id column name, e.g. obs_names")
    parser.add_argument("--output_dir", default=None, help="output LitData directory. In single-run mode this directory directly contains trn/val/tst and metadata files.")
    parser.add_argument("--run_id", default=None, help="Optional human-readable run identifier stored in metadata; not used for file-name discovery.")
    parser.add_argument("--use_seed_subdir", action=argparse.BooleanOptionalAction, default=None, help="Legacy compatibility: write output_dir/seed_<seed> instead of writing directly to output_dir.")
    parser.add_argument("--bulk_mi_threshold", type=float, default=None, help="Optional bulk MI threshold. Omit to keep all valid bulk NPZ edges unless --bulk_edge_quantile is set.")
    parser.add_argument("--bulk_edge_quantile", type=float, default=None, help="Optional bulk MI quantile threshold. Omit to disable quantile edge pruning.")
    parser.add_argument("--bulk_topk_per_node", type=int, default=None, help="Optional bulk top-k pruning per node. Omit or set <=0 to disable.")
    parser.add_argument("--max_bulk_edges", type=int, default=None, help="Optional canonical undirected-edge cap. Omit or set <=0 to disable.")
    parser.add_argument(
        "--graph_scope",
        default=None,
        choices=["fixed_full", "sample_subgraph"],
        help="sample_subgraph writes Run04-style active sample graphs; fixed_full writes the full guide graph for every sample.",
    )
    parser.add_argument(
        "--node_selection",
        default=None,
        choices=["all", "top_abs", "nonzero_top_abs"],
        help="Node selection policy used when --graph_scope sample_subgraph.",
    )
    parser.add_argument(
        "--strict_nonzero_subgraph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When enabled with nonzero_top_abs, keep strictly abs(x)>value_threshold nodes only; no coverage/force/min-node/fallback additions.",
    )
    parser.add_argument("--value_threshold", type=float, default=None, help="Expression threshold for nonzero_top_abs sample node selection.")
    parser.add_argument("--max_nodes_per_sample", type=int, default=None, help="Maximum active nodes retained per sample subgraph.")
    parser.add_argument("--min_nodes_per_sample", type=int, default=None, help="Minimum active nodes retained per sample subgraph.")
    parser.add_argument(
        "--drop_isolated_nodes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Drop selected nodes that have no induced guide-graph edge in sample_subgraph mode.",
    )
    parser.add_argument(
        "--coverage_enhancement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable coverage-aware low-frequency/bridge node enhancement for sample_subgraph mode.",
    )
    parser.add_argument("--coverage_fraction", type=float, default=None, help="Fraction of max_nodes reserved for low-frequency active-node coverage enhancement.")
    parser.add_argument("--bridge_fraction", type=float, default=None, help="Fraction of max_nodes reserved for old-new bridge-node enhancement.")
    parser.add_argument(
        "--coverage_only_expressed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When enabled, coverage/bridge enhancement only samples nodes with abs(expression)>value_threshold.",
    )
    parser.add_argument("--coverage_random_seed", type=int, default=None, help="Random seed for deterministic coverage-aware node sampling.")
    parser.add_argument(
        "--force_include_gene",
        action="append",
        default=None,
        metavar="FEATURE",
        help="Force a feature/gene into every sample subgraph when present. Can be repeated.",
    )
    parser.add_argument(
        "--force_include_gene_pair",
        nargs=2,
        action="append",
        default=None,
        metavar=("FEATURE_A", "FEATURE_B"),
        help="Force both genes into every sample subgraph and try to force-keep their global guide edge. Can be repeated.",
    )
    parser.add_argument(
        "--force_include_missing_action",
        default=None,
        choices=["warn", "error"],
        help="warn or error if a force-included gene is absent from selected bulk features.",
    )
    parser.add_argument(
        "--force_include_budget_exempt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If enabled, force-included genes may increase sample nodes beyond max_nodes_per_sample.",
    )
    parser.add_argument(
        "--force_keep_edge_pair",
        nargs=2,
        action="append",
        default=None,
        metavar=("FEATURE_A", "FEATURE_B"),
        help="Force a biologically important bulk feature pair into the final guide graph. Can be repeated.",
    )
    parser.add_argument(
        "--x_transform",
        default=None,
        choices=["none", "log1p"],
        help="Expression-value transform before writing GData.x. Default from config: log1p, matching DeepTAN Run04 if_log1p=True.",
    )
    parser.add_argument(
        "--x_log1p_negative_action",
        default=None,
        choices=["error", "clip_zero"],
        help="How to handle negative values before log1p. Default: error.",
    )
    parser.add_argument("--x_clip_min", type=float, default=None, help="Optional lower clipping bound after x_transform.")
    parser.add_argument("--x_clip_max", type=float, default=None, help="Optional upper clipping bound after x_transform.")

    parser.add_argument("--seed", type=int, default=None, help="Legacy numeric seed used only for pattern-based input discovery or --use_seed_subdir. Custom-path single-run users can omit it.")
    args = parser.parse_args()

    cfg = _read_optional_yaml_config(args.config)
    cfg = apply_direct_cli_overrides(cfg, args)
    validate_direct_cli_config(cfg)

    setup_deeptan_path(cfg.get("local_deeptan_src", ""))
    logger.info("Trait-aware BulkExpand-DeepTAN single-run dataset builder")
    logger.info(f"Output LitData directory: {cfg['data']['output_root']}")
    os.makedirs(cfg["data"]["output_root"], exist_ok=True)

    direct_paths = cfg["data"].get("direct_split_paths") or {}
    legacy_seeds = [args.seed] if args.seed is not None else list(cfg["data"].get("seeds", []))

    if direct_paths:
        # Custom-path single-run mode. No filename pattern seed is needed.
        # Use 0 as an internal provenance value unless the user supplied --seed.
        seed = int(args.seed) if args.seed is not None else 0
        logger.info("=" * 80)
        logger.info(f"Processing custom-path single run | run_id={cfg['data'].get('run_id', 'bulk_run')} | internal_seed={seed}")
        process_seed(seed, cfg)
    else:
        # Backward-compatible pattern mode. This still supports old split_{seed}_{split}.parquet layouts.
        if not legacy_seeds:
            raise ValueError(
                "No direct split paths were provided and data.seeds is empty. "
                "For the GitHub-facing custom-path workflow, provide --trn_parquet/--val_parquet/--tst_parquet. "
                "For legacy pattern mode, provide --seed or data.seeds in the YAML config."
            )
        if len(legacy_seeds) > 1 and not bool(cfg["data"].get("use_seed_subdir", False)):
            raise ValueError(
                "Multiple legacy seeds would overwrite the same output_dir in single-run mode. "
                "Set data.use_seed_subdir=true or process one seed at a time."
            )
        for seed in legacy_seeds:
            logger.info("=" * 80)
            logger.info(f"Processing legacy pattern seed {seed}")
            process_seed(int(seed), cfg)


if __name__ == "__main__":
    main()
