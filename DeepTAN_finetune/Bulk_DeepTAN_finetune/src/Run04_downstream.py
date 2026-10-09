#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
==============================================================================
1.Trait-aware bulk network downstream analysis
==============================================================================

Purpose
-------
Run non-plotting downstream analysis for an extracted trait-aware bulk network:

  1. Read the extracted edge table.
  2. Use abs(w_ft) as the non-negative edge strength for module detection,
     hub identification and edge filtering.
  3. Detect functional network modules on the selected edge set.
     By default, all non-zero edges are used.
  4. Resolve raw Leiden communities smaller than --min_module_size on a
     raw-community graph using a frozen-core constrained maximum spanning forest.
     RB/configuration-model delta-Q is the primary assignment score, with
     size-normalized cross-module abs(w_ft) strength retained as secondary evidence.
     Small disconnected coreless components still below the minimum are explicitly
     left unassigned and excluded from Hub selection.
  5. Identify module hub genes by intra-module weighted degree.
  6. Export module-level gene sets, hub-gene sets and audit tables for downstream
     biological interpretation and external enrichment analysis.

This script intentionally contains no publication-figure or network-layout logic.
It does NOT run GO enrichment; it exports gene-list files that can be used by
R/clusterProfiler, g:Profiler, agriGO, or other external tools.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import warnings
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import polars as pl

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Basic utilities
# =============================================================================


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_gene_list(path: str, genes: Iterable[str]) -> None:
    with open(path, "w") as f:
        for g in sorted(set(str(x) for x in genes)):
            f.write(g + "\n")


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downstream module, hub, gene-set and audit analysis for an extracted trait-aware bulk network."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--edge_table", default=None, help="Path to bulk_trait_edge_table.parquet")
    input_group.add_argument("--network_dir", default=None, help="Directory containing bulk_trait_edge_table.parquet")

    parser.add_argument("--out_dir", required=True, help="Output directory for downstream results")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducible community detection and deterministic analysis steps")

    # Network construction and module detection.
    parser.add_argument("--weight_col", default="w_ft", help="Signed edge-weight column. Edge strength is abs(weight_col). Default: w_ft")
    parser.add_argument(
        "--module_top_frac",
        type=float,
        default=1.0,
        help="Fraction of strongest edges by abs(weight_col) used for module detection. Default 1.0 uses all non-zero edges.",
    )
    parser.add_argument(
        "--min_abs_weight",
        type=float,
        default=None,
        help="Optional minimum abs(weight_col) for module detection. Default: no threshold.",
    )
    parser.add_argument(
        "--min_weight",
        dest="min_abs_weight",
        type=float,
        default=None,
        help="Alias of --min_abs_weight for backward compatibility.",
    )
    parser.add_argument("--min_module_size", type=int, default=50, help="Minimum raw Leiden-community size defining a frozen core-module anchor. Smaller raw communities are assigned through a core-constrained maximum spanning forest whose primary edge score is RB/configuration-model delta-Q; size-normalized abs(weight_col) cross strength is secondary evidence. Coreless disconnected components still below the minimum remain unassigned and are excluded from Hub selection. Default: 50")
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.3,
        help="Community-detection resolution for Leiden/Louvain when available.",
    )
    parser.add_argument(
        "--allow_community_fallback",
        action="store_true",
        help="Allow Louvain/greedy-modularity fallback only if Leiden cannot run. Default: disabled.",
    )
    parser.add_argument(
        "--reassign_top_k",
        type=int,
        default=10,
        help="Number of strongest block-to-module edges used as secondary local-affinity evidence during small-community reassignment. Default: 10",
    )
    parser.add_argument(
        "--refine_max_iter",
        type=int,
        default=0,
        help="Retained for command-line compatibility. Frozen-core maximum-spanning-forest assignment does not run post-hoc boundary refinement, so original Leiden core modules remain unchanged. Default: 0.",
    )

    # Hub definition.
    parser.add_argument("--hub_top_frac", type=float, default=0.10, help="Fraction of nodes per module selected as hubs")
    parser.add_argument("--min_hubs", type=int, default=1, help="Minimum hubs per non-small module")
    parser.add_argument("--max_hubs", type=int, default=100000, help="Maximum hubs per module")

    return parser.parse_args()


def resolve_edge_table(args: argparse.Namespace) -> str:
    if args.edge_table:
        path = os.path.abspath(os.path.expanduser(args.edge_table))
        if not os.path.exists(path):
            raise FileNotFoundError(f"edge_table not found: {path}")
        return path

    network_dir = os.path.abspath(os.path.expanduser(args.network_dir))
    candidates = [
        os.path.join(network_dir, "bulk_trait_edge_table.parquet"),
        os.path.join(network_dir, "bulk_trait_csn_edge_table.parquet"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No edge table found in network_dir. Tried: " + ", ".join(candidates))


# =============================================================================
# Input loading and graph construction
# =============================================================================


def load_edge_table(edge_table_path: str, weight_col: str) -> pl.DataFrame:
    df = pl.read_parquet(edge_table_path)

    required = ["gene_i_name", "gene_j_name", weight_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in edge table: {missing}")

    if "delta_w" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("delta_w"))
    if "abs_delta_w" not in df.columns:
        df = df.with_columns(pl.col("delta_w").abs().alias("abs_delta_w"))
    if "gene_i_type" not in df.columns:
        df = df.with_columns(pl.lit("unknown").alias("gene_i_type"))
    if "gene_j_type" not in df.columns:
        df = df.with_columns(pl.lit("unknown").alias("gene_j_type"))

    df = df.with_columns(pl.col(weight_col).abs().alias("edge_strength"))
    df = (
        df
        .filter(pl.col("gene_i_name") != pl.col("gene_j_name"))
        .filter(pl.col(weight_col).is_not_null())
        .filter(pl.col("edge_strength").is_not_null())
        .filter(pl.col("edge_strength") > 0)
    )
    if df.height == 0:
        raise ValueError("No non-zero edges remain after loading the edge table.")
    return df


def get_node_type_map(df: pl.DataFrame) -> Dict[str, str]:
    node_type: Dict[str, str] = {}
    for row in df.select(["gene_i_name", "gene_i_type", "gene_j_name", "gene_j_type"]).iter_rows(named=True):
        node_type.setdefault(str(row["gene_i_name"]), str(row.get("gene_i_type", "unknown")))
        node_type.setdefault(str(row["gene_j_name"]), str(row.get("gene_j_type", "unknown")))
    return node_type


def select_module_edges(df: pl.DataFrame, top_frac: float, min_abs_weight: Optional[float]) -> pl.DataFrame:
    d = df
    if min_abs_weight is not None:
        d = d.filter(pl.col("edge_strength") >= float(min_abs_weight))

    if d.height == 0:
        raise ValueError("No edges remain for module detection after filtering.")

    top_frac = float(top_frac)
    if top_frac <= 0 or top_frac > 1:
        raise ValueError("module_top_frac must be in (0, 1].")

    if top_frac < 1.0:
        k = max(1, int(math.ceil(d.height * top_frac)))
        d = d.sort("edge_strength", descending=True).head(k)

    return d


def build_graph_from_edges(df: pl.DataFrame, signed_weight_col: str) -> nx.Graph:
    G = nx.Graph()
    for r in df.iter_rows(named=True):
        u = str(r["gene_i_name"])
        v = str(r["gene_j_name"])
        strength = safe_float(r.get("edge_strength"), 0.0)
        if u == v or strength <= 0:
            continue
        signed_w = safe_float(r.get(signed_weight_col), 0.0)
        delta = safe_float(r.get("delta_w"), 0.0)
        attrs = {
            "weight": strength,
            "edge_strength": strength,
            "signed_weight": signed_w,
            "delta_w": delta,
        }
        if G.has_edge(u, v):
            if strength > G[u][v].get("weight", 0.0):
                G[u][v].update(attrs)
        else:
            G.add_edge(u, v, **attrs)
    return G


def add_all_observed_nodes(G: nx.Graph, df: pl.DataFrame) -> None:
    nodes = set(df["gene_i_name"].to_list()) | set(df["gene_j_name"].to_list())
    G.add_nodes_from(str(x) for x in nodes)


# =============================================================================
# Community detection
# =============================================================================


def detect_modules(
    G: nx.Graph,
    resolution: float,
    seed: int,
    allow_fallback: bool = False,
) -> Tuple[Dict[str, int], str]:
    """Detect raw communities with weighted Leiden/RBConfiguration.

    Publication-oriented runs require Leiden by default. A different community
    algorithm is used only when the user explicitly passes
    ``--allow_community_fallback``; this prevents environment-dependent silent
    changes in the formal analysis method.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("The selected analysis network contains no active nodes.")
    if G.number_of_edges() == 0:
        raise ValueError("The selected analysis network contains no edges.")

    leiden_error: Optional[Exception] = None
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore

        nodes = sorted(str(n) for n in G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        edges = [(node_to_idx[str(u)], node_to_idx[str(v)]) for u, v in G.edges()]
        weights = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges()]
        ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        ig_graph.es["weight"] = weights
        part = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=int(seed),
        )
        raw_part = {
            nodes[i]: int(cid)
            for cid, community in enumerate(part)
            for i in community
        }
        return raw_part, "Leiden/RBConfigurationVertexPartition"
    except Exception as exc:
        leiden_error = exc

    if not allow_fallback:
        raise RuntimeError(
            "Weighted Leiden community detection failed. Publication-oriented "
            "runs do not silently substitute another algorithm. Install/check "
            "python-igraph and leidenalg, or explicitly pass "
            "--allow_community_fallback for exploratory use. "
            f"Original Leiden error: {leiden_error!r}"
        ) from leiden_error

    try:
        import community as community_louvain  # type: ignore
        raw_part = community_louvain.best_partition(
            G,
            weight="weight",
            resolution=float(resolution),
            random_state=int(seed),
        )
        return (
            {str(k): int(v) for k, v in raw_part.items()},
            "Louvain/python-louvain (explicit fallback)",
        )
    except Exception:
        communities = nx.algorithms.community.greedy_modularity_communities(
            G,
            weight="weight",
        )
        part: Dict[str, int] = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                part[str(node)] = int(cid)
        return part, "NetworkX/greedy_modularity (explicit fallback; resolution ignored)"


def diagnose_raw_communities(
    raw_part: Dict[str, int],
    all_nodes: Iterable[str],
    min_module_size: int,
    method: str,
    resolution: float,
    max_print: int = 30,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, int]]:
    """Print and tabulate raw community structure before small-module merging."""
    all_nodes_s = sorted(set(str(n) for n in all_nodes))
    by_cid: Dict[int, List[str]] = defaultdict(list)
    missing_nodes: List[str] = []

    for node in all_nodes_s:
        if node in raw_part:
            by_cid[int(raw_part[node])].append(node)
        else:
            missing_nodes.append(node)

    rows = []
    for cid, nodes in by_cid.items():
        size = len(nodes)
        rows.append({
            "raw_community_id": int(cid),
            "n_genes": int(size),
            "passes_min_module_size": bool(size >= int(min_module_size)),
        })

    rows.sort(key=lambda r: (-int(r["n_genes"]), int(r["raw_community_id"])))
    raw_summary_df = pl.DataFrame(rows) if rows else pl.DataFrame({
        "raw_community_id": pl.Series([], dtype=pl.Int64),
        "n_genes": pl.Series([], dtype=pl.Int64),
        "passes_min_module_size": pl.Series([], dtype=pl.Boolean),
    })

    membership_rows = []
    for cid, nodes in by_cid.items():
        size = len(nodes)
        retained = size >= int(min_module_size)
        for gene in nodes:
            membership_rows.append({
                "gene_name": gene,
                "raw_community_id": int(cid),
                "raw_community_size": int(size),
                "passes_min_module_size": bool(retained),
            })
    for gene in missing_nodes:
        membership_rows.append({
            "gene_name": gene,
            "raw_community_id": -1,
            "raw_community_size": 0,
            "passes_min_module_size": False,
        })
    raw_membership_df = pl.DataFrame(membership_rows) if membership_rows else pl.DataFrame()

    sizes = [int(r["n_genes"]) for r in rows]
    retained_sizes = [s for s in sizes if s >= int(min_module_size)]
    small_sizes = [s for s in sizes if s < int(min_module_size)]

    exact_small_counts = {k: sum(1 for s in sizes if s == k) for k in range(1, 11)}
    bucket_counts = {
        "size_11_19": sum(1 for s in sizes if 11 <= s <= 19),
        "size_20_49": sum(1 for s in sizes if 20 <= s <= 49),
        "size_50_99": sum(1 for s in sizes if 50 <= s <= 99),
        "size_100_plus": sum(1 for s in sizes if s >= 100),
    }

    stats = {
        "raw_communities_total": len(sizes),
        "raw_communities_retained": len(retained_sizes),
        "raw_communities_below_min_size": len(small_sizes),
        "raw_nodes_in_retained_communities": int(sum(retained_sizes)),
        "raw_nodes_in_small_communities": int(sum(small_sizes)),
        "raw_nodes_missing_assignment": len(missing_nodes),
        "largest_raw_community_size": max(sizes) if sizes else 0,
        "smallest_raw_community_size": min(sizes) if sizes else 0,
    }

    print("=" * 80)
    print("RAW COMMUNITY DIAGNOSTICS -- before trait-aware constrained reassignment")
    print("=" * 80)
    print(f"community_method              : {method}")
    print(f"resolution                    : {resolution}")
    print(f"min_module_size               : {int(min_module_size)}")
    print(f"raw_communities_total         : {stats['raw_communities_total']}")
    print(f"raw_communities_retained      : {stats['raw_communities_retained']}  (size >= {int(min_module_size)})")
    print(f"raw_communities_below_min     : {stats['raw_communities_below_min_size']}  (size < {int(min_module_size)})")
    print(f"genes_in_retained_raw_modules : {stats['raw_nodes_in_retained_communities']:,}")
    print(f"genes_in_small_raw_modules    : {stats['raw_nodes_in_small_communities']:,}")
    print(f"genes_missing_raw_assignment  : {stats['raw_nodes_missing_assignment']:,}")
    print(f"largest_raw_community_size    : {stats['largest_raw_community_size']}")
    print(f"smallest_raw_community_size   : {stats['smallest_raw_community_size']}")

    print("-" * 80)
    print("Raw community size frequencies:")
    for k in range(1, 11):
        print(f"  size == {k:2d}: {exact_small_counts[k]:4d} communities")
    print(f"  size 11-19: {bucket_counts['size_11_19']:4d} communities")
    print(f"  size 20-49: {bucket_counts['size_20_49']:4d} communities")
    print(f"  size 50-99: {bucket_counts['size_50_99']:4d} communities")
    print(f"  size >=100: {bucket_counts['size_100_plus']:4d} communities")

    print("-" * 80)
    n_show = min(int(max_print), len(rows))
    print(f"Largest raw communities (showing {n_show} of {len(rows)}):")
    for rank, row in enumerate(rows[:n_show], start=1):
        status = "CORE" if bool(row["passes_min_module_size"]) else "REASSIGN"
        print(
            f"  #{rank:03d}  raw_C{int(row['raw_community_id']):04d}  "
            f"n_genes={int(row['n_genes']):5d}  {status}"
        )
    if len(rows) > n_show:
        print(f"  ... {len(rows) - n_show} additional raw communities are written to CSV.")
    print("=" * 80)

    return raw_summary_df, raw_membership_df, stats


def _build_raw_community_map(
    raw_part: Dict[str, int],
    all_nodes: Iterable[str],
) -> Tuple[Dict[int, List[str]], Dict[str, int], Dict[str, int]]:
    """Return raw-community members plus per-gene raw id/size maps.

    Missing assignments are extremely unusual because all observed nodes are added to
    the module graph before Leiden is called. If a backend nevertheless omits a node,
    that node is represented as its own synthetic negative-id singleton so that every
    observed gene still participates in the constrained refinement.
    """
    by_cid: Dict[int, List[str]] = defaultdict(list)
    missing = []
    for node in sorted(set(str(n) for n in all_nodes)):
        if node in raw_part:
            by_cid[int(raw_part[node])].append(node)
        else:
            missing.append(node)

    next_synth = -1
    for node in missing:
        while next_synth in by_cid:
            next_synth -= 1
        by_cid[next_synth] = [node]
        next_synth -= 1

    by_cid = {int(cid): sorted(set(nodes)) for cid, nodes in by_cid.items()}
    raw_cid_of: Dict[str, int] = {}
    raw_size_of: Dict[str, int] = {}
    for cid, nodes in by_cid.items():
        for gene in nodes:
            raw_cid_of[gene] = int(cid)
            raw_size_of[gene] = int(len(nodes))
    return by_cid, raw_cid_of, raw_size_of


def _weighted_degree_map(G: nx.Graph) -> Dict[str, float]:
    return {str(n): float(v) for n, v in G.degree(weight="weight")}


def _module_degree_sums(
    modules: Dict[str, List[str]],
    weighted_degree: Dict[str, float],
) -> Dict[str, float]:
    return {
        mid: float(sum(weighted_degree.get(str(g), 0.0) for g in genes))
        for mid, genes in modules.items()
    }


def _rb_partition_quality(
    G: nx.Graph,
    modules: Dict[str, List[str]],
    resolution: float,
) -> float:
    """Normalized RB/configuration-model quality equivalent to weighted modularity.

    Leiden's RBConfigurationVertexPartition maximizes the same partition ordering up
    to a constant scale. Using this normalized form lets us report interpretable
    before/after values and compute exact merge/move deltas for the constrained step.
    """
    m = float(G.size(weight="weight"))
    if m <= 0.0:
        return 0.0
    weighted_degree = _weighted_degree_map(G)
    q = 0.0
    for genes in modules.values():
        genes_s = [str(g) for g in genes]
        w_in = float(G.subgraph(genes_s).size(weight="weight"))
        k_sum = float(sum(weighted_degree.get(g, 0.0) for g in genes_s))
        q += (w_in / m) - float(resolution) * (k_sum / (2.0 * m)) ** 2
    return float(q)


def _module_trait_profiles(
    G_analysis: nx.Graph,
    modules: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """Characterize each module's internal signed/trait-direction pattern.

    These profiles are secondary evidence only. The primary assignment objective is
    the same non-negative abs(w_ft) RB modularity optimized by Leiden; using delta_w
    and signed w_ft only as tie/support evidence avoids double-counting trait signal.
    """
    profiles: Dict[str, Dict[str, float]] = {}
    for mid, genes in modules.items():
        sub = G_analysis.subgraph(genes)
        total_strength = 0.0
        signed_positive = 0.0
        delta_positive = 0.0
        n_edges = 0
        for _u, _v, d in sub.edges(data=True):
            w = max(0.0, safe_float(d.get("weight"), 0.0))
            if w <= 0.0:
                continue
            total_strength += w
            if safe_float(d.get("signed_weight"), 0.0) > 0.0:
                signed_positive += w
            if safe_float(d.get("delta_w"), 0.0) > 0.0:
                delta_positive += w
            n_edges += 1
        if total_strength > 0.0:
            signed_frac = signed_positive / total_strength
            delta_frac = delta_positive / total_strength
        else:
            signed_frac = 0.5
            delta_frac = 0.5
        profiles[mid] = {
            "signed_positive_weight_frac": float(signed_frac),
            "delta_positive_weight_frac": float(delta_frac),
            "internal_trait_edge_count": int(n_edges),
        }
    return profiles


def _block_structural_stats_by_module(
    G: nx.Graph,
    block_nodes: Sequence[str],
    module_of: Dict[str, str],
    top_k: int,
) -> Dict[str, Dict[str, object]]:
    """Aggregate selected-network connectivity from one block to current modules."""
    block = set(str(x) for x in block_nodes)
    acc: Dict[str, Dict[str, object]] = defaultdict(lambda: {
        "cross_abs_weight": 0.0,
        "n_edges": 0,
        "strengths": [],
    })
    for u in block:
        if u not in G:
            continue
        for v, d in G[u].items():
            v_s = str(v)
            if v_s in block:
                continue
            mid = module_of.get(v_s)
            if mid is None:
                continue
            w = max(0.0, safe_float(d.get("weight"), 0.0))
            if w <= 0.0:
                continue
            acc[mid]["cross_abs_weight"] = float(acc[mid]["cross_abs_weight"]) + w
            acc[mid]["n_edges"] = int(acc[mid]["n_edges"]) + 1
            acc[mid]["strengths"].append(w)

    out: Dict[str, Dict[str, object]] = {}
    k = max(1, int(top_k))
    for mid, d in acc.items():
        strengths = sorted((float(x) for x in d["strengths"]), reverse=True)
        top = strengths[:min(k, len(strengths))]
        out[mid] = {
            "cross_abs_weight": float(d["cross_abs_weight"]),
            "n_edges": int(d["n_edges"]),
            "topk_mean_abs_weight": float(np.mean(top)) if top else 0.0,
        }
    return out


def _block_trait_stats_by_module(
    G_analysis: nx.Graph,
    block_nodes: Sequence[str],
    module_of: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """Aggregate signed-w_ft and delta_w direction on block-to-module edges."""
    block = set(str(x) for x in block_nodes)
    acc: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "total_strength": 0.0,
        "signed_positive": 0.0,
        "delta_positive": 0.0,
        "n_edges": 0.0,
    })
    for u in block:
        if u not in G_analysis:
            continue
        for v, d in G_analysis[u].items():
            v_s = str(v)
            if v_s in block:
                continue
            mid = module_of.get(v_s)
            if mid is None:
                continue
            w = max(0.0, safe_float(d.get("weight"), 0.0))
            if w <= 0.0:
                continue
            acc[mid]["total_strength"] += w
            if safe_float(d.get("signed_weight"), 0.0) > 0.0:
                acc[mid]["signed_positive"] += w
            if safe_float(d.get("delta_w"), 0.0) > 0.0:
                acc[mid]["delta_positive"] += w
            acc[mid]["n_edges"] += 1.0

    out: Dict[str, Dict[str, float]] = {}
    for mid, d in acc.items():
        total = float(d["total_strength"])
        out[mid] = {
            "signed_positive_weight_frac": float(d["signed_positive"] / total) if total > 0 else 0.5,
            "delta_positive_weight_frac": float(d["delta_positive"] / total) if total > 0 else 0.5,
            "trait_edge_count": int(d["n_edges"]),
        }
    return out


def _candidate_evidence(
    *,
    block_nodes: Sequence[str],
    source_mid: Optional[str],
    modules: Dict[str, List[str]],
    module_of: Dict[str, str],
    G_module: nx.Graph,
    G_analysis: nx.Graph,
    weighted_degree: Dict[str, float],
    module_degree: Dict[str, float],
    resolution: float,
    top_k: int,
    trait_profiles: Dict[str, Dict[str, float]],
) -> List[Dict[str, object]]:
    """Evaluate all legal target modules for one block.

    Primary criterion is the exact normalized RB quality change. For forced merging
    of a small raw community, the best target is therefore the largest gain or, if
    every merge is unfavorable, the smallest unavoidable RB-quality loss. Local
    strongest-edge concentration and trait-direction agreement are deterministic
    secondary evidence and are exported for auditability.
    """
    m = float(G_module.size(weight="weight"))
    if m <= 0.0:
        return []

    block = [str(x) for x in block_nodes]
    block_degree = float(sum(weighted_degree.get(g, 0.0) for g in block))
    structural = _block_structural_stats_by_module(G_module, block, module_of, top_k)
    trait_stats = _block_trait_stats_by_module(G_analysis, block, module_of)
    total_cross = float(sum(float(d.get("cross_abs_weight", 0.0)) for d in structural.values()))

    source_cross = 0.0
    if source_mid is not None:
        source_cross = float(structural.get(source_mid, {}).get("cross_abs_weight", 0.0))

    rows: List[Dict[str, object]] = []
    for target_mid in sorted(modules):
        if source_mid is not None and target_mid == source_mid:
            continue

        target_cross = float(structural.get(target_mid, {}).get("cross_abs_weight", 0.0))
        target_k = float(module_degree.get(target_mid, 0.0))

        # A block may only be assigned/moved to a module to which it has at
        # least one selected-network edge. This prevents the minimum-size
        # constraint from merging disconnected components purely because a
        # cross-component merge has the "least bad" RB penalty.
        if target_cross <= 0.0:
            continue

        if source_mid is None:
            # Exact RB quality delta for merging two previously separate communities.
            delta_q = (target_cross / m) - float(resolution) * (
                block_degree * target_k / (2.0 * m * m)
            )
        else:
            source_k = float(module_degree.get(source_mid, 0.0))
            delta_degree_term = (
                (source_k - block_degree) ** 2
                + (target_k + block_degree) ** 2
                - source_k ** 2
                - target_k ** 2
            )
            delta_q = ((target_cross - source_cross) / m) - float(resolution) * (
                delta_degree_term / (4.0 * m * m)
            )

        sd = structural.get(target_mid, {})
        td = trait_stats.get(target_mid, {})
        prof = trait_profiles.get(target_mid, {})
        delta_frac = float(td.get("delta_positive_weight_frac", 0.5))
        signed_frac = float(td.get("signed_positive_weight_frac", 0.5))
        delta_consistency = 0.0
        signed_consistency = 0.0
        if int(td.get("trait_edge_count", 0)) > 0:
            delta_consistency = 1.0 - abs(
                delta_frac - float(prof.get("delta_positive_weight_frac", 0.5))
            )
            signed_consistency = 1.0 - abs(
                signed_frac - float(prof.get("signed_positive_weight_frac", 0.5))
            )

        rows.append({
            "target_module": target_mid,
            "delta_q": float(delta_q),
            "cross_abs_weight": float(target_cross),
            "strength_fraction": float(target_cross / total_cross) if total_cross > 0 else 0.0,
            "topk_mean_abs_weight": float(sd.get("topk_mean_abs_weight", 0.0)),
            "n_selected_edges_to_module": int(sd.get("n_edges", 0)),
            "delta_positive_weight_frac_to_module": float(delta_frac),
            "signed_positive_weight_frac_to_module": float(signed_frac),
            "delta_direction_consistency": float(delta_consistency),
            "signed_direction_consistency": float(signed_consistency),
        })
    return rows


def _rank_candidate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Deterministic lexicographic ranking with RB quality as the primary objective."""
    return sorted(
        (dict(r) for r in rows),
        key=lambda r: (
            -float(r.get("delta_q", -1e300)),
            -float(r.get("topk_mean_abs_weight", 0.0)),
            -float(r.get("strength_fraction", 0.0)),
            -float(r.get("delta_direction_consistency", 0.0)),
            -float(r.get("signed_direction_consistency", 0.0)),
            str(r.get("target_module", "")),
        ),
    )


def _empty_small_assignment_df() -> pl.DataFrame:
    """Empty audit table for frozen-core maximum-spanning-forest assignment."""
    return pl.DataFrame({
        "raw_community_id": pl.Series([], dtype=pl.Int64),
        "raw_community_size": pl.Series([], dtype=pl.Int64),
        "assigned_module": pl.Series([], dtype=pl.Utf8),
        "final_module_size": pl.Series([], dtype=pl.Int64),
        "target_module_size": pl.Series([], dtype=pl.Int64),
        "assignment_basis": pl.Series([], dtype=pl.Utf8),
        "assignment_status": pl.Series([], dtype=pl.Utf8),
        "component_total_size": pl.Series([], dtype=pl.Int64),
        "exclusion_reason": pl.Series([], dtype=pl.Utf8),
        "component_has_original_core": pl.Series([], dtype=pl.Boolean),
        "anchor_raw_community_id": pl.Series([], dtype=pl.Int64),
        "anchor_raw_community_size": pl.Series([], dtype=pl.Int64),
        "strongest_neighbor_raw_community_id": pl.Series([], dtype=pl.Int64),
        "strongest_neighbor_normalized_strength": pl.Series([], dtype=pl.Float64),
        "forest_first_neighbor_raw_community_id": pl.Series([], dtype=pl.Int64),
        "path_to_anchor": pl.Series([], dtype=pl.Utf8),
        "path_length": pl.Series([], dtype=pl.Int64),
        "path_bottleneck_normalized_strength": pl.Series([], dtype=pl.Float64),
        "normalized_cross_strength": pl.Series([], dtype=pl.Float64),
        "second_best_normalized_cross_strength": pl.Series([], dtype=pl.Float64),
        "normalized_strength_margin": pl.Series([], dtype=pl.Float64),
        "cross_abs_weight": pl.Series([], dtype=pl.Float64),
        "strength_fraction": pl.Series([], dtype=pl.Float64),
        "topk_mean_abs_weight": pl.Series([], dtype=pl.Float64),
        "n_selected_edges_to_module": pl.Series([], dtype=pl.Int64),
        # Retained for backward-compatible diagnostics only. They do not determine
        # frozen-core maximum-spanning-forest assignment.
        "rb_delta_q": pl.Series([], dtype=pl.Float64),
        "second_best_rb_delta_q": pl.Series([], dtype=pl.Float64),
        "rb_delta_q_margin": pl.Series([], dtype=pl.Float64),
        "rb_expected_cross_weight": pl.Series([], dtype=pl.Float64),
        "rb_resolution_adjusted_expected_cross_weight": pl.Series([], dtype=pl.Float64),
        "rb_observed_expected_ratio": pl.Series([], dtype=pl.Float64),
        "delta_direction_consistency": pl.Series([], dtype=pl.Float64),
        "signed_direction_consistency": pl.Series([], dtype=pl.Float64),
    })


def _constrained_assign_small_raw_communities(
    *,
    raw_part: Dict[str, int],
    all_nodes: Iterable[str],
    G_module: nx.Graph,
    G_analysis: nx.Graph,
    min_module_size: int,
    resolution: float,
    top_k: int,
) -> Tuple[
    Dict[str, str],
    Dict[str, List[str]],
    pl.DataFrame,
    Dict[int, List[str]],
    Dict[str, int],
    Dict[str, int],
    set,
    float,
    float,
]:
    """Assign small Leiden communities with an RB-corrected frozen-core forest.

    The Leiden partition is treated as a raw-community graph:

      * every raw Leiden community is one super-node;
      * an edge exists between two super-nodes only when at least one selected-network
        gene-gene edge connects the corresponding raw communities;
      * the primary static edge score is the exact pairwise RB/configuration-model
        quality change for merging two raw communities:

            delta_Q_RB(A, B)
                = W_AB / m
                  - resolution * K_A * K_B / (2 * m^2)

        where W_AB is the total cross-community abs(w_ft), K_A and K_B are
        raw-community weighted-degree/strength sums, and m is total network edge
        weight. This explicitly penalizes the extra cross-weight expected merely
        because a candidate target has large network volume/strength.
      * size-normalized cross strength

            W_AB / sqrt(|A| * |B|)

        is retained only as deterministic secondary evidence/tie-breaking.

    Raw communities with size >= ``min_module_size`` are frozen core anchors. Their
    original genes are never moved, split, or merged with another original core.

    All raw-community edges are sorted globally by descending RB delta-Q and processed
    with a Kruskal-style disjoint-set algorithm. A candidate edge is accepted unless it
    would connect two components that already contain two different original cores.
    Because ``min_module_size`` is an explicit post-processing constraint, negative
    delta-Q edges are not automatically discarded: if every legal connection is
    unfavorable, the least damaging (largest) delta-Q connection is considered first.
    Thus the procedure removes target-volume preference from the primary ranking without
    silently changing the existing minimum-size assignment semantics.

    The rooted/core-constrained maximum-spanning-forest rule preserves three properties:

      1) small communities may connect through small-small paths before reaching a core;
      2) the edge scores are static, so there is no dynamic merge-order dependence;
      3) two original Leiden core modules can never be merged by post-processing.

    A connected raw-community component that contains no original core is allowed to
    become a new final module only when its combined gene count reaches
    ``min_module_size``. A disconnected coreless component that remains below the
    threshold is explicitly left unassigned and excluded from Hub selection.

    RB quality is still reported before/after for auditability, but RB delta-Q is now
    also the PRIMARY small-community assignment score.
    """
    del G_analysis  # Assignment is intentionally based on selected-network strength only.

    raw_by_cid, raw_cid_of, raw_size_of = _build_raw_community_map(raw_part, all_nodes)
    if not raw_by_cid:
        return {}, {}, _empty_small_assignment_df(), {}, {}, {}, set(), 0.0, 0.0

    effective_min = max(2, int(min_module_size))
    ordered_raw = sorted(raw_by_cid.items(), key=lambda kv: (-len(kv[1]), int(kv[0])))
    raw_sizes = {int(cid): int(len(nodes)) for cid, nodes in ordered_raw}

    # Original Leiden communities meeting the threshold are immutable anchors.
    core = [(int(cid), list(nodes)) for cid, nodes in ordered_raw if len(nodes) >= effective_min]
    core_cids = set(cid for cid, _nodes in core)
    core_order = [cid for cid, _nodes in core]

    raw_modules_for_q = {
        f"raw_C{int(cid):04d}": list(nodes)
        for cid, nodes in ordered_raw
    }
    raw_q = _rb_partition_quality(G_module, raw_modules_for_q, resolution)

    # RB/configuration-model quantities used to remove target-module volume bias.
    # G_module stores abs(weight_col) as the non-negative edge weight.
    m_total = float(G_module.size(weight="weight"))
    if m_total <= 0.0:
        raise ValueError("Selected module network has zero total edge weight.")
    weighted_degree = _weighted_degree_map(G_module)
    raw_strength = {
        int(cid): float(sum(weighted_degree.get(str(g), 0.0) for g in genes))
        for cid, genes in ordered_raw
    }

    # -------------------------------------------------------------------------
    # Aggregate the gene-level selected network into a static raw-community graph.
    # -------------------------------------------------------------------------
    pair_acc: Dict[Tuple[int, int], Dict[str, object]] = defaultdict(lambda: {
        "cross_abs_weight": 0.0,
        "n_edges": 0,
        "strengths": [],
    })

    for u, v, d in G_module.edges(data=True):
        u_s = str(u)
        v_s = str(v)
        cid_u = raw_cid_of.get(u_s)
        cid_v = raw_cid_of.get(v_s)
        if cid_u is None or cid_v is None or int(cid_u) == int(cid_v):
            continue
        w = max(0.0, safe_float(d.get("weight"), 0.0))
        if w <= 0.0:
            continue
        a, b = sorted((int(cid_u), int(cid_v)))
        acc = pair_acc[(a, b)]
        acc["cross_abs_weight"] = float(acc["cross_abs_weight"]) + w
        acc["n_edges"] = int(acc["n_edges"]) + 1
        acc["strengths"].append(float(w))

    k = max(1, int(top_k))
    community_edges: List[Dict[str, object]] = []
    incident_edges: Dict[int, List[Dict[str, object]]] = defaultdict(list)

    for (a, b), acc in pair_acc.items():
        cross = float(acc["cross_abs_weight"])
        n_edges = int(acc["n_edges"])
        if cross <= 0.0 or n_edges <= 0:
            continue
        strengths = sorted((float(x) for x in acc["strengths"]), reverse=True)
        top = strengths[:min(k, len(strengths))]
        topk_mean = float(np.mean(top)) if top else 0.0
        denom = math.sqrt(float(raw_sizes[a]) * float(raw_sizes[b]))
        normalized = cross / denom if denom > 0.0 else 0.0

        k_a = float(raw_strength.get(int(a), 0.0))
        k_b = float(raw_strength.get(int(b), 0.0))
        expected_cross = (k_a * k_b) / (2.0 * m_total)
        resolution_adjusted_expected = float(resolution) * expected_cross
        rb_delta_q = (cross / m_total) - (
            float(resolution) * k_a * k_b / (2.0 * m_total * m_total)
        )
        observed_expected_ratio = (
            cross / expected_cross if expected_cross > 0.0 else float("inf")
        )

        row = {
            "cid_a": int(a),
            "cid_b": int(b),
            "size_a": int(raw_sizes[a]),
            "size_b": int(raw_sizes[b]),
            "raw_strength_a": float(k_a),
            "raw_strength_b": float(k_b),
            "rb_expected_cross_weight": float(expected_cross),
            "rb_resolution_adjusted_expected_cross_weight": float(resolution_adjusted_expected),
            "rb_observed_expected_ratio": float(observed_expected_ratio),
            "rb_delta_q": float(rb_delta_q),
            "normalized_cross_strength": float(normalized),
            "cross_abs_weight": float(cross),
            "n_edges": int(n_edges),
            "topk_mean_abs_weight": float(topk_mean),
        }
        community_edges.append(row)
        incident_edges[a].append(row)
        incident_edges[b].append(row)

    # Fixed global ordering: RB/configuration-model delta-Q is PRIMARY.
    # Size-normalized strength and local strongest-edge support are secondary.
    community_edges.sort(key=lambda r: (
        -float(r["rb_delta_q"]),
        -float(r["normalized_cross_strength"]),
        -float(r["topk_mean_abs_weight"]),
        -float(r["cross_abs_weight"]),
        -int(r["n_edges"]),
        int(r["cid_a"]),
        int(r["cid_b"]),
    ))

    # -------------------------------------------------------------------------
    # Core-constrained Kruskal maximum spanning forest.
    # -------------------------------------------------------------------------
    parent = {int(cid): int(cid) for cid in raw_by_cid}
    rank = {int(cid): 0 for cid in raw_by_cid}
    component_core: Dict[int, Optional[int]] = {
        int(cid): (int(cid) if int(cid) in core_cids else None)
        for cid in raw_by_cid
    }

    def find(x: int) -> int:
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return ra
        core_a = component_core.get(ra)
        core_b = component_core.get(rb)
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
            core_a, core_b = core_b, core_a
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        component_core[ra] = core_a if core_a is not None else core_b
        component_core.pop(rb, None)
        return ra

    accepted_edges: List[Dict[str, object]] = []
    forest_adj: Dict[int, List[Tuple[int, Dict[str, object]]]] = defaultdict(list)

    for edge in community_edges:
        a = int(edge["cid_a"])
        b = int(edge["cid_b"])
        ra = find(a)
        rb = find(b)
        if ra == rb:
            continue
        core_a = component_core.get(ra)
        core_b = component_core.get(rb)

        # The defining frozen-core constraint: never connect two different anchors.
        if core_a is not None and core_b is not None and int(core_a) != int(core_b):
            continue

        union(a, b)
        accepted = dict(edge)
        accepted_edges.append(accepted)
        forest_adj[a].append((b, accepted))
        forest_adj[b].append((a, accepted))

    # Rebuild final components after path compression.
    component_cids: Dict[int, List[int]] = defaultdict(list)
    for cid in sorted(raw_by_cid):
        component_cids[find(int(cid))].append(int(cid))

    component_core_final: Dict[int, Optional[int]] = {}
    for root, cids in component_cids.items():
        anchors = sorted(cid for cid in cids if cid in core_cids)
        if len(anchors) > 1:
            raise RuntimeError(
                "Internal consistency error: a frozen-core forest component contains "
                "multiple original core communities: " + ", ".join(
                    f"C{cid:04d}" for cid in anchors
                )
            )
        component_core_final[root] = anchors[0] if anchors else None

    # A coreless connected component is defensible as a new formal module only if
    # the accumulated small communities together satisfy the requested minimum size.
    # Smaller coreless components are NOT forced into an unrelated module.  They are
    # explicitly left unassigned, audited below, and excluded from Hub selection.
    unassigned_coreless = []
    for root, cids in component_cids.items():
        if component_core_final[root] is not None:
            continue
        n_genes = sum(raw_sizes[cid] for cid in cids)
        if n_genes < effective_min:
            unassigned_coreless.append((root, list(cids), int(n_genes)))

    unassigned_roots = {int(root) for root, _cids, _n in unassigned_coreless}
    unassigned_component_size = {
        int(root): int(n_genes)
        for root, _cids, n_genes in unassigned_coreless
    }
    if unassigned_coreless:
        details = []
        for _root, cids, n_genes in unassigned_coreless[:10]:
            labels = ",".join(f"C{cid:04d}(n={raw_sizes[cid]})" for cid in sorted(cids))
            details.append(f"[{labels}] total_n={n_genes}")
        extra = "" if len(unassigned_coreless) <= 10 else f"; plus {len(unassigned_coreless)-10} more"
        print(
            "UNASSIGNED disconnected components below min_module_size: "
            + "; ".join(details) + extra
        )

    # -------------------------------------------------------------------------
    # Name final modules while preserving every original core module id/order.
    # -------------------------------------------------------------------------
    root_to_mid: Dict[int, str] = {}
    modules: Dict[str, List[str]] = {}

    core_mid_by_cid = {
        int(cid): f"M{rank_idx:02d}"
        for rank_idx, cid in enumerate(core_order, start=1)
    }

    for core_cid in core_order:
        root = find(core_cid)
        mid = core_mid_by_cid[core_cid]
        root_to_mid[root] = mid

    coreless_roots = [
        root for root, anchor in component_core_final.items()
        if anchor is None and int(root) not in unassigned_roots
    ]
    coreless_roots.sort(key=lambda root: (
        -sum(raw_sizes[cid] for cid in component_cids[root]),
        min(component_cids[root]),
    ))
    next_mid = len(core_order) + 1
    for root in coreless_roots:
        root_to_mid[root] = f"M{next_mid:02d}"
        next_mid += 1

    module_of: Dict[str, str] = {}
    for root, cids in component_cids.items():
        if int(root) in unassigned_roots:
            # These genes deliberately have no module id.  Because compute_hubs()
            # iterates only over ``modules``, they can never enter Hub selection.
            continue
        mid = root_to_mid[root]
        genes = sorted({
            str(g)
            for cid in cids
            for g in raw_by_cid[cid]
        })
        modules[mid] = genes
        for gene in genes:
            module_of[gene] = mid

    # Verify that every original core is intact and maps to its own final module.
    for core_cid, core_nodes in core:
        expected_mid = core_mid_by_cid[core_cid]
        if not set(str(g) for g in core_nodes).issubset(set(modules.get(expected_mid, []))):
            raise RuntimeError(
                f"Internal consistency error: original core C{core_cid:04d} was altered."
            )

    too_small = {
        mid: len(genes)
        for mid, genes in modules.items()
        if len(genes) < effective_min
    }
    if too_small:
        raise RuntimeError(
            "Internal consistency error: final modules below min_module_size remain: "
            + ", ".join(f"{mid}={n}" for mid, n in sorted(too_small.items()))
        )

    # -------------------------------------------------------------------------
    # Build an audit table explaining each small raw community's final assignment.
    # The accepted edges form a forest, so a path to the component's core is unique.
    # -------------------------------------------------------------------------
    edge_lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
    for edge in accepted_edges:
        a = int(edge["cid_a"])
        b = int(edge["cid_b"])
        edge_lookup[(min(a, b), max(a, b))] = edge

    def forest_path(start_cid: int, goal_cid: int) -> List[int]:
        if start_cid == goal_cid:
            return [start_cid]
        queue = [start_cid]
        prev: Dict[int, Optional[int]] = {start_cid: None}
        qi = 0
        while qi < len(queue):
            cur = queue[qi]
            qi += 1
            for nb, _edge in forest_adj.get(cur, []):
                if nb in prev:
                    continue
                prev[nb] = cur
                if nb == goal_cid:
                    queue = []
                    qi = 0
                    break
                queue.append(nb)
            if goal_cid in prev:
                break
        if goal_cid not in prev:
            return []
        path = []
        cur: Optional[int] = goal_cid
        while cur is not None:
            path.append(int(cur))
            cur = prev[cur]
        return list(reversed(path))

    audit_rows = []
    for cid, nodes in sorted(ordered_raw, key=lambda kv: int(kv[0])):
        cid = int(cid)
        if cid in core_cids:
            continue

        root = find(cid)
        is_unassigned = int(root) in unassigned_roots
        anchor_cid = component_core_final[root]
        component_total_n = int(sum(raw_sizes[x] for x in component_cids[root]))
        if is_unassigned:
            mid = ""
            final_size = 0
        else:
            mid = root_to_mid[root]
            final_size = len(modules[mid])

        local_edges = sorted(
            incident_edges.get(cid, []),
            key=lambda r: (
                -float(r["rb_delta_q"]),
                -float(r["normalized_cross_strength"]),
                -float(r["topk_mean_abs_weight"]),
                -float(r["cross_abs_weight"]),
                -int(r["n_edges"]),
                int(r["cid_a"]),
                int(r["cid_b"]),
            ),
        )
        strongest = local_edges[0] if local_edges else None
        second_local = local_edges[1] if len(local_edges) > 1 else strongest
        if strongest is not None:
            strongest_nb = int(strongest["cid_b"] if int(strongest["cid_a"]) == cid else strongest["cid_a"])
            strongest_norm = float(strongest["normalized_cross_strength"])
            second_norm = float(second_local["normalized_cross_strength"]) if second_local is not None else strongest_norm
            strongest_rb = float(strongest.get("rb_delta_q", 0.0))
            second_rb = float(second_local.get("rb_delta_q", strongest_rb)) if second_local is not None else strongest_rb
        else:
            strongest_nb = -1
            strongest_norm = 0.0
            second_norm = 0.0
            strongest_rb = 0.0
            second_rb = 0.0

        if is_unassigned:
            path = [cid]
            basis = "unassigned_coreless_disconnected_component_below_min_size"
            anchor_size = 0
            assignment_status = "unassigned_disconnected"
            exclusion_reason = "no_path_to_original_core_and_component_total_below_min_module_size"
        elif anchor_cid is not None:
            path = forest_path(cid, int(anchor_cid))
            basis = "rb_delta_q_primary_core_constrained_maximum_spanning_forest_to_frozen_core"
            anchor_size = raw_sizes[int(anchor_cid)]
            assignment_status = "assigned"
            exclusion_reason = ""
        else:
            # This final module is made entirely from mutually connected raw communities
            # that were individually below the threshold but together reach it.
            path = [cid]
            basis = "rb_delta_q_primary_coreless_small_community_component_reached_min_size"
            anchor_size = final_size
            assignment_status = "assigned"
            exclusion_reason = ""

        path_edges = []
        for a, b in zip(path[:-1], path[1:]):
            edge = edge_lookup.get((min(a, b), max(a, b)))
            if edge is not None:
                path_edges.append(edge)

        first_edge = path_edges[0] if path_edges else strongest
        if first_edge is not None:
            first_nb = int(
                first_edge["cid_b"]
                if int(first_edge["cid_a"]) == cid
                else first_edge["cid_a"]
            ) if (int(first_edge["cid_a"]) == cid or int(first_edge["cid_b"]) == cid) else -1
            first_norm = float(first_edge["normalized_cross_strength"])
            first_cross = float(first_edge["cross_abs_weight"])
            first_n_edges = int(first_edge["n_edges"])
            first_topk = float(first_edge["topk_mean_abs_weight"])
            first_rb = float(first_edge.get("rb_delta_q", 0.0))
            first_expected = float(first_edge.get("rb_expected_cross_weight", 0.0))
            first_expected_gamma = float(
                first_edge.get("rb_resolution_adjusted_expected_cross_weight", 0.0)
            )
            first_oe = float(first_edge.get("rb_observed_expected_ratio", 0.0))
        else:
            first_nb = -1
            first_norm = 0.0
            first_cross = 0.0
            first_n_edges = 0
            first_topk = 0.0
            first_rb = 0.0
            first_expected = 0.0
            first_expected_gamma = 0.0
            first_oe = 0.0

        bottleneck = (
            min(float(e["normalized_cross_strength"]) for e in path_edges)
            if path_edges else first_norm
        )
        total_incident_cross = sum(float(e["cross_abs_weight"]) for e in local_edges)
        strength_fraction = (
            first_cross / total_incident_cross if total_incident_cross > 0.0 else 0.0
        )

        audit_rows.append({
            "raw_community_id": cid,
            "raw_community_size": int(len(nodes)),
            "assigned_module": mid,
            "final_module_size": int(final_size),
            "target_module_size": int(anchor_size),
            "assignment_basis": basis,
            "assignment_status": assignment_status,
            "component_total_size": int(component_total_n),
            "exclusion_reason": exclusion_reason,
            "component_has_original_core": bool(anchor_cid is not None),
            "anchor_raw_community_id": int(anchor_cid) if anchor_cid is not None else -1,
            "anchor_raw_community_size": int(raw_sizes[int(anchor_cid)]) if anchor_cid is not None else 0,
            "strongest_neighbor_raw_community_id": int(strongest_nb),
            "strongest_neighbor_normalized_strength": float(strongest_norm),
            "forest_first_neighbor_raw_community_id": int(first_nb),
            "path_to_anchor": ">".join(f"C{x:04d}" for x in path) if path else "",
            "path_length": max(0, len(path) - 1),
            "path_bottleneck_normalized_strength": float(bottleneck),
            "normalized_cross_strength": float(first_norm),
            "second_best_normalized_cross_strength": float(second_norm),
            "normalized_strength_margin": float(strongest_norm - second_norm),
            "cross_abs_weight": float(first_cross),
            "strength_fraction": float(strength_fraction),
            "topk_mean_abs_weight": float(first_topk),
            "n_selected_edges_to_module": int(first_n_edges),
            "rb_delta_q": float(first_rb),
            "second_best_rb_delta_q": float(second_rb),
            "rb_delta_q_margin": float(first_rb - second_rb),
            "rb_expected_cross_weight": float(first_expected),
            "rb_resolution_adjusted_expected_cross_weight": float(first_expected_gamma),
            "rb_observed_expected_ratio": float(first_oe),
            "delta_direction_consistency": 0.0,
            "signed_direction_consistency": 0.0,
        })

    constrained_q = _rb_partition_quality(G_module, modules, resolution)
    assignment_df = pl.DataFrame(audit_rows) if audit_rows else _empty_small_assignment_df()
    return (
        module_of,
        modules,
        assignment_df,
        raw_by_cid,
        raw_cid_of,
        raw_size_of,
        core_cids,
        raw_q,
        constrained_q,
    )


def _build_refinement_units(
    raw_by_cid: Dict[int, List[str]],
    core_cids: set,
) -> List[Tuple[str, List[str]]]:
    """Core-community genes may move individually; small raw communities stay atomic."""
    units: List[Tuple[str, List[str]]] = []
    for cid, genes in sorted(raw_by_cid.items(), key=lambda kv: int(kv[0])):
        if int(cid) in core_cids:
            for gene in sorted(genes):
                units.append((f"gene:{gene}", [str(gene)]))
        else:
            units.append((f"raw:{int(cid)}", sorted(str(g) for g in genes)))
    return units


def _empty_refinement_df() -> pl.DataFrame:
    return pl.DataFrame({
        "iteration": pl.Series([], dtype=pl.Int64),
        "unit_id": pl.Series([], dtype=pl.Utf8),
        "unit_size": pl.Series([], dtype=pl.Int64),
        "from_module": pl.Series([], dtype=pl.Utf8),
        "to_module": pl.Series([], dtype=pl.Utf8),
        "rb_delta_q": pl.Series([], dtype=pl.Float64),
        "cross_abs_weight": pl.Series([], dtype=pl.Float64),
        "strength_fraction": pl.Series([], dtype=pl.Float64),
        "topk_mean_abs_weight": pl.Series([], dtype=pl.Float64),
        "delta_direction_consistency": pl.Series([], dtype=pl.Float64),
        "signed_direction_consistency": pl.Series([], dtype=pl.Float64),
    })


def _refine_constrained_partition(
    *,
    module_of: Dict[str, str],
    modules: Dict[str, List[str]],
    raw_by_cid: Dict[int, List[str]],
    core_cids: set,
    G_module: nx.Graph,
    G_analysis: nx.Graph,
    min_module_size: int,
    resolution: float,
    top_k: int,
    max_iter: int,
) -> Tuple[Dict[str, str], Dict[str, List[str]], pl.DataFrame, int]:
    """Deterministic positive-gain local refinement under a minimum-size constraint.

    Every accepted move has strictly positive exact RB-quality gain, so the procedure
    cannot cycle. Small Leiden communities remain intact as blocks; genes from core
    communities are allowed to adjust individually at module boundaries. Core modules
    are never allowed to shrink below the effective minimum size; a disconnected small
    module that was explicitly retained is not forced to grow or merge.
    """
    if int(max_iter) <= 0:
        return (
            {str(g): str(mid) for g, mid in module_of.items()},
            {str(mid): sorted(str(g) for g in genes) for mid, genes in modules.items()},
            _empty_refinement_df(),
            0,
        )

    effective_min = max(2, int(min_module_size))
    gain_tol = 1e-12
    weighted_degree = _weighted_degree_map(G_module)
    units = _build_refinement_units(raw_by_cid, core_cids)
    rows = []
    passes_run = 0

    # Mutable sets make repeated moves cheap; output is converted back to sorted lists.
    module_sets = {mid: set(str(g) for g in genes) for mid, genes in modules.items()}
    module_of = {str(g): str(mid) for g, mid in module_of.items()}

    for iteration in range(1, max(0, int(max_iter)) + 1):
        passes_run = iteration
        modules_snapshot = {mid: sorted(nodes) for mid, nodes in module_sets.items()}
        module_degree = _module_degree_sums(modules_snapshot, weighted_degree)
        trait_profiles = _module_trait_profiles(G_analysis, modules_snapshot)
        proposals = []

        for unit_id, nodes in units:
            nodes = [str(g) for g in nodes]
            source_mids = {module_of.get(g) for g in nodes}
            if len(source_mids) != 1 or None in source_mids:
                continue
            source = str(next(iter(source_mids)))
            if len(module_sets[source]) - len(nodes) < effective_min:
                continue

            candidates = _candidate_evidence(
                block_nodes=nodes,
                source_mid=source,
                modules=modules_snapshot,
                module_of=module_of,
                G_module=G_module,
                G_analysis=G_analysis,
                weighted_degree=weighted_degree,
                module_degree=module_degree,
                resolution=resolution,
                top_k=top_k,
                trait_profiles=trait_profiles,
            )
            ranked = _rank_candidate_rows(candidates)
            if ranked and float(ranked[0]["delta_q"]) > gain_tol:
                proposals.append((unit_id, nodes, source, ranked[0]))

        if not proposals:
            break

        # Best-gain proposals are attempted first. Each one is re-evaluated against
        # the updated partition before acceptance, so every recorded delta is exact
        # for the state in which the move was actually made.
        proposals.sort(key=lambda x: (
            -float(x[3]["delta_q"]),
            -float(x[3]["topk_mean_abs_weight"]),
            -float(x[3]["strength_fraction"]),
            str(x[0]),
        ))

        moved_this_pass = 0
        for unit_id, nodes, _source_snapshot, _best_snapshot in proposals:
            source_mids = {module_of.get(g) for g in nodes}
            if len(source_mids) != 1 or None in source_mids:
                continue
            source = str(next(iter(source_mids)))
            if len(module_sets[source]) - len(nodes) < effective_min:
                continue

            current_modules = {mid: sorted(vals) for mid, vals in module_sets.items()}
            # Exact RB move gain depends on current module degree sums. Trait profiles
            # are deliberately held fixed within one pass: they are secondary evidence,
            # while recomputing dense internal trait profiles after every accepted move
            # would add major cost without changing the primary RB objective.
            current_degree = _module_degree_sums(current_modules, weighted_degree)
            candidates = _candidate_evidence(
                block_nodes=nodes,
                source_mid=source,
                modules=current_modules,
                module_of=module_of,
                G_module=G_module,
                G_analysis=G_analysis,
                weighted_degree=weighted_degree,
                module_degree=current_degree,
                resolution=resolution,
                top_k=top_k,
                trait_profiles=trait_profiles,
            )
            ranked = _rank_candidate_rows(candidates)
            if not ranked:
                continue
            best = ranked[0]
            if float(best["delta_q"]) <= gain_tol:
                continue

            target = str(best["target_module"])
            if target == source:
                continue
            for g in nodes:
                module_sets[source].discard(g)
                module_sets[target].add(g)
                module_of[g] = target

            rows.append({
                "iteration": int(iteration),
                "unit_id": str(unit_id),
                "unit_size": int(len(nodes)),
                "from_module": source,
                "to_module": target,
                "rb_delta_q": float(best["delta_q"]),
                "cross_abs_weight": float(best["cross_abs_weight"]),
                "strength_fraction": float(best["strength_fraction"]),
                "topk_mean_abs_weight": float(best["topk_mean_abs_weight"]),
                "delta_direction_consistency": float(best["delta_direction_consistency"]),
                "signed_direction_consistency": float(best["signed_direction_consistency"]),
            })
            moved_this_pass += 1

        if moved_this_pass == 0:
            break

    final_modules = {mid: sorted(nodes) for mid, nodes in module_sets.items()}
    refinement_df = pl.DataFrame(rows) if rows else _empty_refinement_df()
    return module_of, final_modules, refinement_df, passes_run


def trait_aware_refine_modules(
    *,
    raw_part: Dict[str, int],
    all_nodes: Iterable[str],
    G_module: nx.Graph,
    G_analysis: nx.Graph,
    min_module_size: int,
    resolution: float,
    top_k: int,
    max_iter: int,
) -> Tuple[
    Dict[str, str],
    Dict[str, List[str]],
    pl.DataFrame,
    pl.DataFrame,
    Dict[str, int],
    Dict[str, int],
    Dict[str, str],
    Dict[str, float],
]:
    """Frozen-core community assignment after the initial Leiden partition.

    ``max_iter`` is intentionally ignored for assignment. The previous optional
    boundary-refinement stage could move genes out of original core communities, which
    conflicts with the frozen-core requirement. The command-line argument is kept only
    for backward compatibility.
    """
    (
        module_of,
        modules,
        small_assignment_df,
        raw_by_cid,
        raw_cid_of,
        raw_size_of,
        core_cids,
        raw_q,
        constrained_q,
    ) = _constrained_assign_small_raw_communities(
        raw_part=raw_part,
        all_nodes=all_nodes,
        G_module=G_module,
        G_analysis=G_analysis,
        min_module_size=min_module_size,
        resolution=resolution,
        top_k=top_k,
    )

    if int(max_iter) > 0:
        print(
            "WARNING: --refine_max_iter is ignored under frozen-core "
            "maximum-spanning-forest assignment so original Leiden core modules "
            "cannot be altered."
        )

    initial_module_of = dict(module_of)
    refinement_df = _empty_refinement_df()
    passes_run = 0
    final_q = _rb_partition_quality(G_module, modules, resolution)
    small_rows = list(small_assignment_df.iter_rows(named=True)) if small_assignment_df.height > 0 else []
    n_small_unassigned = sum(
        1 for r in small_rows
        if str(r.get("assignment_status", "")) == "unassigned_disconnected"
    )
    n_small_reassigned = len(small_rows) - n_small_unassigned
    n_unassigned_genes = sum(
        int(r.get("raw_community_size", 0) or 0)
        for r in small_rows
        if str(r.get("assignment_status", "")) == "unassigned_disconnected"
    )

    quality = {
        "raw_rb_quality": float(raw_q),
        "after_small_reassignment_rb_quality": float(constrained_q),
        "final_rb_quality": float(final_q),
        "refinement_passes_run": int(passes_run),
        "refinement_moves": 0,
        "small_raw_communities_reassigned": int(n_small_reassigned),
        "small_raw_communities_retained_disconnected": 0,
        "small_raw_communities_unassigned_disconnected": int(n_small_unassigned),
        "unassigned_disconnected_genes": int(n_unassigned_genes),
        "original_core_communities_frozen": int(len(core_cids)),
    }
    return (
        module_of,
        modules,
        small_assignment_df,
        refinement_df,
        raw_cid_of,
        raw_size_of,
        initial_module_of,
        quality,
    )


# =============================================================================
# Hub identification and summaries
# =============================================================================


def choose_hub_count(n_nodes: int, hub_top_frac: float, min_hubs: int, max_hubs: int) -> int:
    if n_nodes <= 0:
        return 0
    n = int(math.ceil(n_nodes * float(hub_top_frac)))
    n = max(int(min_hubs), n)
    n = min(int(max_hubs), n)
    return min(n_nodes, n)


def compute_hubs(G_full: nx.Graph, modules: Dict[str, List[str]], hub_top_frac: float, min_hubs: int, max_hubs: int) -> Tuple[Dict[str, List[str]], pl.DataFrame]:
    hub_sets: Dict[str, List[str]] = {}
    rows = []

    global_wdeg = dict(G_full.degree(weight="weight"))
    global_deg = dict(G_full.degree())

    for mid, genes in sorted(modules.items()):
        sub = G_full.subgraph(genes).copy()
        intra_wdeg = dict(sub.degree(weight="weight"))
        intra_deg = dict(sub.degree())

        ranked = sorted(
            genes,
            key=lambda g: (float(intra_wdeg.get(g, 0.0)), int(intra_deg.get(g, 0))),
            reverse=True,
        )
        n_hub = choose_hub_count(len(genes), hub_top_frac, min_hubs, max_hubs)
        hubs = ranked[:n_hub]
        hub_sets[mid] = hubs

        for rank, gene in enumerate(hubs, start=1):
            incident_delta = [safe_float(G_full[gene][nb].get("delta_w"), 0.0) for nb in G_full.neighbors(gene)] if gene in G_full else []
            rows.append({
                "module_id": mid,
                "hub_rank": rank,
                "gene_name": gene,
                "within_abs_weighted_degree": round(float(intra_wdeg.get(gene, 0.0)), 8),
                "within_degree": int(intra_deg.get(gene, 0)),
                "global_abs_weighted_degree": round(float(global_wdeg.get(gene, 0.0)), 8),
                "global_degree": int(global_deg.get(gene, 0)),
                "mean_incident_delta_w": round(float(np.mean(incident_delta)), 8) if incident_delta else 0.0,
            })

    return hub_sets, pl.DataFrame(rows) if rows else pl.DataFrame()


def summarize_modules(G_full: nx.Graph, modules: Dict[str, List[str]], hub_sets: Dict[str, List[str]], node_type: Dict[str, str]) -> pl.DataFrame:
    rows = []
    for mid, genes in sorted(modules.items()):
        sub = G_full.subgraph(genes).copy()
        strengths = [float(d.get("weight", 0.0)) for _, _, d in sub.edges(data=True)]
        signed = [safe_float(d.get("signed_weight"), 0.0) for _, _, d in sub.edges(data=True)]
        deltas = [safe_float(d.get("delta_w"), 0.0) for _, _, d in sub.edges(data=True)]
        n_old = sum(1 for g in genes if node_type.get(g, "unknown") == "old")
        n_new = sum(1 for g in genes if node_type.get(g, "unknown") == "new")
        n_unknown = len(genes) - n_old - n_new
        rows.append({
            "module_id": mid,
            "n_genes": len(genes),
            "n_hubs": len(hub_sets.get(mid, [])),
            "top_hubs": ";".join(hub_sets.get(mid, [])[:10]),
            "n_internal_edges": sub.number_of_edges(),
            "internal_density": round(float(nx.density(sub)), 8) if sub.number_of_nodes() > 1 else 0.0,
            "mean_abs_w_ft_internal": round(float(np.mean(strengths)), 8) if strengths else 0.0,
            "median_abs_w_ft_internal": round(float(np.median(strengths)), 8) if strengths else 0.0,
            "mean_signed_w_ft_internal": round(float(np.mean(signed)), 8) if signed else 0.0,
            "mean_delta_w_internal": round(float(np.mean(deltas)), 8) if deltas else 0.0,
            "positive_delta_frac_internal": round(float(np.mean(np.array(deltas) > 0)), 8) if deltas else 0.0,
            "n_old_nodes": n_old,
            "n_new_nodes": n_new,
            "n_unknown_nodes": n_unknown,
        })
    return pl.DataFrame(rows)


def make_module_assignment_table(modules: Dict[str, List[str]], node_type: Dict[str, str], G_full: nx.Graph) -> pl.DataFrame:
    rows = []
    global_wdeg = dict(G_full.degree(weight="weight"))
    global_deg = dict(G_full.degree())
    for mid, genes in sorted(modules.items()):
        for gene in genes:
            rows.append({
                "gene_name": gene,
                "module_id": mid,
                "node_type": node_type.get(gene, "unknown"),
                "global_abs_weighted_degree": round(float(global_wdeg.get(gene, 0.0)), 8),
                "global_degree": int(global_deg.get(gene, 0)),
            })
    return pl.DataFrame(rows)


# =============================================================================
# Export gene sets
# =============================================================================


def export_gene_sets(out_dir: str, modules: Dict[str, List[str]], hub_sets: Dict[str, List[str]]) -> None:
    gene_set_dir = ensure_dir(os.path.join(out_dir, "gene_sets"))
    module_gene_dir = ensure_dir(os.path.join(gene_set_dir, "module_genes"))
    module_hub_dir = ensure_dir(os.path.join(gene_set_dir, "module_hubs"))

    for mid, genes in sorted(modules.items()):
        write_gene_list(os.path.join(module_gene_dir, f"{mid}_genes.txt"), genes)
        write_gene_list(os.path.join(module_hub_dir, f"{mid}_hubs.txt"), hub_sets.get(mid, []))


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    edge_table = resolve_edge_table(args)
    out_dir = ensure_dir(os.path.abspath(os.path.expanduser(args.out_dir)))
    tables_dir = ensure_dir(os.path.join(out_dir, "tables"))

    print("=" * 80)
    print("Trait-aware bulk network downstream analysis (non-plotting)")
    print("=" * 80)
    print(f"edge_table        : {edge_table}")
    print(f"out_dir           : {out_dir}")
    print(f"signed_weight_col : {args.weight_col}")
    print(f"analysis_weight   : abs({args.weight_col})")
    print(f"module_top_frac   : {args.module_top_frac}")
    print(f"min_abs_weight    : {args.min_abs_weight}")
    print(f"min_module_size   : {args.min_module_size}")
    print(f"resolution        : {args.resolution}")
    print(f"allow_fallback    : {args.allow_community_fallback}")
    print(f"reassign_top_k    : {args.reassign_top_k}")
    print(f"refine_max_iter   : {args.refine_max_iter}")

    df = load_edge_table(edge_table, args.weight_col)
    node_type = get_node_type_map(df)

    G_full = build_graph_from_edges(df, signed_weight_col=args.weight_col)
    add_all_observed_nodes(G_full, df)

    module_edges = select_module_edges(df, args.module_top_frac, args.min_abs_weight)
    G_module = build_graph_from_edges(module_edges, signed_weight_col=args.weight_col)

    # Module detection and small-community reassignment use only nodes that remain
    # connected in the selected module edge set, matching the constrained implementation.
    # G_full is deliberately kept unchanged and is still used below for the original
    # weighted-degree Hub ranking.
    observed_nodes = sorted(str(n) for n in G_full.nodes())
    active_nodes = sorted(str(n) for n in G_module.nodes())
    active_set = set(active_nodes)
    excluded_nodes = [n for n in observed_nodes if n not in active_set]

    if G_module.number_of_edges() == 0 or G_module.number_of_nodes() == 0:
        raise ValueError(
            "The selected module network is empty. Relax --module_top_frac "
            "and/or --min_abs_weight."
        )

    print(f"observed_nodes    : {G_full.number_of_nodes():,}")
    print(f"full_edges        : {G_full.number_of_edges():,}")
    print(f"module_nodes      : {G_module.number_of_nodes():,}")
    print(f"excluded_nodes    : {len(excluded_nodes):,}")
    print(f"module_edges      : {G_module.number_of_edges():,}")

    raw_part, community_method = detect_modules(
        G_module,
        resolution=args.resolution,
        seed=args.seed,
        allow_fallback=args.allow_community_fallback,
    )

    # Diagnostic output only: report the raw Leiden community structure before
    # small-community reassignment. This does not alter module membership.
    raw_summary_df, raw_membership_df, raw_stats = diagnose_raw_communities(
        raw_part=raw_part,
        all_nodes=active_nodes,
        min_module_size=args.min_module_size,
        method=community_method,
        resolution=args.resolution,
    )

    (
        module_of,
        modules,
        small_assignment_df,
        refinement_df,
        raw_cid_of,
        raw_size_of,
        initial_module_of,
        quality_stats,
    ) = trait_aware_refine_modules(
        raw_part=raw_part,
        all_nodes=active_nodes,
        G_module=G_module,
        G_analysis=G_module,
        min_module_size=args.min_module_size,
        resolution=args.resolution,
        top_k=args.reassign_top_k,
        max_iter=args.refine_max_iter,
    )

    hub_sets, hub_df = compute_hubs(
        G_full=G_full,
        modules=modules,
        hub_top_frac=args.hub_top_frac,
        min_hubs=args.min_hubs,
        max_hubs=args.max_hubs,
    )

    module_summary = summarize_modules(G_full, modules, hub_sets, node_type)
    assignment = make_module_assignment_table(modules, node_type, G_full)

    # Active selected-network genes absent from module_of are deliberately unassigned
    # disconnected micro-components below min_module_size.  They are not modules and
    # therefore cannot enter Hub selection or module gene-set exports.
    unassigned_nodes = sorted(set(active_nodes) - set(module_of.keys()))
    audit_by_cid = {}
    if small_assignment_df.height > 0:
        for row in small_assignment_df.iter_rows(named=True):
            if str(row.get("assignment_status", "")) == "unassigned_disconnected":
                audit_by_cid[int(row.get("raw_community_id", -1))] = row

    global_wdeg = dict(G_full.degree(weight="weight"))
    global_deg = dict(G_full.degree())
    unassigned_rows = []
    for gene in unassigned_nodes:
        cid = int(raw_cid_of.get(gene, -1))
        audit = audit_by_cid.get(cid, {})
        unassigned_rows.append({
            "gene_name": gene,
            "raw_community_id": cid,
            "raw_community_size": int(raw_size_of.get(gene, 0)),
            "component_total_size": int(audit.get("component_total_size", raw_size_of.get(gene, 0)) or 0),
            "status": "unassigned_disconnected",
            "reason": str(audit.get(
                "exclusion_reason",
                "no_path_to_original_core_and_component_total_below_min_module_size",
            )),
            "node_type": node_type.get(gene, "unknown"),
            "global_abs_weighted_degree": round(float(global_wdeg.get(gene, 0.0)), 8),
            "global_degree": int(global_deg.get(gene, 0)),
        })
    unassigned_df = pl.DataFrame(unassigned_rows) if unassigned_rows else pl.DataFrame({
        "gene_name": pl.Series([], dtype=pl.Utf8),
        "raw_community_id": pl.Series([], dtype=pl.Int64),
        "raw_community_size": pl.Series([], dtype=pl.Int64),
        "component_total_size": pl.Series([], dtype=pl.Int64),
        "status": pl.Series([], dtype=pl.Utf8),
        "reason": pl.Series([], dtype=pl.Utf8),
        "node_type": pl.Series([], dtype=pl.Utf8),
        "global_abs_weighted_degree": pl.Series([], dtype=pl.Float64),
        "global_degree": pl.Series([], dtype=pl.Int64),
    })

    total_module_nodes = sum(len(v) for v in modules.values())
    total_modules = len(modules)

    module_count_summary = pl.DataFrame([
        {"metric": "observed_nodes", "value": str(G_full.number_of_nodes())},
        {"metric": "module_active_nodes", "value": str(G_module.number_of_nodes())},
        {"metric": "excluded_nodes_after_module_edge_filter", "value": str(len(excluded_nodes))},
        {"metric": "assigned_nodes_total", "value": str(total_module_nodes)},
        {"metric": "unassigned_disconnected_nodes", "value": str(len(unassigned_nodes))},
        {"metric": "modules_total", "value": str(total_modules)},
        {"metric": "community_method", "value": str(community_method)},
        {"metric": "small_raw_modules_policy", "value": "frozen_core_constrained_maximum_spanning_forest"},
        {"metric": "small_raw_modules_reassigned", "value": str(int(quality_stats.get("small_raw_communities_reassigned", 0)))},
        {"metric": "small_raw_modules_retained_disconnected", "value": str(int(quality_stats.get("small_raw_communities_retained_disconnected", 0)))},
        {"metric": "small_raw_modules_unassigned_disconnected", "value": str(int(quality_stats.get("small_raw_communities_unassigned_disconnected", 0)))},
        {"metric": "unassigned_disconnected_genes", "value": str(int(quality_stats.get("unassigned_disconnected_genes", 0)))},
        {"metric": "min_module_size", "value": str(int(args.min_module_size))},
    ])

    assignment.write_csv(os.path.join(tables_dir, "module_assignment.csv"))
    module_summary.write_csv(os.path.join(tables_dir, "module_summary.csv"))
    module_count_summary.write_csv(os.path.join(tables_dir, "module_count_summary.csv"))
    if small_assignment_df.height > 0:
        small_assignment_df.write_csv(os.path.join(tables_dir, "small_module_assignment_audit.csv"))
    unassigned_df.write_csv(os.path.join(tables_dir, "unassigned_disconnected_genes.csv"))
    if hub_df.height > 0:
        hub_df.write_csv(os.path.join(tables_dir, "hub_gene_summary.csv"))
    module_edges.write_parquet(os.path.join(tables_dir, "module_edges.parquet"))

    export_gene_sets(out_dir, modules, hub_sets)

    n_hubs = sum(len(v) for v in hub_sets.values())

    print("-" * 80)
    print("FINAL MODULES -- after frozen-core constrained reassignment")
    print(f"community_method              : {community_method}")
    print(f"raw_communities_detected      : {raw_stats['raw_communities_total']}")
    print(f"raw_core_communities          : {raw_stats['raw_communities_retained']}")
    print(f"raw_small_communities         : {raw_stats['raw_communities_below_min_size']}")
    print(f"small_communities_reassigned  : {int(quality_stats.get('small_raw_communities_reassigned', 0))}")
    print(f"small_communities_unassigned  : {int(quality_stats.get('small_raw_communities_unassigned_disconnected', 0))}")
    print(f"raw_RB_quality                : {float(quality_stats.get('raw_rb_quality', 0.0)):.8f}")
    print(f"after_small_reassign_RB_Q     : {float(quality_stats.get('after_small_reassignment_rb_quality', 0.0)):.8f}")
    print(f"final_RB_quality              : {float(quality_stats.get('final_rb_quality', 0.0)):.8f}")
    print(f"modules_detected_final        : {total_modules}")
    print(f"selected_active_nodes         : {G_module.number_of_nodes():,}")
    print(f"assigned_nodes_total          : {total_module_nodes:,} / {G_module.number_of_nodes():,} selected active nodes")
    print(f"unassigned_disconnected       : {len(unassigned_nodes):,}")
    print(f"excluded_nodes                : {len(excluded_nodes):,}")
    print(
        f"small_modules_policy          : original raw modules >= {args.min_module_size} are frozen; "
        "smaller raw communities are assigned through a core-constrained maximum "
        "spanning forest on the raw-community graph using RB/configuration-model "
        "delta-Q as the PRIMARY edge score. sum(abs(w_ft))/sqrt(n_A*n_B) is retained "
        "only as secondary evidence. Small-small paths are allowed; two frozen original "
        "cores are never merged. If every legal merge is RB-unfavorable, the least "
        "damaging (largest delta-Q) edge is considered first rather than favoring target size. "
        "Coreless disconnected components that still remain below min_module_size are "
        "explicitly unassigned and excluded from Hub selection."
    )
    print(f"hub_genes_total               : {n_hubs}")
    print("hub_selection_method          : intra-module weighted degree = sum(abs(w_ft)); top fraction within each final module")
    print("final module sizes:")
    for mid, genes in sorted(modules.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        print(f"  {mid}: {len(genes):,} genes | hubs={len(hub_sets.get(mid, [])):,}")
    print("outputs:")
    print("  tables/module_assignment.csv")
    print("  tables/module_summary.csv")
    print("  tables/module_count_summary.csv")
    print("  tables/small_module_assignment_audit.csv")
    print("  tables/unassigned_disconnected_genes.csv")
    print("  tables/hub_gene_summary.csv")
    print("  tables/module_edges.parquet")
    print("  gene_sets/module_genes/Mxx_genes.txt")
    print("  gene_sets/module_hubs/Mxx_hubs.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()
