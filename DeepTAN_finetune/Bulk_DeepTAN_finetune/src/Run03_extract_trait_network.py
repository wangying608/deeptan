#!/usr/bin/env python3
r"""
==============================================================================
Run03: Extract Trait-Aware BulkExpand-DeepTAN Latent Network
==============================================================================

Purpose
-------
Extract a trait-aware bulk-specific latent gene/feature association network from
one BulkExpand-DeepTAN model fine-tuned on bulk trait data.

Pipeline position
-----------------
Run01: user-defined bulk files -> bulk-expanded LitData
Run02: LitData -> trait-aware bulk fine-tuned DeepTAN checkpoint
Run03: LitData + fine-tuned checkpoint -> trait-aware latent gene network
Run04: trait-aware network -> downstream module / hub / gene-set analysis

Core extraction workflow
------------------------
This script performs network extraction only. It does not rebuild LitData and
does not fine-tune the model.

Main steps:
  1. Load the Run01 bulk-expanded LitData and expanded_metadata.pkl.
  2. Reconstruct an expanded-pretrained baseline model from the scRNA-pretrained
     DeepTAN checkpoint and the Run01 expanded metadata.
  3. Load the Run02 fine-tuned TraitBulkDeepTAN checkpoint.
  4. Extract node-level representations after the NodeEmbedding stack:
       identity embedding + feature projection + fusion MLP + GAT layers + norm.
  5. Average each gene/feature embedding across all bulk samples in which that
     node is observed.
  6. Compute latent edge weights with cosine similarity:
       w_ij = cos_sim(mean_h_i, mean_h_j)
  7. Compute trait-aware delta weights:
       delta_w_ij = w_ij(fine_tuned) - w_ij(expanded_pretrained_baseline)
  8. Save the complete edge table, gene summary, embeddings, raw arrays, and
     extraction metadata.

Important interpretation
------------------------
The output is a latent gene/feature association network induced by trait-aware
fine-tuning. It is not a raw expression correlation network and it should not be
interpreted as a causal regulatory network without additional evidence.

Expected single-run input layout
--------------------------------
The LitData directory should be the output of Run01:

    /path/to/litdata/
        expanded_metadata.pkl
        edge_source.parquet
        trn/
        val/
        tst/

The fine-tuned checkpoint should be produced by Run02:

    /path/to/finetune_output/
        best_model.ckpt or FT16_epoch=xx_val_loss=yyyy.ckpt
        finetune_metadata.pkl   # optional but useful for exact config recovery

Recommended usage
-----------------
python bulk_trait_network_extract_single.py \
    --config configs/bulk_trait_network_single.yaml

Optional command-line overrides:

python bulk_trait_network_extract_single.py \
    --config configs/bulk_trait_network_single.yaml \
    --litdata_dir /path/to/litdata \
    --finetuned_ckpt /path/to/best_model.ckpt \
    --output_dir /path/to/network_output

Compatibility
-------------
Legacy multi-seed options such as --seed, --all_seeds, bulk_runs, and
bulk_run_order are still supported for old experiments, but they are no longer
recommended as the main GitHub-facing workflow.

Author: DeepTAN Bulk Network Extraction Pipeline
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import pickle
import re
import sys
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import litdata
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch_geometric.data import Batch as GBatch
from torch_geometric.data import Data as GData
from torch_geometric.utils import subgraph as pyg_subgraph

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Default configuration values
# =============================================================================

DEFAULT_EXTRACT_CONFIG: Dict[str, Any] = {
    "local_deeptan_src": "",
    "pretrained_ckpt": "",

    # ------------------------------------------------------------------
    # Recommended single-run interface.
    # ------------------------------------------------------------------
    # Run01 output directory. It should directly contain:
    #   expanded_metadata.pkl, trn/, val/, tst/
    "litdata_dir": "",

    # Run02 fine-tuned checkpoint. Explicitly setting this is recommended.
    # If omitted, the script can optionally search under finetune_output_dir.
    "finetuned_ckpt": "",

    # Optional Run02 output directory used only for automatic checkpoint search.
    # In single-run mode, the script first looks for finetune_metadata.pkl and
    # then falls back to the checkpoint filename with the lowest parsed val_loss.
    "finetune_output_dir": "",

    # Network extraction output root. Results are written to:
    #   output_dir/trait_network/
    "output_dir": "",

    # Human-readable run label written to metadata only. It is not used for
    # resolving file paths.
    "run_id": "bulk_trait_network",

    # Random seed for deterministic baseline reconstruction / extraction.
    # This is not a data split seed.
    "seed": 42,

    # ------------------------------------------------------------------
    # Legacy compatibility fields. These are kept so old multi-seed configs
    # still work, but they are not needed for the single-run GitHub workflow.
    # ------------------------------------------------------------------
    "dataset_root": "",
    "bulk_runs": {},
    "bulk_run_order": [],
    "default_seed": 42,
    "seeds": [42],
    "trait_csn_output_dir": "",

    "run_name": "TraitBulkDeepTAN",
    "accelerator": "gpu",
    "devices": 1,
    "precision": "32-true",
    "batch_size": 32,
    "n_workers": 8,

    # Dynamic import for the fine-tuning module containing TraitBulkDeepTAN.
    "finetune_module_dir": "",
    "finetune_module_name": "run_02_trait_aware_bulk_finetune_single",
    "finetune_class_name": "TraitBulkDeepTAN",

    # Extraction behavior.
    "extract_splits": ["trn", "val", "tst"],
    "max_batches": None,
    "abort_if_node_encoder_identical": True,
    "save_baseline_sample_embeddings": True,
    "save_finetuned_sample_embeddings": True,
    "edge_weight_method": "cosine_similarity",
}


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml_config(path: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_EXTRACT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, user_cfg)
    else:
        raise FileNotFoundError(f"Config not found: {path}")
    return cfg


def setup_local_deeptan(local_src: str) -> None:
    if local_src and local_src not in sys.path:
        sys.path.insert(0, local_src)


# =============================================================================
# Generic utility functions
# =============================================================================


def collate_fn(data_list: List[GData]) -> GBatch:
    return GBatch.from_data_list(data_list)


def _param_hash(module: nn.Module) -> str:
    h = hashlib.md5()
    for p in module.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _tensor_hash(t: torch.Tensor) -> str:
    h = hashlib.md5()
    h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _seed_everything(seed: int) -> None:
    try:
        from lightning import seed_everything

        seed_everything(seed, workers=True)
    except Exception:
        torch.manual_seed(seed)
        np.random.seed(seed)


def _natural_val_loss(path: str) -> float:
    name = os.path.basename(path)
    # Compatible with FT16_42_epoch=31_val_loss=0.1148.ckpt
    m = re.search(r"val_loss=([0-9eE+\-.]+)", name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return float("inf")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_yaml_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.detach().cpu().item())
        return obj.detach().cpu().numpy().tolist()
    return obj


# =============================================================================
# Dynamic model loading and checkpoint resolution
# =============================================================================


def _resolve_finetune_module(config: Dict[str, Any]):
    module_dir = config.get("finetune_module_dir") or ""
    module_name = config.get("finetune_module_name") or "run_02_trait_aware_bulk_finetune"
    class_name = config.get("finetune_class_name") or "TraitBulkDeepTAN"

    if module_dir and module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Cannot import fine-tune module '{module_name}'. "
            "Pass --finetune_module_dir to the directory containing your "
            "run_02_trait_aware_bulk_finetune.py script, or set "
            "finetune_module_dir / finetune_module_name in the YAML."
        ) from e

    if not hasattr(module, class_name):
        raise AttributeError(
            f"Cannot find class '{class_name}' in module '{module_name}'. "
            "Check finetune_class_name."
        )

    return module, getattr(module, class_name)


def load_pretrained_deeptan(pretrained_ckpt: str, map_location: str = "cpu") -> nn.Module:
    if not pretrained_ckpt or not os.path.exists(pretrained_ckpt):
        raise FileNotFoundError(f"pretrained_ckpt not found: {pretrained_ckpt}")

    from deeptan.graph.model import DeepTAN  # type: ignore

    model_dir = os.path.dirname(pretrained_ckpt)
    hparams_path = os.path.join(model_dir, "version_0", "hparams.yaml")
    kwargs = {"map_location": map_location}
    if os.path.exists(hparams_path):
        kwargs["hparams_file"] = hparams_path

    model = DeepTAN.load_from_checkpoint(pretrained_ckpt, **kwargs)
    model.eval()
    logger.info(
        f"Loaded scRNA pretrained DeepTAN: {pretrained_ckpt}\n"
        f"  old_nodes={getattr(model, 'num_all_nodes', 'NA')} "
        f"label_dim={getattr(model, 'output_g_label_dim', 'NA')}"
    )
    return model


def read_seed_metadata(seed_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta_path = os.path.join(seed_dir, "expanded_metadata.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"expanded_metadata.pkl not found: {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    qc_path = os.path.join(seed_dir, "qc_report.json")
    qc = {}
    if os.path.exists(qc_path):
        with open(qc_path, "r") as f:
            qc = json.load(f)

    required = ["dict_node_names", "old_dict_node_names", "num_old_nodes"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise KeyError(f"expanded_metadata.pkl missing keys: {missing}")
    return meta, qc


def _load_training_config_from_checkpoint_or_metadata(
    ft_ckpt_path: str,
    train_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Use the exact training config when available, else fall back to YAML."""
    ckpt_dir = os.path.dirname(ft_ckpt_path)
    meta_path = os.path.join(ckpt_dir, "finetune_metadata.pkl")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            cfg = meta.get("config")
            if isinstance(cfg, dict):
                logger.info(f"Loaded training config from {meta_path}")
                return cfg
        except Exception as e:
            logger.warning(f"Failed to read finetune_metadata.pkl: {e}")

    try:
        ckpt = torch.load(ft_ckpt_path, map_location="cpu")
        hp = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
        # Lightning stores constructor args; in this script the bulk model uses `cfg`.
        if isinstance(hp.get("cfg"), dict):
            logger.info("Loaded training config from checkpoint hyper_parameters['cfg']")
            return hp["cfg"]
        if isinstance(hp.get("config"), dict):
            logger.info("Loaded training config from checkpoint hyper_parameters['config']")
            return hp["config"]
    except Exception as e:
        logger.warning(f"Failed to inspect checkpoint hyper_parameters: {e}")

    logger.warning("Using YAML config as training config fallback.")
    return train_cfg


def find_best_checkpoint(train_output_dir: str, seed: int) -> Optional[str]:
    """Find the best checkpoint for one seed under a fine-tuning output root.

    Supports both layouts:
      1) train_output_dir/seed_43/*.ckpt
      2) train_output_dir/<run_id>/seed_43/*.ckpt

    Priority:
      1) finetune_metadata.pkl: best_ckpt
      2) lowest val_loss parsed from checkpoint filename
    """
    if not train_output_dir:
        return None

    train_output_dir = os.path.abspath(os.path.expanduser(str(train_output_dir)))
    if not os.path.isdir(train_output_dir):
        return None

    run_key = f"seed_{int(seed)}"
    candidate_dirs: List[str] = []

    # Direct legacy/simple layout: output_root/seed_43
    direct = os.path.join(train_output_dir, run_key)
    if os.path.isdir(direct):
        candidate_dirs.append(direct)

    # If the given directory itself is seed_43.
    if os.path.basename(os.path.normpath(train_output_dir)) == run_key:
        candidate_dirs.append(train_output_dir)

    # Nested layout: output_root/1/seed_43, output_root/2/seed_43, ...
    for root, dirs, _files in os.walk(train_output_dir):
        if os.path.basename(os.path.normpath(root)) == run_key:
            candidate_dirs.append(root)
        # Avoid descending too deeply after seed directories are found.
        dirs[:] = [d for d in dirs if d != run_key]

    # De-duplicate while preserving order.
    seen_dirs = set()
    candidate_dirs = [
        d for d in candidate_dirs
        if not (d in seen_dirs or seen_dirs.add(d))
    ]

    for seed_out in candidate_dirs:
        meta_path = os.path.join(seed_out, "finetune_metadata.pkl")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                best = meta.get("best_ckpt")
                if best and os.path.exists(best):
                    logger.info(f"Resolved best checkpoint from finetune_metadata.pkl: {best}")
                    return str(best)
            except Exception as e:
                logger.warning(f"Failed to read {meta_path}: {e}")

    ckpts = []
    for seed_out in candidate_dirs:
        for root, _dirs, files in os.walk(seed_out):
            for f in files:
                if f.endswith(".ckpt"):
                    ckpts.append(os.path.join(root, f))

    if not ckpts:
        return None

    ckpts.sort(key=_natural_val_loss)
    logger.info(f"Resolved checkpoint by filename val_loss: {ckpts[0]}")
    return ckpts[0]


def _seed_key(seed: int) -> str:
    return f"seed_{int(seed)}"


def _parse_seed_value(x: Any) -> int:
    """Parse 43, '43', 'seed_43', or 'seed-43' into integer 43."""
    if isinstance(x, int):
        return int(x)
    text = str(x).strip()
    m = re.search(r"(?:seed[_-]?)?(\d+)$", text)
    if not m:
        raise ValueError(f"Cannot parse seed value: {x}")
    return int(m.group(1))


def _get_bulk_run_cfg(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    bulk_runs = cfg.get("bulk_runs") or {}
    if not isinstance(bulk_runs, dict):
        return {}

    key = _seed_key(seed)
    run_cfg = bulk_runs.get(key)
    if run_cfg is None:
        run_cfg = bulk_runs.get(str(int(seed)), {})

    return run_cfg if isinstance(run_cfg, dict) else {}


def _looks_like_litdata_dir(path: str) -> bool:
    """Return True if path directly looks like a Run01 LitData output."""
    if not path:
        return False
    path = os.path.abspath(os.path.expanduser(str(path)))
    return (
        os.path.exists(os.path.join(path, "expanded_metadata.pkl"))
        and os.path.isdir(os.path.join(path, "trn"))
        and os.path.isdir(os.path.join(path, "val"))
        and os.path.isdir(os.path.join(path, "tst"))
    )


def _is_single_run_mode(cfg: Dict[str, Any], args: argparse.Namespace) -> bool:
    """Whether to use the GitHub-facing single-run path semantics."""
    return bool(
        getattr(args, "litdata_dir", None)
        or getattr(args, "finetuned_ckpt", None)
        or cfg.get("litdata_dir")
        or cfg.get("finetuned_ckpt")
    )


def resolve_seed_dir(seed: int, cfg: Dict[str, Any], args: argparse.Namespace) -> str:
    """Resolve the LitData directory.

    Recommended single-run priority:
      1) --litdata_dir
      2) config.litdata_dir
      3) --dataset_root, if it directly contains expanded_metadata.pkl/trn/val/tst
      4) config.dataset_root, if it directly contains expanded_metadata.pkl/trn/val/tst

    Legacy fallback:
      - bulk_runs.seed_43.litdata
      - dataset_root/seed_43
    """
    run_key = _seed_key(seed)
    bulk_run = _get_bulk_run_cfg(cfg, seed)

    candidates: List[str] = []
    if getattr(args, "litdata_dir", None):
        candidates.append(str(args.litdata_dir))
    if cfg.get("litdata_dir"):
        candidates.append(str(cfg["litdata_dir"]))

    # --dataset_root / dataset_root are accepted as direct single-run LitData roots
    # when they already contain expanded_metadata.pkl and trn/val/tst.
    if getattr(args, "dataset_root", None):
        candidates.append(str(args.dataset_root))
    if cfg.get("dataset_root"):
        candidates.append(str(cfg["dataset_root"]))

    for cand in candidates:
        root = os.path.abspath(os.path.expanduser(cand))
        if _looks_like_litdata_dir(root):
            return root

    # Legacy explicit per-run LitData path.
    if bulk_run.get("litdata"):
        seed_dir = os.path.abspath(os.path.expanduser(str(bulk_run["litdata"])))
        if os.path.isdir(seed_dir):
            return seed_dir
        raise FileNotFoundError(f"LitData directory not found: {seed_dir}")

    # Legacy dataset_root/seed_xx layout.
    legacy_root = None
    if getattr(args, "dataset_root", None):
        legacy_root = str(args.dataset_root)
    elif cfg.get("dataset_root"):
        legacy_root = str(cfg["dataset_root"])

    if legacy_root:
        legacy_root = os.path.abspath(os.path.expanduser(legacy_root))
        seed_dir = os.path.join(legacy_root, run_key)
        if os.path.isdir(seed_dir):
            return seed_dir

    raise FileNotFoundError(
        "Cannot resolve LitData directory. For the single-run workflow, set "
        "litdata_dir in the YAML or pass --litdata_dir. The directory must "
        "directly contain expanded_metadata.pkl, trn/, val/, and tst/."
    )

def _find_best_checkpoint_single(train_output_dir: str) -> Optional[str]:
    """Find a best checkpoint under a single-run fine-tuning output directory."""
    if not train_output_dir:
        return None
    root = os.path.abspath(os.path.expanduser(str(train_output_dir)))
    if not os.path.isdir(root):
        return None

    # Preferred: Run02 metadata explicitly records the best checkpoint.
    for meta_root, _dirs, files in os.walk(root):
        if "finetune_metadata.pkl" not in files:
            continue
        meta_path = os.path.join(meta_root, "finetune_metadata.pkl")
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            best = meta.get("best_ckpt")
            if best and os.path.exists(best):
                logger.info(f"Resolved best checkpoint from finetune_metadata.pkl: {best}")
                return str(best)
        except Exception as e:
            logger.warning(f"Failed to read {meta_path}: {e}")

    ckpts = []
    for ckpt_root, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".ckpt"):
                ckpts.append(os.path.join(ckpt_root, f))

    if not ckpts:
        return None
    ckpts.sort(key=_natural_val_loss)
    logger.info(f"Resolved checkpoint by filename val_loss: {ckpts[0]}")
    return ckpts[0]


def resolve_ft_checkpoint(seed: int, cfg: Dict[str, Any], args: argparse.Namespace) -> str:
    """Resolve the fine-tuned checkpoint.

    Recommended single-run priority:
      1) --finetuned_ckpt
      2) --ft_ckpt legacy alias
      3) config.finetuned_ckpt
      4) automatic search under config.finetune_output_dir

    Legacy fallback:
      - bulk_runs.seed_43.ft_ckpt
      - finetune_output_dir/seed_43 or nested seed_43 directories
    """
    run_key = _seed_key(seed)
    bulk_run = _get_bulk_run_cfg(cfg, seed)

    ft_ckpt = (
        getattr(args, "finetuned_ckpt", None)
        or getattr(args, "ft_ckpt", None)
        or cfg.get("finetuned_ckpt")
        or cfg.get("ft_ckpt")
        or bulk_run.get("ft_ckpt")
    )
    if ft_ckpt:
        ft_ckpt = os.path.abspath(os.path.expanduser(str(ft_ckpt)))
        if not os.path.exists(ft_ckpt):
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {ft_ckpt}")
        return ft_ckpt

    # Single-run automatic search.
    if cfg.get("finetune_output_dir") and (_is_single_run_mode(cfg, args) or not cfg.get("bulk_runs")):
        found = _find_best_checkpoint_single(str(cfg["finetune_output_dir"]))
        if found:
            return found

    # Legacy seed-aware automatic search.
    search_roots: List[str] = []
    for k in ("finetune_output_dir", "train_output_dir"):
        if cfg.get(k):
            search_roots.append(str(cfg[k]))

    if cfg.get("output_dir") and not cfg.get("bulk_runs") and not _is_single_run_mode(cfg, args):
        search_roots.append(str(cfg["output_dir"]))

    for root in search_roots:
        found = find_best_checkpoint(root, seed)
        if found:
            return found

    searched = ", ".join(search_roots) if search_roots else "none"
    raise FileNotFoundError(
        "Could not resolve the fine-tuned checkpoint. For the single-run workflow, "
        "set finetuned_ckpt in the YAML or pass --finetuned_ckpt. Alternatively, "
        "set finetune_output_dir so the script can search for a checkpoint. "
        f"Legacy run key={run_key}; searched: {searched}"
    )

def resolve_output_root(cfg: Dict[str, Any], args: argparse.Namespace) -> str:
    """Resolve output root for extracted networks."""
    if getattr(args, "output_dir", None):
        return os.path.abspath(os.path.expanduser(args.output_dir))
    if cfg.get("output_dir"):
        return os.path.abspath(os.path.expanduser(str(cfg["output_dir"])))
    if cfg.get("trait_csn_output_dir"):
        return os.path.abspath(os.path.expanduser(str(cfg["trait_csn_output_dir"])))
    return os.path.abspath(os.path.expanduser("network_output"))


def resolve_run_id(cfg: Dict[str, Any], args: argparse.Namespace) -> str:
    """Resolve a human-readable run identifier for metadata and logs."""
    rid = getattr(args, "run_id", None) or cfg.get("run_id") or cfg.get("run_name") or "bulk_trait_network"
    return str(rid)

def resolve_requested_seeds(cfg: Dict[str, Any], args: argparse.Namespace) -> List[int]:
    """Resolve CLI seed selection.

    Supports:
      - --seed 43
      - --all_seeds with bulk_run_order: [seed_42, seed_43, ...]
      - --all_seeds with bulk_runs keys
      - legacy config.seeds
      - default_seed
    """
    if args.all_seeds:
        if args.ft_ckpt:
            raise ValueError("--ft_ckpt is only valid with a single --seed, not --all_seeds")

        if cfg.get("bulk_run_order"):
            return [_parse_seed_value(x) for x in cfg["bulk_run_order"]]

        bulk_runs = cfg.get("bulk_runs") or {}
        if isinstance(bulk_runs, dict) and bulk_runs:
            return sorted(_parse_seed_value(k) for k in bulk_runs.keys())

        return [int(s) for s in cfg.get("seeds", [cfg.get("seed", 42)])]

    return [
        int(
            args.seed
            if args.seed is not None
            else cfg.get("default_seed", cfg.get("seed", 42))
        )
    ]


def instantiate_expanded_baseline(
    pretrained_model: nn.Module,
    meta: Dict[str, Any],
    seed_dir: str,
    cfg: Dict[str, Any],
    finetune_cls,
    seed: int,
) -> nn.Module:
    """Expanded-pretrained baseline before bulk trait fine-tuning."""
    _seed_everything(seed)
    model = finetune_cls(copy.deepcopy(pretrained_model), meta, seed_dir, cfg)
    model.eval()
    return model


def apply_exact_initial_embedding_from_checkpoint(
    baseline_model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> bool:
    """Use persisted initial old/new embedding buffers from the checkpoint.

    TraitBulkDeepTAN stores:
      - old_embedding_anchor: pretrained old gene embedding
      - new_embedding_init: initial embedding for newly added bulk nodes/features

    This makes the baseline embedding exactly match the initialization used for
    fine-tuning, rather than relying only on RNG reproducibility.
    """
    embed_key = "amsgp.node_embedding_layers.embed.weight"
    if embed_key not in dict(baseline_model.named_parameters()):
        # Named parameter lookup below works more reliably than direct dict access
        pass

    if "old_embedding_anchor" not in state_dict or "new_embedding_init" not in state_dict:
        logger.warning(
            "Checkpoint does not contain old_embedding_anchor/new_embedding_init. "
            "Baseline new embeddings will use deterministic re-initialization."
        )
        return False

    old = state_dict["old_embedding_anchor"].detach().cpu()
    new = state_dict["new_embedding_init"].detach().cpu()
    full = torch.cat([old, new], dim=0)

    emb = baseline_model.amsgp.node_embedding_layers.embed.weight
    if tuple(emb.shape) != tuple(full.shape):
        logger.warning(
            f"Cannot install exact initial embedding: baseline embed shape={tuple(emb.shape)}, "
            f"checkpoint initial shape={tuple(full.shape)}"
        )
        return False

    with torch.no_grad():
        emb.copy_(full.to(device=emb.device, dtype=emb.dtype))
    logger.success(
        "Installed exact initial expanded embedding into baseline "
        f"(hash={_tensor_hash(emb)})"
    )
    return True


def load_finetuned_trait_bulk_model(
    pretrained_model: nn.Module,
    meta: Dict[str, Any],
    seed_dir: str,
    train_cfg: Dict[str, Any],
    ft_ckpt_path: str,
    finetune_cls,
    seed: int,
    map_location: str = "cpu",
) -> Tuple[nn.Module, Dict[str, torch.Tensor], Dict[str, Any]]:
    if not ft_ckpt_path or not os.path.exists(ft_ckpt_path):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {ft_ckpt_path}")

    actual_cfg = _load_training_config_from_checkpoint_or_metadata(ft_ckpt_path, train_cfg)
    _seed_everything(seed)
    model = finetune_cls(copy.deepcopy(pretrained_model), meta, seed_dir, actual_cfg)

    ckpt = torch.load(ft_ckpt_path, map_location=map_location)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a state_dict-like object")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if not k.startswith("_anchor_")]
    if real_missing:
        logger.warning(f"Missing keys (non-anchor) while loading fine-tuned model: {real_missing[:20]}")
    if unexpected:
        logger.warning(f"Unexpected keys while loading fine-tuned model: {unexpected[:20]}")

    model.eval()
    logger.info(f"Loaded fine-tuned TraitBulkDeepTAN: {ft_ckpt_path}")
    return model, state_dict, actual_cfg


# =============================================================================
# LitData loading
# =============================================================================


def make_dataloader(seed_dir: str, split: str, cfg: Dict[str, Any]):
    split_dir = os.path.join(seed_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"LitData split directory not found: {split_dir}")
    ds = litdata.StreamingDataset(split_dir, max_cache_size="10GB")
    return litdata.StreamingDataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 32)),
        num_workers=int(cfg.get("n_workers", 4)),
        collate_fn=collate_fn,
    )


def make_dataloaders(seed_dir: str, splits: Sequence[str], cfg: Dict[str, Any]) -> List[Tuple[str, Any]]:
    out = []
    for split in splits:
        out.append((split, make_dataloader(seed_dir, split, cfg)))
    return out


# =============================================================================
# Node-level embedding extraction
# =============================================================================


def _split_node_names(batch_node_names: Any, counts: List[int]) -> List[List[str]]:
    if isinstance(batch_node_names, (list, tuple)) and len(batch_node_names) > 0 and isinstance(batch_node_names[0], list):
        return [list(x) for x in batch_node_names]

    if isinstance(batch_node_names, (list, tuple)):
        flat = list(batch_node_names)
    else:
        flat = list(batch_node_names)

    out = []
    start = 0
    for c in counts:
        out.append([str(x) for x in flat[start : start + c]])
        start += c
    return out


@torch.no_grad()
def _extract_node_embeddings_from_batch(
    model_amsgp: nn.Module,
    batch: GData,
    dict_node_names: Dict[str, int],
    device: torch.device,
) -> Dict[str, Any]:
    """Execute the NodeEmbedding stack for each graph in a PyG batch.

    This intentionally does not run the full AMSGP pooling, decoder, or trait
    predictor. It extracts the same node-level representation used upstream of
    AMSGP pooling:
        embed(id) + feature_proj(x) -> fusion_mlp -> GAT layers -> norm

    If a sample-level graph has no edges, the function keeps the node embeddings
    via the same fallback idea as AMSGP: embed + feature_proj + fusion_mlp + norm,
    without GAT propagation.
    """
    node_embedding_layers = model_amsgp.node_embedding_layers

    node_batch = getattr(
        batch,
        "batch",
        torch.zeros(batch.x.size(0), dtype=torch.long, device=device),
    )
    x = batch.x.float().to(device)
    edge_index = batch.edge_index.long().to(device)
    node_batch = node_batch.to(device)

    unique_batches, _inverse, counts = torch.unique(
        node_batch, return_inverse=True, return_counts=True
    )
    node_indices_list = torch.split(
        torch.arange(node_batch.size(0), device=device), counts.tolist()
    )
    node_names_splits = _split_node_names(batch.node_names, counts.tolist())

    obs_names = getattr(batch, "obs_name", None)
    if obs_names is None:
        obs_list = [f"sample_{i}" for i in range(len(node_indices_list))]
    elif isinstance(obs_names, (list, tuple)):
        obs_list = [str(x) for x in obs_names]
    else:
        try:
            obs_list = [str(x) for x in list(obs_names)]
        except Exception:
            obs_list = [str(obs_names)]

    results = {
        "node_embeddings": [],
        "node_gene_ids": [],
        "edge_indices_local": [],
        "node_names_per_graph": [],
        "obs_names": [],
    }

    for graph_i, (node_indices, graph_node_names) in enumerate(zip(node_indices_list, node_names_splits)):
        if node_indices.numel() == 0:
            continue

        valid_mask = [n in dict_node_names for n in graph_node_names]
        if not all(valid_mask):
            n_invalid = len(valid_mask) - int(sum(valid_mask))
            logger.warning(
                f"Graph {graph_i}: {n_invalid}/{len(graph_node_names)} node names not in expanded dict; skipping graph."
            )
            continue

        sub_x = x[node_indices]
        sub_edge_index, _ = pyg_subgraph(
            node_indices,
            edge_index,
            relabel_nodes=True,
            num_nodes=x.size(0),
        )

        ids = torch.tensor(
            [int(dict_node_names[n]) for n in graph_node_names],
            device=device,
            dtype=torch.long,
        )

        node_embs = node_embedding_layers.embed(ids)
        x_proj = node_embedding_layers.feature_proj(sub_x.unsqueeze(-1)).squeeze(1)
        fused = torch.cat([node_embs, x_proj], dim=-1)
        fused = node_embedding_layers.fusion_mlp(fused)

        if sub_edge_index.numel() > 0:
            for layer in node_embedding_layers._layers:
                fused = layer(fused, sub_edge_index)

        fused = node_embedding_layers.norm(fused)

        results["node_embeddings"].append(fused.detach().cpu())
        results["node_gene_ids"].append(ids.detach().cpu())
        results["edge_indices_local"].append(sub_edge_index.detach().cpu())
        results["node_names_per_graph"].append(list(graph_node_names))
        results["obs_names"].append(obs_list[graph_i] if graph_i < len(obs_list) else f"sample_{graph_i}")

    return results


@torch.no_grad()
def extract_signals_from_dataloaders(
    model: nn.Module,
    dataloaders: List[Tuple[str, Any]],
    dict_node_names: Dict[str, int],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    if not hasattr(model, "amsgp"):
        raise ValueError("Model does not have .amsgp")

    model.eval()
    model.amsgp.eval()

    gene_sum: Dict[int, np.ndarray] = {}
    gene_count: Dict[int, int] = defaultdict(int)
    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    sample_embeddings: List[np.ndarray] = []
    sample_obs_names: List[str] = []

    n_samples = 0
    n_batches_total = 0
    t0 = time.time()

    for split_name, dataloader in dataloaders:
        split_samples = 0
        for _batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and n_batches_total >= int(max_batches):
                break

            batch = batch.to(device)
            try:
                sig = _extract_node_embeddings_from_batch(
                    model.amsgp, batch, dict_node_names, device=device
                )
            except Exception as e:
                logger.warning(f"Extraction failed at split={split_name}, batch={n_batches_total}: {e}")
                continue

            for gi in range(len(sig["node_embeddings"])):
                embs = sig["node_embeddings"][gi].numpy().astype(np.float64, copy=False)
                gids = sig["node_gene_ids"][gi].numpy().astype(np.int64, copy=False)
                edge_index_local = sig["edge_indices_local"][gi].numpy().astype(np.int64, copy=False)

                if embs.size == 0 or gids.size == 0:
                    continue

                # Sample-level bulk embedding by mean pooling node embeddings.
                sample_embeddings.append(embs.mean(axis=0).astype(np.float32))
                sample_obs_names.append(str(sig["obs_names"][gi]))

                for local_i, gid in enumerate(gids.tolist()):
                    gid = int(gid)
                    if gid not in gene_sum:
                        gene_sum[gid] = np.zeros(embs.shape[1], dtype=np.float64)
                    gene_sum[gid] += embs[local_i]
                    gene_count[gid] += 1

                n_local_nodes = len(gids)
                if edge_index_local.size > 0:
                    for ei in range(edge_index_local.shape[1]):
                        src_local = int(edge_index_local[0, ei])
                        dst_local = int(edge_index_local[1, ei])
                        if src_local >= n_local_nodes or dst_local >= n_local_nodes:
                            continue
                        a = int(gids[src_local])
                        b = int(gids[dst_local])
                        if a == b:
                            continue
                        pair = (min(a, b), max(a, b))
                        edge_counts[pair] += 1

                n_samples += 1
                split_samples += 1

            n_batches_total += 1
            if n_batches_total % 20 == 0:
                logger.info(
                    f"    [{n_batches_total} batches, {n_samples} samples, "
                    f"{len(gene_count)} observed genes/features, {len(edge_counts)} edges, "
                    f"{time.time() - t0:.1f}s]"
                )

        logger.info(f"    split={split_name}: {split_samples} samples")

    hidden_dim = 0
    if gene_sum:
        hidden_dim = next(iter(gene_sum.values())).shape[0]

    logger.info(
        f"  Extraction complete: samples={n_samples}, genes/features={len(gene_count)}, "
        f"edges={len(edge_counts)}, dim={hidden_dim}, elapsed={time.time() - t0:.1f}s"
    )

    return {
        "gene_sum": gene_sum,
        "gene_count": dict(gene_count),
        "edge_counts": dict(edge_counts),
        "sample_embeddings": np.stack(sample_embeddings, axis=0) if sample_embeddings else np.empty((0, hidden_dim), dtype=np.float32),
        "sample_obs_names": np.asarray(sample_obs_names, dtype=object),
        "n_samples": n_samples,
        "hidden_dim": hidden_dim,
    }


# =============================================================================
# Aggregation and latent network construction
# =============================================================================


def aggregate_gene_means(
    signals: Dict[str, Any],
    n_total_nodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    hidden_dim = int(signals.get("hidden_dim", 0))
    mean_emb = np.zeros((n_total_nodes, hidden_dim), dtype=np.float32)
    counts = np.zeros(n_total_nodes, dtype=np.int64)

    gene_sum: Dict[int, np.ndarray] = signals["gene_sum"]
    gene_count: Dict[int, int] = signals["gene_count"]
    for gid, s in gene_sum.items():
        c = int(gene_count.get(gid, 0))
        if c > 0:
            mean_emb[int(gid)] = (s / c).astype(np.float32)
            counts[int(gid)] = c

    return mean_emb, counts


def compute_edge_cosine_weights(
    mean_emb: np.ndarray,
    edge_counts: Dict[Tuple[int, int], int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not edge_counts:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    edge_list = sorted(edge_counts.keys())
    edge_array = np.asarray(edge_list, dtype=np.int64)
    seen_counts = np.asarray([edge_counts[p] for p in edge_list], dtype=np.int64)
    w = np.zeros(edge_array.shape[0], dtype=np.float64)

    for i, (a, b) in enumerate(edge_array):
        ha = mean_emb[int(a)]
        hb = mean_emb[int(b)]
        na = float(np.linalg.norm(ha))
        nb = float(np.linalg.norm(hb))
        if na > 1e-10 and nb > 1e-10:
            w[i] = float(np.dot(ha, hb) / (na * nb))

    return edge_array, w, seen_counts


def align_and_delta(
    edges_pre: np.ndarray,
    w_pre: np.ndarray,
    edge_seen_pre: np.ndarray,
    edges_ft: np.ndarray,
    w_ft: np.ndarray,
    edge_seen_ft: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre = {
        (int(edges_pre[i, 0]), int(edges_pre[i, 1])): (float(w_pre[i]), int(edge_seen_pre[i]))
        for i in range(len(edges_pre))
    }
    ft = {
        (int(edges_ft[i, 0]), int(edges_ft[i, 1])): (float(w_ft[i]), int(edge_seen_ft[i]))
        for i in range(len(edges_ft))
    }
    common = sorted(set(pre.keys()) & set(ft.keys()))
    if not common:
        raise RuntimeError("No common edges between expanded-pretrained baseline and fine-tuned model")

    common_edges = np.asarray(common, dtype=np.int64)
    w_pre_aligned = np.asarray([pre[k][0] for k in common], dtype=np.float64)
    w_ft_aligned = np.asarray([ft[k][0] for k in common], dtype=np.float64)
    seen_pre = np.asarray([pre[k][1] for k in common], dtype=np.int64)
    seen_ft = np.asarray([ft[k][1] for k in common], dtype=np.int64)
    delta = w_ft_aligned - w_pre_aligned
    return common_edges, w_pre_aligned, w_ft_aligned, delta, seen_pre, seen_ft


def build_edge_table(
    common_edges: np.ndarray,
    w_pre: np.ndarray,
    w_ft: np.ndarray,
    delta_w: np.ndarray,
    edge_seen_pre: np.ndarray,
    edge_seen_ft: np.ndarray,
    gene_counts_pre: np.ndarray,
    gene_counts_ft: np.ndarray,
    gene_id_to_name: Dict[int, str],
    num_old_nodes: int,
) -> pl.DataFrame:
    rows = []
    for idx in range(common_edges.shape[0]):
        a = int(common_edges[idx, 0])
        b = int(common_edges[idx, 1])
        ca_pre = int(gene_counts_pre[a]) if a < len(gene_counts_pre) else 0
        cb_pre = int(gene_counts_pre[b]) if b < len(gene_counts_pre) else 0
        ca_ft = int(gene_counts_ft[a]) if a < len(gene_counts_ft) else 0
        cb_ft = int(gene_counts_ft[b]) if b < len(gene_counts_ft) else 0
        d = float(delta_w[idx])
        rows.append(
            {
                "gene_i_id": a,
                "gene_j_id": b,
                "gene_i_name": gene_id_to_name.get(a, f"node_{a}"),
                "gene_j_name": gene_id_to_name.get(b, f"node_{b}"),
                "gene_i_type": "old" if a < num_old_nodes else "new",
                "gene_j_type": "old" if b < num_old_nodes else "new",
                "w_pre": round(float(w_pre[idx]), 8),
                "w_ft": round(float(w_ft[idx]), 8),
                "delta_w": round(d, 8),
                "abs_delta_w": round(abs(d), 8),
                "delta_sign": "positive" if d > 0 else ("negative" if d < 0 else "zero"),
                "edge_seen_pre": int(edge_seen_pre[idx]),
                "edge_seen_ft": int(edge_seen_ft[idx]),
                "edge_seen_min": int(min(edge_seen_pre[idx], edge_seen_ft[idx])),
                "gene_i_count_pre": ca_pre,
                "gene_j_count_pre": cb_pre,
                "gene_i_count_ft": ca_ft,
                "gene_j_count_ft": cb_ft,
                "gene_count_min_pre": int(min(ca_pre, cb_pre)),
                "gene_count_min_ft": int(min(ca_ft, cb_ft)),
            }
        )

    return pl.DataFrame(rows)


def build_gene_table(
    gene_id_to_name: Dict[int, str],
    counts_pre: np.ndarray,
    counts_ft: np.ndarray,
    mean_pre: np.ndarray,
    mean_ft: np.ndarray,
    num_old_nodes: int,
) -> pl.DataFrame:
    rows = []
    n = len(gene_id_to_name)
    for gid in range(n):
        pre_norm = float(np.linalg.norm(mean_pre[gid])) if gid < mean_pre.shape[0] else 0.0
        ft_norm = float(np.linalg.norm(mean_ft[gid])) if gid < mean_ft.shape[0] else 0.0
        rows.append(
            {
                "gene_id": gid,
                "gene_name": gene_id_to_name.get(gid, f"node_{gid}"),
                "node_type": "old" if gid < num_old_nodes else "new",
                "count_pre": int(counts_pre[gid]) if gid < len(counts_pre) else 0,
                "count_ft": int(counts_ft[gid]) if gid < len(counts_ft) else 0,
                "embedding_norm_pre": round(pre_norm, 8),
                "embedding_norm_ft": round(ft_norm, 8),
                "embedding_norm_delta": round(ft_norm - pre_norm, 8),
            }
        )
    return pl.DataFrame(rows)


# =============================================================================
# Output writing
# =============================================================================


def save_outputs(
    out_dir: str,
    seed: int,
    ft_ckpt_path: str,
    seed_dir: str,
    config_path: str,
    cfg: Dict[str, Any],
    actual_train_cfg: Dict[str, Any],
    meta: Dict[str, Any],
    qc: Dict[str, Any],
    edge_df: pl.DataFrame,
    gene_df: pl.DataFrame,
    mean_pre: np.ndarray,
    mean_ft: np.ndarray,
    signals_pre: Dict[str, Any],
    signals_ft: Dict[str, Any],
    common_edges: np.ndarray,
    w_pre: np.ndarray,
    w_ft: np.ndarray,
    delta_w: np.ndarray,
    hashes: Dict[str, str],
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    edge_path = os.path.join(out_dir, "bulk_trait_edge_table.parquet")
    gene_path = os.path.join(out_dir, "gene_node_summary.csv")
    edge_df.write_parquet(edge_path)
    gene_df.write_csv(gene_path)

    np.save(os.path.join(out_dir, "mean_embeddings_expanded_pretrained.npy"), mean_pre)
    np.save(os.path.join(out_dir, "mean_embeddings_finetuned.npy"), mean_ft)

    if bool(cfg.get("save_baseline_sample_embeddings", True)):
        np.save(os.path.join(out_dir, "bulk_embeddings_expanded_pretrained.npy"), signals_pre["sample_embeddings"])
    if bool(cfg.get("save_finetuned_sample_embeddings", True)):
        np.save(os.path.join(out_dir, "bulk_embeddings_finetuned.npy"), signals_ft["sample_embeddings"])
    np.save(os.path.join(out_dir, "bulk_obs_names.npy"), signals_ft["sample_obs_names"])

    raw_dir = os.path.join(out_dir, "raw_signals")
    os.makedirs(raw_dir, exist_ok=True)
    np.save(os.path.join(raw_dir, "common_edges.npy"), common_edges)
    np.save(os.path.join(raw_dir, "w_pre.npy"), w_pre)
    np.save(os.path.join(raw_dir, "w_ft.npy"), w_ft)
    np.save(os.path.join(raw_dir, "delta_w.npy"), delta_w)

    metadata = {
        "task": "Trait-aware BulkExpand-DeepTAN latent network extraction",
        "run_id": cfg.get("run_id", "bulk_trait_network"),
        "seed": seed,
        "config_path": config_path,
        "seed_dir": seed_dir,
        "fine_tuned_checkpoint": ft_ckpt_path,
        "n_expanded_nodes": int(len(meta["dict_node_names"])),
        "n_old_nodes": int(meta.get("num_old_nodes", 0)),
        "n_new_nodes": int(len(meta["dict_node_names"]) - int(meta.get("num_old_nodes", 0))),
        "n_samples_pre": int(signals_pre["n_samples"]),
        "n_samples_ft": int(signals_ft["n_samples"]),
        "n_edges_total": int(edge_df.height),
        "edge_weight_method": "cosine_similarity_of_cross_sample_mean_node_embeddings",
        "delta_definition": "delta_w = w_finetuned - w_expanded_pretrained_baseline",
        "data_scope": cfg.get("extract_splits", ["trn", "val", "tst"]),
        "interpretation": (
            "Latent gene association shift induced by FT16 trait-aware bulk fine-tuning. "
            "This is not raw expression correlation and not causal regulation."
        ),
        "hashes": hashes,
        "stats": {
            "w_pre_min": _safe_float(np.min(w_pre)) if len(w_pre) else None,
            "w_pre_max": _safe_float(np.max(w_pre)) if len(w_pre) else None,
            "w_ft_min": _safe_float(np.min(w_ft)) if len(w_ft) else None,
            "w_ft_max": _safe_float(np.max(w_ft)) if len(w_ft) else None,
            "delta_w_mean": _safe_float(np.mean(delta_w)) if len(delta_w) else None,
            "delta_w_std": _safe_float(np.std(delta_w)) if len(delta_w) else None,
            "delta_w_median": _safe_float(np.median(delta_w)) if len(delta_w) else None,
            "delta_w_positive_frac": _safe_float(np.mean(delta_w > 0)) if len(delta_w) else None,
            "delta_w_negative_frac": _safe_float(np.mean(delta_w < 0)) if len(delta_w) else None,
        },
        "qc_from_dataset_builder": qc,
        "training_config_used_for_model_reconstruction": actual_train_cfg,
    }
    with open(os.path.join(out_dir, "bulk_trait_metadata.yaml"), "w") as f:
        yaml.dump(_to_yaml_safe(metadata), f, allow_unicode=True, default_flow_style=False)

    logger.info("\n" + "─" * 60)
    logger.info(f"Bulk trait network output directory: {out_dir}")
    for item in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, item)
        if os.path.isfile(p):
            logger.info(f"  {item} ({os.path.getsize(p) / 1024:.1f} KB)")
        else:
            logger.info(f"  {item}/")
    logger.info("─" * 60)
    return out_dir


# =============================================================================
# Main extraction pipeline
# =============================================================================


def process_one_seed(
    seed: int,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    finetune_cls,
) -> Optional[str]:
    local_deeptan_src = cfg.get("local_deeptan_src") or args.local_deeptan_src or ""
    setup_local_deeptan(local_deeptan_src)

    # Resolve the LitData directory. In the recommended single-run workflow this
    # is config.litdata_dir or --litdata_dir and should directly contain
    # expanded_metadata.pkl, trn/, val/, and tst/.
    seed_dir = resolve_seed_dir(seed, cfg, args)

    meta, qc = read_seed_metadata(seed_dir)
    n_all = len(meta["dict_node_names"])
    num_old = int(meta["num_old_nodes"])
    gene_id_to_name = {int(v): str(k) for k, v in meta["dict_node_names"].items()}

    pretrained_ckpt = args.pretrained_ckpt or cfg.get("pretrained_ckpt")
    if not pretrained_ckpt:
        raise ValueError("pretrained_ckpt is required")

    # Resolve checkpoint from:
    #   --ft_ckpt > bulk_runs.seed_43.ft_ckpt > finetune_output_dir auto-search
    ft_ckpt = resolve_ft_checkpoint(seed, cfg, args)

    # Device selection.
    if args.device:
        device = torch.device(args.device)
    elif str(cfg.get("accelerator", "auto")).lower() == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Use extraction batch/n_workers overrides if provided.
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.n_workers is not None:
        cfg["n_workers"] = int(args.n_workers)
    if args.max_batches is not None:
        cfg["max_batches"] = int(args.max_batches)

    extract_splits = args.splits or cfg.get("extract_splits", ["trn", "val", "tst"])
    max_batches = cfg.get("max_batches", None)

    logger.info("=" * 70)
    run_id = resolve_run_id(cfg, args)
    logger.info(f"Trait-aware bulk network extraction | run_id={run_id}, seed={seed}")
    logger.info(f"  litdata_dir: {seed_dir}")
    logger.info(f"  pretrained_ckpt: {pretrained_ckpt}")
    logger.info(f"  ft_ckpt: {ft_ckpt}")
    logger.info(f"  expanded nodes: total={n_all}, old={num_old}, new={n_all - num_old}")
    logger.info(f"  splits: {extract_splits}")
    logger.info("=" * 70)

    dataloaders = make_dataloaders(seed_dir, extract_splits, cfg)

    pretrained = load_pretrained_deeptan(pretrained_ckpt, map_location="cpu")

    # Load fine-tuned first so we can use its persisted initial embedding buffers
    # to make the expanded-pretrained baseline exact.
    ft_model, ft_state, actual_train_cfg = load_finetuned_trait_bulk_model(
        pretrained_model=pretrained,
        meta=meta,
        seed_dir=seed_dir,
        train_cfg=cfg,
        ft_ckpt_path=ft_ckpt,
        finetune_cls=finetune_cls,
        seed=seed,
        map_location="cpu",
    )

    baseline_model = instantiate_expanded_baseline(
        pretrained_model=pretrained,
        meta=meta,
        seed_dir=seed_dir,
        cfg=actual_train_cfg,
        finetune_cls=finetune_cls,
        seed=seed,
    )
    apply_exact_initial_embedding_from_checkpoint(baseline_model, ft_state)

    baseline_hash = _param_hash(baseline_model.amsgp.node_embedding_layers)
    ft_hash = _param_hash(ft_model.amsgp.node_embedding_layers)
    hashes = {
        "node_encoder_expanded_pretrained": baseline_hash,
        "node_encoder_finetuned": ft_hash,
        "baseline_embed": _tensor_hash(baseline_model.amsgp.node_embedding_layers.embed.weight),
        "finetuned_embed": _tensor_hash(ft_model.amsgp.node_embedding_layers.embed.weight),
    }

    if baseline_hash == ft_hash:
        msg = (
            "Node encoder parameters are identical between expanded-pretrained baseline and fine-tuned model. "
            "The extracted delta network would likely be zero or uninformative."
        )
        if bool(cfg.get("abort_if_node_encoder_identical", True)) and not args.allow_identical_node_encoder:
            raise RuntimeError(msg)
        logger.warning(msg)
    else:
        logger.success(
            f"Node encoder differs: baseline={baseline_hash}, fine_tuned={ft_hash}"
        )

    # Move models only after checkpoint/baseline construction.
    baseline_model = baseline_model.to(device)
    ft_model = ft_model.to(device)

    # Extract baseline signals.
    logger.info("\n[Step 1a] Extracting expanded-pretrained baseline node embeddings ...")
    signals_pre = extract_signals_from_dataloaders(
        baseline_model,
        dataloaders,
        dict_node_names=meta["dict_node_names"],
        device=device,
        max_batches=max_batches,
    )
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Extract fine-tuned signals.
    logger.info("\n[Step 1b] Extracting fine-tuned node embeddings ...")
    signals_ft = extract_signals_from_dataloaders(
        ft_model,
        dataloaders,
        dict_node_names=meta["dict_node_names"],
        device=device,
        max_batches=max_batches,
    )
    del ft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate and compute weights.
    logger.info("\n[Step 2] Aggregating gene/feature mean embeddings and cosine weights ...")
    mean_pre, counts_pre = aggregate_gene_means(signals_pre, n_total_nodes=n_all)
    mean_ft, counts_ft = aggregate_gene_means(signals_ft, n_total_nodes=n_all)

    edges_pre, w_pre_all, edge_seen_pre = compute_edge_cosine_weights(mean_pre, signals_pre["edge_counts"])
    edges_ft, w_ft_all, edge_seen_ft = compute_edge_cosine_weights(mean_ft, signals_ft["edge_counts"])

    logger.info(
        f"  baseline edges={len(edges_pre)}, ft edges={len(edges_ft)}, "
        f"observed genes baseline={(counts_pre > 0).sum()}/{n_all}, ft={(counts_ft > 0).sum()}/{n_all}"
    )

    logger.info("\n[Step 3] Computing delta weights on common edges ...")
    common_edges, w_pre, w_ft, delta_w, common_seen_pre, common_seen_ft = align_and_delta(
        edges_pre, w_pre_all, edge_seen_pre, edges_ft, w_ft_all, edge_seen_ft
    )
    logger.info(
        f"  common_edges={len(common_edges)}, "
        f"delta_w mean={delta_w.mean():.6f}, std={delta_w.std():.6f}, "
        f"positive={(delta_w > 0).mean():.2%}, negative={(delta_w < 0).mean():.2%}"
    )

    logger.info("\n[Step 4] Building edge and gene tables ...")
    edge_df = build_edge_table(
        common_edges=common_edges,
        w_pre=w_pre,
        w_ft=w_ft,
        delta_w=delta_w,
        edge_seen_pre=common_seen_pre,
        edge_seen_ft=common_seen_ft,
        gene_counts_pre=counts_pre,
        gene_counts_ft=counts_ft,
        gene_id_to_name=gene_id_to_name,
        num_old_nodes=num_old,
    )
    gene_df = build_gene_table(
        gene_id_to_name=gene_id_to_name,
        counts_pre=counts_pre,
        counts_ft=counts_ft,
        mean_pre=mean_pre,
        mean_ft=mean_ft,
        num_old_nodes=num_old,
    )

    # Sort edge table by absolute delta for convenient downstream inspection.
    edge_df = edge_df.sort("abs_delta_w", descending=True)

    output_root = resolve_output_root(cfg, args)
    if _is_single_run_mode(cfg, args) and not args.all_seeds and not cfg.get("bulk_runs"):
        out_dir = os.path.join(output_root, "trait_network")
    else:
        # Legacy multi-seed output layout.
        out_dir = os.path.join(output_root, f"seed_{seed}", "trait_network")

    logger.info("\n[Step 5] Saving outputs ...")
    return save_outputs(
        out_dir=out_dir,
        seed=seed,
        ft_ckpt_path=ft_ckpt,
        seed_dir=seed_dir,
        config_path=args.config,
        cfg=cfg,
        actual_train_cfg=actual_train_cfg,
        meta=meta,
        qc=qc,
        edge_df=edge_df,
        gene_df=gene_df,
        mean_pre=mean_pre,
        mean_ft=mean_ft,
        signals_pre=signals_pre,
        signals_ft=signals_ft,
        common_edges=common_edges,
        w_pre=w_pre,
        w_ft=w_ft,
        delta_w=delta_w,
        hashes=hashes,
    )


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run03: extract a trait-aware latent gene/feature network from one BulkExpand-DeepTAN fine-tuned model."
    )
    parser.add_argument("--config", required=True, help="YAML config for Run03 bulk trait-network extraction.")

    # Recommended single-run overrides.
    parser.add_argument("--litdata_dir", default=None, help="Run01 LitData directory containing expanded_metadata.pkl and trn/val/tst.")
    parser.add_argument("--finetuned_ckpt", default=None, help="Run02 fine-tuned TraitBulkDeepTAN checkpoint path.")
    parser.add_argument("--output_dir", default=None, help="Output root for the extracted trait-aware network.")
    parser.add_argument("--run_id", default=None, help="Human-readable run ID written only to metadata and logs.")

    # Legacy aliases / multi-seed compatibility.
    parser.add_argument("--seed", type=int, default=None, help="Legacy data split seed, e.g. 43. Not needed for the recommended single-run workflow.")
    parser.add_argument("--all_seeds", action="store_true", help="Legacy mode: process all seeds from config.bulk_run_order, config.bulk_runs, or config.seeds.")
    parser.add_argument("--ft_ckpt", default=None, help="Legacy alias for --finetuned_ckpt.")
    parser.add_argument("--dataset_root", default=None, help="Legacy LitData root. If it directly contains expanded_metadata.pkl and trn/val/tst, it is used as the LitData directory.")

    parser.add_argument("--pretrained_ckpt", default=None, help="Override the scRNA-pretrained DeepTAN checkpoint path.")
    parser.add_argument("--device", default=None, help="Explicit device, for example cuda:0 or cpu.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="Debug mode: cap the total number of batches across all selected splits.")
    parser.add_argument("--splits", nargs="+", default=None, choices=["trn", "val", "tst"], help="LitData splits used for network extraction.")
    parser.add_argument("--local_deeptan_src", default=None, help="Override the local DeepTAN source path.")
    parser.add_argument("--finetune_module_dir", default=None, help="Directory containing the Run02 fine-tuning module.")
    parser.add_argument("--finetune_module_name", default=None, help="Python module name containing TraitBulkDeepTAN.")
    parser.add_argument("--finetune_class_name", default=None, help="Fine-tuning class name. Default: TraitBulkDeepTAN.")
    parser.add_argument(
        "--allow_identical_node_encoder",
        action="store_true",
        help="Do not abort if baseline and fine-tuned node encoder hashes are identical.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    if args.local_deeptan_src:
        cfg["local_deeptan_src"] = args.local_deeptan_src
    setup_local_deeptan(cfg.get("local_deeptan_src", ""))

    if args.finetune_module_dir:
        cfg["finetune_module_dir"] = args.finetune_module_dir
    if args.finetune_module_name:
        cfg["finetune_module_name"] = args.finetune_module_name
    if args.finetune_class_name:
        cfg["finetune_class_name"] = args.finetune_class_name

    _module, finetune_cls = _resolve_finetune_module(cfg)

    outputs = {}
    single_run = _is_single_run_mode(cfg, args) and not args.all_seeds

    if single_run:
        # In the GitHub-facing workflow, cfg.seed is only a reproducibility seed
        # used for deterministic baseline construction. It is not a data split ID.
        extraction_seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
        out = process_one_seed(extraction_seed, cfg=copy.deepcopy(cfg), args=args, finetune_cls=finetune_cls)
        if out:
            outputs[resolve_run_id(cfg, args)] = out
    else:
        # Legacy multi-seed mode retained for old experiments.
        seeds = resolve_requested_seeds(cfg, args)
        for seed in seeds:
            try:
                out = process_one_seed(seed, cfg=copy.deepcopy(cfg), args=args, finetune_cls=finetune_cls)
                if out:
                    outputs[f"seed_{seed}"] = out
            except Exception as e:
                logger.error(f"Failed seed={seed}: {e}")
                import traceback
                traceback.print_exc()

    if outputs:
        summary_root = resolve_output_root(cfg, args)
        os.makedirs(summary_root, exist_ok=True)
        summary_path = os.path.join(summary_root, "bulk_trait_summary.yaml")
        with open(summary_path, "w") as f:
            yaml.dump(outputs, f, allow_unicode=True, default_flow_style=False)
        logger.success(f"Summary written: {summary_path}")
    else:
        raise RuntimeError("No outputs were generated")


if __name__ == "__main__":
    main()
