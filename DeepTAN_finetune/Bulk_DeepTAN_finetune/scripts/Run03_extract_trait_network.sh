#!/usr/bin/env bash
# ==============================================================================
# Run03: Extract trait-aware latent network from a fine-tuned BulkExpand-DeepTAN
# ==============================================================================
# Usage:
#   bash 03_extract_trait_network.sh configs/bulk_trait_network_single.yaml
#
# This script is the downstream step after:
#   1. Run01: build LitData from user-defined input files
#   2. Run02: fine-tune BulkExpand-DeepTAN on the LitData
#   3. Run03: extract trait-aware latent network using LitData + fine-tuned ckpt
#
# The YAML config should define:
#   litdata_dir       : Run01 output directory containing expanded_metadata.pkl,trn,val,tst
#   finetuned_ckpt    : Run02 checkpoint path
#   pretrained_ckpt   : scRNA pretrained DeepTAN checkpoint path
#   output_dir        : output root; results go to output_dir/trait_network
# ===============================================================================

set -euo pipefail

# Optional: activate your environment here.
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_deeptan_env_name

CONFIG_FILE="${1:-configs/bulk_trait_network_single.yaml}"
PY_SCRIPT="${PY_SCRIPT:-src/bulk_trait_network_extract_single.py}"
LOG_DIR=""
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/extract_trait_network_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Python extraction script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

echo "=============================================================================="
echo "Bulk trait-aware latent network extraction"
echo "Python      : $(which python)"
echo "PY_SCRIPT   : ${PY_SCRIPT}"
echo "CONFIG_FILE : ${CONFIG_FILE}"
echo "LOG_FILE    : ${LOG_FILE}"
echo "=============================================================================="
echo

python -u "${PY_SCRIPT}" \
  --config "${CONFIG_FILE}" \
  2>&1 | tee "${LOG_FILE}"

echo
echo "=============================================================================="
echo "Network extraction finished."
echo "Config   : ${CONFIG_FILE}"
echo "Log file : ${LOG_FILE}"
echo "=============================================================================="
