#!/usr/bin/env bash
# ==============================================================================
# Fine-tune Trait-aware BulkExpand-DeepTAN on a single LitData directory
# ==============================================================================
# Usage:
#   bash 02_finetune_bulk.sh /path/to/finetune_bulk_single.yaml
#
# The config's dataset_root should directly contain:
#   trn/ val/ tst/ expanded_metadata.pkl
# ==============================================================================

set -euo pipefail

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_deeptan_env_name

CONFIG_FILE="${1:-/path/to/configs/finetune_bulk_single.yaml}"
PY_SCRIPT="${PY_SCRIPT:-/path/to/src/run_02_trait_aware_bulk_finetune_single.py}"

LOG_DIR="${LOG_DIR:-logs/finetune}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/finetune_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Python fine-tuning script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

echo "=============================================================================="
echo "Trait-aware BulkExpand-DeepTAN Fine-tuning"
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
echo "Fine-tuning finished."
echo "Config   : ${CONFIG_FILE}"
echo "Log file : ${LOG_FILE}"
echo "=============================================================================="
