#!/usr/bin/env bash
# ==============================================================================
# Run03: Extract Tissue-Specific DeepTAN Network
# ==============================================================================
#
# Purpose:
#   Extract a tissue-specific latent gene network from a tissue fine-tuned
#   DeepTAN checkpoint.
#
# Pipeline position:
#   Run01: user-defined tissue scRNA files -> LitData
#   Run02: LitData -> tissue-specific fine-tuned DeepTAN checkpoint
#   Run03: LitData + fine-tuned checkpoint -> tissue-specific latent network
#
# Usage:
#   1. Edit Section 1 paths, or pass a config file as the first argument.
#
#   2. Make this script executable:
#        chmod +x scripts/Run03_extract_network.sh
#
#   3. Run:
#        bash scripts/Run03_extract_network.sh configs/Run03_extract_network.yaml
#
# Optional overrides after the config path are forwarded to Python:
#        bash scripts/Run03_extract_network.sh configs/Run03_extract_network.yaml \
#          --litdata_dir /path/to/tissue_litdata \
#          --finetuned_ckpt /path/to/best_model.ckpt \
#          --output_dir /path/to/tissue_network_output
#
# Notes:
#   - DEEPTAN_SRC is needed only if DeepTAN is not installed in the active
#     Python environment.
#   - FINETUNE_MODULE_DIR should contain the Run02 fine-tuning script so that
#     Run03 can dynamically import the LightningModule class.
# ==============================================================================

set -euo pipefail


# ------------------------------------------------------------------------------
# 0. Optional: activate conda / mamba environment
# ------------------------------------------------------------------------------
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_deeptan_env_name


# ==============================================================================
# 1. User-configurable paths
# ==============================================================================

# Local DeepTAN source checkout.
# Leave empty if DeepTAN is already installed in the active Python environment.
DEEPTAN_SRC="/path/to/deeptan-dev/src"

# Directory containing the Run02 tissue fine-tuning module.
# Leave empty if the module is already importable from PYTHONPATH or provided
# correctly in the YAML config.
FINETUNE_MODULE_DIR="/path/to/Tissue_DeepTAN_finetune/src"

# Run03 Python script.
RUN03_SCRIPT="/path/to/Tissue_DeepTAN_finetune/src/Run03_extract_network.py"

# Default config file. This can be overridden by passing a config path as $1.
DEFAULT_CONFIG_FILE="/path/to/Tissue_DeepTAN_finetune/configs/Run03_extract_network.yaml"


# ==============================================================================
# 2. Parse arguments
# ==============================================================================

CONFIG_FILE="${1:-${DEFAULT_CONFIG_FILE}}"

if [[ $# -gt 0 ]]; then
  shift
fi

EXTRA_ARGS=("$@")


# ==============================================================================
# 3. Logging
# ==============================================================================

LOG_DIR="${LOG_DIR:-logs/run03_tissue_network}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run03_extract_tissue_network_$(date +%Y%m%d_%H%M%S).log"


# ==============================================================================
# 4. Sanity checks
# ==============================================================================

check_file_exists() {
  local file_path="$1"
  local description="$2"

  if [[ ! -f "${file_path}" ]]; then
    echo "[ERROR] Missing ${description}: ${file_path}" >&2
    exit 1
  fi
}

check_file_exists "${RUN03_SCRIPT}" "Run03 Python script"
check_file_exists "${CONFIG_FILE}" "Run03 YAML config"

if [[ -n "${DEEPTAN_SRC}" ]]; then
  if [[ ! -d "${DEEPTAN_SRC}" ]]; then
    echo "[ERROR] DEEPTAN_SRC does not exist: ${DEEPTAN_SRC}" >&2
    exit 1
  fi
  export DEEPTAN_SRC
fi

if [[ -n "${FINETUNE_MODULE_DIR}" ]]; then
  if [[ ! -d "${FINETUNE_MODULE_DIR}" ]]; then
    echo "[ERROR] FINETUNE_MODULE_DIR does not exist: ${FINETUNE_MODULE_DIR}" >&2
    exit 1
  fi
  export FINETUNE_MODULE_DIR
fi


# ==============================================================================
# 5. Print run information
# ==============================================================================

echo "=============================================================================="
echo "Run03 | Extract Tissue-Specific DeepTAN Network"
echo "Python               : $(which python)"
echo "RUN03_SCRIPT         : ${RUN03_SCRIPT}"
echo "CONFIG_FILE          : ${CONFIG_FILE}"
echo "DEEPTAN_SRC          : ${DEEPTAN_SRC:-<use environment package>}"
echo "FINETUNE_MODULE_DIR  : ${FINETUNE_MODULE_DIR:-<from config or PYTHONPATH>}"
echo "LOG_FILE             : ${LOG_FILE}"
echo "EXTRA_ARGS           : ${EXTRA_ARGS[*]:-<none>}"
echo "=============================================================================="
echo


# ==============================================================================
# 6. Run extraction
# ==============================================================================

python -u "${RUN03_SCRIPT}" \
  --config "${CONFIG_FILE}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"


echo
echo "=============================================================================="
echo "Run03 finished."
echo "Check the output_dir configured in:"
echo "  ${CONFIG_FILE}"
echo
echo "Expected single-run output:"
echo "  output_dir/detsn/detsn_edge_table.parquet"
echo "  output_dir/detsn/detsn_metadata.yaml"
echo "  output_dir/detsn/mean_embeddings_pretrained.npy"
echo "  output_dir/detsn/mean_embeddings_finetuned.npy"
echo "  output_dir/detsn/gene_id_map.csv"
echo "=============================================================================="
