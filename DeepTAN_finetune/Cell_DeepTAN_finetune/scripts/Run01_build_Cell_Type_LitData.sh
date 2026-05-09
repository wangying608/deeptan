#!/usr/bin/env bash
# ==============================================================================
# Run01: Build Cell-Type LitData for DeepTAN Fine-Tuning
# ==============================================================================
# Purpose:
#   Convert user-provided train/validation/test scRNA expression parquet files
#   into the LitData directory required by Run02 cell-type fine-tuning.
#
# Pipeline position:
#   Run01: user-defined scRNA files -> LitData
#   Run02: LitData -> cell-type fine-tuned DeepTAN checkpoint
#   Run03: LitData + checkpoint -> cell-specific network
#
# Usage:
#   1. Edit Section 1 paths.
#   2. Make this script executable:
#        chmod +x 01_build_cell_litdata_custom.sh
#   3. Run:
#        bash 01_build_cell_litdata_custom.sh
#
# Input naming policy:
#   File names are fully user-defined. The files do NOT need to be named
#   split_42_0.parquet or follow any internal batch naming convention.
#
# Output:
#   LITDATA_DIR/
#     trn/
#     val/
#     tst/
#     litdata_others2save.pkl
#     litdata_others2save.json
#     gene_cv_weights.csv
#     celltype_onehot.parquet  # if labels are provided/generated
# ============================================================================== 

set -euo pipefail


# ------------------------------------------------------------------------------
# 0. Optional: activate conda / mamba environment
# ------------------------------------------------------------------------------
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_deeptan_env_name


# ==============================================================================
# 1. User-configurable paths
# ===============================================================================

# Local DeepTAN source checkout. This is exported as DEEPTAN_SRC and consumed by
# the Python script. Leave empty if DeepTAN is already installed in the active
# Python environment.
DEEPTAN_SRC=""

# Run01 Python script.
RUN01_SCRIPT="/path/to/run_01_build_cell_litdata_userfriendly.py"

# Pretraining resources. These come from the scRNA DeepTAN pretraining stage.
PRETRAINED_TRN_NPZ="/path/to/pretrained/trn.npz"
PRETRAINED_PKL="/path/to/pretrained/others2save.pkl"

# Cell-type/run name. This is only used for metadata and readable logs.
CELL_TYPE_NAME="ExampleCellType"

# User-defined train/validation/test expression files.
# Required format:
#   - one cell/sample ID column, e.g. obs_names / obs_name / barcode / cell_id;
#   - numeric gene expression columns.
TRN_PARQUET="/path/to/train_expression.parquet"
VAL_PARQUET="/path/to/validation_expression.parquet"
TST_PARQUET="/path/to/test_expression.parquet"

# Optional label inputs.
# Use one of the following modes:
#   A) Existing one-hot label parquet:
#        LABELS_PARQUET="/path/to/celltype_onehot.parquet"
#
#   B) A cell-type column already present in train/val/test parquet:
#        CELLTYPE_COL="cell_type"
#
#   C) External annotation CSV:
#        CELLTYPE_CSV="/path/to/celltype_annotation.csv"
#
# For recon_only fine-tuning, labels are optional.
LABELS_PARQUET=""
CELLTYPE_COL=""
CELLTYPE_CSV=""

# Output LitData directory. Run02 should use this path in config.tissues.<name>.litdata.
LITDATA_DIR="/path/to/cell_litdata"


# ==============================================================================
# 2. Conversion parameters
# ===============================================================================

BATCH_SIZE=32
N_WORKERS=8
EDGE_WEIGHT_THRESHOLD=0.0
RANDOM_SEED=42

# Set to true to skip gene_cv_weights.csv calculation.
SKIP_CV=false


# ==============================================================================
# 3. Logging
# ===============================================================================

LOG_DIR="${LITDATA_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run01_build_cell_litdata_$(date +%Y%m%d_%H%M%S).log"


# ==============================================================================
# 4. Sanity checks
# ===============================================================================

check_file_exists() {
  local file_path="$1"
  local description="$2"

  if [[ ! -f "${file_path}" ]]; then
    echo "[ERROR] Missing ${description}: ${file_path}" >&2
    exit 1
  fi
}

check_optional_file_exists() {
  local file_path="$1"
  local description="$2"

  if [[ -n "${file_path}" && ! -f "${file_path}" ]]; then
    echo "[ERROR] Missing ${description}: ${file_path}" >&2
    exit 1
  fi
}

check_file_exists "${RUN01_SCRIPT}" "Run01 Python script"
check_file_exists "${PRETRAINED_TRN_NPZ}" "pretrained train npz"
check_file_exists "${PRETRAINED_PKL}" "pretrained metadata pkl"
check_file_exists "${TRN_PARQUET}" "train expression parquet"
check_file_exists "${VAL_PARQUET}" "validation expression parquet"
check_file_exists "${TST_PARQUET}" "test expression parquet"
check_optional_file_exists "${LABELS_PARQUET}" "one-hot labels parquet"
check_optional_file_exists "${CELLTYPE_CSV}" "cell-type annotation csv"

mkdir -p "${LITDATA_DIR}"

if [[ -n "${DEEPTAN_SRC}" ]]; then
  if [[ ! -d "${DEEPTAN_SRC}" ]]; then
    echo "[ERROR] DEEPTAN_SRC does not exist: ${DEEPTAN_SRC}" >&2
    exit 1
  fi
  export DEEPTAN_SRC
fi


# ==============================================================================
# 5. Print run information
# ===============================================================================

echo "=============================================================================="
echo "Run01 | Build Cell-Type LitData"
echo "Python              : $(which python)"
echo "RUN01_SCRIPT        : ${RUN01_SCRIPT}"
echo "DEEPTAN_SRC         : ${DEEPTAN_SRC:-<use environment package>}"
echo "CELL_TYPE_NAME      : ${CELL_TYPE_NAME}"
echo "TRN_PARQUET         : ${TRN_PARQUET}"
echo "VAL_PARQUET         : ${VAL_PARQUET}"
echo "TST_PARQUET         : ${TST_PARQUET}"
echo "LABELS_PARQUET      : ${LABELS_PARQUET:-<not provided>}"
echo "CELLTYPE_COL        : ${CELLTYPE_COL:-<not provided>}"
echo "CELLTYPE_CSV        : ${CELLTYPE_CSV:-<not provided>}"
echo "LITDATA_DIR         : ${LITDATA_DIR}"
echo "LOG_FILE            : ${LOG_FILE}"
echo "=============================================================================="
echo


# ==============================================================================
# 6. Build command
# ===============================================================================

CMD=(
  python -u "${RUN01_SCRIPT}"
  --pretrained_trn_npz "${PRETRAINED_TRN_NPZ}"
  --pretrained_pkl "${PRETRAINED_PKL}"
  --trn_parquet "${TRN_PARQUET}"
  --val_parquet "${VAL_PARQUET}"
  --tst_parquet "${TST_PARQUET}"
  --tissue_name "${CELL_TYPE_NAME}"
  --output_dir "${LITDATA_DIR}"
  --bs "${BATCH_SIZE}"
  --thre_mi "${EDGE_WEIGHT_THRESHOLD}"
  --n_workers "${N_WORKERS}"
  --seed "${RANDOM_SEED}"
  --validate_output
)

if [[ -n "${LABELS_PARQUET}" ]]; then
  CMD+=(--labels_parquet "${LABELS_PARQUET}")
fi

if [[ -n "${CELLTYPE_COL}" ]]; then
  CMD+=(--celltype_col "${CELLTYPE_COL}")
fi

if [[ -n "${CELLTYPE_CSV}" ]]; then
  CMD+=(--celltype_csv "${CELLTYPE_CSV}")
fi

if [[ "${SKIP_CV}" == "true" ]]; then
  CMD+=(--skip_cv)
fi


# ==============================================================================
# 7. Run conversion
# ===============================================================================

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"


echo
echo "=============================================================================="
echo "Run01 finished."
echo "LitData directory:"
echo "  ${LITDATA_DIR}"
echo
echo "Use this path in Run02 config:"
echo "  tissues:"
echo "    ${CELL_TYPE_NAME}:"
echo "      litdata: \"${LITDATA_DIR}\""
echo "=============================================================================="
