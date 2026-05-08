#!/usr/bin/env bash
# ==============================================================================
# Build single-run Bulk LitData for Trait-aware BulkExpand-DeepTAN Fine-tuning
# ==============================================================================
# Usage:
#   1. Edit the paths in section 1.
#   2. Run:
#        bash 01_build_litdata_custom.sh
#
# This script does not require any fixed input filename pattern. The user may use
# arbitrary file names as long as the file contents match the required format.
# Output is written directly to LITDATA_DIR, not LITDATA_DIR/seed_xx.
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# 0. Optional environment activation
# ------------------------------------------------------------------------------
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_deeptan_env_name

# ------------------------------------------------------------------------------
# 1. User-configurable paths
# ------------------------------------------------------------------------------

RUN_ID="FT16_bulk_run"

PY_SCRIPT="/path/to/src/run_01_build_trait_bulk_litdata_single.py"

# Run01 output directory. After success, this directory directly contains:
#   trn/ val/ tst/ expanded_metadata.pkl qc_report.json
LITDATA_DIR="/path/to/output/litdata"

PRETRAINED_TRN_NPZ="/path/to/pretrained/split_train_old_graph.npz"
PRETRAINED_PKL="/path/to/pretrained/others2save.pkl"

BULK_NMIC_NPZ="/path/to/your_bulk_train_nmic.npz"
BULK_NMIC_PARQUET="/path/to/your_bulk_train_nmic_companion.parquet"

TRN_PARQUET="/path/to/your_train_expression.parquet"
VAL_PARQUET="/path/to/your_validation_expression.parquet"
TST_PARQUET="/path/to/your_test_expression.parquet"

PHENOTYPE_PARQUET="/path/to/your_phenotype.parquet"
PHENOTYPE_COL="xxx"
OBS_COL="obs_names"

# ------------------------------------------------------------------------------
# 2. Graph and expression settings
# ------------------------------------------------------------------------------

GRAPH_SCOPE="sample_subgraph"
NODE_SELECTION="nonzero_top_abs"
VALUE_THRESHOLD="1e-8"
MAX_NODES_PER_SAMPLE=0
MIN_NODES_PER_SAMPLE=0
BULK_TOPK_PER_NODE=0
MAX_BULK_EDGES=0
X_TRANSFORM="log1p"

# ------------------------------------------------------------------------------
# 3. Logs and checks
# ------------------------------------------------------------------------------

LOG_DIR="${LITDATA_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/build_litdata_$(date +%Y%m%d_%H%M%S).log"

check_file_exists() {
  local file_path="$1"
  local description="$2"
  if [[ ! -f "${file_path}" ]]; then
    echo "[ERROR] Missing ${description}: ${file_path}" >&2
    exit 1
  fi
}

check_file_exists "${PY_SCRIPT}" "Run01 Python script"
check_file_exists "${PRETRAINED_TRN_NPZ}" "pretrained train NMIC npz"
check_file_exists "${PRETRAINED_PKL}" "pretrained metadata pkl"
check_file_exists "${BULK_NMIC_NPZ}" "bulk train NMIC npz"
check_file_exists "${BULK_NMIC_PARQUET}" "bulk NMIC companion parquet"
check_file_exists "${TRN_PARQUET}" "train expression parquet"
check_file_exists "${VAL_PARQUET}" "validation expression parquet"
check_file_exists "${TST_PARQUET}" "test expression parquet"
check_file_exists "${PHENOTYPE_PARQUET}" "phenotype parquet"

mkdir -p "${LITDATA_DIR}"

# ------------------------------------------------------------------------------
# 4. Run
# ------------------------------------------------------------------------------

echo "=============================================================================="
echo "Trait-aware BulkExpand-DeepTAN LitData Builder"
echo "RUN_ID      : ${RUN_ID}"
echo "Python      : $(which python)"
echo "PY_SCRIPT   : ${PY_SCRIPT}"
echo "LITDATA_DIR : ${LITDATA_DIR}"
echo "LOG_FILE    : ${LOG_FILE}"
echo "=============================================================================="
echo

python -u "${PY_SCRIPT}" \
  --pretrained_trn_npz "${PRETRAINED_TRN_NPZ}" \
  --pretrained_pkl "${PRETRAINED_PKL}" \
  --bulk_trn_npz "${BULK_NMIC_NPZ}" \
  --bulk_nmic_parquet "${BULK_NMIC_PARQUET}" \
  --trn_parquet "${TRN_PARQUET}" \
  --val_parquet "${VAL_PARQUET}" \
  --tst_parquet "${TST_PARQUET}" \
  --phenotype_parquet "${PHENOTYPE_PARQUET}" \
  --phenotype_col "${PHENOTYPE_COL}" \
  --obs_col "${OBS_COL}" \
  --output_dir "${LITDATA_DIR}" \
  --run_id "${RUN_ID}" \
  --nmic_companion_check_action error \
  --graph_scope "${GRAPH_SCOPE}" \
  --node_selection "${NODE_SELECTION}" \
  --strict_nonzero_subgraph \
  --value_threshold "${VALUE_THRESHOLD}" \
  --max_nodes_per_sample "${MAX_NODES_PER_SAMPLE}" \
  --min_nodes_per_sample "${MIN_NODES_PER_SAMPLE}" \
  --no-drop_isolated_nodes \
  --no-coverage_enhancement \
  --bulk_topk_per_node "${BULK_TOPK_PER_NODE}" \
  --max_bulk_edges "${MAX_BULK_EDGES}" \
  --x_transform "${X_TRANSFORM}" \
  2>&1 | tee "${LOG_FILE}"

echo
echo "=============================================================================="
echo "Finished building LitData."
echo "Output   : ${LITDATA_DIR}"
echo "Log file : ${LOG_FILE}"
echo "=============================================================================="
