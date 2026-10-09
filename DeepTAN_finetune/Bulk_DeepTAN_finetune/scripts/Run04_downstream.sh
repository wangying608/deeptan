#!/usr/bin/env bash
# ==============================================================================
# Run04: Trait-Aware Bulk Network Downstream Analysis
# ==============================================================================
#
# Purpose:
#   Run non-plotting downstream module / hub / gene-set analysis on the
#   trait-aware bulk network extracted by Run03.
#
# Pipeline position:
#   Run01: raw bulk files -> LitData
#   Run02: LitData -> fine-tuned BulkExpand-DeepTAN checkpoint
#   Run03: LitData + checkpoint -> trait-aware latent network
#   Run04: trait-aware network -> module / hub / gene-set analysis
#   Run05: Run04 outputs -> publication network figure
#
# Core Run04 analysis:
#   1. Read the Run03 trait-aware edge table.
#   2. Use abs(weight_col), normally abs(w_ft), as non-negative edge strength.
#   3. Detect raw weighted communities with Leiden/RBConfiguration.
#   4. Treat sufficiently large raw Leiden communities as frozen core modules.
#   5. Resolve smaller raw communities using a frozen-core constrained
#      maximum-spanning forest on the raw-community graph.
#   6. Use RB/configuration-model delta-Q as the PRIMARY reassignment score.
#   7. Leave disconnected coreless components below MIN_MODULE_SIZE unassigned.
#   8. Identify module Hub genes by intra-module weighted degree.
#   9. Export module tables, reassignment audits, and gene sets.
#
# Important:
#   - This script does NOT generate network figures.
#   - GO/KEGG enrichment is NOT performed here.
#   - RANDOM_SEED controls reproducibility; it is NOT a data-split seed.
#
# Recommended repository layout:
#
#   project/
#   ├── bulk_network_downstream_analysis.py
#   └── 04_run_downstream_analysis.sh
#
# Usage:
#   1. Edit NETWORK_DIR and OUT_DIR below.
#
#   2. Make executable:
#        chmod +x 04_run_downstream_analysis.sh
#
#   3. Run:
#        ./04_run_downstream_analysis.sh
#
#      or:
#        bash 04_run_downstream_analysis.sh
#
# ==============================================================================

set -euo pipefail


# ==============================================================================
# 0. Optional: activate conda / mamba environment
# ==============================================================================
#
# Uncomment and edit if needed.
#
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_deeptan_env_name


# ==============================================================================
# 1. Script and user-configurable paths
# ==============================================================================

# Directory containing this shell script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Python executable.
# Can also be overridden externally, e.g.:
#   PYTHON_BIN=/path/to/python ./04_run_downstream_analysis.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

# Run04 Python analysis script.
# By default, it is expected to be in the same directory as this .sh file.
PY_SCRIPT="${SCRIPT_DIR}/bulk_network_downstream_analysis.py"

# Run03 trait-network directory.
#
# It should normally contain one of:
#   bulk_trait_edge_table.parquet
#   bulk_trait_csn_edge_table.parquet
NETWORK_DIR="/path/to/network_output/trait_network"

# Run04 output directory.
#
# Main outputs will be written under:
#   ${OUT_DIR}/tables/
#   ${OUT_DIR}/gene_sets/
#   ${OUT_DIR}/logs/
OUT_DIR="/path/to/downstream_output"


# ==============================================================================
# 2. Network and module-analysis parameters
# ==============================================================================

# Reproducibility seed.
#
# This affects stochastic community detection / deterministic analysis steps.
# It is NOT a LitData split ID.
RANDOM_SEED=42


# ------------------------------------------------------------------------------
# Edge weight
# ------------------------------------------------------------------------------

# Signed edge-weight column in the Run03 edge table.
#
# Recommended/default:
#   w_ft
#
# Analysis strength is:
#   abs(WEIGHT_COL)
#
# NOTE:
#   w_ft is the intended primary network weight for the current workflow.
WEIGHT_COL="w_ft"


# ------------------------------------------------------------------------------
# Module-detection edge selection
# ------------------------------------------------------------------------------

# Fraction of strongest edges retained for module detection.
#
# 1.0:
#   use all non-zero edges after MIN_ABS_WEIGHT filtering.
#
# Example:
#   0.20 = strongest 20% of remaining edges.
MODULE_TOP_FRAC=1.0

# Optional minimum absolute edge strength.
#
# Leave empty to disable:
MIN_ABS_WEIGHT=""

# Example:
# MIN_ABS_WEIGHT="0.10"


# ------------------------------------------------------------------------------
# Leiden / community detection
# ------------------------------------------------------------------------------

# Minimum raw Leiden-community size defining a frozen core-module anchor.
#
# Raw communities >= this threshold:
#   become frozen core anchors.
#
# Raw communities < this threshold:
#   are resolved on the raw-community graph using the frozen-core constrained
#   maximum-spanning-forest procedure.
#
# Important:
#   Small communities are NOT simply merged into the largest module.
#
#   Coreless disconnected components that still remain smaller than this
#   threshold are explicitly left unassigned and excluded from Hub selection.
MIN_MODULE_SIZE=50

# Leiden/RBConfiguration resolution parameter.
RESOLUTION=1.3


# ------------------------------------------------------------------------------
# Small-community reassignment
# ------------------------------------------------------------------------------

# Number of strongest block-to-module/raw-community connections retained as
# secondary local-affinity evidence.
#
# The PRIMARY assignment criterion remains RB/configuration-model delta-Q.
REASSIGN_TOP_K=10


# ------------------------------------------------------------------------------
# Community-detection fallback policy
# ------------------------------------------------------------------------------

# 0:
#   Require weighted Leiden.
#   Recommended for reproducible / publication-oriented analyses.
#
# 1:
#   If Leiden fails, explicitly allow Louvain or NetworkX greedy-modularity
#   fallback.
#
# IMPORTANT:
#   Keeping this at 0 prevents silent changes in the formal community-detection
#   method across computing environments.
ALLOW_COMMUNITY_FALLBACK=0


# ------------------------------------------------------------------------------
# Hub-gene definition
# ------------------------------------------------------------------------------

# Fraction of genes within each final assigned module selected as Hub genes.
#
# Hub ranking is based on intra-module weighted degree:
#
#   sum_j abs(w_ft_ij)
#
HUB_TOP_FRAC=0.10

# Minimum number of Hubs per assigned module.
MIN_HUBS=1

# Maximum number of Hubs per assigned module.
MAX_HUBS=100000


# ==============================================================================
# 3. Helper functions
# ==============================================================================

die() {
  echo "[ERROR] $*" >&2
  exit 1
}


check_file_exists() {
  local file_path="$1"
  local description="$2"

  [[ -f "${file_path}" ]] || \
    die "Missing ${description}: ${file_path}"
}


check_dir_exists() {
  local dir_path="$1"
  local description="$2"

  [[ -d "${dir_path}" ]] || \
    die "Missing ${description}: ${dir_path}"
}


# ==============================================================================
# 4. Sanity checks
# ==============================================================================

# Python executable.
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  die "Python executable not found: ${PYTHON_BIN}"
fi

# Run04 Python script.
check_file_exists \
  "${PY_SCRIPT}" \
  "Run04 downstream-analysis Python script"

# Run03 network directory.
check_dir_exists \
  "${NETWORK_DIR}" \
  "Run03 trait-network directory"


# ------------------------------------------------------------------------------
# Check Run03 edge table
# ------------------------------------------------------------------------------

EDGE_TABLE_PRIMARY="${NETWORK_DIR}/bulk_trait_edge_table.parquet"
EDGE_TABLE_LEGACY="${NETWORK_DIR}/bulk_trait_csn_edge_table.parquet"

if [[ -f "${EDGE_TABLE_PRIMARY}" ]]; then
  DETECTED_EDGE_TABLE="${EDGE_TABLE_PRIMARY}"
elif [[ -f "${EDGE_TABLE_LEGACY}" ]]; then
  DETECTED_EDGE_TABLE="${EDGE_TABLE_LEGACY}"
else
  die \
    "No Run03 edge table found in ${NETWORK_DIR}. Expected either:
  ${EDGE_TABLE_PRIMARY}
  ${EDGE_TABLE_LEGACY}"
fi


# ------------------------------------------------------------------------------
# Check core Python dependencies
# ------------------------------------------------------------------------------

if ! "${PYTHON_BIN}" -c \
  'import networkx, numpy, polars' \
  >/dev/null 2>&1
then
  die \
    "Missing required Python package(s).
Required core packages include:
  networkx
  numpy
  polars"
fi


# ------------------------------------------------------------------------------
# Check Leiden dependencies when strict Leiden mode is enabled
# ------------------------------------------------------------------------------

if [[ "${ALLOW_COMMUNITY_FALLBACK}" == "0" ]]; then
  if ! "${PYTHON_BIN}" -c \
    'import igraph, leidenalg' \
    >/dev/null 2>&1
  then
    die \
      "Weighted Leiden is required because ALLOW_COMMUNITY_FALLBACK=0.

Missing or unavailable package(s):
  python-igraph
  leidenalg

Install/check these packages, or explicitly set:
  ALLOW_COMMUNITY_FALLBACK=1

for exploratory fallback analysis."
  fi
fi


# ------------------------------------------------------------------------------
# Validate selected switches
# ------------------------------------------------------------------------------

if [[ "${ALLOW_COMMUNITY_FALLBACK}" != "0" \
   && "${ALLOW_COMMUNITY_FALLBACK}" != "1" ]]; then
  die "ALLOW_COMMUNITY_FALLBACK must be either 0 or 1."
fi


# ==============================================================================
# 5. Prepare output and logging directories
# ==============================================================================

mkdir -p "${OUT_DIR}"

LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/downstream_analysis_$(date +%Y%m%d_%H%M%S).log"


# ==============================================================================
# 6. Print run configuration
# ==============================================================================

echo "=============================================================================="
echo "Run04: Trait-Aware Bulk Network Downstream Analysis"
echo "=============================================================================="
echo
echo "Software:"
echo "  Python executable          : $(command -v "${PYTHON_BIN}")"
echo "  Python version             : $("${PYTHON_BIN}" --version 2>&1)"
echo "  Run04 script               : ${PY_SCRIPT}"
echo
echo "Input / output:"
echo "  NETWORK_DIR                : ${NETWORK_DIR}"
echo "  detected edge table        : ${DETECTED_EDGE_TABLE}"
echo "  OUT_DIR                    : ${OUT_DIR}"
echo "  LOG_FILE                   : ${LOG_FILE}"
echo
echo "Network analysis:"
echo "  RANDOM_SEED                : ${RANDOM_SEED}"
echo "  WEIGHT_COL                 : ${WEIGHT_COL}"
echo "  analysis edge strength     : abs(${WEIGHT_COL})"
echo "  MODULE_TOP_FRAC            : ${MODULE_TOP_FRAC}"
echo "  MIN_ABS_WEIGHT             : ${MIN_ABS_WEIGHT:-<disabled>}"
echo
echo "Community detection:"
echo "  MIN_MODULE_SIZE            : ${MIN_MODULE_SIZE}"
echo "  RESOLUTION                 : ${RESOLUTION}"
echo "  REASSIGN_TOP_K             : ${REASSIGN_TOP_K}"
echo "  ALLOW_COMMUNITY_FALLBACK   : ${ALLOW_COMMUNITY_FALLBACK}"
echo
echo "Hub selection:"
echo "  HUB_TOP_FRAC               : ${HUB_TOP_FRAC}"
echo "  MIN_HUBS                   : ${MIN_HUBS}"
echo "  MAX_HUBS                   : ${MAX_HUBS}"
echo
echo "Small-community policy:"
echo "  frozen cores               : raw Leiden communities >= ${MIN_MODULE_SIZE}"
echo "  primary reassignment score : RB/configuration-model delta-Q"
echo "  disconnected undersized    : left unassigned"
echo "=============================================================================="
echo


# ==============================================================================
# 7. Build Run04 command
# ==============================================================================

CMD=(
  "${PYTHON_BIN}"
  -u
  "${PY_SCRIPT}"

  --network_dir
  "${NETWORK_DIR}"

  --out_dir
  "${OUT_DIR}"

  --seed
  "${RANDOM_SEED}"

  --weight_col
  "${WEIGHT_COL}"

  --module_top_frac
  "${MODULE_TOP_FRAC}"

  --min_module_size
  "${MIN_MODULE_SIZE}"

  --resolution
  "${RESOLUTION}"

  --reassign_top_k
  "${REASSIGN_TOP_K}"

  --hub_top_frac
  "${HUB_TOP_FRAC}"

  --min_hubs
  "${MIN_HUBS}"

  --max_hubs
  "${MAX_HUBS}"
)


# ------------------------------------------------------------------------------
# Optional minimum absolute edge threshold
# ------------------------------------------------------------------------------

if [[ -n "${MIN_ABS_WEIGHT}" ]]; then
  CMD+=(
    --min_abs_weight
    "${MIN_ABS_WEIGHT}"
  )
fi


# ------------------------------------------------------------------------------
# Optional explicit community-detection fallback
# ------------------------------------------------------------------------------

if [[ "${ALLOW_COMMUNITY_FALLBACK}" == "1" ]]; then
  CMD+=(
    --allow_community_fallback
  )
fi


# ==============================================================================
# 8. Print exact command
# ==============================================================================

echo "Command:"
printf '  %q' "${CMD[@]}"
echo
echo


# ==============================================================================
# 9. Run downstream analysis
# ==============================================================================

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"


# ==============================================================================
# 10. Finish
# ==============================================================================

echo
echo "=============================================================================="
echo "Run04 downstream analysis finished successfully."
echo "=============================================================================="
echo
echo "Output directory:"
echo "  ${OUT_DIR}"
echo
echo "Main analysis outputs:"
echo "  ${OUT_DIR}/tables/module_assignment.csv"
echo "  ${OUT_DIR}/tables/module_summary.csv"
echo "  ${OUT_DIR}/tables/module_count_summary.csv"
echo "  ${OUT_DIR}/tables/unassigned_disconnected_genes.csv"
echo "  ${OUT_DIR}/tables/module_edges.parquet"
echo
echo "Conditional analysis outputs:"
echo "  ${OUT_DIR}/tables/small_module_assignment_audit.csv"
echo "      Written when small-community assignment records exist."
echo
echo "  ${OUT_DIR}/tables/hub_gene_summary.csv"
echo "      Written when Hub genes are identified."
echo
echo "Gene-set outputs:"
echo "  ${OUT_DIR}/gene_sets/module_genes/"
echo "  ${OUT_DIR}/gene_sets/module_hubs/"
echo
echo "Log file:"
echo "  ${LOG_FILE}"
echo
echo "Next pipeline stage:"
echo "  Run05 can use the Run04 module / Hub results for network visualization."
echo
echo "=============================================================================="