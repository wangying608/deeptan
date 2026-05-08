# Bulk-DeepTAN Fine-tuning

> **A practical workflow for adapting scRNA-pretrained DeepTAN to user-defined bulk omics datasets and extracting trait-aware gene networks.**  
> This repository provides a complete pipeline for **data formatting → model fine-tuning → network extraction → downstream analysis**.

---

## 1. Overview
`Bulk_DeepTAN_finetune` is organized as a user-facing workflow. Users provide their own bulk omics files, phenotype table, guide graph, and pretrained DeepTAN checkpoint. The pipeline then produces a fine-tuned model and a trait-aware network for downstream interpretation.

```text
User-defined bulk omics files
        + phenotype table
        + NMIC guide graph
        + scRNA-pretrained DeepTAN checkpoint
                    │
                    ▼
Run01  Build Bulk-DeepTAN LitData
                    │
                    ▼
Run02  Fine-tune DeepTAN on the user bulk dataset
                    │
                    ▼
Run03  extract a trait-aware latent network
                    │
                    ▼
Run04  Partition the network into co-functional modules and identify hub genes
```

The pipeline is intended for studies where prediction is not the endpoint. Its main purpose is to support **trait-associated gene prioritization and network-level biological interpretation**.

---

## 2. Repository structure

```text
Bulk_DeepTAN_finetune/
├── configs/
│   ├── Run02_Bulk-Type_finetune.yaml
│   └── Run03_extract_trait_network.yaml
│
├── scripts/
│   ├── Run01_build_Bulk-DeepTAN_LitData.sh
│   ├── Run02_Bulk-Type_finetune.sh
│   ├── Run03_extract_trait_network.sh
│   └── Run04_downstream.sh
│
├── src/
│   ├── Run01_build_Bulk-DeepTAN_LitData.py
│   ├── Run02_Bulk-Type_finetune.py
│   ├── Run03_extract_trait_network.py
│   └── Run04_downstream.py
│
└── README.md
```

### 2.1 File roles

| File | Purpose | Typical user action |
|---|---|---|
| `configs/Run02_Bulk-Type_finetune.yaml` | Configuration for bulk fine-tuning | Edit model paths, data paths, and training parameters |
| `configs/Run03_extract_trait_network.yaml` | Configuration for network extraction | Edit checkpoint, LitData, and output paths |
| `scripts/Run01_build_Bulk-DeepTAN_LitData.sh` | Shell wrapper for data formatting | Edit user file paths and run |
| `scripts/Run02_Bulk-Type_finetune.sh` | Shell wrapper for fine-tuning | Point to the Run02 config and run |
| `scripts/Run03_extract_trait_network.sh` | Shell wrapper for network extraction | Point to the Run03 config and run |
| `scripts/Run04_downstream.sh` | Shell wrapper for downstream analysis | Edit network input and output paths |
| `src/Run01_build_Bulk-DeepTAN_LitData.py` | Main data-construction program | Usually no modification needed |
| `src/Run02_Bulk-Type_finetune.py` | Main fine-tuning program | Usually no modification needed |
| `src/Run03_extract_trait_network.py` | Main network-extraction program | Usually no modification needed |
| `src/Run04_downstream.py` | Main downstream-analysis program | Usually no modification needed |

---

## 3. Environment

Use an isolated Python environment and make sure the local DeepTAN source tree is available.

```bash
conda activate deeptan
cd Bulk_DeepTAN_finetune
```

Check installation:

```bash
python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)

import deeptan
print("DeepTAN import: OK")
PY
```

If DeepTAN cannot be imported, check the source path in your YAML configuration:

```yaml
local_deeptan_src: "/path/to/deeptan-dev/src"
```

---

## 4. Input files

The pipeline is designed around **user-defined file names and paths**. The examples below use descriptive names only; users may replace them with their own file names.

### 4.1 Required user files

| File type | Example name | Description |
|---|---|---|
| Training bulk matrix | `bulk_train.parquet` | Samples × genes/features table used for training |
| Validation bulk matrix | `bulk_valid.parquet` | Samples × genes/features table used for validation |
| Test bulk matrix | `bulk_test.parquet` | Samples × genes/features table used for final evaluation |
| Bulk NMIC graph | `bulk_train_nmic.npz` | Guide graph inferred from the training bulk matrix |
| NMIC companion table | `bulk_train_nmic_features.parquet` | Metadata table containing sample and feature names corresponding to the NMIC file |
| Phenotype table | `phenotype_table.parquet` | Continuous trait labels matched to user samples |
| Pretrained checkpoint | `best_model.ckpt` | scRNA-pretrained DeepTAN checkpoint |
| Optional pretrained metadata | `pretrained_metadata.pkl` | Node vocabulary metadata, if required by your Run01 setup |

### 4.2 Expected columns

Bulk matrix files should contain:

```text
sample_id   feature_1   feature_2   ...   feature_N
```

Phenotype table should contain at least:

```text
sample_id   trait_value
```

The actual column names are configurable. Keep them consistent across Run01 and Run02.

### 4.3 Recommended input organization

A clear project-specific structure makes downstream debugging easier:

```text
my_bulk_project/
├── input/
│   ├── bulk_train.parquet
│   ├── bulk_valid.parquet
│   ├── bulk_test.parquet
│   ├── bulk_train_nmic.npz
│   ├── bulk_train_nmic_features.parquet
│   └── phenotype_table.parquet
│
├── pretrained/
│   └── best_model.ckpt
│
├── litdata/
├── finetune_output/
├── trait_network/
└── downstream/
```

This structure is not mandatory. The scripts accept custom paths.

---

## 5. Run01: build Bulk-DeepTAN LitData

Run01 converts user files into the graph-structured LitData format consumed by DeepTAN.

### 5.1 Configure paths

Edit:

```bash
scripts/Run01_build_Bulk-DeepTAN_LitData.sh
```

Point the script to your input files, for example:

```bash
BULK_TRAIN=/path/to/my_bulk_project/input/bulk_train.parquet
BULK_VALID=/path/to/my_bulk_project/input/bulk_valid.parquet
BULK_TEST=/path/to/my_bulk_project/input/bulk_test.parquet

BULK_NMIC=/path/to/my_bulk_project/input/bulk_train_nmic.npz
BULK_NMIC_PARQUET=/path/to/my_bulk_project/input/bulk_train_nmic_features.parquet

PHENOTYPE=/path/to/my_bulk_project/input/phenotype_table.parquet
PRETRAINED_CKPT=/path/to/my_bulk_project/pretrained/best_model.ckpt

OUTPUT_DIR=/path/to/my_bulk_project/litdata
```

### 5.2 Run

```bash
bash scripts/Run01_build_Bulk-DeepTAN_LitData.sh
```

### 5.3 Expected output

```text
litdata/
├── expanded_metadata.pkl
├── expanded_metadata.json
├── feature_mapping.parquet
├── edge_source.parquet
├── edge_source_undirected.parquet
├── ecotype_ft16_labels.parquet
├── expanded_vocabulary.parquet
├── trn/
├── val/
└── tst/
```

| File / directory | Description |
|---|---|
| `expanded_metadata.pkl` | Main metadata file required by Run02 |
| `expanded_metadata.json` | Human-readable metadata summary |
| `feature_mapping.parquet` | Mapping between bulk features and expanded node indices |
| `edge_source.parquet` | Edge table for the bulk-expanded guide graph |
| `edge_source_undirected.parquet` | Canonical undirected edge table |
| `ecotype_ft16_labels.parquet` | Matched continuous trait labels used by the LitData samples |
| `expanded_vocabulary.parquet` | Expanded node vocabulary with old/new node annotations |
| `trn/` | Training LitData |
| `val/` | Validation LitData |
| `tst/` | Test LitData |

Before training, verify that `expanded_metadata.pkl`, `trn/`, `val/`, and `tst/` exist. It is also recommended to inspect `feature_mapping.parquet`, `edge_source.parquet`, and `ecotype_ft16_labels.parquet` to confirm that feature mapping, graph construction, and trait-label matching are consistent with the intended dataset.

---

## 6. Run02: fine-tune DeepTAN on bulk data

Run02 adapts the scRNA-pretrained DeepTAN model to the user bulk dataset.

### 6.1 Configure fine-tuning

Edit:

```bash
configs/Run02_Bulk-Type_finetune.yaml
```

A typical configuration:

```yaml
local_deeptan_src: "/path/to/deeptan-dev/src"
pretrained_ckpt: "/path/to/my_bulk_project/pretrained/best_model.ckpt"

dataset_root: "/path/to/my_bulk_project/litdata"
output_dir: "/path/to/my_bulk_project/finetune_output"
run_name: "my_bulk_trait_finetune"

# Random seed for training reproducibility.
seed: 42

accelerator: "gpu"
devices: 1
precision: "16-mixed"

batch_size: 32
n_workers: 12

lr_multipliers:
  old_embedding_effective: 0.03
  new_embedding_effective: 1.0
  feature_proj: 0.10
  fusion_mlp: 0.10
  gat_layers: 0.20
  pooling: 0.20
  ge_decoder: 1.00
  trait_predictor: 1.00

loss:
  lambda_reg: 1.0
  lambda_recon: 0.70
  lambda_anchor: 0.02
  lambda_old_embedding_anchor: 0.02
  lambda_new_smoothness: 0.01
  use_zero_penalty: false
  lambda_zero: 0.0

task_balance:
  strategy: "cycle"
  warmup_recon_epochs: 3
  cycle: ["recon", "regress", "joint"]
```

If your version of the script supports processing multiple named runs, define them in the corresponding wrapper or config according to your local implementation. For the standard single-project layout above, `dataset_root` should point directly to the LitData directory generated by Run01.

### 6.2 Run

Using the wrapper:

```bash
bash scripts/Run02_Bulk-Type_finetune.sh
```

or directly:

```bash
python src/Run02_Bulk-Type_finetune.py \
  --config configs/Run02_Bulk-Type_finetune.yaml
```

### 6.3 Expected output

```text
finetune_output/
├── FT16_epoch=XX_val_loss=YYYY.ckpt
├── finetune_metadata.pkl
├── test_label_regression_predictions.csv
├── test_label_regression_metrics.json
├── test_recon_metrics.json
├── test_reconstruction_metrics.json
├── test_metrics.json
└── tensorboard/
```

| File | Description |
|---|---|
| `.ckpt` | Fine-tuned model checkpoint |
| `finetune_metadata.pkl` | Training config, best checkpoint path, source data paths, and run metadata |
| `test_label_regression_predictions.csv` | Per-sample observed and predicted trait values |
| `test_label_regression_metrics.json` | Trait-regression metrics |
| `test_recon_metrics.json` | Feature-reconstruction metrics |
| `test_metrics.json` | Combined metric file for both tasks |
| `tensorboard/` | Lightning training logs |

---

## 7. Fine-tuning objective

Run02 expands the pretrained model to the bulk feature space and fine-tunes it using trait regression and observed-feature reconstruction.

```text
loss =
    λ_reg   × trait_regression_MSE
  + λ_recon × observed_feature_reconstruction_MSE
  + λ_anchor × pretrained_parameter_anchor
  + λ_old_embedding_anchor × old_embedding_anchor
  + λ_new_smoothness × new_embedding_smoothness
  + λ_zero × zero_penalty
```

| Term | Meaning |
|---|---|
| `trait_regression_MSE` | Graph-level regression loss for the target quantitative trait |
| `observed_feature_reconstruction_MSE` | Reconstruction loss for observed nodes in each bulk sample graph |
| `pretrained_parameter_anchor` | Regularizes model parameters toward the pretrained state |
| `old_embedding_anchor` | Protects embeddings copied from the scRNA-pretrained model |
| `new_embedding_smoothness` | Stabilizes newly introduced bulk-feature embeddings |
| `zero_penalty` | Optional penalty that pushes absent/zero-node predictions toward zero |

The default schedule is:

```text
recon → regress → joint
```

This gives the model separate reconstruction-focused, trait-focused, and joint-optimization phases.

---

## 8. Key parameters

### `lambda_reg`

```yaml
lambda_reg: 1.0
```

Weight of the trait-regression objective. Usually keep this at `1.0`.

### `lambda_recon`

```yaml
lambda_recon: 0.70
```

Weight of the feature-reconstruction objective. This is usually the main knob for balancing trait prediction and feature reconstruction.

Recommended search path:

```text
0.50 → 0.60 → 0.70 → 0.80
```

Use small steps. If reconstruction improves but trait metrics degrade, return to the previous value.

### `ge_decoder`

```yaml
lr_multipliers:
  ge_decoder: 1.00
```

Learning-rate multiplier for the reconstruction decoder. For bulk fine-tuning, values around `0.8–1.0` are recommended.

### `task_balance`

```yaml
task_balance:
  strategy: "cycle"
  warmup_recon_epochs: 3
  cycle: ["recon", "regress", "joint"]
```

The default cycle is recommended for the first experiments. A reconstruction-heavy cycle can be tested later:

```yaml
cycle: ["recon", "recon", "regress", "joint"]
```

but it may reduce trait-regression performance.

### `use_zero_penalty`

```yaml
use_zero_penalty: false
lambda_zero: 0.0
```

Keep this disabled by default. In large expanded vocabularies, penalizing all absent/zero nodes can dominate the loss and make predictions overly conservative.

---

## 9. Practical tuning strategy

Start with:

```yaml
lr_multipliers:
  ge_decoder: 1.00

loss:
  lambda_reg: 1.0
  lambda_recon: 0.50
  use_zero_penalty: false
```

If trait metrics remain stable and reconstruction is still weak, gradually test:

```text
lambda_recon = 0.60
lambda_recon = 0.70
lambda_recon = 0.80
```

Do not change `lambda_recon`, `warmup_recon_epochs`, `cycle`, and `use_zero_penalty` all at once. One-factor-at-a-time changes make results easier to interpret.

---

## 10. Evaluation metrics

Run02 reports two task groups.

### 10.1 Trait regression

```text
label_regression_MSE
label_regression_MAE
label_regression_PCC
label_regression_RMSE
```

### 10.2 Feature reconstruction

```text
recon_MSE
recon_MAE
recon_PCC
recon_RMSE
```

For model comparison, report both tasks. If comparing with a from-scratch baseline, use the same input files, phenotype labels, and evaluation protocol whenever possible.

---

## 11. Run03: extract trait-aware network

Run03 converts the fine-tuned model into a latent gene/feature association network.

### 11.1 Configure

Edit:

```bash
configs/Run03_extract_trait_network.yaml
```

Example:

```yaml
pretrained_ckpt: "/path/to/my_bulk_project/pretrained/best_model.ckpt"
finetuned_ckpt: "/path/to/my_bulk_project/finetune_output/best_model.ckpt"
litdata_dir: "/path/to/my_bulk_project/litdata"
output_dir: "/path/to/my_bulk_project/trait_network"
```

### 11.2 Run

```bash
bash scripts/Run03_extract_trait_network.sh
```

or:

```bash
python src/Run03_extract_trait_network.py \
  --config configs/Run03_extract_trait_network.yaml
```

### 11.3 Output

```text
trait_network/
├── bulk_trait_edge_table.parquet
├── gene_node_summary.csv
├── bulk_trait_metadata.yaml
├── mean_embeddings_expanded_pretrained.npy
├── mean_embeddings_finetuned.npy
└── raw_signals/
```

Common fields in `bulk_trait_edge_table.parquet`:

| Field | Meaning |
|---|---|
| `gene_i_name`, `gene_j_name` | Edge endpoints |
| `w_pre` | Latent association before bulk fine-tuning |
| `w_ft` | Latent association after bulk fine-tuning |
| `delta_w` | Fine-tuning-induced edge-weight change |
| `abs_delta_w` | Absolute edge-weight change |
| `gene_i_type`, `gene_j_type` | Node type annotation, such as old/new |
| `edge_seen_min` | Edge support or coverage-related statistic |
| `gene_count_min_ft` | Node coverage or occurrence-related statistic |

Suggested usage:

| Goal | Recommended edge signal |
|---|---|
| Build the fine-tuned trait-aware network | `w_ft` |
| Identify fine-tuning-induced network changes | `delta_w` or `abs_delta_w` |
| Run robust module analysis | thresholded `abs(w_ft)` or `abs(delta_w)` |
| Prioritize candidate genes | combine edge strength, module membership, hub degree, and prior biological knowledge |

---

## 12. Run04: downstream analysis

Run04 converts the extracted network into interpretable tables and gene sets.

### 12.1 Run

Edit:

```bash
scripts/Run04_downstream.sh
```

Run:

```bash
bash scripts/Run04_downstream.sh
```

### 12.2 Output

```text
downstream/
├── tables/
│   ├── module_summary.csv
│   ├── node_table.csv
│   ├── hub_table.csv
│   └── edge_table_filtered.csv
├── gene_sets/
│   ├── module_*.txt
│   └── hubs_*.txt
└── logs/
```

Recommended checks:

- Module sizes should be reasonable.
- Hub genes should not be dominated by obvious artifacts.
- Exported gene sets should contain enough genes for enrichment analysis.
- Network results should be interpreted together with trait relevance and biological prior knowledge.

---

## 13. Reproducibility checklist

Before running:

- [ ] `local_deeptan_src` points to the correct DeepTAN source tree.
- [ ] `pretrained_ckpt` points to the intended checkpoint.
- [ ] User input files exist and have expected columns.
- [ ] `dataset_root` points to the LitData generated by Run01.
- [ ] `output_dir` will not overwrite important previous results.
- [ ] Key hyperparameters are reflected in the output directory name.

After Run01:

- [ ] Confirm that `expanded_metadata.pkl` exists.
- [ ] Confirm that `trn/`, `val/`, and `tst/` exist.
- [ ] Check feature mapping in `feature_mapping.parquet`.
- [ ] Check guide-graph edges in `edge_source.parquet`.
- [ ] Check trait-label matching in `ecotype_ft16_labels.parquet`.
- [ ] Check old/new node annotations in `expanded_vocabulary.parquet`.

After Run02:

- [ ] Inspect `test_metrics.json`.
- [ ] Confirm that both trait-regression and reconstruction metrics are present.
- [ ] Verify the selected checkpoint path in `finetune_metadata.pkl`.
- [ ] Check TensorBoard curves for abnormal loss behavior.

After Run03:

- [ ] Confirm that `bulk_trait_edge_table.parquet` exists.
- [ ] Check that `w_ft`, `w_pre`, and `delta_w` are non-degenerate.
- [ ] Confirm that node names are correctly decoded.
- [ ] Verify that the checkpoint used for extraction is the intended checkpoint.

After Run04:

- [ ] Check module count and module sizes.
- [ ] Inspect top hub genes.
- [ ] Confirm gene-set export.
- [ ] Record the edge-weight column used for downstream analysis.

---

## 14. FAQ

### Q1. Run02 cannot find `expanded_metadata.pkl`.

Check whether `dataset_root` points to the LitData directory generated by Run01. The directory should contain:

```text
expanded_metadata.pkl
trn/
val/
tst/
```

### Q2. Trait prediction improves but reconstruction is weak.

This is a common multi-task trade-off. Start by increasing decoder adaptation:

```yaml
lr_multipliers:
  ge_decoder: 1.00

loss:
  lambda_recon: 0.50
```

Then test `0.60`, `0.70`, and `0.80` if trait metrics remain stable.

### Q3. Should I enable `use_zero_penalty`?

Usually no. Keep it disabled unless full-matrix zero reconstruction is explicitly required.

### Q4. Should I use a reconstruction-heavy cycle?

Not as the first step. First tune `lambda_recon`. If reconstruction remains weak and trait performance is stable, then test:

```yaml
cycle: ["recon", "recon", "regress", "joint"]
```

### Q5. Which checkpoint should be used for network extraction?

Use the checkpoint that matches the downstream purpose. For balanced model performance, use the validation-loss best checkpoint. If the analysis is trait-first, a trait-metric-selected checkpoint can also be used, but record the selection rule.

---

## 15. Recommended experiment naming

Use output names that encode the key settings:

```text
output_lrecon050_decoder100
output_lrecon060_decoder100
output_lrecon070_decoder100
output_lrecon080_decoder100
output_cycle_rrgj_lrecon070
```

Useful fields to record:

```text
trait name
dataset name
pretrained checkpoint ID
lambda_recon
decoder learning-rate multiplier
task cycle
zero-penalty setting
```

---

## 16. Minimal command summary

```bash
# Run01: build LitData
bash scripts/Run01_build_Bulk-DeepTAN_LitData.sh

# Run02: fine-tune DeepTAN on bulk data
python src/Run02_Bulk-Type_finetune.py \
  --config configs/Run02_Bulk-Type_finetune.yaml

# Run03: extract trait-aware latent network
python src/Run03_extract_trait_network.py \
  --config configs/Run03_extract_trait_network.yaml

# Run04: downstream module and hub analysis
bash scripts/Run04_downstream.sh
```

---

## 17. Questions and support

If you encounter issues, please open an issue with the following information:

1. the command you ran;
2. the relevant config file;
3. the error message or log excerpt;
4. the expected output directory structure;
5. the versions of Python, PyTorch, and DeepTAN source used.

Clear issue reports make debugging much faster and help improve the pipeline for future users.
