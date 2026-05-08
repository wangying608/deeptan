# Cell-DeepTAN Fine-tuning

> **A practical workflow for adapting an scRNA-pretrained DeepTAN model to user-defined cell-type scRNA datasets and extracting cell-specific latent gene networks.**  
> This repository provides a complete pipeline for **data formatting → cell-type fine-tuning → cell-specific network extraction**.

---

## 1. Overview

`Cell_DeepTAN_finetune` is organized as a user-facing workflow. Users provide their own cell-type scRNA expression files and an scRNA-pretrained DeepTAN checkpoint. The pipeline converts the data into DeepTAN-compatible LitData, fine-tunes the pretrained model in a reconstruction-focused manner, and extracts a cell-specific latent gene network.

```text
User-defined cell-type scRNA files
        + scRNA-pretrained graph resources
        + scRNA-pretrained DeepTAN checkpoint
                    │
                    ▼
Run01  Build Cell-Type DeepTAN LitData
                    │
                    ▼
Run02  Fine-tune DeepTAN on the user cell-type dataset
                    │
                    ▼
Run03  Extract a cell-specific network
```

This pipeline is designed for **cell-type-specific model adaptation and network-level biological interpretation**. In the default configuration, cell-type fine-tuning uses `recon_only: true`, meaning that the model focuses on gene-expression reconstruction rather than classification.

---

## 2. Repository structure

```text
Cell_DeepTAN_finetune/
├── configs/
│   ├── Run02_Cell-Type_finetune.yaml
│   └── Run03_extract_network.yaml
│
├── scripts/
│   ├── Run01_build_Cell-Type_LitData.sh
│   ├── Run02_Cell-Type_finetune.sh
│   └── Run03_extract_network.sh
│
├── src/
│   ├── Run01_build_Cell-Type_LitData.py
│   ├── Run02_Cell-Type_finetune.py
│   └── Run03_extract_network.py
│
└── README.md
```

### 2.1 File roles

| File | Purpose | Typical user action |
|---|---|---|
| `configs/Run02_Cell-Type_finetune.yaml` | Configuration for cell-type fine-tuning | Edit checkpoint, LitData, output, and training parameters |
| `configs/Run03_extract_network.yaml` | Configuration for cell-specific network extraction | Edit pretrained checkpoint, fine-tuned checkpoint, LitData, and output paths |
| `scripts/Run01_build_Cell-Type_LitData.sh` | Shell wrapper for data formatting | Edit user input paths and run |
| `scripts/Run02_Cell-Type_finetune.sh` | Shell wrapper for fine-tuning | Point to the Run02 config and run |
| `scripts/Run03_extract_network.sh` | Shell wrapper for network extraction | Point to the Run03 config and run |
| `src/Run01_build_Cell-Type_LitData.py` | Main LitData-construction program | Usually no modification needed |
| `src/Run02_Cell-Type_finetune.py` | Main cell-type fine-tuning program | Usually no modification needed |
| `src/Run03_extract_network.py` | Main cell-specific network-extraction program | Usually no modification needed |

---

## 3. Environment

Use an isolated Python environment and make sure the local DeepTAN source tree is available.

```bash
conda activate deeptan
cd Cell_DeepTAN_finetune
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

If DeepTAN cannot be imported, set the local source path in the shell wrappers:

```bash
DEEPTAN_SRC="/path/to/deeptan-dev/src"
```

or make sure DeepTAN is installed in the active Python environment.

---

## 4. Input files

The pipeline uses **user-defined file names and paths**. Users do not need to follow internal naming conventions such as `split_42_0.parquet`.

### 4.1 Required user files for Run01

| File type | Example name | Description |
|---|---|---|
| Training scRNA expression table | `cell_train.parquet` | Cells × genes expression table used for training |
| Validation scRNA expression table | `cell_valid.parquet` | Cells × genes expression table used for validation |
| Test scRNA expression table | `cell_test.parquet` | Cells × genes expression table used for final evaluation |
| Pretrained training graph file | `pretrained_trn.npz` | Pretraining-stage NMIC/graph file used to reuse the pretrained graph skeleton |
| Pretrained metadata file | `others2save.pkl` | Pretraining metadata containing the pretrained node vocabulary |
| Pretrained checkpoint | `best_model.ckpt` | scRNA-pretrained DeepTAN checkpoint used by Run02 and Run03 |

### 4.2 Optional label files

Cell-type fine-tuning usually runs with:

```yaml
recon_only: true
```

In this mode, classification is not the objective. Labels are therefore optional. If labels are available, Run01 can use one of the following:

| Label input mode | Description |
|---|---|
| `LABELS_PARQUET` | Existing one-hot label parquet |
| `CELLTYPE_COL` | A cell-type column already present in the expression parquet files |
| `CELLTYPE_CSV` | External annotation CSV containing cell IDs and labels |

### 4.3 Expected expression table format

The train/validation/test parquet files should contain:

```text
obs_names   gene_1   gene_2   ...   gene_N
cell_001    0.0      1.2            0.5
cell_002    2.1      0.0            3.3
...
```

The cell identifier column may be named, for example:

```text
obs_names
obs_name
barcode
cell_id
index
```

All gene expression columns should be numeric.

### 4.4 Recommended input organization

```text
my_cell_project/
├── input/
│   ├── cell_train.parquet
│   ├── cell_valid.parquet
│   ├── cell_test.parquet
│   └── celltype_onehot.parquet        # optional
│
├── pretrained/
│   ├── pretrained_trn.npz
│   ├── others2save.pkl
│   └── best_model.ckpt
│
├── litdata/
├── finetune_output/
└── cell_network/
```

This structure is recommended but not required. The scripts accept custom paths.

---

## 5. Run01: build Cell-Type DeepTAN LitData

Run01 converts user-provided scRNA expression files into DeepTAN-compatible LitData. It reuses the graph skeleton and node vocabulary from the scRNA-pretrained model resources.

### 5.1 Configure paths

Edit:

```bash
scripts/Run01_build_Cell-Type_LitData.sh
```

Typical fields to modify:

```bash
DEEPTAN_SRC="/path/to/deeptan-dev/src"

RUN01_SCRIPT="/path/to/Cell_DeepTAN_finetune/src/Run01_build_Cell-Type_LitData.py"

PRETRAINED_TRN_NPZ="/path/to/my_cell_project/pretrained/pretrained_trn.npz"
PRETRAINED_PKL="/path/to/my_cell_project/pretrained/others2save.pkl"

CELL_TYPE_NAME="ExampleCellType"

TRN_PARQUET="/path/to/my_cell_project/input/cell_train.parquet"
VAL_PARQUET="/path/to/my_cell_project/input/cell_valid.parquet"
TST_PARQUET="/path/to/my_cell_project/input/cell_test.parquet"

LABELS_PARQUET=""
CELLTYPE_COL=""
CELLTYPE_CSV=""

LITDATA_DIR="/path/to/my_cell_project/litdata"
```

### 5.2 Run

```bash
bash scripts/Run01_build_Cell-Type_LitData.sh
```

### 5.3 Expected output

```text
litdata/
├── trn/
├── val/
├── tst/
├── litdata_others2save.pkl
├── litdata_others2save.json
├── gene_cv_weights.csv
└── celltype_onehot.parquet      # if labels are provided or generated
```

| File / directory | Description |
|---|---|
| `trn/` | Training LitData |
| `val/` | Validation LitData |
| `tst/` | Test LitData |
| `litdata_others2save.pkl` | Main metadata file required by Run02 |
| `litdata_others2save.json` | Human-readable metadata summary |
| `gene_cv_weights.csv` | Gene-level CV weights used for reconstruction loss weighting |
| `celltype_onehot.parquet` | Optional one-hot cell-type label table |

Before training, confirm that `trn/`, `val/`, `tst/`, and `litdata_others2save.pkl` exist.

---

## 6. Run02: fine-tune DeepTAN on a cell-type dataset

Run02 adapts the scRNA-pretrained DeepTAN model to a user-defined cell-type dataset.

### 6.1 Configure fine-tuning

Edit:

```bash
configs/Run02_Cell-Type_finetune.yaml
```

A typical single-run configuration:

```yaml
pretrained_ckpt: "/path/to/my_cell_project/pretrained/best_model.ckpt"
nmic_npz: "data/NMIC.npz"

output_dir: "/path/to/my_cell_project/finetune_output"

recon_only: true

tissues:
  ExampleCellType:
    litdata: "/path/to/my_cell_project/litdata"

tissue_order:
  - ExampleCellType

base_lr: 3.0e-5

lr_multipliers:
  embed: 0.0
  feature_proj: 0.1
  fusion_mlp: 0.1
  gat_layers: 0.2
  pooling: 0.2
  ge_decoder: 0.3
  g_label_predictor: 0.0

weight_decay: 1.0e-5
warmup_epochs: 3
max_epochs: 99
min_epochs: 12
es_patience: 15
accumulate_grad_batches: 1
gradient_clip_val: 1.0

lambda_anchor: 0.02
focal_gamma: 2.0
loss_zero_coeff: 0.5

scheduler: "cosine_warm_restarts"
cosine_T0: 15
cosine_T_mult: 1
plateau_patience: 5
plateau_factor: 0.5
min_lr: 1.0e-7

accelerator: "gpu"
devices: 1
precision: "16-mixed"
n_workers: 8
batch_size: 32

seed: 42
```

### 6.2 Run

Using the wrapper:

```bash
bash scripts/Run02_Cell-Type_finetune.sh configs/Run02_Cell-Type_finetune.yaml
```

or directly:

```bash
python src/Run02_Cell-Type_finetune.py \
  --config configs/Run02_Cell-Type_finetune.yaml
```

Optional:

```bash
bash scripts/Run02_Cell-Type_finetune.sh configs/Run02_Cell-Type_finetune.yaml --skip_tsa
```

### 6.3 Expected output

A typical output directory is:

```text
finetune_output/
└── ExampleCellType/
    ├── checkpoints or *.ckpt
    ├── finetune_metadata.pkl
    ├── tensorboard/
    └── tissue_specificity/       # if TSA is enabled
```

The exact checkpoint name may include epoch and validation loss.

---

## 7. Cell-type fine-tuning objective

In the recommended cell-type workflow:

```yaml
recon_only: true
```

This means that fine-tuning focuses on gene-expression reconstruction:

```text
loss =
    reconstruction_loss
  + anchor_loss
```

The classification branch is retained for checkpoint compatibility but is not used as the primary objective.

| Term | Meaning |
|---|---|
| `reconstruction_loss` | CV-weighted reconstruction loss on observed gene expression values, with optional zero-expression penalty |
| `anchor_loss` | Regularizes trainable parameters toward the pretrained state |
| `gene_cv_weights.csv` | Gene-level weights computed from the Run01 training split |
| `g_label_predictor` | Classification head; frozen by default in recon-only mode |

### 7.1 Why recon-only?

For cell-type-specific fine-tuning, a dataset often contains one target cell type or one focused cell-group subset. In that setting, classification is usually not the meaningful endpoint. Reconstruction-focused adaptation is used to adjust the pretrained representation to the target cell-type expression distribution.

---

## 8. Key parameters

### `recon_only`

```yaml
recon_only: true
```

Recommended default for cell-type fine-tuning. It disables classification-loss optimization and avoids spending epochs on label-focused phases.

### `ge_decoder`

```yaml
lr_multipliers:
  ge_decoder: 0.3
```

Learning-rate multiplier for the reconstruction decoder. This is the main component adapted during reconstruction-focused fine-tuning.

### `g_label_predictor`

```yaml
lr_multipliers:
  g_label_predictor: 0.0
```

Recommended in recon-only mode. It keeps the classification head frozen.

### `lambda_anchor`

```yaml
lambda_anchor: 0.02
```

Anchors the fine-tuned model to the pretrained parameter state. This helps prevent destructive drift when adapting to a single cell type.

### `loss_zero_coeff`

```yaml
loss_zero_coeff: 0.5
```

Controls the penalty on predicted expression for zero or absent nodes. Use caution when increasing this value.

### `seed`

```yaml
seed: 42
```

Training random seed for reproducibility. It is not a data split identifier in the single-run workflow.

---

## 9. Run03: extract a cell-specific latent network

Run03 extracts a cell-specific latent gene network from the pretrained and fine-tuned models.

### 9.1 Configure

Edit:

```bash
configs/Run03_extract_network.yaml
```

Typical single-run configuration:

```yaml
pretrained_ckpt: "/path/to/my_cell_project/pretrained/best_model.ckpt"
finetuned_ckpt: "/path/to/my_cell_project/finetune_output/ExampleCellType/best_model.ckpt"

litdata_dir: "/path/to/my_cell_project/litdata"

cell_group_name: "ExampleCellType"
split_id: null

output_dir: "/path/to/my_cell_project/cell_network"
simple_output: true

finetune_module_dir: "/path/to/Cell_DeepTAN_finetune/src"
finetune_module_name: "Run02_Cell-Type_finetune"
finetune_class_name: "DeepTANFineTune"

accelerator: "gpu"
devices: 1
precision: "32-true"
batch_size: 32
n_workers: 8
```

### 9.2 Run

Using the wrapper:

```bash
bash scripts/Run03_extract_network.sh configs/Run03_extract_network.yaml
```

or directly:

```bash
python src/Run03_extract_network.py \
  --config configs/Run03_extract_network.yaml
```

Optional overrides:

```bash
bash scripts/Run03_extract_network.sh configs/Run03_extract_network.yaml \
  --litdata_dir /path/to/my_cell_project/litdata \
  --finetuned_ckpt /path/to/my_cell_project/finetune_output/ExampleCellType/best_model.ckpt \
  --output_dir /path/to/my_cell_project/cell_network
```

### 9.3 Expected output

With:

```yaml
simple_output: true
```

Run03 writes:

```text
cell_network/
├── csn_summary.yaml
└── csn/
    ├── csn_edge_table.parquet
    ├── csn_metadata.yaml
    ├── mean_embeddings_pretrained.npy
    ├── mean_embeddings_finetuned.npy
    ├── cell_embeddings_finetuned.npy
    ├── gene_id_map.csv
    └── raw_signals/
```

Common fields in `csn_edge_table.parquet`:

| Field | Meaning |
|---|---|
| `gene_i_id`, `gene_j_id` | Edge endpoint IDs |
| `gene_i_name`, `gene_j_name` | Edge endpoint gene names |
| `w_pre` | Latent cosine-similarity edge weight from the pretrained model |
| `w_ft` | Latent cosine-similarity edge weight from the fine-tuned model |
| `delta_w` | Fine-tuning-induced edge change: `w_ft - w_pre` |

Suggested usage:

| Goal | Recommended signal |
|---|---|
| Build the fine-tuned cell-specific network | `w_ft` |
| Identify cell-type adaptation effects | `delta_w` or `abs(delta_w)` |
| Prioritize cell-type-specific candidate genes | combine module membership, hub degree, `delta_w`, and biological prior knowledge |
| Visualize cell-level states | use `cell_embeddings_finetuned.npy` |

---

## 10. Reproducibility checklist

Before running:

- [ ] `DEEPTAN_SRC` points to the correct DeepTAN source tree, or DeepTAN is installed.
- [ ] `PRETRAINED_TRN_NPZ` and `PRETRAINED_PKL` are from the intended pretraining run.
- [ ] `pretrained_ckpt` points to the intended scRNA-pretrained checkpoint.
- [ ] User scRNA parquet files exist and contain expected cell ID and numeric gene columns.
- [ ] Output directories will not overwrite important previous results.

After Run01:

- [ ] Confirm that `trn/`, `val/`, `tst/`, and `litdata_others2save.pkl` exist.
- [ ] Check `gene_cv_weights.csv`.
- [ ] Check gene coverage warnings in the Run01 log.
- [ ] If labels are used, confirm `celltype_onehot.parquet`.

After Run02:

- [ ] Confirm that a fine-tuned `.ckpt` was created.
- [ ] Inspect `finetune_metadata.pkl`.
- [ ] Check TensorBoard curves for abnormal loss behavior.
- [ ] Verify that recon-only behavior is intended.

After Run03:

- [ ] Confirm that `csn/csn_edge_table.parquet` exists.
- [ ] Check that `w_pre`, `w_ft`, and `delta_w` are non-degenerate.
- [ ] Confirm that `csn/gene_id_map.csv` contains expected gene names.
- [ ] Confirm that `csn/cell_embeddings_finetuned.npy` exists if cell embedding output is needed.

---

## 11. FAQ

### Q1. Do my input files need to be named `split_42_0.parquet`?

No. The GitHub-facing workflow uses explicit user-defined paths. Your files can have any names as long as the script variables point to the correct files.

### Q2. What is the difference between `seed` and `split_id`?

`seed` is a random seed used for reproducibility during training. In the single-run workflow, it is not a data split identifier.

`split_id` is an optional run label for legacy or batch experiments. Leave it as `null` for normal single-run use.

### Q3. Why is `recon_only: true` recommended?

For cell-type-specific adaptation, classification is usually not the main endpoint. Reconstruction-focused fine-tuning adapts the pretrained representation to the target cell-type expression distribution.

### Q4. Do I need labels?

Not necessarily. In recon-only mode, labels are optional. If you provide labels, they can still be stored in LitData and used for metadata or optional analyses, but they are not required for the core reconstruction-focused fine-tuning objective.

### Q5. Which checkpoint should be used for Run03?

Use the best fine-tuned checkpoint generated by Run02. Prefer the validation-loss best checkpoint unless you have a specific model-selection rule.

### Q6. Run03 reports that GAT weights are identical. What does that mean?

Run03 checks whether the fine-tuned GAT weights differ from the pretrained weights. If they are identical, the checkpoint path may be wrong or the fine-tuned weights may not have loaded correctly. Verify `finetuned_ckpt`.

---

## 12. Minimal command summary

```bash
# Run01: build Cell-Type LitData
bash scripts/Run01_build_Cell-Type_LitData.sh

# Run02: fine-tune DeepTAN on the cell-type dataset
bash scripts/Run02_Cell-Type_finetune.sh configs/Run02_Cell-Type_finetune.yaml

# Run03: extract cell-specific latent network
bash scripts/Run03_extract_network.sh configs/Run03_extract_network.yaml
```

---

## 13. Questions and support

If you encounter issues, please open an issue with the following information:

1. the command you ran;
2. the relevant config file;
3. the error message or log excerpt;
4. the expected output directory structure;
5. the versions of Python, PyTorch, and DeepTAN source used.

Clear issue reports make debugging faster and help improve the pipeline for future users.
