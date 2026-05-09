# Tissue-DeepTAN Fine-tuning

> **A practical workflow for adapting an scRNA-pretrained DeepTAN model to user-defined tissue scRNA datasets and extracting tissue-specific latent gene networks.**  
> This repository provides a complete pipeline for **data formatting → tissue-specific fine-tuning → tissue-specific network extraction**.

---

## 1. Overview

`Tissue_DeepTAN_finetune` is organized as a user-facing workflow. Users provide their own tissue-level scRNA expression files, cell-type annotations, pretrained graph resources, and an scRNA-pretrained DeepTAN checkpoint. The pipeline converts the tissue scRNA data into DeepTAN-compatible LitData, fine-tunes the pretrained model on the target tissue, and extracts a tissue-specific latent gene network.

```text
User-defined tissue scRNA files
        + cell-type annotation
        + scRNA-pretrained graph resources
        + scRNA-pretrained DeepTAN checkpoint
                    │
                    ▼
Run01  Build Tissue_Type DeepTAN LitData
                    │
                    ▼
Run02  Fine-tune DeepTAN on the user tissue dataset
                    │
                    ▼
Run03  Extract a tissue-specific network
```

This pipeline is designed for **tissue-specific model adaptation and network-level biological interpretation**. Unlike the cell-type fine-tuning workflow, tissue-specific fine-tuning usually retains both **gene-expression reconstruction** and **within-tissue cell-type classification**, because each tissue dataset may contain multiple cell types.

---

## 2. Repository structure

```text
Tissue_DeepTAN_finetune/
├── configs/
│   ├── Run02_Tissue_Type_finetune.yaml
│   └── Run03_extract_network.yaml
│
├── scripts/
│   ├── Run01_build_Tissue_Type_LitData.sh
│   ├── Run02_Tissue_Type_finetune.sh
│   └── Run03_extract_network.sh
│
├── src/
│   ├── Run01_build_Tissue_Type_LitData.py
│   ├── Run02_Tissue_Type_finetune.py
│   └── Run03_extract_network.py
│
└── README.md
```

### 2.1 File roles

| File | Purpose | Typical user action |
|---|---|---|
| `configs/Run02_Tissue_Type_finetune.yaml` | Configuration for tissue-specific fine-tuning | Edit checkpoint, LitData, output, and training parameters |
| `configs/Run03_extract_network.yaml` | Configuration for tissue-specific network extraction | Edit pretrained checkpoint, fine-tuned checkpoint, LitData, and output paths |
| `scripts/Run01_build_Tissue_Type_LitData.sh` | Shell wrapper for data formatting | Edit user input paths and run |
| `scripts/Run02_Tissue_Type_finetune.sh` | Shell wrapper for fine-tuning | Point to the Run02 config and run |
| `scripts/Run03_extract_network.sh` | Shell wrapper for network extraction | Point to the Run03 config and run |
| `src/Run01_build_Tissue_Type_LitData.py` | Main LitData-construction program | Usually no modification needed |
| `src/Run02_Tissue_Type_finetune.py` | Main tissue-specific fine-tuning program | Usually no modification needed |
| `src/Run03_extract_network.py` | Main tissue-specific network-extraction program | Usually no modification needed |

---

## 3. Environment

Use an isolated Python environment and make sure the local DeepTAN source tree is available.

```bash
conda activate deeptan
cd Tissue_DeepTAN_finetune
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
| Training tissue scRNA expression table | `tissue_train.parquet` | Cells × genes expression table used for training |
| Validation tissue scRNA expression table | `tissue_valid.parquet` | Cells × genes expression table used for validation |
| Test tissue scRNA expression table | `tissue_test.parquet` | Cells × genes expression table used for final evaluation |
| Pretrained training graph file | `pretrained_trn.npz` | Pretraining-stage NMIC/graph file used to reuse the pretrained graph skeleton |
| Pretrained metadata file | `others2save.pkl` | Pretraining metadata containing the pretrained node vocabulary |
| Pretrained checkpoint | `best_model.ckpt` | scRNA-pretrained DeepTAN checkpoint used by Run02 and Run03 |

### 4.2 Cell-type label files

Tissue-specific fine-tuning normally uses cell-type labels, because the tissue dataset may contain multiple cell populations. Run01 supports one of the following label input modes:

| Label input mode | Description |
|---|---|
| `LABELS_PARQUET` | Existing one-hot label parquet |
| `CELLTYPE_COL` | A cell-type column already present in the expression parquet files |
| `CELLTYPE_CSV` | External annotation CSV containing cell IDs and labels |

If your tissue dataset contains only one cell population, classification may be less informative. In that case, check whether your local Run02 configuration should be adjusted for reconstruction-focused fine-tuning.

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

All gene expression columns should be numeric. If using `CELLTYPE_COL`, the annotation column should be present in the input parquet files and should not be interpreted as a gene column.

### 4.4 Recommended input organization

```text
my_tissue_project/
├── input/
│   ├── tissue_train.parquet
│   ├── tissue_valid.parquet
│   ├── tissue_test.parquet
│   └── celltype_onehot.parquet        # optional if using CELLTYPE_COL or CELLTYPE_CSV
│
├── pretrained/
│   ├── pretrained_trn.npz
│   ├── others2save.pkl
│   └── best_model.ckpt
│
├── litdata/
├── finetune_output/
└── tissue_network/
```

This structure is recommended but not required. The scripts accept custom paths.

---

## 5. Run01: build Tissue_Type DeepTAN LitData

Run01 converts user-provided tissue scRNA expression files into DeepTAN-compatible LitData. It reuses the graph skeleton and node vocabulary from the scRNA-pretrained model resources.

### 5.1 Configure paths

Edit:

```bash
scripts/Run01_build_Tissue_Type_LitData.sh
```

Typical fields to modify:

```bash
DEEPTAN_SRC="/path/to/deeptan-dev/src"

RUN01_SCRIPT="/path/to/Tissue_DeepTAN_finetune/src/Run01_build_Tissue_Type_LitData.py"

PRETRAINED_TRN_NPZ="/path/to/my_tissue_project/pretrained/pretrained_trn.npz"
PRETRAINED_PKL="/path/to/my_tissue_project/pretrained/others2save.pkl"

TISSUE_NAME="ExampleTissue"

TRN_PARQUET="/path/to/my_tissue_project/input/tissue_train.parquet"
VAL_PARQUET="/path/to/my_tissue_project/input/tissue_valid.parquet"
TST_PARQUET="/path/to/my_tissue_project/input/tissue_test.parquet"

LABELS_PARQUET="/path/to/my_tissue_project/input/celltype_onehot.parquet"
CELLTYPE_COL=""
CELLTYPE_CSV=""

LITDATA_DIR="/path/to/my_tissue_project/litdata"
```

Use only one label source unless you intentionally want to override the default behavior.

### 5.2 Run

```bash
bash scripts/Run01_build_Tissue_Type_LitData.sh
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
└── celltype_onehot.parquet
```

| File / directory | Description |
|---|---|
| `trn/` | Training LitData |
| `val/` | Validation LitData |
| `tst/` | Test LitData |
| `litdata_others2save.pkl` | Main metadata file required by Run02 |
| `litdata_others2save.json` | Human-readable metadata summary |
| `gene_cv_weights.csv` | Gene-level CV weights used for reconstruction loss weighting |
| `celltype_onehot.parquet` | One-hot cell-type label table |

Before training, confirm that `trn/`, `val/`, `tst/`, `litdata_others2save.pkl`, `gene_cv_weights.csv`, and `celltype_onehot.parquet` exist.

---

## 6. Run02: fine-tune DeepTAN on a tissue dataset

Run02 adapts the scRNA-pretrained DeepTAN model to a user-defined tissue dataset.

### 6.1 Configure fine-tuning

Edit:

```bash
configs/Run02_Tissue_Type_finetune.yaml
```

A typical single-run configuration:

```yaml
pretrained_ckpt: "/path/to/my_tissue_project/pretrained/best_model.ckpt"
nmic_npz: "data/NMIC.npz"

output_dir: "/path/to/my_tissue_project/finetune_output"

pretrained_labels_parquet: ""

tissues:
  ExampleTissue:
    litdata: "/path/to/my_tissue_project/litdata"

tissue_order:
  - ExampleTissue

base_lr: 3.0e-5

lr_multipliers:
  embed: 0.0
  feature_proj: 0.1
  fusion_mlp: 0.1
  gat_layers: 0.2
  pooling: 0.2
  ge_decoder: 0.3
  g_label_predictor: 1.0

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
bash scripts/Run02_Tissue_Type_finetune.sh configs/Run02_Tissue_Type_finetune.yaml
```

or directly:

```bash
python src/Run02_Tissue_Type_finetune.py \
  --config configs/Run02_Tissue_Type_finetune.yaml
```

Optional:

```bash
bash scripts/Run02_Tissue_Type_finetune.sh configs/Run02_Tissue_Type_finetune.yaml --skip_tsa
```

### 6.3 Expected output

A typical output directory is:

```text
finetune_output/
└── ExampleTissue/
    ├── checkpoints or *.ckpt
    ├── finetune_metadata.pkl
    ├── tensorboard/
    └── tissue_specificity/       # if TSA is enabled
```

The exact checkpoint name may include epoch and validation loss.

---

## 7. Tissue-specific fine-tuning objective

Tissue-specific fine-tuning usually combines reconstruction, cell-type classification, and parameter anchoring:

```text
loss =
    reconstruction_loss
  + label_classification_loss
  + anchor_loss
```

| Term | Meaning |
|---|---|
| `reconstruction_loss` | CV-weighted reconstruction loss on observed gene expression values, with optional zero-expression penalty |
| `label_classification_loss` | Cell-type classification loss within the tissue dataset |
| `anchor_loss` | Regularizes trainable parameters toward the pretrained state |
| `gene_cv_weights.csv` | Gene-level weights computed from the Run01 training split |
| `celltype_onehot.parquet` | One-hot label table used for tissue cell-type classification |

The default training logic uses epoch-level task balancing inherited from the original tissue fine-tuning script, including reconstruction-focused, label-focused, and joint phases.

---

## 8. Key parameters

### `g_label_predictor`

```yaml
lr_multipliers:
  g_label_predictor: 1.0
```

Learning-rate multiplier for the classification head. In tissue-specific fine-tuning, this is usually trainable because each tissue can contain multiple cell types.

### `ge_decoder`

```yaml
lr_multipliers:
  ge_decoder: 0.3
```

Learning-rate multiplier for the reconstruction decoder.

### `lambda_anchor`

```yaml
lambda_anchor: 0.02
```

Anchors the fine-tuned model to the pretrained parameter state. This helps reduce destructive drift while adapting to a specific tissue.

### `loss_zero_coeff`

```yaml
loss_zero_coeff: 0.5
```

Controls the penalty on predicted expression for zero or absent nodes. Use caution when increasing this value.

### `pretrained_labels_parquet`

```yaml
pretrained_labels_parquet: ""
```

Optional. If provided, it should point to the pretraining-stage `celltype_onehot.parquet`. This allows optional Tissue Specificity Analysis baseline evaluation to match classifier rows by class name.

### `seed`

```yaml
seed: 42
```

Training random seed for reproducibility. It is not a data split identifier in the single-run workflow.

---

## 9. Run03: extract a tissue-specific latent network

Run03 extracts a tissue-specific latent gene network from the pretrained and fine-tuned models.

### 9.1 Configure

Edit:

```bash
configs/Run03_extract_network.yaml
```

Typical single-run configuration:

```yaml
pretrained_ckpt: "/path/to/my_tissue_project/pretrained/best_model.ckpt"
finetuned_ckpt: "/path/to/my_tissue_project/finetune_output/ExampleTissue/best_model.ckpt"

litdata_dir: "/path/to/my_tissue_project/litdata"

tissue_name: "ExampleTissue"
split_id: null

output_dir: "/path/to/my_tissue_project/tissue_network"
simple_output: true

finetune_module_dir: "/path/to/Tissue_DeepTAN_finetune/src"
finetune_module_name: "Run02_Tissue_Type_finetune"
finetune_class_name: "DeepTANFineTune"

accelerator: "gpu"
devices: 1
precision: "32-true"
batch_size: 32
n_workers: 8
```

If your Run03 script dynamically imports the Run02 module, make sure `finetune_module_name` matches the importable Python module name used in your project.

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
  --litdata_dir /path/to/my_tissue_project/litdata \
  --finetuned_ckpt /path/to/my_tissue_project/finetune_output/ExampleTissue/best_model.ckpt \
  --output_dir /path/to/my_tissue_project/tissue_network
```

### 9.3 Expected output

With:

```yaml
simple_output: true
```

Run03 writes:

```text
tissue_network/
├── detsn_summary.yaml
└── detsn/
    ├── detsn_edge_table.parquet
    ├── detsn_metadata.yaml
    ├── mean_embeddings_pretrained.npy
    ├── mean_embeddings_finetuned.npy
    ├── gene_id_map.csv
    └── raw_signals/
```

Common fields in `detsn_edge_table.parquet`:

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
| Build the fine-tuned tissue-specific network | `w_ft` |
| Identify tissue-adaptation effects | `delta_w` or `abs(delta_w)` |
| Prioritize tissue-specific candidate genes | combine module membership, hub degree, `delta_w`, and biological prior knowledge |
| Compare tissue-specific regulatory structure | compare `w_ft`, `delta_w`, and module-level properties across tissues |

---

## 10. Reproducibility checklist

Before running:

- [ ] `DEEPTAN_SRC` points to the correct DeepTAN source tree, or DeepTAN is installed.
- [ ] `PRETRAINED_TRN_NPZ` and `PRETRAINED_PKL` are from the intended pretraining run.
- [ ] `pretrained_ckpt` points to the intended scRNA-pretrained checkpoint.
- [ ] User tissue scRNA parquet files exist and contain expected cell ID and numeric gene columns.
- [ ] A valid label source is provided if classification is intended.
- [ ] Output directories will not overwrite important previous results.

After Run01:

- [ ] Confirm that `trn/`, `val/`, `tst/`, and `litdata_others2save.pkl` exist.
- [ ] Confirm that `celltype_onehot.parquet` exists when classification is intended.
- [ ] Check `gene_cv_weights.csv`.
- [ ] Check gene coverage warnings in the Run01 log.

After Run02:

- [ ] Confirm that a fine-tuned `.ckpt` was created.
- [ ] Inspect `finetune_metadata.pkl`.
- [ ] Check TensorBoard curves for abnormal loss behavior.
- [ ] If TSA is enabled, inspect the `tissue_specificity/` outputs.

After Run03:

- [ ] Confirm that `detsn/detsn_edge_table.parquet` exists.
- [ ] Check that `w_pre`, `w_ft`, and `delta_w` are non-degenerate.
- [ ] Confirm that `detsn/gene_id_map.csv` contains expected gene names.
- [ ] Verify that the fine-tuned checkpoint is the intended one.

---

## 11. FAQ

### Q1. Do my input files need to be named `split_42_0.parquet`?

No. The GitHub-facing workflow uses explicit user-defined paths. Your files can have any names as long as the script variables point to the correct files.

### Q2. What is the difference between `seed` and `split_id`?

`seed` is a random seed used for reproducibility during training. In the single-run workflow, it is not a data split identifier.

`split_id` is an optional run label for legacy or batch experiments. Leave it as `null` for normal single-run use.

### Q3. Do I need cell-type labels?

Usually yes for tissue-specific fine-tuning, because the tissue dataset may include multiple cell types and Run02 can use classification as part of the training objective. If your tissue dataset has only one cell type, classification may not be meaningful and the training objective should be checked accordingly.

### Q4. Which checkpoint should be used for Run03?

Use the best fine-tuned checkpoint generated by Run02. Prefer the validation-loss best checkpoint unless you have a specific model-selection rule.

### Q5. Run03 reports that GAT weights are identical. What does that mean?

Run03 checks whether the fine-tuned GAT weights differ from the pretrained weights. If they are identical, the checkpoint path may be wrong or the fine-tuned weights may not have loaded correctly. Verify `finetuned_ckpt`.

### Q6. Should I use `w_ft` or `delta_w` for downstream analysis?

Use `w_ft` when you want the fine-tuned tissue-specific network itself. Use `delta_w` when you want to focus on fine-tuning-induced changes relative to the pretrained baseline. For candidate gene prioritization, it is often useful to combine both with module membership and biological prior knowledge.

---

## 12. Minimal command summary

```bash
# Run01: build Tissue_Type LitData
bash scripts/Run01_build_Tissue_Type_LitData.sh

# Run02: fine-tune DeepTAN on the tissue dataset
bash scripts/Run02_Tissue_Type_finetune.sh configs/Run02_Tissue_Type_finetune.yaml

# Run03: extract tissue-specific latent network
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
