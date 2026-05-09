# Data description

This directory contains datasets used for DeepTAN pretraining and fine-tuning experiments. The datasets are organized into four folders:

```bash
data/
├── snRNA/
├── tissue_finetune/
├── cell_finetune/
└── bulk/
```

## 1. `snRNA/`

```bash
snRNA/
└── ath_pretrain.h5ad  # Arabidopsis thaliana single-nucleus transcriptome dataset used for DeepTAN pretraining.
```

- This dataset contains 89,654 cell samples.
- The pretrained model learned from this dataset can be further fine-tuned for downstream biological state-specific network construction.

## 2. `tissue_finetune/`

The `tissue_finetune` folder contains Arabidopsis thaliana single-nucleus RNA-seq datasets used for tissue-specific fine-tuning of DeepTAN.

```bash
tissue_finetune/
├── rosette/
│   └── rosette_finetune.h5ad  # Rosette tissue single-nucleus RNA-seq data for fine-tuning.
├── seedling/
│   └── seedling_finetune.h5ad  # Seedling tissue single-nucleus RNA-seq data for fine-tuning.
└── silique/
    └── silique_finetune.h5ad  # Silique tissue single-nucleus RNA-seq data for fine-tuning.
```

- These datasets are used for constructing **tissue-specific networks** with DeepTAN.

- A pretrained DeepTAN model can be fine-tuned separately on rosette, seedling, and silique datasets to capture tissue-specific transcriptomic patterns and infer corresponding biological state-specific networks.

## 3. `cell_finetune/`

The `cell_finetune/` folder contains Arabidopsis thaliana single-nucleus RNA-seq datasets used for cell-type-specific fine-tuning of DeepTAN.

This folder includes 9 cell-type-specific datasets.

```bash
cell_finetune/
├── rosette_Epidermal/
│   └── rosette_Epidermal.h5ad  # Epidermal cell data from rosette leaves.
├── rosette_Stele/  
│   └── rosette_Stele.h5ad  # Stele cell data from rosette leaves.
├── seedling_Epidermal/
│   └── seedling_Epidermal.h5ad  # Epidermal cell data from seedlings.
├── seedling_Stele/
│   └── seedling_Stele.h5ad  # Stele cell data from seedlings.
├── seedling_Mesophyll/
│   └── seedling_Mesophyll.h5ad  # Mesophyll cell data from seedlings.
├── silique_Epidermal/
│   └── silique_Epidermal.h5ad  # Epidermal cell data from siliques.
├── silique_Stele/
│   └── silique_Stele.h5ad  # Stele cell data from siliques.
├── silique_Seed_silique/
│   └── silique_Seed_silique.h5ad  # Seed_silique cell data from siliques.
└── silique_Young_silique/
    └── silique_Young_silique.h5ad  # Young_silique cell data from siliques.
```

- These datasets are used to fine-tune a pretrained DeepTAN model for constructing **cell-type-specific networks**.

## 4. `bulk/`

The `bulk/` folder contains Arabidopsis thaliana transcriptome and methylome data downloaded from the **1001 Genomes Project** database. These data are associated with phenotype information and are used for fine-tuning DeepTAN to construct **trait-associated networks**.

``` bash
bulk/
├── exp_meth.parquet  # Arabidopsis thaliana bulk transcriptome and methylome data.
└── exp_meth_FT16_log1p.parquet  # The same dataset with the flowering time phenotype processed by log1p transformation.
```

These files can be used for:

   - phenotype regression prediction;
   - DeepTAN fine-tuning for flowering time-related **trait-associated networks**.

The log1p-transformed phenotype file is recommended when the phenotype distribution is skewed.

## Data formats

- `.h5ad` files store single-cell transcriptome data in AnnData format and can be loaded using `scanpy`.
- `.parquet` files store bulk transcriptome, methylome, and phenotype data and can be loaded using `pandas`.

## Workflow

- Use `snRNA/ath_pretrain.h5ad` to train a general DeepTAN model from scratch.
- Fine-tune the pretrained model on `tissue_finetune/` datasets to construct tissue-specific networks.
- Fine-tune the pretrained model on `cell_finetune/` datasets to construct cell-type-specific networks.
- Fine-tune the pretrained model on `bulk/` datasets to construct flowering time-related trait-associated networks.

