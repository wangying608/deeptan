# mi2graph

[![Crates.io](https://img.shields.io/crates/v/mi2graph?style=for-the-badge&color=blue)](https://crates.io/crates/mi2graph)
[![Downloads](https://img.shields.io/crates/d/mi2graph?style=for-the-badge&color=green)](https://crates.io/crates/mi2graph)
[![License](https://img.shields.io/crates/l/mi2graph?style=for-the-badge)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg?style=for-the-badge&logo=rust)](https://www.rust-lang.org)

**Generate Maximal Information Coefficient (MIC) relations between features with dynamic feature filtering for graph initialization.**

[Installation](#installation) · [Usage](#usage) · [Algorithm](#algorithm)

---

## Overview

`mi2graph` is a high-performance Rust tool for computing **Maximal Information Coefficient (MIC)** relationships between features in large-scale datasets. It implements dynamic feature filtering using sliding windows to identify optimal correlations and mutual information relationships.

### Key Features

| Feature | Description |
|---------|-------------|
| **Dynamic Sliding Windows** | Adaptive window sizing for optimal correlation detection |
| **Feature Filtering** | Automatic removal of low-variation and redundant features |
| **High Performance** | Parallel processing with Rayon for multi-threaded execution |
| **Memory Efficient** | Streaming processing for large datasets |
| **Standard Formats** | Input/output in Apache Parquet and NumPy NPZ formats |

### Use Cases

- **Bioinformatics**: Gene expression network construction
- **Machine Learning**: Feature selection and graph neural network initialization
- **Data Mining**: Discovering non-linear relationships between variables

> `mi2graph` is used for data preprocessing in our [DeepTAN](https://pypi.org/project/deeptan/) project.

---

## Installation

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) 1.70 or higher

### From Crates.io

```bash
cargo install mi2graph
```

### From Source

```bash
cd mi2graph
cargo build --release
```

The compiled binary will be available at `target/release/mi2graph`.

---

## Usage

### Command Line Interface

```bash
mi2graph -i input.parquet -o output [OPTIONS]
```

### Quick Start

```bash
# Basic usage
mi2graph -i data.parquet -o results

# With all options
mi2graph \
  --input data.parquet \
  --output results \
  --nfeat 5000 \
  --threcv 0.1 \
  --threpcc 0.95 \
  --thremi 0.05 \
  --chksim \
  --threads 8
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-i` | *required* | Input Parquet file path |
| `--output` | `-o` | *required* | Output path prefix |
| `--nfeat` | `-n` | `5000` | Number of top-CV features to retain (0 = use `--threcv` instead) |
| `--threcv` | | `0.1` | Coefficient of variation threshold for feature filtering |
| `--threpcc` | | `0.95` | Pearson correlation threshold for removing redundant features |
| `--thremi` | | `0.05` | Mutual information threshold for edge filtering |
| `--maxwin` | | `0.98` | Maximum window size ratio (window_size / num_samples) |
| `--minwin` | | `0.33` | Minimum window size ratio |
| `--stepwin` | | `0.07` | Window size step ratio |
| `--stepsli` | | `0.07` | Sliding step ratio |
| `--chksim` | `-s` | `false` | Enable detection of similar feature pairs |
| `--threads` | `-t` | `0` | Number of threads (0 = use all available - 1) |

---

## Algorithm

### Steps Explained

#### 1. Coefficient of Variation (CV) Filtering
- Computes CV using dynamic sliding windows
- Removes features with low variation (noise)
- Can use either threshold-based or top-N selection

#### 2. Similar Feature Detection (Optional)
- Uses Pearson correlation with dynamic 2D sliding windows
- Identifies and removes redundant features (highly correlated pairs)

#### 3. Mutual Information Computation
- Computes MIC for all feature pairs using Freedman-Diaconis binning
- Uses dynamic sliding windows to find optimal mutual information
- Results are normalized (NMIC)

#### 4. Edge Filtering
- Removes edges with MI below threshold
- Keeps only features involved in significant relationships

---

## Input/Output Format

### Input

The input must be a **Parquet file** with the following structure:

- **Shape**: `(n_observations, 1 + n_features)`
- **First column**: `obs_names` - observation/sample identifiers
- **Remaining columns**: feature values (numeric, f64)

| obs_names | feature_1 | feature_2 | ... | feature_n |
|-----------|-----------|-----------|-----|-----------|
| sample_1  | 0.5       | 1.2       | ... | 3.4       |
| sample_2  | 0.7       | 0.9       | ... | 2.1       |
| ...       | ...       | ...       | ... | ...       |

### Output

The tool generates two files:

#### 1. NPZ File (`output.npz`)

| Array Name | Shape | Description |
|------------|-------|-------------|
| `mi_values` | `(n_edges,)` | Sorted mutual information values |
| `feat_pairs` | `(n_edges, 2)` | Feature pair indices (edges) |
| `processed_mat` | `(n_features, n_obs)` | Filtered and processed data matrix |
| `mat_feat_indices` | `(n_features,)` | Original indices of retained features |
| `mat_simi_feat_pairs` | `(n_similar_pairs, 2)` | Indices of similar feature pairs |
| `thre_cv`, `thre_pcc`, `thre_mi` | `(1,)` | Threshold parameters used |
| `ratio_*` | `(1,)` | Window configuration parameters |

#### 2. Parquet File (`output.parquet`)

A DataFrame containing the processed matrix with observation names and feature names.

---

## Library Usage

You can also use `mi2graph` as a Rust library:

```rust
use mi2graph::{
    mic_mat_with_data_filter, read_parquet_to_array2d,
    FilterConfig, ProcessingOptions, WindowConfig
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read data
    let (data, obs_names, var_names) = read_parquet_to_array2d("input.parquet")?;
    
    // Configure processing
    let filter_config = FilterConfig {
        thre_cv: 0.1,
        thre_pcc: 0.95,
        thre_mi: 0.05,
    };
    
    let window_config = WindowConfig {
        ratio_min: 0.33,
        ratio_max: 0.98,
        ratio_step: 0.07,
        ratio_slide: 0.07,
    };
    
    let options = ProcessingOptions {
        check_sim: true,
        n_features_to_select: 5000,
        n_threads: 8,
    };
    
    // Run analysis
    mic_mat_with_data_filter(
        "output",
        &data,
        &obs_names,
        &var_names,
        filter_config,
        window_config,
        options,
    )?;
    
    Ok(())
}
```

Add to your `Cargo.toml`:

```toml
[dependencies]
mi2graph = "0.3"
```

---

## License

This project is licensed under either of:

- **MIT License** - See [LICENSE](LICENSE) file
- **Apache License, Version 2.0** - See [LICENSE](LICENSE) file

at your option.