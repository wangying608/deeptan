<div align="center">

# DeepTAN

[![pypi-badge](https://img.shields.io/pypi/v/deeptan)](https://pypi.org/project/deeptan)
[![PyPI Downloads](https://static.pepy.tech/badge/deeptan)](https://pepy.tech/projects/deeptan)

</div>

## About

DeepTAN is a graph-based multi-task framework designed to infer large-scale multi-omics trait-associated networks (TANs) and reconstruct phenotype-specific omics states.

## Installation

```bash
conda create -n deeptan python=3.13 -y
conda activate deeptan

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install deeptan
```

## Usage

...
