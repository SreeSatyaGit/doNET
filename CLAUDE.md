# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepOMAPNet is a deep learning framework for multi-modal single-cell analysis using CITE-seq data. It maps RNA expression to surface protein (ADT) levels while simultaneously performing cell-type identification and disease classification (AML vs. Normal).

## Environment Setup

```bash
# Conda (preferred)
conda env create -f environment.yml
conda activate deepomapnet

# Pip alternative
pip install -r requirements.txt
```

Key dependencies: PyTorch >=2.0.0, PyTorch Geometric >=2.3.0, ScanPy >=1.9.0, AnnData >=0.9.0.

## Running Code

There are no CLI entry points. All workflows are driven through Jupyter notebooks in `Tutorials/`:

```bash
jupyter notebook Tutorials/Training.ipynb    # End-to-end training
jupyter notebook Tutorials/Test.ipynb        # Evaluation + UMAP visualization
jupyter notebook Tutorials/Finetune.ipynb    # Adapter-based transfer learning
jupyter notebook Tutorials/scVI.ipynb        # Comparison with scVI baseline
```

Linting/formatting (from `requirements.txt`):
```bash
black scripts/
flake8 scripts/
mypy scripts/
pytest
```

## Architecture

### Data Flow

```
Raw CITE-seq AnnData (RNA + ADT)
    → CLR normalization (ADT) + Z-score normalization (both)
    → Train/val/test split
    → PCA (50 components) + k-NN graph (k=15) + Leiden clustering
    → PyTorch Geometric Data objects
    → GATWithTransformerFusion model
    → Outputs: ADT predictions, AML classification, fused embeddings
```

### Module Map

| Module | File | Role |
|--------|------|------|
| Data normalization & splitting | `scripts/data_provider/data_preprocessing.py` | CLR/Z-score normalization, `prepare_train_test_anndata()` |
| Graph construction | `scripts/data_provider/graph_data_builder.py` | k-NN graph + PCA → PyG `Data` objects via `build_pyg_data()` / `process_data_with_graphs()` |
| Core model | `scripts/model/doNET.py` | `GATWithTransformerFusion` and supporting classes |
| Training loop | `scripts/trainer/gat_trainer.py` | `train_gat_transformer_fusion()` — multi-task training, AMP, early stopping |

### Model Components (`scripts/model/doNET.py`)

`GATWithTransformerFusion` is composed of:

1. **GraphPositionalEncoding** — topology-aware embeddings using node degree and clustering coefficients
2. **SparseCrossAttentionLayer** — sparse multi-head cross-attention using edge lists (bounded by `max_neighbors`, default 50); use when graphs are large
3. **CrossAttentionLayer** — dense cross-attention variant with layer norm; use for smaller graphs
4. **AdapterLayer** — bottleneck adapter for parameter-efficient fine-tuning (configurable reduction factor, L2 regularization)
5. **TransformerFusion** — stacks cross-attention layers to fuse RNA and ADT modalities; optionally uses adapters

The model produces three outputs: `adt_pred` (protein-level regression), `aml_pred` (binary disease classification), and `fused_embeddings` (latent cell representations). An optional cell-type classification head can be enabled at runtime.

### Training (`scripts/trainer/gat_trainer.py`)

`train_gat_transformer_fusion()` handles:
- Multi-task loss: MSE (ADT regression) + BCE (AML classification), with optional cell-type CE loss
- Stratified splits (default 70/15/15)
- Mixed precision (AMP) + gradient accumulation
- Early stopping with configurable patience
- Graph statistics computation (degree, clustering coefficients) fed into `GraphPositionalEncoding`

## Key Design Decisions

- **Sparse vs. dense attention:** `SparseCrossAttentionLayer` is preferred for scalability on large cell graphs; `CrossAttentionLayer` is the fallback for small graphs.
- **Adapter fine-tuning:** `AdapterLayer` enables transfer learning without full retraining — used in `Tutorials/Finetune.ipynb`.
- **Multi-task learning:** Disease classification and ADT prediction are jointly optimized; loss weights are configurable.
- **Graph construction:** k-NN graph is built in PCA space (50 dims), not raw feature space, to reduce noise.
