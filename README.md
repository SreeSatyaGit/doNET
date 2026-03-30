# DeepOMAPNet: Graph-Attention Multi-Modal Single-Cell Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://github.com/SreeSatyaGit/DeepOMAPNet/actions/workflows/ci.yml/badge.svg" alt="CI">
</p>

**DeepOMAPNet** is a deep learning framework for integrated multi-modal single-cell analysis of CITE-seq data. It jointly maps RNA expression to surface protein (ADT) levels, classifies cell types, and performs disease diagnosis (e.g., AML vs. Normal) by combining **Graph Attention Networks (GAT)** with **Cross-Modal Transformer Fusion**.

---

## Architecture

```
RNA expression (N × G)
        │
   GATConv ×2              ← topology-aware RNA encoding
        │
  GraphPositionalEncoding  ← degree + clustering coefficients
        │
TransformerFusion ×L       ← bidirectional RNA ↔ ADT cross-attention
        │                     (sparse O(E) or dense O(N²))
  ┌─────┴──────┐
ADT Regression  AML Classification
  [N × P]         [N × 1]
```

**Key components** (`scripts/model/doNET.py`):

| Component | Role |
|---|---|
| `GATWithTransformerFusion` | End-to-end model: GAT encoder → TransformerFusion → multi-task heads |
| `SparseCrossAttentionLayer` | O(E) cross-attention over edge lists — scales to >100 k cells |
| `GraphPositionalEncoding` | Injects node degree and clustering coefficient into embeddings |
| `AdapterLayer` | Bottleneck residual (dim → dim/r → dim) for parameter-efficient fine-tuning |
| `TransformerFusion` | Stacks bidirectional cross-attention to fuse RNA and ADT modalities |

---

## Results on Synthetic CITE-seq Benchmark

500 cells (250 Normal, 250 AML) · 30 proteins · 500 genes · CPU, 15 s

| Metric | Value |
|---|---|
| ADT prediction — mean Pearson r | **0.785** |
| ADT prediction — best protein r | 0.948 |
| AML classification — AUC-ROC | **0.836** |
| AML classification — F1 | 0.719 |

Reproduce with:

```bash
python run_experiment.py   # saves figures to results/
```

---

## Installation

```bash
git clone https://github.com/SreeSatyaGit/DeepOMAPNet.git
cd DeepOMAPNet

# Conda (recommended)
conda env create -f environment.yml
conda activate deepomapnet

# or pip
pip install -r requirements.txt
```

**Core dependencies:** PyTorch ≥ 2.0 · PyTorch Geometric ≥ 2.3 · ScanPy ≥ 1.9 · AnnData ≥ 0.9

---

## Tutorials

All end-to-end workflows are in `Tutorials/`:

| Notebook | Purpose |
|---|---|
| `Training.ipynb` | Full training pipeline on real CITE-seq AnnData |
| `Test.ipynb` | Evaluation, UMAP visualization, Pearson/Spearman metrics |
| `Finetune.ipynb` | Adapter-based transfer learning to new datasets |
| `scVI.ipynb` | Comparison with scVI baseline |

---

## Synthetic Data

`scripts/data_provider/synthetic_citeseq.py` provides a biologically realistic CITE-seq generator for benchmarking and testing:

- 7 PBMC + AML cell types with biologically accurate marker profiles
- 30-protein ADT panel (CD3, CD4, CD8, CD14, CD34, CD117, CD33, …)
- Negative-binomial RNA counts + bimodal ADT expression
- Tunable Normal vs. AML proportions

```python
from scripts.data_provider.synthetic_citeseq import generate_citeseq_dataset
ds = generate_citeseq_dataset(n_normal=1000, n_aml=1000, seed=42)
# ds.rna  [N, 500]  log-normalized + z-scored
# ds.adt  [N, 30]   CLR-normalized
```

---

## Testing

```bash
# Full test suite (87 tests)
pytest

# Single test file
pytest tests/test_model_components.py -v

# Single test
pytest tests/test_training.py::TestLossDecreases::test_adt_loss_decreases_over_epochs -v
```

Test coverage:

| File | Tests | Scope |
|---|---|---|
| `test_model_components.py` | 36 | Forward pass, gradients, sparse attention, adapters |
| `test_data_pipeline.py` | 25 | CLR/Z-score normalization, graph validity, split integrity |
| `test_training.py` | 10 | Loss decrease, gradient clipping, reproducibility |
| `test_performance_benchmark.py` | 16 | Pearson r vs baselines, Wilcoxon test, AML AUC |

---

## Repository Structure

```
DeepOMAPNet/
├── scripts/
│   ├── model/
│   │   └── doNET.py                 # GATWithTransformerFusion + all components
│   ├── data_provider/
│   │   ├── data_preprocessing.py    # CLR / Z-score normalization
│   │   ├── graph_data_builder.py    # k-NN graph → PyG Data objects
│   │   └── synthetic_citeseq.py     # Realistic synthetic CITE-seq generator
│   ├── trainer/
│   │   ├── gat_trainer.py           # Multi-task training loop (AMP, early stopping)
│   │   └── fineTune.py              # Adapter-based fine-tuning
│   └── visualizations.py            # Publication-quality plotting utilities
├── tests/                           # pytest test suite (87 tests)
├── Tutorials/                       # Jupyter notebooks
├── R/                               # Supporting R scripts (WNN, preprocessing)
├── research/                        # Autoresearch experiment loop
├── run_experiment.py                # Synthetic data → training → figures
├── environment.yml
└── requirements.txt
```

---

## License

MIT — see `LICENSE`.

---

Developed by the **DeepOMAPNet Contributors**.
