"""
Shared pytest fixtures for DeepOMAPNet tests.

All fixtures use synthetic CITE-seq data so tests run without real files.
Sizes are kept small (300 cells) for fast CPU execution.
"""

import sys
import os
import importlib.util

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Patch: import model directly, bypassing scripts/__init__.py which tries
# to load hardcoded .h5ad files at import time.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO_ROOT, rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


donet = _load_module("doNET", "scripts/model/doNET.py")

GATWithTransformerFusion   = donet.GATWithTransformerFusion
TransformerFusion          = donet.TransformerFusion
SparseCrossAttentionLayer  = donet.SparseCrossAttentionLayer
CrossAttentionLayer        = donet.CrossAttentionLayer
AdapterLayer               = donet.AdapterLayer
GraphPositionalEncoding    = donet.GraphPositionalEncoding
compute_graph_statistics_fast = donet.compute_graph_statistics_fast

# ---------------------------------------------------------------------------
# Synthetic data constants
# ---------------------------------------------------------------------------

N_CELLS   = 300
N_GENES   = 60
N_ADTS    = 12
K_NEIGH   = 8
N_TYPES   = 4
SEED      = 42

HIDDEN    = 32
HEADS     = 2
NHEAD     = 2
N_LAYERS  = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_knn_edges(features: np.ndarray, k: int = K_NEIGH) -> torch.Tensor:
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(features)
    idx = nbrs.kneighbors(features, return_distance=False)
    rows, cols = [], []
    for i, neighbors in enumerate(idx):
        for j in neighbors[1:]:
            rows += [i, j]
            cols += [j, i]
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return torch.unique(edge_index, dim=1)


def make_synthetic_citeseq(seed: int = SEED, n_cells: int = N_CELLS,
                            n_genes: int = N_GENES, n_adts: int = N_ADTS,
                            n_types: int = N_TYPES):
    """
    Generate synthetic CITE-seq data with a known latent factor structure.
    RNA and ADT are correlated through shared latent factors (n_types cell types).
    AML label is set by latent factor dominance.
    """
    rng = np.random.default_rng(seed)
    cell_factors = rng.dirichlet(np.ones(n_types), size=n_cells)  # [N, K]

    W_rna = rng.standard_normal((n_types, n_genes)).astype(np.float32) * 2.0
    mu_rna = np.exp(cell_factors @ W_rna + 0.3)
    rna_counts = rng.poisson(mu_rna).astype(np.float32)
    lib_size = rna_counts.sum(1, keepdims=True) + 1e-8
    rna_norm = np.log1p(rna_counts / lib_size * 1e4)
    rna_norm = ((rna_norm - rna_norm.mean(0)) / (rna_norm.std(0) + 1e-8)).astype(np.float32)

    W_adt = rng.standard_normal((n_types, n_adts)).astype(np.float32) * 1.5
    mu_adt = np.exp(cell_factors @ W_adt * 0.8)
    adt_counts = rng.poisson(mu_adt + 1).astype(np.float32)
    adt_log = np.log(adt_counts + 1.0)
    adt_norm = (adt_log - adt_log.mean(1, keepdims=True)).astype(np.float32)
    adt_norm = ((adt_norm - adt_norm.mean(0)) / (adt_norm.std(0) + 1e-8)).astype(np.float32)

    aml_score = cell_factors[:, 0] + 0.5 * cell_factors[:, 1]
    aml_labels = (aml_score > np.percentile(aml_score, 60)).astype(np.int32)
    celltype_labels = np.argmax(cell_factors, axis=1).astype(np.int64)

    return rna_norm, adt_norm, aml_labels, celltype_labels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_data():
    rna, adt, aml, ctype = make_synthetic_citeseq()
    return rna, adt, aml, ctype


@pytest.fixture(scope="session")
def pyg_data(synthetic_data):
    rna_norm, adt_norm, aml_labels, celltype_labels = synthetic_data
    n_cells = rna_norm.shape[0]

    # PCA-lite: use SVD for graph construction
    U, S, _ = np.linalg.svd(rna_norm, full_matrices=False)
    pca = U[:, :min(20, U.shape[1])] * S[:min(20, len(S))]
    edge_index = make_knn_edges(pca, k=K_NEIGH)

    rna_data = Data(x=torch.tensor(rna_norm), edge_index=edge_index)
    adt_data = Data(x=torch.tensor(adt_norm), edge_index=edge_index)

    node_degrees, clustering_coeffs = compute_graph_statistics_fast(edge_index, n_cells)

    return {
        "rna_data":          rna_data,
        "adt_data":          adt_data,
        "aml_labels":        torch.tensor(aml_labels, dtype=torch.float32),
        "celltype_labels":   torch.tensor(celltype_labels, dtype=torch.long),
        "node_degrees":      node_degrees,
        "clustering_coeffs": clustering_coeffs,
        "n_cells":           n_cells,
        "n_genes":           N_GENES,
        "n_adts":            N_ADTS,
    }


@pytest.fixture(scope="session")
def model(pyg_data):
    torch.manual_seed(SEED)
    m = GATWithTransformerFusion(
        in_channels=pyg_data["n_genes"],
        hidden_channels=HIDDEN,
        out_channels=pyg_data["n_adts"],
        heads=HEADS,
        dropout=0.1,
        nhead=NHEAD,
        num_layers=N_LAYERS,
        use_adapters=True,
        reduction_factor=4,
        use_sparse_attention=True,
        neighborhood_size=K_NEIGH,
        num_cell_types=N_TYPES,
    )
    m.eval()
    return m


@pytest.fixture(scope="session")
def trained_model(pyg_data):
    """
    Lightly trained model (30 epochs) for performance tests.
    Uses the same seed for reproducibility.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d = pyg_data
    m = GATWithTransformerFusion(
        in_channels=d["n_genes"],
        hidden_channels=HIDDEN,
        out_channels=d["n_adts"],
        heads=HEADS,
        dropout=0.1,
        nhead=NHEAD,
        num_layers=N_LAYERS,
        use_adapters=True,
        reduction_factor=4,
        use_sparse_attention=True,
        neighborhood_size=K_NEIGH,
        num_cell_types=N_TYPES,
    )

    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = d["n_cells"]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:int(0.7 * n)]] = True

    for epoch in range(30):
        m.train()
        optimizer.zero_grad(set_to_none=True)
        adt_pred, aml_pred, fused = m(
            d["rna_data"].x, d["rna_data"].edge_index, d["adt_data"].edge_index,
            node_degrees_rna=d["node_degrees"], node_degrees_adt=d["node_degrees"],
            clustering_coeffs_rna=d["clustering_coeffs"], clustering_coeffs_adt=d["clustering_coeffs"],
        )
        loss = (
            torch.nn.functional.mse_loss(adt_pred[train_mask], d["adt_data"].x[train_mask])
            + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
                aml_pred[train_mask].squeeze(), d["aml_labels"][train_mask]
            )
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()

    m.eval()
    return m, train_mask
