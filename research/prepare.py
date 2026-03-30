"""
Fixed data preparation and evaluation for DeepOMAPNet autoresearch.

Provides:
  - Synthetic CITE-seq data (mimics real RNA + ADT structure)
  - k-NN graph construction (PyTorch Geometric)
  - evaluate_nrmse(): the single ground-truth metric (lower = better)

Constants in this file are FIXED — do not modify.
Usage:
    python prepare.py          # verify setup, print dataset stats
"""

import sys
import os
import math
import time

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

NUM_CELLS      = 200    # total cells (small for CPU-only machines)
NUM_GENES      = 80     # RNA input features
NUM_ADTS       = 10     # surface protein markers to predict
K_NEIGHBORS    = 8      # k for k-NN graph
TIME_BUDGET    = 60     # wall-clock training seconds (1 min on CPU)
RANDOM_SEED    = 42
VAL_FRAC       = 0.15
TEST_FRAC      = 0.15

# ---------------------------------------------------------------------------
# Synthetic CITE-seq data generator
# ---------------------------------------------------------------------------

def _make_synthetic_citeseq(seed: int = RANDOM_SEED):
    """
    Generate synthetic RNA + ADT data that mimics real CITE-seq structure:
    - RNA: overdispersed counts (Negative Binomial-like via Poisson-Gamma)
    - ADT: ~30 surface proteins correlated with latent cell-type factors
    - AML labels: binary (roughly 40 % AML, 60 % Normal)

    Returns numpy arrays (float32, already normalized):
        rna_norm  [N, G]  — log1p-normalized RNA
        adt_norm  [N, P]  — CLR-normalized ADT
        aml_labels [N]    — int32 {0,1}
    """
    rng = np.random.default_rng(seed)

    # --- Latent cell-state factors (simulate ~6 cell types) ---
    n_factors = 6
    cell_factors = rng.dirichlet(np.ones(n_factors), size=NUM_CELLS)  # [N, n_factors]

    # AML label: cells with high factor-0 or factor-1 are AML
    aml_score = cell_factors[:, 0] + 0.5 * cell_factors[:, 1]
    aml_threshold = np.percentile(aml_score, 60)
    aml_labels = (aml_score > aml_threshold).astype(np.int32)

    # --- RNA expression: factor loadings + noise ---
    W_rna = rng.standard_normal((n_factors, NUM_GENES)).astype(np.float32) * 2.0
    mu_rna = np.exp(cell_factors @ W_rna + 0.5)  # [N, G] positive
    rna_counts = rng.poisson(mu_rna).astype(np.float32)
    # log1p normalization
    rna_norm = np.log1p(rna_counts / (rna_counts.sum(axis=1, keepdims=True) + 1e-8) * 1e4)
    rna_norm = ((rna_norm - rna_norm.mean(0)) / (rna_norm.std(0) + 1e-8)).astype(np.float32)

    # --- ADT expression: protein levels from same latent factors ---
    W_adt = rng.standard_normal((n_factors, NUM_ADTS)).astype(np.float32) * 1.5
    mu_adt = np.exp(cell_factors @ W_adt * 0.8 + rng.standard_normal((NUM_CELLS, NUM_ADTS)).astype(np.float32) * 0.3)
    adt_counts = rng.poisson(mu_adt + 1).astype(np.float32)
    # CLR normalization (per cell)
    adt_log = np.log(adt_counts + 1.0)
    adt_norm = (adt_log - adt_log.mean(axis=1, keepdims=True)).astype(np.float32)
    adt_norm = ((adt_norm - adt_norm.mean(0)) / (adt_norm.std(0) + 1e-8)).astype(np.float32)

    return rna_norm, adt_norm, aml_labels


# ---------------------------------------------------------------------------
# Graph construction (k-NN in RNA PCA space → PyG Data)
# ---------------------------------------------------------------------------

def build_knn_graph(features: np.ndarray, k: int = K_NEIGHBORS) -> torch.Tensor:
    """Build undirected k-NN graph. Returns edge_index [2, E]."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", n_jobs=-1)
    nbrs.fit(features)
    indices = nbrs.kneighbors(features, return_distance=False)  # [N, k+1]

    rows, cols = [], []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def make_pyg_data(rna_norm, adt_norm):
    """
    Build PyG Data objects for RNA and ADT modalities.
    Both share the same k-NN graph topology (built from RNA features).
    """
    # Simple PCA-like projection for graph: use top-50 PCs via SVD
    n_components = min(50, rna_norm.shape[1] - 1)
    U, S, Vt = np.linalg.svd(rna_norm, full_matrices=False)
    rna_pca = U[:, :n_components] * S[:n_components]

    edge_index = build_knn_graph(rna_pca, k=K_NEIGHBORS)

    rna_data = Data(
        x=torch.tensor(rna_norm, dtype=torch.float32),
        edge_index=edge_index,
    )
    adt_data = Data(
        x=torch.tensor(adt_norm, dtype=torch.float32),
        edge_index=edge_index,
    )
    return rna_data, adt_data


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def make_splits(n: int, aml_labels: np.ndarray, seed: int = RANDOM_SEED):
    """Stratified split — returns boolean masks."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)

    # Stratified by AML label
    aml_idx   = idx[aml_labels == 1]
    norm_idx  = idx[aml_labels == 0]

    def _split(arr):
        arr = rng.permutation(arr)
        n_val  = max(1, int(len(arr) * VAL_FRAC))
        n_test = max(1, int(len(arr) * TEST_FRAC))
        return arr[n_val + n_test:], arr[:n_val], arr[n_val:n_val + n_test]

    tr_a, va_a, te_a = _split(aml_idx)
    tr_n, va_n, te_n = _split(norm_idx)

    train_idx = np.concatenate([tr_a, tr_n])
    val_idx   = np.concatenate([va_a, va_n])
    test_idx  = np.concatenate([te_a, te_n])

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True
    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# Evaluation — FIXED ground truth metric
# ---------------------------------------------------------------------------

def evaluate(model, rna_data, adt_data, aml_labels_tensor,
             val_mask, device,
             node_degrees=None, clustering_coeffs=None) -> dict:
    """
    Evaluate model on the validation set.

    Returns a dict with:
        val_nrmse  — normalized RMSE for ADT prediction (primary, lower = better)
        val_pearson — mean per-protein Pearson r (higher = better, informational)
        val_auc    — AUC-ROC for AML classification (informational)
    """
    model.eval()
    with torch.no_grad():
        adt_pred, aml_pred, _ = model(
            rna_data.x.to(device),
            rna_data.edge_index.to(device),
            adt_data.edge_index.to(device),
            node_degrees_rna=node_degrees,
            node_degrees_adt=node_degrees,
            clustering_coeffs_rna=clustering_coeffs,
            clustering_coeffs_adt=clustering_coeffs,
        )

    vm = val_mask.to(device)
    adt_true = adt_data.x.to(device)[vm].cpu().numpy()
    adt_hat  = adt_pred[vm].cpu().numpy()
    aml_true = aml_labels_tensor.to(device)[vm].cpu().numpy()
    aml_logits = aml_pred[vm].cpu().numpy().squeeze()

    # Normalized RMSE: RMSE / std(target)  — scale-free
    rmse = math.sqrt(((adt_hat - adt_true) ** 2).mean())
    nrmse = rmse / (adt_true.std() + 1e-8)

    # Per-protein Pearson r
    rs = []
    for p in range(adt_true.shape[1]):
        r, _ = pearsonr(adt_true[:, p], adt_hat[:, p])
        if not math.isnan(r):
            rs.append(r)
    mean_pearson = float(np.mean(rs)) if rs else 0.0

    # AUC
    try:
        aml_probs = 1.0 / (1.0 + np.exp(-aml_logits))
        auc = roc_auc_score(aml_true, aml_probs)
    except ValueError:
        auc = 0.5

    return {
        "val_nrmse":   nrmse,
        "val_pearson": mean_pearson,
        "val_auc":     auc,
    }


# ---------------------------------------------------------------------------
# Entry point — sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic CITE-seq dataset...")
    t0 = time.time()
    rna_norm, adt_norm, aml_labels = _make_synthetic_citeseq()
    rna_data, adt_data = make_pyg_data(rna_norm, adt_norm)
    train_mask, val_mask, test_mask = make_splits(NUM_CELLS, aml_labels)
    elapsed = time.time() - t0

    print(f"  RNA  : {rna_norm.shape}  ({rna_norm.dtype})")
    print(f"  ADT  : {adt_norm.shape}  ({adt_norm.dtype})")
    print(f"  AML  : {aml_labels.sum()} / {NUM_CELLS} positive ({aml_labels.mean()*100:.1f}%)")
    print(f"  Edges: {rna_data.edge_index.shape[1]}")
    print(f"  Split: train={train_mask.sum()} / val={val_mask.sum()} / test={test_mask.sum()}")
    print(f"  Prep : {elapsed:.2f}s")
    print("Setup OK.")
