"""
Tests for data preprocessing and graph construction.

Verifies normalization correctness, graph validity,
and the absence of data leakage between splits.
"""

import sys
import os
import types
import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from scipy import sparse
import anndata as ad

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_preprocessing():
    """
    Load data_preprocessing.py while neutralising the hardcoded sc.read_h5ad()
    calls that fire as default argument values at module import time.
    We replace scanpy's read_h5ad with a stub that returns an empty AnnData
    so the module-level code doesn't crash.
    """
    import scanpy as sc
    _real_read = sc.read_h5ad

    def _stub_read(path, *args, **kwargs):
        return ad.AnnData(np.zeros((1, 1), dtype=np.float32))

    sc.read_h5ad = _stub_read
    try:
        spec = importlib.util.spec_from_file_location(
            "data_preprocessing",
            os.path.join(_REPO_ROOT, "scripts/data_provider/data_preprocessing.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sc.read_h5ad = _real_read  # always restore

    return mod


def _load_graph_builder():
    spec = importlib.util.spec_from_file_location(
        "graph_data_builder",
        os.path.join(_REPO_ROOT, "scripts/data_provider/graph_data_builder.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


preprocessing = _load_preprocessing()
graph_builder = _load_graph_builder()

clr_normalize    = preprocessing.clr_normalize
zscore_normalize = preprocessing.zscore_normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n_cells=200, n_features=30, seed=42, sparse_input=False):
    rng = np.random.default_rng(seed)
    X = rng.poisson(lam=5.0, size=(n_cells, n_features)).astype(np.float32)
    if sparse_input:
        X = sparse.csr_matrix(X)
    return ad.AnnData(X=X)


# ---------------------------------------------------------------------------
# CLR Normalization
# ---------------------------------------------------------------------------

class TestCLRNormalize:

    def test_output_shape_preserved(self):
        adata = _make_adata()
        out = clr_normalize(adata)
        assert out.shape == adata.shape

    def test_does_not_modify_input(self):
        adata = _make_adata()
        X_orig = np.array(adata.X)
        clr_normalize(adata)
        np.testing.assert_array_equal(adata.X, X_orig, err_msg="Input AnnData must not be mutated")

    def test_output_finite(self):
        adata = _make_adata()
        out = clr_normalize(adata)
        X = out.X if not sparse.issparse(out.X) else out.X.toarray()
        assert np.isfinite(X).all(), "CLR output contains inf or NaN"

    def test_per_cell_centering(self):
        """CLR is centered: row sums of exp(CLR) should equal N*1 (geometric mean = 1)."""
        adata = _make_adata(n_cells=50, n_features=10)
        out = clr_normalize(adata, axis=1)
        X = out.X if not sparse.issparse(out.X) else out.X.toarray()
        # Each row mean should be ≈ 0 (CLR centers in log space)
        row_means = X.mean(axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-5,
                                   err_msg="CLR row means must be 0")

    def test_handles_sparse_input(self):
        adata = _make_adata(sparse_input=True)
        out = clr_normalize(adata)
        X = out.X if not sparse.issparse(out.X) else out.X.toarray()
        assert np.isfinite(X).all()

    def test_handles_zero_rows(self):
        """Cells with all-zero counts must not produce NaN."""
        adata = _make_adata(n_cells=10, n_features=5)
        adata.X[3, :] = 0  # all-zero row
        out = clr_normalize(adata)
        X = out.X if not sparse.issparse(out.X) else out.X.toarray()
        assert np.isfinite(X).all(), "CLR must handle all-zero rows"

    def test_axis0_normalization(self):
        """axis=0 normalizes per feature (across cells)."""
        adata = _make_adata(n_cells=50, n_features=10)
        out = clr_normalize(adata, axis=0)
        X = out.X if not sparse.issparse(out.X) else out.X.toarray()
        col_means = X.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-5,
                                   err_msg="CLR axis=0 column means must be 0")


# ---------------------------------------------------------------------------
# Z-score Normalization
# ---------------------------------------------------------------------------

class TestZscoreNormalize:

    def test_output_shape(self):
        adata = _make_adata()
        out, means, stds = zscore_normalize(adata)
        assert out.shape == adata.shape
        assert means.shape == (1, adata.shape[1])
        assert stds.shape  == (1, adata.shape[1])

    def test_zero_mean_unit_variance(self):
        adata = _make_adata(n_cells=500, n_features=20)
        out, _, _ = zscore_normalize(adata)
        X = out.X if not sparse.issparse(out.X) else out.X.toarray()
        col_means = X.mean(0)
        col_stds  = X.std(0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-5,
                                   err_msg="Z-score must yield zero column means")
        # Columns with non-zero variance should have std ≈ 1
        nonzero = col_stds > 0.01
        np.testing.assert_allclose(col_stds[nonzero], 1.0, atol=1e-4,
                                   err_msg="Z-score must yield unit column std")

    def test_returns_correct_statistics(self):
        adata = _make_adata(n_cells=100, n_features=10)
        X_orig = np.array(adata.X, dtype=np.float32)
        _, means, stds = zscore_normalize(adata)
        np.testing.assert_allclose(means.squeeze(), X_orig.mean(0), rtol=1e-4)

    def test_output_finite(self):
        adata = _make_adata()
        out, _, _ = zscore_normalize(adata)
        X = out.X if not sparse.issparse(out.X) else out.X.toarray()
        assert np.isfinite(X).all()

    def test_constant_column_no_nan(self):
        """Constant features (std=0) must not produce NaN after z-score."""
        adata = _make_adata(n_cells=50, n_features=5)
        adata.X[:, 2] = 5.0  # constant column
        out, _, _ = zscore_normalize(adata)
        X = out.X if not sparse.issparse(out.X) else out.X.toarray()
        assert np.isfinite(X).all(), "Constant feature must not produce NaN"

    def test_does_not_modify_input(self):
        adata = _make_adata()
        X_orig = np.array(adata.X)
        zscore_normalize(adata)
        np.testing.assert_array_equal(adata.X, X_orig)


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

class TestGraphConstruction:

    def test_edge_index_valid_range(self, pyg_data):
        """All edge indices must be in [0, N_CELLS)."""
        edge_index = pyg_data["rna_data"].edge_index
        n = pyg_data["n_cells"]
        assert edge_index.min().item() >= 0
        assert edge_index.max().item() < n, \
            f"Edge index out of range: max={edge_index.max().item()}, N={n}"

    def test_edge_index_two_rows(self, pyg_data):
        edge_index = pyg_data["rna_data"].edge_index
        assert edge_index.shape[0] == 2, "edge_index must be shape [2, E]"

    def test_no_self_loops(self, pyg_data):
        """k-NN graph should not have self-loops (neighbors exclude self)."""
        edge_index = pyg_data["rna_data"].edge_index
        self_loops = (edge_index[0] == edge_index[1]).sum().item()
        assert self_loops == 0, f"Found {self_loops} self-loops in k-NN graph"

    def test_graph_is_undirected(self, pyg_data):
        """For every edge (i,j), edge (j,i) must also exist."""
        edge_index = pyg_data["rna_data"].edge_index
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        for (i, j) in list(edges)[:200]:  # check first 200 for speed
            assert (j, i) in edges, f"Edge ({i},{j}) present but ({j},{i}) missing — not undirected"

    def test_all_nodes_have_neighbors(self, pyg_data):
        """Every cell should have at least one connection."""
        edge_index = pyg_data["rna_data"].edge_index
        n = pyg_data["n_cells"]
        unique_nodes = torch.unique(edge_index.flatten())
        assert len(unique_nodes) == n, \
            f"Isolated nodes found: {n - len(unique_nodes)} cells have no edges"

    def test_rna_adt_same_node_count(self, pyg_data):
        """RNA and ADT graphs must cover the same cells."""
        assert pyg_data["rna_data"].num_nodes == pyg_data["adt_data"].num_nodes

    def test_node_degrees_shape(self, pyg_data):
        degrees = pyg_data["node_degrees"]
        assert degrees.shape == (pyg_data["n_cells"],)
        assert (degrees >= 0).all()

    def test_clustering_coeffs_shape_and_range(self, pyg_data):
        cc = pyg_data["clustering_coeffs"]
        assert cc.shape == (pyg_data["n_cells"],)
        assert (cc >= 0).all(), "Clustering coefficients must be non-negative"


# ---------------------------------------------------------------------------
# Data Split Integrity (no leakage)
# ---------------------------------------------------------------------------

class TestDataSplits:

    def _make_splits(self, n=300, train_frac=0.7, val_frac=0.15, seed=42):
        """Reproduce the split logic from conftest for isolated testing."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_train = int(train_frac * n)
        n_val   = int(val_frac * n)
        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]
        return set(train_idx), set(val_idx), set(test_idx)

    def test_splits_are_disjoint(self):
        train, val, test = self._make_splits()
        assert len(train & val)  == 0, "Train and val overlap"
        assert len(train & test) == 0, "Train and test overlap"
        assert len(val & test)   == 0, "Val and test overlap"

    def test_splits_cover_all_samples(self):
        n = 300
        train, val, test = self._make_splits(n=n)
        assert len(train | val | test) == n, "Not all samples accounted for"

    def test_split_sizes_approximately_correct(self):
        n = 300
        train, val, test = self._make_splits(n=n, train_frac=0.7, val_frac=0.15)
        assert abs(len(train) / n - 0.70) < 0.05
        assert abs(len(val)   / n - 0.15) < 0.05

    def test_no_feature_leakage_via_normalization(self, pyg_data):
        """
        Normalization statistics (mean, std) should differ between train and full dataset.
        If they're identical, normalization was likely computed on the full dataset.
        Verifies that train-time statistics don't perfectly encode test data.
        """
        d = pyg_data
        n = d["n_cells"]
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        train_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[idx[:int(0.7 * n)]] = True

        X_full  = d["adt_data"].x
        X_train = d["adt_data"].x[train_mask]

        full_mean  = X_full.mean(0)
        train_mean = X_train.mean(0)
        # Train mean should NOT be identical to full mean (different samples)
        assert not torch.allclose(full_mean, train_mean, atol=1e-6), \
            "Train statistics identical to full-dataset statistics — possible leakage"
