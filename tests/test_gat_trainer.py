"""Unit tests for GAT Transformer Fusion Training Module."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from unittest.mock import Mock, patch, MagicMock

from scripts.trainer.gat_trainer import (
    compute_multi_task_loss,
    compute_classification_metrics,
    _create_data_splits,
    _validate_inputs,
    _compute_correlations_vectorized,
    _compute_loss_weights,
    _get_nan_metrics,
    _compute_regression_metrics,
    NormalizationParams,
    GraphStatistics,
    TrainingResult,
    DeviceSwitchRequired,
    MAX_HIDDEN_CHANNELS,
    MAX_ATTENTION_HEADS,
    MAX_TRANSFORMER_LAYERS,
    DEFAULT_REG_WEIGHT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_rna_data():
    """Small RNA PyG Data for testing."""
    num_nodes, num_features, num_edges = 100, 50, 200
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def small_adt_data():
    """Small ADT PyG Data for testing."""
    return Data(x=torch.randn(100, 10))


@pytest.fixture
def binary_labels():
    return np.random.RandomState(42).randint(0, 2, size=100)


@pytest.fixture
def celltype_labels_arr():
    return np.random.RandomState(42).randint(0, 5, size=100)


# ---------------------------------------------------------------------------
# TestDataSplits
# ---------------------------------------------------------------------------


class TestDataSplits:
    def test_basic_split_sizes(self):
        train_mask, val_mask, test_mask = _create_data_splits(
            num_nodes=100, stratify_labels=None,
            train_fraction=0.8, val_fraction=0.1, seed=42
        )
        assert train_mask.sum().item() == 80
        assert val_mask.sum().item() == 10
        assert test_mask.sum().item() == 10

    def test_no_overlap(self):
        train_mask, val_mask, test_mask = _create_data_splits(
            100, None, 0.8, 0.1, seed=42
        )
        assert (train_mask & val_mask).sum().item() == 0
        assert (train_mask & test_mask).sum().item() == 0
        assert (val_mask & test_mask).sum().item() == 0

    def test_reproducibility(self):
        m1 = _create_data_splits(100, None, 0.8, 0.1, seed=7)
        m2 = _create_data_splits(100, None, 0.8, 0.1, seed=7)
        for a, b in zip(m1, m2):
            assert torch.equal(a, b)

    def test_stratified_split(self):
        labels = np.array([0] * 50 + [1] * 50)
        train_mask, _, _ = _create_data_splits(
            100, labels, 0.8, 0.1, seed=42
        )
        train_labels = labels[train_mask.numpy()]
        assert abs(train_labels.mean() - 0.5) < 0.15


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def _call(self, rna, adt, **kwargs):
        defaults = dict(
            rna_anndata=None, adt_anndata=None, aml_labels=None,
            celltype_labels=None, stratify_labels=None,
            train_fraction=0.8, val_fraction=0.1, epochs=10,
            learning_rate=1e-3, weight_decay=1e-4, dropout_rate=0.4
        )
        defaults.update(kwargs)
        _validate_inputs(rna, adt, **defaults)

    def test_valid_passes(self, small_rna_data, small_adt_data):
        self._call(small_rna_data, small_adt_data)  # Should not raise

    def test_wrong_type_rna(self, small_adt_data):
        with pytest.raises(TypeError, match="rna_data must be"):
            self._call("not_data", small_adt_data)

    def test_wrong_type_adt(self, small_rna_data):
        with pytest.raises(TypeError, match="adt_data must be"):
            self._call(small_rna_data, "not_data")

    def test_mismatched_nodes(self, small_rna_data):
        bad_adt = Data(x=torch.randn(50, 10))
        with pytest.raises(ValueError, match="same number of nodes"):
            self._call(small_rna_data, bad_adt)

    def test_invalid_train_fraction(self, small_rna_data, small_adt_data):
        with pytest.raises(ValueError, match="train_fraction must be in"):
            self._call(small_rna_data, small_adt_data, train_fraction=1.5)

    def test_fractions_sum_too_large(self, small_rna_data, small_adt_data):
        with pytest.raises(ValueError, match="must be < 1"):
            self._call(small_rna_data, small_adt_data, train_fraction=0.9, val_fraction=0.2)

    def test_invalid_aml_labels_length(self, small_rna_data, small_adt_data):
        with pytest.raises(ValueError, match="aml_labels length"):
            self._call(small_rna_data, small_adt_data, aml_labels=np.array([0, 1]))

    def test_non_binary_aml_labels(self, small_rna_data, small_adt_data):
        labels = np.zeros(100, dtype=int)
        labels[0] = 5
        with pytest.raises(ValueError, match="binary"):
            self._call(small_rna_data, small_adt_data, aml_labels=labels)

    def test_negative_celltype_labels(self, small_rna_data, small_adt_data):
        labels = np.zeros(100, dtype=int)
        labels[0] = -1
        with pytest.raises(ValueError, match="non-negative"):
            self._call(small_rna_data, small_adt_data, celltype_labels=labels)

    def test_invalid_dropout(self, small_rna_data, small_adt_data):
        with pytest.raises(ValueError, match="dropout_rate"):
            self._call(small_rna_data, small_adt_data, dropout_rate=1.5)

    def test_invalid_learning_rate(self, small_rna_data, small_adt_data):
        with pytest.raises(ValueError, match="learning_rate"):
            self._call(small_rna_data, small_adt_data, learning_rate=0.0)

    def test_invalid_epochs(self, small_rna_data, small_adt_data):
        with pytest.raises(ValueError, match="epochs"):
            self._call(small_rna_data, small_adt_data, epochs=0)


# ---------------------------------------------------------------------------
# TestLossComputation
# ---------------------------------------------------------------------------


class TestLossComputation:
    def test_multi_task_loss_positive(self):
        adt_pred = torch.randn(10, 5)
        adt_target = torch.randn(10, 5)
        aml_pred = torch.randn(10, 1)
        aml_target = torch.randint(0, 2, (10,)).float()

        total, adt, aml = compute_multi_task_loss(
            adt_pred, adt_target, aml_pred, aml_target
        )
        assert total.item() > 0

    def test_multi_task_loss_weighting(self):
        adt_pred = torch.randn(10, 5)
        adt_target = torch.randn(10, 5)
        aml_pred = torch.randn(10, 1)
        aml_target = torch.randint(0, 2, (10,)).float()

        total, adt_l, aml_l = compute_multi_task_loss(
            adt_pred, adt_target, aml_pred, aml_target,
            adt_weight=2.0, classification_weight=0.5
        )
        expected = 2.0 * adt_l + 0.5 * aml_l
        assert torch.isclose(total, expected)

    def test_classification_metrics_perfect(self):
        pred = torch.full((20, 1), 10.0)  # Strong positive logit
        target = torch.ones(20)
        metrics = compute_classification_metrics(pred, target)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0


# ---------------------------------------------------------------------------
# TestLossScheduling
# ---------------------------------------------------------------------------


class TestLossScheduling:
    def test_decay_schedule(self):
        lam, _ = _compute_loss_weights(50, 100, "decay", 0.1, "constant", 1.0)
        assert abs(lam - 0.05) < 1e-6

    def test_constant_schedule(self):
        lam, _ = _compute_loss_weights(50, 100, "constant", 0.1, "constant", 1.0)
        assert abs(lam - 0.1) < 1e-6

    def test_warmup_schedule(self):
        lam, _ = _compute_loss_weights(50, 100, "warmup", 0.1, "constant", 1.0)
        assert abs(lam - 0.05) < 1e-6

    def test_celltype_warmup(self):
        _, cell = _compute_loss_weights(50, 100, "constant", 0.1, "warmup", 1.0)
        assert abs(cell - 0.5) < 1e-6

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown reg_weight_schedule"):
            _compute_loss_weights(1, 100, "invalid", 0.1, "constant", 1.0)

    def test_unknown_celltype_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown celltype_weight_schedule"):
            _compute_loss_weights(1, 100, "constant", 0.1, "invalid", 1.0)


# ---------------------------------------------------------------------------
# TestCorrelationComputation
# ---------------------------------------------------------------------------


class TestCorrelationComputation:
    def test_perfect_correlation(self):
        arr = np.arange(1, 101).reshape(-1, 1).astype(float)
        pearson, spearman = _compute_correlations_vectorized(arr, arr)
        assert abs(pearson - 1.0) < 1e-5
        assert abs(spearman - 1.0) < 1e-5

    def test_zero_std_returns_nan(self):
        arr = np.ones((100, 2))
        pearson, spearman = _compute_correlations_vectorized(arr, arr)
        assert np.isnan(pearson)
        assert np.isnan(spearman)

    def test_multiple_features(self):
        rng = np.random.RandomState(0)
        target = rng.randn(200, 10)
        pred = target + 0.05 * rng.randn(200, 10)
        pearson, spearman = _compute_correlations_vectorized(target, pred)
        assert 0.9 < pearson <= 1.0
        assert 0.9 < spearman <= 1.0

    def test_negative_correlation(self):
        arr = np.arange(1, 101).reshape(-1, 1).astype(float)
        neg = -arr
        pearson, _ = _compute_correlations_vectorized(arr, neg)
        assert pearson < -0.9


# ---------------------------------------------------------------------------
# TestNormalizationParams
# ---------------------------------------------------------------------------


class TestNormalizationParams:
    def test_denormalize(self):
        mean = torch.tensor([[1.0, 2.0]])
        std = torch.tensor([[0.5, 1.0]])
        norm = NormalizationParams(adt_mean=mean, adt_std=std)
        normalized = torch.zeros(3, 2)
        result = norm.denormalize(normalized)
        assert torch.allclose(result, mean.expand(3, 2))

    def test_round_trip(self):
        mean = torch.tensor([[1.0, 2.0]])
        std = torch.tensor([[0.5, 1.0]])
        norm = NormalizationParams(adt_mean=mean, adt_std=std)
        original = torch.randn(10, 2)
        assert torch.allclose(norm.denormalize(norm.normalize(original)), original, atol=1e-5)


# ---------------------------------------------------------------------------
# TestRegressionMetrics
# ---------------------------------------------------------------------------


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        arr = np.random.randn(50, 5)
        mse, rmse, mae, r2 = _compute_regression_metrics(arr, arr)
        assert mse < 1e-10
        assert rmse < 1e-5
        assert r2 > 0.999

    def test_returns_floats(self):
        arr = np.random.randn(50, 5)
        metrics = _compute_regression_metrics(arr, arr)
        for v in metrics:
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# TestTrainingResult
# ---------------------------------------------------------------------------


class TestTrainingResult:
    def _make_result(self, val_r2_list, test_r2_list):
        return TrainingResult(
            model=Mock(),
            rna_data=Mock(),
            adt_data=Mock(),
            history={"val_R2": val_r2_list, "test_R2": test_r2_list},
            normalization=Mock(),
            graph_stats=Mock(),
        )

    def test_get_best_val_r2(self):
        result = self._make_result([0.5, 0.8, 0.7], [0.4, 0.75, 0.7])
        assert result.get_best_val_r2() == 0.8

    def test_get_final_test_r2(self):
        result = self._make_result([0.5, 0.8], [0.4, 0.75])
        assert result.get_final_test_r2() == 0.75


# ---------------------------------------------------------------------------
# TestGetNanMetrics
# ---------------------------------------------------------------------------


class TestGetNanMetrics:
    def test_all_nan(self):
        metrics = _get_nan_metrics()
        for v in metrics.values():
            assert np.isnan(v)

    def test_expected_keys(self):
        metrics = _get_nan_metrics()
        assert "MSE" in metrics
        assert "R2" in metrics
        assert "AML_Accuracy" in metrics
        assert "CellType_Accuracy" in metrics


# ---------------------------------------------------------------------------
# TestDeviceSwitchRequired
# ---------------------------------------------------------------------------


class TestDeviceSwitchRequired:
    def test_is_exception(self):
        exc = DeviceSwitchRequired("test message")
        assert isinstance(exc, Exception)
        assert "test message" in str(exc)


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_caps_are_reasonable(self):
        assert MAX_HIDDEN_CHANNELS > 0
        assert MAX_ATTENTION_HEADS > 0
        assert MAX_TRANSFORMER_LAYERS > 0

    def test_default_reg_weight_positive(self):
        assert DEFAULT_REG_WEIGHT > 0
