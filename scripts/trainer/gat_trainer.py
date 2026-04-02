"""
GAT Transformer Fusion Training Module

This module provides the main training pipeline for DeepOMAPNet,
a Graph Attention Network with Transformer Fusion for RNA-to-ADT mapping
in single-cell CITE-seq data.

Key Features:
    - Multi-task learning: ADT prediction + AML classification + cell type classification
    - Graph attention networks with transformer fusion
    - Automatic mixed precision training
    - Early stopping with best model restoration
    - Stratified data splitting
    - Comprehensive error handling and OOM recovery

Basic Usage:
    >>> from scripts.trainer.gat_trainer import train_gat_transformer_fusion
    >>>
    >>> result = train_gat_transformer_fusion(
    ...     rna_data=rna_pyg,
    ...     adt_data=adt_pyg,
    ...     epochs=100,
    ...     learning_rate=1e-3,
    ...     hidden_channels=64,
    ...     num_heads=4
    ... )
    >>> print(f"Best validation R²: {result.get_best_val_r2():.4f}")
    >>> print(f"Final test R²: {result.get_final_test_r2():.4f}")
    >>> predictions = result.predict_adt(new_rna_features)

Advanced Usage with Multi-Task Learning:
    >>> result = train_gat_transformer_fusion(
    ...     rna_data=rna_pyg,
    ...     adt_data=adt_pyg,
    ...     aml_labels=aml_labels,
    ...     classification_weight=0.5,
    ...     epochs=100
    ... )

Memory-Constrained Training:
    >>> result = train_gat_transformer_fusion(
    ...     rna_data=rna_pyg,
    ...     adt_data=adt_pyg,
    ...     gradient_accumulation_steps=4,
    ...     hidden_channels=32,
    ...     use_cpu_fallback=True,
    ...     epochs=100
    ... )

Exceptions:
    DeviceSwitchRequired: Raised when GPU OOM requires CPU fallback
    ValueError: Raised for invalid input parameters
    TypeError: Raised for incorrect input types
"""

import gc
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import rankdata

# ============================================================================
# TRAINING CONSTANTS
# ============================================================================

# Memory Management
DEFAULT_GPU_MEMORY_FRACTION = 0.5
FALLBACK_GPU_MEMORY_FRACTION = 0.3

# Gradient Clipping
MAX_GRAD_NORM = 1.0

# Regularization
DEFAULT_REG_WEIGHT = 0.05

# Numerical stability
EPSILON = 1e-8

# GPU Memory Caps (for safety)
MAX_HIDDEN_CHANNELS = 64
MAX_ATTENTION_HEADS = 4
MAX_TRANSFORMER_LAYERS = 3

# Model Architecture Defaults
DEFAULT_HIDDEN_CHANNELS = 96
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_ATTENTION_HEADS = 8
DEFAULT_NUM_LAYERS = 3
DEFAULT_DROPOUT = 0.4

# Training Defaults
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 200
DEFAULT_EARLY_STOPPING_PATIENCE = 20

# Data Splits
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_VAL_FRACTION = 0.1

# Evaluation
MIN_STD_THRESHOLD = 1e-8

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class DeviceSwitchRequired(Exception):
    """Raised when training needs to switch to CPU due to OOM or dtype error."""


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class NormalizationParams:
    """Parameters for denormalizing ADT predictions."""

    adt_mean: torch.Tensor
    adt_std: torch.Tensor

    def denormalize(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """Denormalize ADT data."""
        return normalized_data * self.adt_std + self.adt_mean

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize ADT data."""
        return (data - self.adt_mean) / self.adt_std


@dataclass
class GraphStatistics:
    """Graph topology statistics for positional encoding."""

    node_degrees_rna: torch.Tensor
    node_degrees_adt: torch.Tensor
    clustering_coeffs_rna: torch.Tensor
    clustering_coeffs_adt: torch.Tensor


@dataclass
class TrainingResult:
    """Complete result from GAT Transformer Fusion training."""

    model: torch.nn.Module
    rna_data: object
    adt_data: object
    history: Dict[str, list]
    normalization: NormalizationParams
    graph_stats: GraphStatistics

    def get_best_val_r2(self) -> float:
        """Get the best validation R² achieved during training."""
        return max(self.history["val_R2"])

    def get_final_test_r2(self) -> float:
        """Get the final test R²."""
        return self.history["test_R2"][-1]

    def predict_adt(
        self, rna_features: torch.Tensor, denormalize: bool = True
    ) -> torch.Tensor:
        """Make ADT predictions and optionally denormalize."""
        self.model.eval()
        with torch.no_grad():
            pred, _, _ = self.model(
                x=rna_features,
                edge_index_rna=self.rna_data.edge_index,
                edge_index_adt=(
                    self.adt_data.edge_index
                    if hasattr(self.adt_data, "edge_index")
                    else None
                ),
                node_degrees_rna=self.graph_stats.node_degrees_rna,
                node_degrees_adt=self.graph_stats.node_degrees_adt,
                clustering_coeffs_rna=self.graph_stats.clustering_coeffs_rna,
                clustering_coeffs_adt=self.graph_stats.clustering_coeffs_adt,
            )
        if denormalize:
            pred = self.normalization.denormalize(pred)
        return pred

    def save(self, path: str) -> None:
        """Save training result to disk."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "history": self.history,
                "normalization": {
                    "adt_mean": self.normalization.adt_mean,
                    "adt_std": self.normalization.adt_std,
                },
                "graph_stats": {
                    "node_degrees_rna": self.graph_stats.node_degrees_rna,
                    "node_degrees_adt": self.graph_stats.node_degrees_adt,
                    "clustering_coeffs_rna": self.graph_stats.clustering_coeffs_rna,
                    "clustering_coeffs_adt": self.graph_stats.clustering_coeffs_adt,
                },
            },
            path,
        )

    def __iter__(self):
        """Support legacy 10-value tuple unpacking.

        Yields values in the same order as the original return tuple:
        model, rna_data, adt_data, history,
        adt_mean, adt_std,
        node_degrees_rna, node_degrees_adt,
        clustering_coeffs_rna, clustering_coeffs_adt
        """
        yield self.model
        yield self.rna_data
        yield self.adt_data
        yield self.history
        yield self.normalization.adt_mean
        yield self.normalization.adt_std
        yield self.graph_stats.node_degrees_rna
        yield self.graph_stats.node_degrees_adt
        yield self.graph_stats.clustering_coeffs_rna
        yield self.graph_stats.clustering_coeffs_adt


# ============================================================================
# PUBLIC API
# ============================================================================


def compute_multi_task_loss(
    adt_pred: torch.Tensor,
    adt_target: torch.Tensor,
    aml_pred: torch.Tensor,
    aml_target: torch.Tensor,
    adt_weight: float = 1.0,
    classification_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined loss for ADT prediction and AML classification tasks.

    Args:
        adt_pred: ADT predictions [N, num_markers]
        adt_target: ADT targets [N, num_markers]
        aml_pred: AML predictions [N, 1]
        aml_target: AML labels [N] (0=Normal, 1=AML)
        adt_weight: Weight for ADT regression loss
        classification_weight: Weight for classification loss

    Returns:
        Tuple of (total_loss, adt_loss, aml_loss)
    """
    adt_loss = F.mse_loss(adt_pred, adt_target)
    aml_loss = F.binary_cross_entropy_with_logits(
        aml_pred.squeeze(), aml_target.float()
    )
    total_loss = adt_weight * adt_loss + classification_weight * aml_loss
    return total_loss, adt_loss, aml_loss


def compute_classification_metrics(
    aml_pred: torch.Tensor, aml_target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute classification metrics for AML prediction.

    Args:
        aml_pred: AML predictions [N, 1] (logits)
        aml_target: AML labels [N] (0=Normal, 1=AML)

    Returns:
        Dictionary of classification metrics
    """
    aml_probs = torch.sigmoid(aml_pred).cpu().numpy().squeeze()
    aml_pred_binary = (aml_probs > 0.5).astype(int)
    aml_target_np = aml_target.cpu().numpy()

    accuracy = accuracy_score(aml_target_np, aml_pred_binary)
    precision = precision_score(aml_target_np, aml_pred_binary, zero_division=0)
    recall = recall_score(aml_target_np, aml_pred_binary, zero_division=0)
    f1 = f1_score(aml_target_np, aml_pred_binary, zero_division=0)

    try:
        auc_roc = roc_auc_score(aml_target_np, aml_probs)
    except ValueError:
        auc_roc = 0.5

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
    }


def setup_training_logger(
    log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """Configure logger for training.

    Args:
        log_file: Optional path to save training logs.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def train_gat_transformer_fusion(
    rna_data,
    adt_data,
    aml_labels=None,
    rna_anndata=None,
    adt_anndata=None,
    epochs: int = DEFAULT_EPOCHS,
    use_cpu_fallback: bool = False,
    seed: int = 42,
    stratify_labels: Optional[np.ndarray] = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    adt_weight: float = 1.0,
    classification_weight: float = 1.0,
    dropout_rate: float = DEFAULT_DROPOUT,
    hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
    num_heads: int = DEFAULT_NUM_HEADS,
    num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
    num_layers: int = DEFAULT_NUM_LAYERS,
    use_mixed_precision: bool = True,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    num_cell_types: Optional[int] = None,
    celltype_labels: Optional[np.ndarray] = None,
    celltype_weight: float = 1.0,
    use_neighbor_sampling: bool = True,
    batch_size: int = 2048,
    num_workers: int = 4,
    gradient_accumulation_steps: int = 1,
    reg_weight_schedule: str = "decay",
    reg_weight_init: float = DEFAULT_REG_WEIGHT,
    celltype_weight_schedule: str = "constant",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
) -> TrainingResult:
    """
    Train GAT Transformer Fusion model for RNA-to-ADT mapping with multi-task learning.

    Args:
        rna_data: PyTorch Geometric Data with RNA features and graph structure.
            Must have attributes: x, edge_index, num_nodes.
        adt_data: PyTorch Geometric Data with ADT targets.
            Must have attributes: x, num_nodes (must match rna_data).
        aml_labels: Binary labels for AML classification (0=Normal, 1=AML).
            Shape: (num_nodes,). If provided, enables AML classification.
        rna_anndata: AnnData for RNA preprocessing. If None, uses rna_data.x directly.
        adt_anndata: AnnData for ADT preprocessing (should be CLR-normalized).
            If None, uses adt_data.x directly.
        epochs: Number of training epochs. Must be >= 1.
        use_cpu_fallback: Automatically fall back to CPU on GPU OOM.
        seed: Random seed for reproducibility.
        stratify_labels: Labels for stratified train/val/test splitting.
        train_fraction: Fraction of data for training. Must be in (0, 1).
        val_fraction: Fraction for validation. train_fraction + val_fraction < 1.
        learning_rate: Learning rate for AdamW. Must be > 0.
        weight_decay: L2 weight decay for AdamW. Must be >= 0.
        adt_weight: Weight for ADT regression loss.
        classification_weight: Weight for AML classification loss.
        dropout_rate: Dropout rate. Must be in [0, 1).
        hidden_channels: Hidden channels (capped at MAX_HIDDEN_CHANNELS=32 with warning).
        num_heads: GAT attention heads (capped at MAX_ATTENTION_HEADS=2 with warning).
        num_attention_heads: Transformer attention heads (capped at 2 with warning).
        num_layers: Transformer layers (capped at MAX_TRANSFORMER_LAYERS=1 with warning).
        use_mixed_precision: Use AMP training. Disabled automatically on dtype errors.
        early_stopping_patience: Evaluation checkpoints without improvement before stop.
        num_cell_types: Number of cell types. Required if celltype_labels is provided.
        celltype_labels: Integer cell type labels. Shape: (num_nodes,).
        celltype_weight: Weight for cell type classification loss.
        gradient_accumulation_steps: Steps to accumulate gradients (reduces memory).
        reg_weight_schedule: Regularization schedule: 'decay', 'constant', or 'warmup'.
        reg_weight_init: Initial regularization weight.
        celltype_weight_schedule: Cell type loss schedule: 'constant', 'warmup', 'decay'.
        log_file: Optional path to save training logs.
        log_level: Logging level (logging.DEBUG, INFO, WARNING, ERROR).

    Returns:
        TrainingResult with trained model, data, history, and helper methods.

    Raises:
        TypeError: If rna_data or adt_data are not PyTorch Geometric Data objects.
        ValueError: If input validation fails.
        RuntimeError: If CUDA errors occur that cannot be recovered.

    Examples:
        Basic usage:
            >>> result = train_gat_transformer_fusion(rna_data=rna_pyg, adt_data=adt_pyg, epochs=100)
            >>> print(f"Test R²: {result.get_final_test_r2():.4f}")

        Multi-task learning:
            >>> result = train_gat_transformer_fusion(
            ...     rna_data=rna_pyg, adt_data=adt_pyg,
            ...     aml_labels=aml_labels,
            ...     celltype_labels=celltype_labels, num_cell_types=10,
            ...     epochs=100
            ... )
    """
    setup_training_logger(log_file, log_level)
    logger.info(f"Starting GAT training: {rna_data.num_nodes:,} nodes, {rna_data.num_edges:,} edges")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_per_process_memory_fraction(DEFAULT_GPU_MEMORY_FRACTION)

    initial_device = torch.device(
        "cuda" if torch.cuda.is_available() and not use_cpu_fallback else "cpu"
    )

    _validate_inputs(
        rna_data,
        adt_data,
        rna_anndata,
        adt_anndata,
        aml_labels,
        celltype_labels,
        stratify_labels,
        train_fraction,
        val_fraction,
        epochs,
        learning_rate,
        weight_decay,
        dropout_rate,
    )

    rna_input_dim = rna_anndata.shape[1] if rna_anndata is not None else rna_data.x.size(1)
    num_nodes = rna_data.num_nodes

    train_mask, val_mask, test_mask = _create_data_splits(
        num_nodes, stratify_labels, train_fraction, val_fraction, seed
    )
    rna_data.train_mask = train_mask
    rna_data.val_mask = val_mask
    rna_data.test_mask = test_mask

    if rna_anndata is not None:
        rna_input_dim = _preprocess_rna_data(rna_data, rna_anndata)

    adt_mean, adt_std = _preprocess_adt_data(adt_data, adt_anndata)
    adt_output_dim = adt_data.x.size(1)

    if aml_labels is not None:
        if isinstance(aml_labels, torch.Tensor):
            aml_labels_np = aml_labels.cpu().numpy()
        else:
            aml_labels_np = np.array(aml_labels)
        aml_labels = aml_labels_np

    # Apply caps with warnings (Phase 2.1)
    if hidden_channels > MAX_HIDDEN_CHANNELS:
        logger.warning(
            f"Reducing hidden_channels from {hidden_channels} to {MAX_HIDDEN_CHANNELS} "
            "for GPU memory constraints"
        )
        hidden_channels = MAX_HIDDEN_CHANNELS

    if num_heads > MAX_ATTENTION_HEADS:
        logger.warning(
            f"Reducing num_heads from {num_heads} to {MAX_ATTENTION_HEADS} "
            "for GPU memory constraints"
        )
        num_heads = MAX_ATTENTION_HEADS

    if num_attention_heads > MAX_ATTENTION_HEADS:
        logger.warning(
            f"Reducing num_attention_heads from {num_attention_heads} to {MAX_ATTENTION_HEADS} "
            "for GPU memory constraints"
        )
        num_attention_heads = MAX_ATTENTION_HEADS

    if num_layers > MAX_TRANSFORMER_LAYERS:
        logger.warning(
            f"Reducing num_layers from {num_layers} to {MAX_TRANSFORMER_LAYERS} "
            "for GPU memory constraints"
        )
        num_layers = MAX_TRANSFORMER_LAYERS

    logger.info(
        f"Training config: epochs={epochs}, lr={learning_rate}, "
        f"hidden_channels={hidden_channels}, num_heads={num_heads}, "
        f"num_layers={num_layers}"
    )

    # Determine final device before model init (Phase 1.1)
    device = _determine_device(rna_data, initial_device)

    model = _initialize_model(
        rna_input_dim,
        adt_output_dim,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_attention_heads=num_attention_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        device=device,
        num_cell_types=num_cell_types,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    rna_data, adt_data, device = _move_data_to_device(rna_data, adt_data, device)
    model = model.to(device)

    adt_mean = adt_mean.to(device)
    adt_std = adt_std.to(device)

    if aml_labels is not None:
        aml_labels = torch.tensor(
            aml_labels.astype(np.float32), dtype=torch.float32, device=device
        )

    if celltype_labels is not None:
        celltype_labels = torch.tensor(celltype_labels, dtype=torch.long, device=device)

    node_degrees_rna, clustering_coeffs_rna = _compute_graph_statistics(
        rna_data.edge_index, num_nodes
    )
    node_degrees_adt, clustering_coeffs_adt = _compute_graph_statistics(
        adt_data.edge_index if hasattr(adt_data, "edge_index") else rna_data.edge_index,
        num_nodes,
    )

    # Attach stats to data objects for loader to pick up (Phase 2.3 - Scaling)
    rna_data.node_degrees = node_degrees_rna.cpu()
    rna_data.clustering_coeffs = clustering_coeffs_rna.cpu()
    adt_data.node_degrees = node_degrees_adt.cpu()
    adt_data.clustering_coeffs = clustering_coeffs_adt.cpu()
    if aml_labels is not None:
        rna_data.aml_labels = aml_labels.cpu()
    if celltype_labels is not None:
        rna_data.celltype_labels = celltype_labels.cpu()

    optimizer, scheduler, criterion, scaler = _setup_training_components(
        model, learning_rate, weight_decay, use_mixed_precision, device
    )

    # Setup Neighbor Sampling Loaders (Phase 2.3)
    train_loader = None
    if use_neighbor_sampling:
        try:
            # NeighborLoader's worker processes require pyg-lib or torch-sparse.
            # Probe for either before constructing the loader so we can fall back cleanly.
            try:
                import pyg_lib  # noqa: F401
                _has_backend = True
            except ImportError:
                try:
                    import torch_sparse  # noqa: F401
                    _has_backend = True
                except ImportError:
                    _has_backend = False

            if not _has_backend:
                raise ImportError(
                    "'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'"
                )

            from torch_geometric.loader import NeighborLoader
            logger.info(f"Setting up NeighborLoader with batch_size={batch_size}")
            # Attach ADT features onto the RNA graph so each mini-batch carries
            # the correct ADT targets for its sampled nodes.
            rna_data.adt_x = adt_data.x
            train_loader = NeighborLoader(
                rna_data,
                num_neighbors=[15, 10], # Sample for 2 GAT layers
                batch_size=batch_size,
                input_nodes=rna_data.train_mask,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False
            )
        except (ImportError, Exception) as e:
            logger.warning(
                f"NeighborLoader unavailable ({e}); falling back to full-graph training. "
                "Install 'pyg-lib' or 'torch-sparse' to enable neighbor sampling."
            )
            train_loader = None

    training_history = _run_training_loop(
        model=model,
        rna_data=rna_data,
        adt_data=adt_data,
        train_loader=train_loader, # New loader arg
        aml_labels=aml_labels,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        scaler=scaler,
        node_degrees_rna=node_degrees_rna,
        node_degrees_adt=node_degrees_adt,
        clustering_coeffs_rna=clustering_coeffs_rna,
        clustering_coeffs_adt=clustering_coeffs_adt,
        adt_mean=adt_mean,
        adt_std=adt_std,
        adt_weight=adt_weight,
        classification_weight=classification_weight,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        use_mixed_precision=use_mixed_precision,
        device=device,
        celltype_labels=celltype_labels,
        celltype_weight=celltype_weight,
        gradient_accumulation_steps=gradient_accumulation_steps,
        reg_weight_schedule=reg_weight_schedule,
        reg_weight_init=reg_weight_init,
        celltype_weight_schedule=celltype_weight_schedule,
    )

    _log_final_metrics(
        model,
        rna_data,
        adt_data,
        aml_labels,
        adt_mean,
        adt_std,
        node_degrees_rna,
        node_degrees_adt,
        clustering_coeffs_rna,
        clustering_coeffs_adt,
        use_mixed_precision,
        device,
    )

    return TrainingResult(
        model=model,
        rna_data=rna_data,
        adt_data=adt_data,
        history=training_history,
        normalization=NormalizationParams(adt_mean=adt_mean, adt_std=adt_std),
        graph_stats=GraphStatistics(
            node_degrees_rna=node_degrees_rna,
            node_degrees_adt=node_degrees_adt,
            clustering_coeffs_rna=clustering_coeffs_rna,
            clustering_coeffs_adt=clustering_coeffs_adt,
        ),
    )


# ============================================================================
# VALIDATION
# ============================================================================


def _validate_inputs(
    rna_data,
    adt_data,
    rna_anndata,
    adt_anndata,
    aml_labels,
    celltype_labels,
    stratify_labels,
    train_fraction: float,
    val_fraction: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dropout_rate: float,
) -> None:
    """Validate all input parameters before training.

    Raises:
        TypeError: For incorrect input types.
        ValueError: For invalid parameter values or shape mismatches.
    """
    from torch_geometric.data import Data

    if not isinstance(rna_data, Data):
        raise TypeError(
            f"rna_data must be torch_geometric.data.Data, got {type(rna_data)}"
        )
    if not isinstance(adt_data, Data):
        raise TypeError(
            f"adt_data must be torch_geometric.data.Data, got {type(adt_data)}"
        )
    if rna_data.x is None:
        raise ValueError("rna_data.x (RNA features) cannot be None")
    if adt_data.x is None:
        raise ValueError("adt_data.x (ADT features) cannot be None")
    if not hasattr(rna_data, "edge_index") or rna_data.edge_index is None:
        raise ValueError("rna_data must have edge_index attribute")
    if rna_data.num_nodes != adt_data.num_nodes:
        raise ValueError(
            f"RNA and ADT must have same number of nodes. "
            f"Got RNA: {rna_data.num_nodes}, ADT: {adt_data.num_nodes}"
        )

    num_nodes = rna_data.num_nodes

    if rna_anndata is not None and rna_anndata.shape[0] != num_nodes:
        raise ValueError(
            f"rna_anndata has {rna_anndata.shape[0]} cells but rna_data has {num_nodes} nodes"
        )
    if adt_anndata is not None and adt_anndata.shape[0] != num_nodes:
        raise ValueError(
            f"adt_anndata has {adt_anndata.shape[0]} cells but adt_data has {num_nodes} nodes"
        )

    if aml_labels is not None:
        aml_arr = (
            aml_labels.cpu().numpy()
            if torch.is_tensor(aml_labels)
            else np.array(aml_labels)
        )
        if len(aml_arr) != num_nodes:
            raise ValueError(
                f"aml_labels length ({len(aml_arr)}) must match number of nodes ({num_nodes})"
            )
        unique_labels = np.unique(aml_arr)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                f"aml_labels must be binary (0/1), got unique values: {unique_labels}"
            )

    if celltype_labels is not None:
        ct_arr = (
            celltype_labels.cpu().numpy()
            if torch.is_tensor(celltype_labels)
            else np.array(celltype_labels)
        )
        if len(ct_arr) != num_nodes:
            raise ValueError(
                f"celltype_labels length ({len(ct_arr)}) must match number of nodes ({num_nodes})"
            )
        if ct_arr.min() < 0:
            raise ValueError("celltype_labels must be non-negative integers")

    if stratify_labels is not None and len(stratify_labels) != num_nodes:
        raise ValueError(
            f"stratify_labels length ({len(stratify_labels)}) must match "
            f"number of nodes ({num_nodes})"
        )

    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if train_fraction + val_fraction >= 1:
        raise ValueError(
            f"train_fraction ({train_fraction}) + val_fraction ({val_fraction}) "
            "must be < 1 to leave data for testing"
        )
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if not 0 <= dropout_rate < 1:
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

    logger.info("Input validation passed")


# ============================================================================
# DEVICE MANAGEMENT (Phase 1.1)
# ============================================================================


def _determine_device(rna_data, initial_device: torch.device) -> torch.device:
    """Determine final device by attempting a small data transfer.

    Args:
        rna_data: RNA data object used for GPU memory probe.
        initial_device: The requested device.

    Returns:
        The final device to use (may fall back to CPU on OOM).
    """
    if initial_device.type != "cuda":
        return initial_device

    try:
        _test = rna_data.x[:100].to(initial_device)
        del _test
        torch.cuda.empty_cache()
        return initial_device
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            logger.warning("GPU OOM detected during device check, falling back to CPU")
            return torch.device("cpu")
        raise


def _move_data_to_device(
    rna_data, adt_data, device: torch.device
) -> Tuple[object, object, torch.device]:
    """Move data objects to device. Model should already be on device.

    Args:
        rna_data: RNA PyG Data object.
        adt_data: ADT PyG Data object.
        device: Target device.

    Returns:
        Tuple of (rna_data, adt_data, final_device).
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            rna_data = rna_data.to(device)
            torch.cuda.empty_cache()
            adt_data = adt_data.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.set_per_process_memory_fraction(FALLBACK_GPU_MEMORY_FRACTION)

                try:
                    rna_data = rna_data.to(device)
                    adt_data = adt_data.to(device)
                except RuntimeError as e2:
                    if "out of memory" in str(e2).lower():
                        logger.warning("GPU OOM during data transfer, falling back to CPU")
                        device = torch.device("cpu")
                        rna_data = rna_data.cpu()
                        adt_data = adt_data.cpu()
                    else:
                        raise
            else:
                raise
    else:
        rna_data = rna_data.to(device)
        adt_data = adt_data.to(device)

    return rna_data, adt_data, device


# ============================================================================
# DATA HELPERS
# ============================================================================


def _create_data_splits(
    num_nodes: int,
    stratify_labels: Optional[np.ndarray],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create train/validation/test masks.

    Args:
        num_nodes: Total number of graph nodes.
        stratify_labels: Optional labels for stratified splitting.
        train_fraction: Fraction for training set.
        val_fraction: Fraction for validation set.
        seed: Random seed.

    Returns:
        Tuple of boolean masks (train_mask, val_mask, test_mask).
    """
    indices = np.arange(num_nodes)

    if stratify_labels is not None:
        from sklearn.model_selection import StratifiedShuffleSplit

        stratify_labels = np.asarray(stratify_labels)
        sss1 = StratifiedShuffleSplit(
            n_splits=1, train_size=train_fraction, random_state=seed
        )
        train_idx, rest_idx = next(sss1.split(indices, stratify_labels))

        rest_labels = stratify_labels[rest_idx]
        val_size = int(val_fraction * num_nodes)
        sss2 = StratifiedShuffleSplit(
            n_splits=1, train_size=val_size / len(rest_idx), random_state=seed
        )
        val_rel, test_rel = next(sss2.split(rest_idx, rest_labels))
        val_idx = rest_idx[val_rel]
        test_idx = rest_idx[test_rel]
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        n_train = int(train_fraction * num_nodes)
        n_val = int(val_fraction * num_nodes)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def _initialize_model(
    rna_input_dim: int,
    adt_output_dim: int,
    hidden_channels: int,
    num_heads: int,
    num_attention_heads: int,
    num_layers: int,
    dropout_rate: float,
    device: torch.device,
    num_cell_types: Optional[int] = None,
) -> torch.nn.Module:
    """Initialize the GATWithTransformerFusion model.

    Args:
        rna_input_dim: Input feature dimension from RNA.
        adt_output_dim: Output dimension for ADT predictions.
        hidden_channels: Hidden channel size.
        num_heads: GAT attention heads.
        num_attention_heads: Transformer attention heads.
        num_layers: Number of transformer layers.
        dropout_rate: Dropout rate.
        device: Target device.
        num_cell_types: Optional number of cell types for classification head.

    Returns:
        Initialized model on the given device.
    """
    from scripts.model.doNET import GATWithTransformerFusion

    model = GATWithTransformerFusion(
        in_channels=rna_input_dim,
        hidden_channels=hidden_channels,
        out_channels=adt_output_dim,
        heads=num_heads,
        dropout=dropout_rate,
        nhead=num_attention_heads,
        num_layers=num_layers,
        use_adapters=True,
        reduction_factor=4,
        adapter_l2_reg=5e-5,
        use_positional_encoding=True,
        num_cell_types=num_cell_types,
    ).to(device)

    return model


def _preprocess_rna_data(rna_data, rna_anndata) -> int:
    """Preprocess RNA data using AnnData object.

    Args:
        rna_data: PyTorch Geometric data object.
        rna_anndata: AnnData object for RNA preprocessing.

    Returns:
        Updated RNA input dimension.
    """
    if hasattr(rna_anndata.X, "toarray"):
        rna_tensor = torch.tensor(rna_anndata.X.toarray(), dtype=torch.float32)
    else:
        rna_tensor = torch.tensor(rna_anndata.X, dtype=torch.float32)

    rna_data.x = rna_tensor
    return rna_data.x.size(1)


def _preprocess_adt_data(
    adt_data, adt_anndata=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and prepare ADT data for training.

    Note: If data comes from data_preprocessing.py, it is already CLR + z-score normalized.

    Args:
        adt_data: PyTorch Geometric data object.
        adt_anndata: Optional AnnData for preprocessing. If None, uses adt_data.x directly.

    Returns:
        Tuple of (mean, std) tensors for denormalization.
    """
    if adt_anndata is not None:
        if hasattr(adt_anndata.X, "toarray"):
            adt_tensor = torch.tensor(adt_anndata.X.toarray(), dtype=torch.float32)
        else:
            adt_tensor = torch.tensor(adt_anndata.X, dtype=torch.float32)
        adt_data.x = adt_tensor

        adt_mean = adt_data.x.mean(dim=0, keepdim=True)
        adt_std = adt_data.x.std(dim=0, keepdim=True) + EPSILON

        if abs(adt_mean.mean().item()) < 0.01 and abs(adt_std.mean().item() - 1.0) < 0.01:
            return adt_mean, adt_std
    else:
        adt_mean = adt_data.x.mean(dim=0, keepdim=True)
        adt_std = adt_data.x.std(dim=0, keepdim=True) + EPSILON

    adt_data.x = (adt_data.x - adt_mean) / adt_std
    return adt_mean, adt_std


def _compute_graph_statistics(
    edge_index: torch.Tensor, num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute graph statistics for positional encoding.

    Args:
        edge_index: Edge index tensor of shape [2, num_edges].
        num_nodes: Number of graph nodes.

    Returns:
        Tuple of (node_degrees, clustering_coefficients).
    """
    from scripts.model.doNET import compute_graph_statistics_fast

    return compute_graph_statistics_fast(edge_index, num_nodes)


def _setup_training_components(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    use_mixed_precision: bool,
    device: torch.device,
) -> Tuple:
    """Setup optimizer, scheduler, loss function, and gradient scaler.

    Returns:
        Tuple of (optimizer, scheduler, criterion, scaler).
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(
        enabled=(use_mixed_precision and device.type == "cuda")
    )
    return optimizer, scheduler, criterion, scaler


# ============================================================================
# LOSS SCHEDULING (Phase 2.2)
# ============================================================================


def _compute_loss_weights(
    epoch: int,
    epochs: int,
    reg_schedule: str,
    reg_init: float,
    cell_schedule: str,
    cell_weight: float,
) -> Tuple[float, float]:
    """Compute loss weights based on scheduling strategy.

    Args:
        epoch: Current epoch (1-indexed).
        epochs: Total number of epochs.
        reg_schedule: 'decay', 'constant', or 'warmup' for regularization weight.
        reg_init: Initial regularization weight.
        cell_schedule: 'constant', 'warmup', or 'decay' for cell type weight.
        cell_weight: Base cell type classification weight.

    Returns:
        Tuple of (reg_lambda, cell_lambda).

    Raises:
        ValueError: If an unknown schedule string is provided.
    """
    progress = epoch / epochs

    if reg_schedule == "decay":
        reg_lambda = reg_init * (1 - progress)
    elif reg_schedule == "constant":
        reg_lambda = reg_init
    elif reg_schedule == "warmup":
        reg_lambda = reg_init * progress
    else:
        raise ValueError(f"Unknown reg_weight_schedule: {reg_schedule!r}")

    if cell_schedule == "constant":
        cell_lambda = cell_weight
    elif cell_schedule == "warmup":
        cell_lambda = cell_weight * progress
    elif cell_schedule == "decay":
        cell_lambda = cell_weight * (1 - progress)
    else:
        raise ValueError(f"Unknown celltype_weight_schedule: {cell_schedule!r}")

    return reg_lambda, cell_lambda


# ============================================================================
# TRAINING LOOP
# ============================================================================


def _run_training_loop(
    model: torch.nn.Module,
    rna_data,
    adt_data,
    train_loader: Optional[object], # New arg
    aml_labels: Optional[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: torch.nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    node_degrees_rna: torch.Tensor,
    node_degrees_adt: torch.Tensor,
    clustering_coeffs_rna: torch.Tensor,
    clustering_coeffs_adt: torch.Tensor,
    adt_mean: torch.Tensor,
    adt_std: torch.Tensor,
    adt_weight: float,
    classification_weight: float,
    epochs: int,
    early_stopping_patience: int,
    use_mixed_precision: bool,
    device: torch.device,
    celltype_labels: Optional[torch.Tensor],
    celltype_weight: float,
    gradient_accumulation_steps: int = 1,
    reg_weight_schedule: str = "decay",
    reg_weight_init: float = DEFAULT_REG_WEIGHT,
    celltype_weight_schedule: str = "constant",
) -> Dict[str, List]:
    """Run the main training loop with early stopping.

    Returns:
        Dictionary of training history keyed by metric name.
    """
    best_val_r2 = float("-inf")
    best_state = None
    bad_epochs = 0
    mixed_precision_disabled = False

    training_history: Dict[str, List] = {
        "epoch": [],
        "train_loss": [],
        "reg_loss": [],
        "aml_loss": [],
        "celltype_loss": [],
        "val_MSE": [],
        "val_R2": [],
        "test_MSE": [],
        "test_R2": [],
        "val_AML_Accuracy": [],
        "val_AML_F1": [],
        "test_AML_Accuracy": [],
        "test_AML_F1": [],
    }

    for epoch in range(1, epochs + 1):
        current_mixed_precision = use_mixed_precision and not mixed_precision_disabled

        try:
            adt_loss, reg_loss, aml_loss, celltype_ce = _training_step(
                model=model,
                rna_data=rna_data,
                adt_data=adt_data,
                train_loader=train_loader, # Pass loader
                aml_labels=aml_labels,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                node_degrees_rna=node_degrees_rna,
                node_degrees_adt=node_degrees_adt,
                clustering_coeffs_rna=clustering_coeffs_rna,
                clustering_coeffs_adt=clustering_coeffs_adt,
                adt_mean=adt_mean,
                adt_std=adt_std,
                adt_weight=adt_weight,
                classification_weight=classification_weight,
                epoch=epoch,
                epochs=epochs,
                use_mixed_precision=current_mixed_precision,
                device=device,
                celltype_labels=celltype_labels,
                celltype_weight=celltype_weight,
                gradient_accumulation_steps=gradient_accumulation_steps,
                reg_weight_schedule=reg_weight_schedule,
                reg_weight_init=reg_weight_init,
                celltype_weight_schedule=celltype_weight_schedule,
            )
        except DeviceSwitchRequired as exc:
            logger.warning(f"Switching to CPU: {exc}")
            device = torch.device("cpu")
            model = model.cpu()
            rna_data = rna_data.cpu()
            adt_data = adt_data.cpu()
            adt_mean = adt_mean.cpu()
            adt_std = adt_std.cpu()
            node_degrees_rna = node_degrees_rna.cpu()
            node_degrees_adt = node_degrees_adt.cpu()
            clustering_coeffs_rna = clustering_coeffs_rna.cpu()
            clustering_coeffs_adt = clustering_coeffs_adt.cpu()
            if aml_labels is not None:
                aml_labels = aml_labels.cpu()
            if celltype_labels is not None:
                celltype_labels = celltype_labels.cpu()
            mixed_precision_disabled = True
            continue  # Retry epoch on CPU

        if not current_mixed_precision and use_mixed_precision:
            mixed_precision_disabled = True

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            if device.type == "cuda":
                torch.cuda.empty_cache()

            eval_mixed = use_mixed_precision and not mixed_precision_disabled
            val_metrics = _evaluate_model(
                model, rna_data, adt_data, aml_labels, adt_mean, adt_std,
                node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
                rna_data.val_mask, current_mixed_precision, device,
                celltype_labels,
            )
            test_metrics = _evaluate_model(
                model, rna_data, adt_data, aml_labels, adt_mean, adt_std,
                node_degrees_rna, node_degrees_adt, clustering_coeffs_rna, clustering_coeffs_adt,
                rna_data.test_mask, current_mixed_precision, device,
                celltype_labels,
            )

            training_history["epoch"].append(epoch)
            training_history["train_loss"].append(adt_loss)
            training_history["reg_loss"].append(reg_loss)
            training_history["aml_loss"].append(aml_loss)
            training_history["celltype_loss"].append(celltype_ce)
            training_history["val_MSE"].append(val_metrics["MSE"])
            training_history["val_R2"].append(val_metrics["R2"])
            training_history["test_MSE"].append(test_metrics["MSE"])
            training_history["test_R2"].append(test_metrics["R2"])
            training_history["val_AML_Accuracy"].append(val_metrics["AML_Accuracy"])
            training_history["val_AML_F1"].append(val_metrics["AML_F1"])
            training_history["test_AML_Accuracy"].append(test_metrics["AML_Accuracy"])
            training_history["test_AML_F1"].append(test_metrics["AML_F1"])
            training_history.setdefault("val_CellType_Accuracy", []).append(
                val_metrics["CellType_Accuracy"]
            )
            training_history.setdefault("val_CellType_F1", []).append(
                val_metrics["CellType_F1"]
            )
            training_history.setdefault("test_CellType_Accuracy", []).append(
                test_metrics["CellType_Accuracy"]
            )
            training_history.setdefault("test_CellType_F1", []).append(
                test_metrics["CellType_F1"]
            )

            cell_val_acc = val_metrics["CellType_Accuracy"]
            cell_test_acc = test_metrics["CellType_Accuracy"]
            logger.info(
                f"Epoch {epoch:03d} | "
                f"ADT Loss {adt_loss:.6f} AML Loss {aml_loss:.6f} Reg Loss {reg_loss:.6f} | "
                f"Val MSE {val_metrics['MSE']:.6f} R² {val_metrics['R2']:.4f} | "
                f"Test MSE {test_metrics['MSE']:.6f} R² {test_metrics['R2']:.4f} | "
                f"Val AML Acc {val_metrics['AML_Accuracy']:.3f} F1 {val_metrics['AML_F1']:.3f} | "
                f"Test AML Acc {test_metrics['AML_Accuracy']:.3f} F1 {test_metrics['AML_F1']:.3f} | "
                f"Val Cell Acc {cell_val_acc:.3f} | Test Cell Acc {cell_test_acc:.3f}"
            )

            scheduler.step(val_metrics["MSE"])

            if val_metrics["R2"] > best_val_r2:
                best_val_r2 = val_metrics["R2"]

                # Clean up old checkpoint before creating new one (Phase 1.2)
                if best_state is not None:
                    del best_state
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no val R² improvement for {early_stopping_patience} checks)"
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return training_history


# ============================================================================
# TRAINING STEP
# ============================================================================


def _forward_pass(
    model: torch.nn.Module,
    rna_data,
    adt_data,
    node_degrees_rna: torch.Tensor,
    node_degrees_adt: torch.Tensor,
    clustering_coeffs_rna: torch.Tensor,
    clustering_coeffs_adt: torch.Tensor,
    use_mixed_precision: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform forward pass through model.

    Returns:
        Tuple of (adt_pred, aml_pred, fused).
    """
    with torch.cuda.amp.autocast(
        enabled=(use_mixed_precision and device.type == "cuda")
    ):
        adt_pred, aml_pred, fused = model(
            x=rna_data.x,
            edge_index_rna=rna_data.edge_index,
            edge_index_adt=(
                adt_data.edge_index if hasattr(adt_data, "edge_index") else None
            ),
            node_degrees_rna=node_degrees_rna,
            node_degrees_adt=node_degrees_adt,
            clustering_coeffs_rna=clustering_coeffs_rna,
            clustering_coeffs_adt=clustering_coeffs_adt,
        )
    return adt_pred, aml_pred, fused


def _compute_training_loss(
    model: torch.nn.Module,
    adt_pred: torch.Tensor,
    aml_pred: torch.Tensor,
    fused: torch.Tensor,
    adt_data,
    aml_labels: Optional[torch.Tensor],
    celltype_labels: Optional[torch.Tensor],
    train_mask: torch.Tensor,
    criterion: torch.nn.Module,
    adt_weight: float,
    classification_weight: float,
    celltype_weight_scaled: float,
    reg_lambda: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined training loss.

    Returns:
        Tuple of (total_loss, adt_loss, aml_loss, celltype_ce, reg_loss).
    """
    # In mini-batch mode the batch object is the RNA subgraph; ADT targets are
    # stored as adt_x (attached before loader construction). Fall back to .x
    # for the full-graph path where adt_data is the actual ADT Data object.
    adt_target = adt_data.adt_x if hasattr(adt_data, "adt_x") else adt_data.x
    # Cast outputs to float32: model runs in float16 under AMP autocast but loss
    # computation must be float32 to avoid mixed-dtype backward errors.
    adt_pred = adt_pred.float()
    aml_pred = aml_pred.float()
    fused = fused.float()
    if aml_labels is not None:
        total_loss, adt_loss, aml_loss = compute_multi_task_loss(
            adt_pred[train_mask],
            adt_target[train_mask],
            aml_pred[train_mask],
            aml_labels[train_mask],
            adt_weight,
            classification_weight,
        )
    else:
        adt_loss = criterion(adt_pred[train_mask], adt_target[train_mask])
        aml_loss = torch.tensor(0.0, device=device)
        total_loss = adt_loss

    if (
        hasattr(model, "celltype_head")
        and model.celltype_head is not None
        and celltype_labels is not None
    ):
        logits = model.predict_celltypes(fused)
        celltype_ce = F.cross_entropy(logits[train_mask], celltype_labels[train_mask])
    else:
        celltype_ce = torch.tensor(0.0, device=device)

    reg_loss = model.get_total_reg_loss()
    total_loss = total_loss + reg_lambda * reg_loss + celltype_weight_scaled * celltype_ce

    return total_loss, adt_loss, aml_loss, celltype_ce, reg_loss


def _training_step(
    model: torch.nn.Module,
    rna_data,
    adt_data,
    train_loader: Optional[object], # New arg
    aml_labels: Optional[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    node_degrees_rna: torch.Tensor,
    node_degrees_adt: torch.Tensor,
    clustering_coeffs_rna: torch.Tensor,
    clustering_coeffs_adt: torch.Tensor,
    adt_mean: torch.Tensor,
    adt_std: torch.Tensor,
    adt_weight: float,
    classification_weight: float,
    epoch: int,
    epochs: int,
    use_mixed_precision: bool,
    device: torch.device,
    celltype_labels: Optional[torch.Tensor],
    celltype_weight: float,
    gradient_accumulation_steps: int = 1,
    reg_weight_schedule: str = "decay",
    reg_weight_init: float = DEFAULT_REG_WEIGHT,
    celltype_weight_schedule: str = "constant",
) -> Tuple[float, float, float, float]:
    """Perform one training step, either full-graph or mini-batch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    reg_lambda, cell_lambda = _compute_loss_weights(
        epoch, epochs, reg_weight_schedule, reg_weight_init, 
        celltype_weight_schedule, celltype_weight
    )

    if train_loader is not None:
        total_adt, total_reg, total_aml, total_cell = 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            # Use precomputed stats from batch
            adt_pred, aml_pred, fused = _forward_pass(
                model, batch, batch, # Passing batch as both rna and adt
                batch.node_degrees, batch.node_degrees, # Approximating ADT stats
                batch.clustering_coeffs, batch.clustering_coeffs,
                use_mixed_precision, device
            )
            
            # Map labels to batch if needed
            batch_aml = batch.aml_labels if hasattr(batch, 'aml_labels') else None
            batch_ct = batch.celltype_labels if hasattr(batch, 'celltype_labels') else None
            
            total_loss, adt_l, aml_l, cell_l, reg_l = _compute_training_loss(
                model, adt_pred, aml_pred, fused, 
                batch, # adt_data is the batch here
                batch_aml, batch_ct,
                torch.ones(batch.num_nodes, dtype=torch.bool, device=device), # Use all nodes in batch
                criterion, adt_weight, classification_weight, cell_lambda, reg_lambda, device
            )
            
            scaled_loss = total_loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            total_adt += adt_l.item()
            total_reg += reg_l.item()
            total_aml += aml_l.item()
            total_cell += cell_l.item()
            n_batches += 1
            
        return total_adt/n_batches, total_reg/n_batches, total_aml/n_batches, total_cell/n_batches

    # Fallback to full-graph training logic
    try:
        adt_pred, aml_pred, fused = _forward_pass(
            model, rna_data, adt_data, node_degrees_rna, node_degrees_adt,
            clustering_coeffs_rna, clustering_coeffs_adt, use_mixed_precision, device
        )
        total_loss, adt_l, aml_l, cell_l, reg_l = _compute_training_loss(
            model, adt_pred, aml_pred, fused, adt_data, aml_labels, celltype_labels,
            rna_data.train_mask, criterion, adt_weight, classification_weight, cell_lambda, reg_lambda, device
        )
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        return adt_l.item(), reg_l.item(), aml_l.item(), cell_l.item()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device.type == "cuda":
            raise DeviceSwitchRequired("GPU OOM in full-graph training")
        raise e


# ============================================================================
# EVALUATION
# ============================================================================


def _compute_correlations_vectorized(
    target_np: np.ndarray, pred_np: np.ndarray
) -> Tuple[float, float]:
    """Compute Pearson and Spearman correlations efficiently using vectorized ops.

    Args:
        target_np: Ground truth array of shape [N, num_features].
        pred_np: Predicted array of shape [N, num_features].

    Returns:
        Tuple of (mean_pearson, mean_spearman).
    """
    target_std = np.std(target_np, axis=0)
    pred_std = np.std(pred_np, axis=0)
    valid_mask = (target_std > MIN_STD_THRESHOLD) & (pred_std > MIN_STD_THRESHOLD)

    if not valid_mask.any():
        return float("nan"), float("nan")

    t = target_np[:, valid_mask]
    p = pred_np[:, valid_mask]

    t_centered = t - np.mean(t, axis=0)
    p_centered = p - np.mean(p, axis=0)

    numerator = np.sum(t_centered * p_centered, axis=0)
    denominator = np.sqrt(np.sum(t_centered**2, axis=0)) * np.sqrt(
        np.sum(p_centered**2, axis=0)
    )
    pearson_corrs = numerator / (denominator + EPSILON)

    t_ranks = np.apply_along_axis(rankdata, 0, t)
    p_ranks = np.apply_along_axis(rankdata, 0, p)
    tr_centered = t_ranks - np.mean(t_ranks, axis=0)
    pr_centered = p_ranks - np.mean(p_ranks, axis=0)
    num_rank = np.sum(tr_centered * pr_centered, axis=0)
    den_rank = np.sqrt(np.sum(tr_centered**2, axis=0)) * np.sqrt(
        np.sum(pr_centered**2, axis=0)
    )
    spearman_corrs = num_rank / (den_rank + EPSILON)

    return float(np.nanmean(pearson_corrs)), float(np.nanmean(spearman_corrs))


def _get_nan_metrics() -> Dict[str, float]:
    """Return a dict of NaN metrics for empty evaluation sets."""
    return {
        k: float("nan")
        for k in [
            "MSE",
            "RMSE",
            "MAE",
            "R2",
            "MeanPearson",
            "MeanSpearman",
            "AML_Accuracy",
            "AML_Precision",
            "AML_Recall",
            "AML_F1",
            "AML_AUC",
            "CellType_Accuracy",
            "CellType_F1",
        ]
    }


def _compute_regression_metrics(
    target_np: np.ndarray, pred_np: np.ndarray
) -> Tuple[float, float, float, float]:
    """Compute regression metrics (MSE, RMSE, MAE, R²).

    Returns:
        Tuple of (mse, rmse, mae, r2).
    """
    mse = mean_squared_error(target_np, pred_np)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(target_np, pred_np)
    r2 = r2_score(target_np.reshape(-1), pred_np.reshape(-1))
    return float(mse), rmse, float(mae), float(r2)


def _compute_aml_metrics(
    aml_pred: torch.Tensor,
    aml_labels: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute AML classification metrics for a data split.

    Returns:
        Dict with AML_Accuracy, AML_Precision, AML_Recall, AML_F1, AML_AUC.
    """
    if aml_labels is not None:
        metrics = compute_classification_metrics(aml_pred[mask], aml_labels[mask])
        return {
            "AML_Accuracy": metrics["accuracy"],
            "AML_Precision": metrics["precision"],
            "AML_Recall": metrics["recall"],
            "AML_F1": metrics["f1_score"],
            "AML_AUC": metrics["auc_roc"],
        }
    return {
        "AML_Accuracy": float("nan"),
        "AML_Precision": float("nan"),
        "AML_Recall": float("nan"),
        "AML_F1": float("nan"),
        "AML_AUC": float("nan"),
    }


def _compute_celltype_metrics(
    model: torch.nn.Module,
    fused: torch.Tensor,
    celltype_labels: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """Compute cell type classification accuracy and macro F1.

    Returns:
        Tuple of (accuracy, macro_f1).
    """
    if (
        hasattr(model, "celltype_head")
        and model.celltype_head is not None
        and celltype_labels is not None
    ):
        with torch.no_grad():
            logits = model.predict_celltypes(fused.clone())
        preds_np = torch.argmax(logits, dim=-1)[mask].detach().cpu().numpy()
        y_true = celltype_labels[mask].detach().cpu().numpy()
        return (
            float(accuracy_score(y_true, preds_np)),
            float(f1_score(y_true, preds_np, average="macro", zero_division=0)),
        )
    return float("nan"), float("nan")


def _evaluate_model(
    model: torch.nn.Module,
    rna_data,
    adt_data,
    aml_labels: Optional[torch.Tensor],
    adt_mean: torch.Tensor,
    adt_std: torch.Tensor,
    node_degrees_rna: torch.Tensor,
    node_degrees_adt: torch.Tensor,
    clustering_coeffs_rna: torch.Tensor,
    clustering_coeffs_adt: torch.Tensor,
    mask: torch.Tensor,
    use_mixed_precision: bool,
    device: torch.device,
    celltype_labels: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Evaluate model on a specific data split.

    Args:
        mask: Boolean node mask selecting the evaluation split.

    Returns:
        Dictionary of metric name to float value.
    """
    if mask.sum().item() == 0:
        return _get_nan_metrics()

    model.eval()

    try:
        with torch.inference_mode():
            adt_pred, aml_pred, fused = _forward_pass(
                model,
                rna_data,
                adt_data,
                node_degrees_rna,
                node_degrees_adt,
                clustering_coeffs_rna,
                clustering_coeffs_adt,
                use_mixed_precision,
                device,
            )
    except RuntimeError as e:
        err_str = str(e).lower()
        if "found dtype float but expected half" in err_str or "dtype" in err_str:
            with torch.inference_mode():
                adt_pred, aml_pred, fused = _forward_pass(
                    model,
                    rna_data,
                    adt_data,
                    node_degrees_rna,
                    node_degrees_adt,
                    clustering_coeffs_rna,
                    clustering_coeffs_adt,
                    False,
                    device,
                )
        else:
            raise

    adt_pred_denorm = adt_pred[mask] * adt_std + adt_mean
    adt_target = adt_data.x[mask] * adt_std + adt_mean

    adt_target_np = adt_target.detach().cpu().numpy()
    adt_pred_np = adt_pred_denorm.detach().cpu().numpy()

    mse, rmse, mae, r2 = _compute_regression_metrics(adt_target_np, adt_pred_np)
    mean_pearson, mean_spearman = _compute_correlations_vectorized(
        adt_target_np, adt_pred_np
    )
    aml_metrics = _compute_aml_metrics(aml_pred, aml_labels, mask)
    celltype_acc, celltype_f1 = _compute_celltype_metrics(
        model, fused, celltype_labels, mask
    )

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MeanPearson": mean_pearson,
        "MeanSpearman": mean_spearman,
        **aml_metrics,
        "CellType_Accuracy": celltype_acc,
        "CellType_F1": celltype_f1,
    }


# ============================================================================
# REPORTING
# ============================================================================


def _log_final_metrics(
    model: torch.nn.Module,
    rna_data,
    adt_data,
    aml_labels: Optional[torch.Tensor],
    adt_mean: torch.Tensor,
    adt_std: torch.Tensor,
    node_degrees_rna: torch.Tensor,
    node_degrees_adt: torch.Tensor,
    clustering_coeffs_rna: torch.Tensor,
    clustering_coeffs_adt: torch.Tensor,
    use_mixed_precision: bool,
    device: torch.device,
) -> None:
    """Log final metrics for all data splits."""
    logger.info("\n" + "=" * 80)
    logger.info("FINAL METRICS")
    logger.info("=" * 80)

    splits = [
        ("Train", rna_data.train_mask),
        ("Val", rna_data.val_mask),
        ("Test", rna_data.test_mask),
    ]

    for split_name, mask in splits:
        metrics = _evaluate_model(
            model,
            rna_data,
            adt_data,
            aml_labels,
            adt_mean,
            adt_std,
            node_degrees_rna,
            node_degrees_adt,
            clustering_coeffs_rna,
            clustering_coeffs_adt,
            mask,
            use_mixed_precision,
            device,
        )

        logger.info(
            f"  {split_name:5s} | "
            f"MSE {metrics['MSE']:.6f}  RMSE {metrics['RMSE']:.6f}  "
            f"MAE {metrics['MAE']:.6f}  R² {metrics['R2']:.4f}  "
            f"r_mean {metrics['MeanPearson']:.3f}  ρ_mean {metrics['MeanSpearman']:.3f}"
        )

        if not np.isnan(metrics["AML_Accuracy"]):
            logger.info(
                f"         | "
                f"AML Acc {metrics['AML_Accuracy']:.3f}  "
                f"Precision {metrics['AML_Precision']:.3f}  "
                f"Recall {metrics['AML_Recall']:.3f}  "
                f"F1 {metrics['AML_F1']:.3f}  AUC {metrics['AML_AUC']:.3f}"
            )
