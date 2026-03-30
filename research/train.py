"""
DeepOMAPNet autoresearch training script.
Mirrors the autoresearch pattern: fixed time budget, single metric, one file to edit.

Usage:
    cd DeepOMAPNet/research
    python train.py

Primary metric: val_nrmse (normalized RMSE for ADT prediction) — LOWER IS BETTER.

This is the ONLY file the agent edits. Everything is fair game:
model architecture, optimizer, hyperparameters, loss weights, etc.
Do NOT modify prepare.py.
"""

import os
import sys
import math
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn.functional as F

# Ensure the parent package is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)

from prepare import (
    _make_synthetic_citeseq, make_pyg_data, make_splits,
    evaluate, NUM_CELLS, NUM_GENES, NUM_ADTS, TIME_BUDGET, RANDOM_SEED,
)

# Import model directly (avoids scripts/__init__.py which loads hardcoded data paths)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "doNET", os.path.join(_repo_root, "scripts", "model", "doNET.py")
)
_donet = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_donet)
GATWithTransformerFusion = _donet.GATWithTransformerFusion
compute_graph_statistics_fast = _donet.compute_graph_statistics_fast

# ---------------------------------------------------------------------------
# Hyperparameters — EDIT THESE
# ---------------------------------------------------------------------------

# Model architecture
HIDDEN_CHANNELS    = 32      # embedding dimension (kept small for CPU)
NUM_HEADS          = 2       # GAT attention heads
NUM_TRANSFORMER_HEADS = 2    # transformer cross-attention heads
NUM_LAYERS         = 1       # transformer fusion layers
DROPOUT            = 0.2     # dropout rate
USE_ADAPTERS       = False   # disable adapters to reduce memory
REDUCTION_FACTOR   = 4       # adapter bottleneck ratio
USE_SPARSE_ATTN    = True    # sparse vs dense cross-attention
NEIGHBORHOOD_SIZE  = 8       # max neighbors for sparse attention

# Optimization
LEARNING_RATE      = 1e-3
WEIGHT_DECAY       = 1e-4
ADT_LOSS_WEIGHT    = 1.0     # weight for ADT regression loss
AML_LOSS_WEIGHT    = 0.5     # weight for AML classification loss
USE_AMP            = False   # mixed precision (True only if CUDA available)

# Training loop
BATCH_SIZE         = NUM_CELLS   # full-graph training (transductive)
GRAD_CLIP          = 1.0

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    # MPS lacks scatter_reduce, so always use CPU on Mac
    DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Data ---
    rna_norm, adt_norm, aml_labels = _make_synthetic_citeseq(seed=RANDOM_SEED)
    rna_data, adt_data = make_pyg_data(rna_norm, adt_norm)
    train_mask, val_mask, test_mask = make_splits(NUM_CELLS, aml_labels, seed=RANDOM_SEED)

    rna_data = rna_data.to(DEVICE)
    adt_data = adt_data.to(DEVICE)
    aml_labels_t = torch.tensor(aml_labels, dtype=torch.float32, device=DEVICE)

    node_degrees, clustering_coeffs = compute_graph_statistics_fast(
        rna_data.edge_index, NUM_CELLS
    )

    # --- Model ---
    model = GATWithTransformerFusion(
        in_channels=NUM_GENES,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=NUM_ADTS,
        heads=NUM_HEADS,
        dropout=DROPOUT,
        nhead=NUM_TRANSFORMER_HEADS,
        num_layers=NUM_LAYERS,
        use_adapters=USE_ADAPTERS,
        reduction_factor=REDUCTION_FACTOR,
        use_sparse_attention=USE_SPARSE_ATTN,
        neighborhood_size=NEIGHBORHOOD_SIZE,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # AMP only on CUDA

    # LR schedule: linear warmup (10 %) + cosine decay
    def lr_lambda(step, total_steps):
        warmup = max(1, int(0.1 * total_steps))
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    # --- Training loop (fixed time budget) ---
    train_start = time.perf_counter()
    step = 0
    best_val_nrmse = float("inf")
    peak_memory_mb = 0.0

    while True:
        elapsed = time.perf_counter() - train_start
        if elapsed >= TIME_BUDGET:
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_steps_estimate = max(1, int(TIME_BUDGET / max(0.01, elapsed / max(1, step)))) if step > 0 else 10000

        with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
            adt_pred, aml_pred, _ = model(
                rna_data.x,
                rna_data.edge_index,
                adt_data.edge_index,
                node_degrees_rna=node_degrees,
                node_degrees_adt=node_degrees,
                clustering_coeffs_rna=clustering_coeffs,
                clustering_coeffs_adt=clustering_coeffs,
            )

            tm = train_mask.to(DEVICE)
            adt_loss = F.mse_loss(adt_pred[tm], adt_data.x[tm])
            aml_loss = F.binary_cross_entropy_with_logits(
                aml_pred[tm].squeeze(), aml_labels_t[tm]
            )
            reg_loss = model.get_total_reg_loss()
            loss = ADT_LOSS_WEIGHT * adt_loss + AML_LOSS_WEIGHT * aml_loss + reg_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # LR schedule
        lr_scale = lr_lambda(step, total_steps_estimate)
        for pg in optimizer.param_groups:
            pg["lr"] = LEARNING_RATE * lr_scale

        # Track peak memory
        if DEVICE.type == "cuda":
            peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated() / 1e6)

        step += 1

    training_seconds = time.perf_counter() - train_start

    # --- Evaluation ---
    metrics = evaluate(
        model, rna_data, adt_data, aml_labels_t,
        val_mask, DEVICE, node_degrees, clustering_coeffs,
    )

    total_seconds = time.perf_counter() - train_start

    # --- Report (mirrors autoresearch output format) ---
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("---")
    print(f"val_nrmse:         {metrics['val_nrmse']:.6f}")
    print(f"val_pearson:       {metrics['val_pearson']:.6f}")
    print(f"val_auc:           {metrics['val_auc']:.6f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_memory_mb:    {peak_memory_mb:.1f}")
    print(f"num_steps:         {step}")
    print(f"num_params_M:      {num_params:.2f}")
    print(f"hidden_channels:   {HIDDEN_CHANNELS}")
    print(f"num_layers:        {NUM_LAYERS}")
    print(f"device:            {DEVICE}")


if __name__ == "__main__":
    run()
