"""
DeepOMAPNet — Publication-quality visualization utilities.

All functions accept numpy arrays and return matplotlib Figure objects so they
can be embedded in notebooks or saved to disk.

Typical usage::

    from scripts.visualizations import (
        plot_training_curves, plot_umap, plot_protein_scatter,
        plot_pearson_barplot, plot_aml_roc, plot_celltype_heatmap,
    )
    fig = plot_training_curves(history)
    fig.savefig("training.pdf", dpi=300, bbox_inches="tight")
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# Publication palette — colorblind-safe (Wong 2011)
PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]


# ---------------------------------------------------------------------------
# Training dynamics
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "DeepOMAPNet — Training Dynamics",
) -> plt.Figure:
    """
    Plot train loss, validation ADT metrics, and AML AUC over epochs.

    Parameters
    ----------
    history : dict with keys ``train_loss``, ``val_adt_loss``,
              ``val_pearson``, ``val_auc``
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#2196F3", lw=1.5)
    ax.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["val_adt_loss"], color="#F44336", lw=1.5, label="Val MSE")
    ax.plot(epochs, history["val_pearson"],  color="#4CAF50", lw=1.5, label="Val Pearson r")
    ax.set_title("Validation ADT Metrics", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history["val_auc"], color="#9C27B0", lw=1.5)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random")
    ax.set_ylim(0, 1.05)
    ax.set_title("AML Classification AUC", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC-ROC"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UMAP embedding panels
# ---------------------------------------------------------------------------

def plot_umap(
    umap_coords: np.ndarray,
    celltype_labels: np.ndarray,
    aml_labels: np.ndarray,
    adt_matrix: np.ndarray,
    adt_names: List[str],
    celltype_names: Optional[List[str]] = None,
    highlight_protein: Optional[str] = "CD34",
    title: str = "DeepOMAPNet — Fused Embedding Space",
) -> plt.Figure:
    """
    Three-panel UMAP: cell type / AML status / marker protein expression.

    Parameters
    ----------
    umap_coords      : [N, 2] 2-D UMAP coordinates
    celltype_labels  : [N]   integer cell-type indices
    aml_labels       : [N]   binary (0=Normal, 1=AML)
    adt_matrix       : [N, P] CLR-normalized ADT values
    adt_names        : protein names (length P)
    highlight_protein: protein to show in panel C (default CD34)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pt_kw = dict(s=10, alpha=0.75, rasterized=True, linewidths=0)

    # Panel A — cell type
    ax = axes[0]
    n_types = int(celltype_labels.max()) + 1
    ct_names = celltype_names or [f"CT{i}" for i in range(n_types)]
    for i in range(n_types):
        mask = celltype_labels == i
        ax.scatter(*umap_coords[mask].T, c=PALETTE[i % len(PALETTE)],
                   label=ct_names[i], **pt_kw)
    ax.set_title("Cell Type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, markerscale=2.5, bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    # Panel B — AML status
    ax = axes[1]
    for lbl, name, c in zip([0, 1], ["Normal", "AML"], ["#2196F3", "#F44336"]):
        mask = aml_labels == lbl
        ax.scatter(*umap_coords[mask].T, c=c, label=name, **pt_kw)
    ax.set_title("AML Status", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2.5, frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    # Panel C — marker protein
    ax = axes[2]
    prot = highlight_protein if (highlight_protein and highlight_protein in adt_names) else adt_names[0]
    p_idx = adt_names.index(prot)
    sc = ax.scatter(*umap_coords.T, c=adt_matrix[:, p_idx],
                    cmap="RdYlBu_r", **pt_kw)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="CLR expression")
    ax.set_title(f"{prot} Expression", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-protein scatter (predicted vs true)
# ---------------------------------------------------------------------------

def plot_protein_scatter(
    adt_true: np.ndarray,
    adt_pred: np.ndarray,
    adt_names: Optional[List[str]] = None,
    max_proteins: int = 12,
    ncols: int = 4,
) -> plt.Figure:
    """
    Grid of scatter plots (true vs predicted) for each protein.

    Returns a Figure with up to ``max_proteins`` panels.
    """
    n_prot = min(max_proteins, adt_true.shape[1])
    nrows = math.ceil(n_prot / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
    axes = np.array(axes).flatten()

    for i in range(n_prot):
        ax = axes[i]
        true = adt_true[:, i]
        pred = adt_pred[:, i]
        r, _ = pearsonr(true, pred)
        r = 0.0 if math.isnan(r) else r

        ax.scatter(true, pred, s=8, alpha=0.45,
                   color=PALETTE[i % len(PALETTE)], rasterized=True)
        m, b = np.polyfit(true, pred, 1)
        xs = np.linspace(true.min(), true.max(), 50)
        ax.plot(xs, m * xs + b, "k--", lw=1, alpha=0.7)

        name = adt_names[i] if (adt_names and i < len(adt_names)) else f"ADT {i}"
        ax.set_title(f"{name}  r = {r:.3f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("True", fontsize=8); ax.set_ylabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.2)

    for j in range(n_prot, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("DeepOMAPNet — Predicted vs True Protein Expression (test set)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-protein Pearson r bar chart
# ---------------------------------------------------------------------------

def plot_pearson_barplot(
    adt_true: np.ndarray,
    adt_pred: np.ndarray,
    adt_names: Optional[List[str]] = None,
) -> plt.Figure:
    """Ranked per-protein Pearson r bar chart."""
    n_prot = adt_true.shape[1]
    rs = []
    for i in range(n_prot):
        r, _ = pearsonr(adt_true[:, i], adt_pred[:, i])
        rs.append(0.0 if math.isnan(r) else float(r))

    names = [adt_names[i] if (adt_names and i < len(adt_names)) else f"ADT {i}"
             for i in range(n_prot)]
    order  = np.argsort(rs)[::-1]
    rs_s   = [rs[i] for i in order]
    nm_s   = [names[i] for i in order]
    clr_s  = [PALETTE[i % len(PALETTE)] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, n_prot * 0.65), 5))
    ax.bar(range(n_prot), rs_s, color=clr_s, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(n_prot))
    ax.set_xticklabels(nm_s, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_ylim(-0.1, 1.0)
    ax.set_title(
        f"Per-Protein Pearson r — mean = {np.mean(rs):.3f}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# AML ROC + confusion matrix
# ---------------------------------------------------------------------------

def plot_aml_roc(
    aml_true: np.ndarray,
    aml_logits: np.ndarray,
) -> plt.Figure:
    """Two-panel figure: ROC curve + confusion matrix at optimal threshold."""
    probs = 1.0 / (1.0 + np.exp(-aml_logits))
    fpr, tpr, thresholds = roc_curve(aml_true, probs)
    auc = roc_auc_score(aml_true, probs)

    best_idx = int(np.argmax(tpr - fpr))
    best_thr = float(thresholds[best_idx])
    pred_lbl = (probs >= best_thr).astype(int)
    acc = accuracy_score(aml_true, pred_lbl)
    f1  = f1_score(aml_true, pred_lbl, zero_division=0)
    cm  = confusion_matrix(aml_true, pred_lbl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(fpr, tpr, color="#F44336", lw=2, label=f"DeepOMAPNet (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], s=90, zorder=5, color="black",
               label=f"Threshold = {best_thr:.2f}")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("AML — ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#F44336")

    ax = axes[1]
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    tick_labels = ["Normal", "AML"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(tick_labels, fontsize=10)
    for (r, c), val in np.ndenumerate(cm):
        ax.text(c, r, str(val), ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white" if val > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, shrink=0.75)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix  (Acc={acc:.2f}, F1={f1:.2f})",
                 fontsize=12, fontweight="bold")

    fig.suptitle("DeepOMAPNet — AML Disease Classification",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cell-type × protein expression heatmap
# ---------------------------------------------------------------------------

def plot_celltype_heatmap(
    adt_matrix: np.ndarray,
    celltype_labels: np.ndarray,
    adt_names: Optional[List[str]] = None,
    celltype_names: Optional[List[str]] = None,
    max_proteins: int = 20,
) -> plt.Figure:
    """
    Row-normalised mean surface protein expression per cell type.

    Rows = cell types, Columns = proteins.
    """
    n_types = int(celltype_labels.max()) + 1
    n_prot  = min(max_proteins, adt_matrix.shape[1])

    mean_expr = np.zeros((n_types, n_prot))
    for i in range(n_types):
        mask = celltype_labels == i
        if mask.any():
            mean_expr[i] = adt_matrix[mask, :n_prot].mean(0)

    # Row normalise to [0, 1]
    rmin = mean_expr.min(1, keepdims=True)
    rmax = mean_expr.max(1, keepdims=True)
    norm = (mean_expr - rmin) / (rmax - rmin + 1e-8)

    prot_labels = [adt_names[i] if (adt_names and i < len(adt_names)) else f"ADT {i}"
                   for i in range(n_prot)]
    ct_labels   = [celltype_names[i] if (celltype_names and i < len(celltype_names)) else f"CT{i}"
                   for i in range(n_types)]

    fig, ax = plt.subplots(figsize=(max(10, n_prot * 0.6), n_types * 0.9 + 1.5))
    im = ax.imshow(norm, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Row-normalised mean expression")

    ax.set_xticks(range(n_prot))
    ax.set_xticklabels(prot_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(ct_labels, fontsize=9)
    ax.set_title("Mean Surface Protein Expression per Cell Type",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
