# run_experiment.py
"""
DeepOMAPNet — Full training + visualization on realistic synthetic CITE-seq data.

Generates a biologically-realistic PBMC+AML dataset (negative-binomial RNA,
bimodal ADT, 7 cell types), trains GATWithTransformerFusion for 150 epochs,
and saves publication-quality figures to results/.

Usage:
    python run_experiment.py

Figures saved to results/:
    01_training_curves.png
    02_umap_embeddings.png
    03_protein_scatter.png
    04_pearson_per_protein.png
    05_aml_roc.png
    06_confusion_matrix.png
    07_celltype_heatmap.png
"""

import os
import sys
import math
import time
import importlib.util
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    f1_score, accuracy_score,
)
from scipy.stats import pearsonr, spearmanr
from torch_geometric.data import Data

# ── matplotlib (non-interactive backend for scripts) ──────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# ── UMAP (optional — skip gracefully if not installed) ────────────────────────
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[warn] umap-learn not installed — UMAP figure will be skipped")

# ── seaborn (optional) ────────────────────────────────────────────────────────
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_REPO, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model (bypass scripts/__init__.py which loads hardcoded .h5ad paths)
# ---------------------------------------------------------------------------

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_donet = _load_module("doNET", os.path.join(_REPO, "scripts", "model", "doNET.py"))
GATWithTransformerFusion    = _donet.GATWithTransformerFusion
compute_graph_statistics_fast = _donet.compute_graph_statistics_fast

_synth = _load_module(
    "synthetic_citeseq",
    os.path.join(_REPO, "scripts", "data_provider", "synthetic_citeseq.py"),
)
generate_citeseq_dataset = _synth.generate_citeseq_dataset
CELL_TYPE_NAMES           = _synth.CELL_TYPE_NAMES
ADT_PANEL                 = _synth.ADT_PANEL

# ---------------------------------------------------------------------------
# Config — kept small so CPU memory stays under control
# ---------------------------------------------------------------------------

N_NORMAL        = 250   # normal donor cells
N_AML           = 250   # AML patient cells
SEED            = 42

# Model (tiny for CPU)
HIDDEN          = 48
HEADS           = 2
NHEAD           = 2
NLAYERS         = 2
DROPOUT         = 0.1
USE_ADAPTERS    = False
K_NEIGH         = 10
NEIGHBORHOOD_SZ = 10

# Training
EPOCHS          = 150
LR              = 3e-3
WEIGHT_DECAY    = 1e-4
ADT_WEIGHT      = 1.0
AML_WEIGHT      = 0.5
GRAD_CLIP       = 1.0
PATIENCE        = 25     # early-stopping patience

DEVICE = torch.device("cpu")   # MPS lacks scatter_reduce

# ---------------------------------------------------------------------------
# Dataset → PyG
# ---------------------------------------------------------------------------

def build_knn_graph(features: np.ndarray, k: int) -> torch.Tensor:
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", n_jobs=-1)
    nbrs.fit(features)
    idx = nbrs.kneighbors(features, return_distance=False)
    rows, cols = [], []
    for i, neighbors in enumerate(idx):
        for j in neighbors[1:]:
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return torch.unique(edge_index, dim=1)


def prepare_data(seed=SEED):
    print(f"Generating synthetic CITE-seq data  ({N_NORMAL} normal + {N_AML} AML cells) …")
    ds = generate_citeseq_dataset(n_normal=N_NORMAL, n_aml=N_AML, seed=seed)

    # PCA for k-NN graph topology
    U, S, _ = np.linalg.svd(ds.rna, full_matrices=False)
    pca = (U[:, :min(20, U.shape[1])] * S[:min(20, len(S))]).astype(np.float32)

    edge_index = build_knn_graph(pca, k=K_NEIGH)
    rna_data = Data(x=torch.tensor(ds.rna, dtype=torch.float32), edge_index=edge_index)
    adt_data = Data(x=torch.tensor(ds.adt, dtype=torch.float32), edge_index=edge_index)

    N = ds.n_cells
    nd, cc = compute_graph_statistics_fast(edge_index, N)

    aml_t    = torch.tensor(ds.aml_label, dtype=torch.float32)
    ctype_t  = torch.tensor(ds.celltype_label, dtype=torch.long)

    # Stratified train / val / test split
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    for label in np.unique(ds.aml_label):
        grp = idx[ds.aml_label == label]
        grp = rng.permutation(grp)
        nv = max(1, int(0.15 * len(grp)))
        nt = max(1, int(0.15 * len(grp)))
        val_mask[grp[:nv]]          = True
        test_mask[grp[nv:nv + nt]]  = True
        train_mask[grp[nv + nt:]]   = True

    print(f"  RNA  : {ds.rna.shape}  ADT : {ds.adt.shape}")
    print(f"  Edges: {edge_index.shape[1]}  "
          f"Train/Val/Test = {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")

    return rna_data, adt_data, aml_t, ctype_t, nd, cc, train_mask, val_mask, test_mask, ds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(rna_data, adt_data, aml_t, ctype_t, nd, cc,
          train_mask, val_mask, n_genes, n_adts):

    torch.manual_seed(SEED)
    model = GATWithTransformerFusion(
        in_channels=n_genes,
        hidden_channels=HIDDEN,
        out_channels=n_adts,
        heads=HEADS,
        dropout=DROPOUT,
        nhead=NHEAD,
        num_layers=NLAYERS,
        use_adapters=USE_ADAPTERS,
        use_sparse_attention=True,
        neighborhood_size=NEIGHBORHOOD_SZ,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01
    )

    history = {
        "train_loss": [], "val_adt_loss": [], "val_pearson": [], "val_auc": [],
    }

    best_val   = float("inf")
    best_state = None
    patience_count = 0

    rna_data = rna_data.to(DEVICE)
    adt_data = adt_data.to(DEVICE)
    aml_t    = aml_t.to(DEVICE)
    tm       = train_mask.to(DEVICE)
    vm       = val_mask.to(DEVICE)

    t0 = time.perf_counter()
    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE}) …")

    for epoch in range(1, EPOCHS + 1):
        # ── train step ──────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad(set_to_none=True)

        adt_pred, aml_pred, _ = model(
            rna_data.x, rna_data.edge_index,
            node_degrees_rna=nd, clustering_coeffs_rna=cc,
        )

        adt_loss = F.mse_loss(adt_pred[tm], adt_data.x[tm])
        aml_loss = F.binary_cross_entropy_with_logits(
            aml_pred[tm].squeeze(), aml_t[tm]
        )
        reg_loss = model.get_total_reg_loss()
        loss = ADT_WEIGHT * adt_loss + AML_WEIGHT * aml_loss + reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        history["train_loss"].append(loss.item())

        # ── validation ──────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            adt_p, aml_p, _ = model(
                rna_data.x, rna_data.edge_index,
                node_degrees_rna=nd, clustering_coeffs_rna=cc,
            )

        vadt_true = adt_data.x[vm].cpu().numpy()
        vadt_pred = adt_p[vm].cpu().numpy()
        vaml_true = aml_t[vm].cpu().numpy()
        vaml_logits = aml_p[vm].squeeze().cpu().numpy()

        val_adt_loss = float(F.mse_loss(
            torch.tensor(vadt_pred), torch.tensor(vadt_true)
        ))
        rs = [pearsonr(vadt_true[:, p], vadt_pred[:, p])[0]
              for p in range(vadt_true.shape[1])]
        rs = [r for r in rs if not math.isnan(r)]
        mean_r = float(np.mean(rs)) if rs else 0.0

        try:
            probs = 1.0 / (1.0 + np.exp(-vaml_logits))
            auc = roc_auc_score(vaml_true, probs)
        except ValueError:
            auc = 0.5

        history["val_adt_loss"].append(val_adt_loss)
        history["val_pearson"].append(mean_r)
        history["val_auc"].append(auc)

        # ── logging ─────────────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t0
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={loss.item():.4f}  "
                  f"val_mse={val_adt_loss:.4f}  "
                  f"val_r={mean_r:.4f}  "
                  f"auc={auc:.4f}  "
                  f"({elapsed:.0f}s)")

        # ── early stopping ───────────────────────────────────────────────────
        if val_adt_loss < best_val - 1e-5:
            best_val   = val_adt_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
                break

    total_time = time.perf_counter() - t0
    print(f"\nTraining done in {total_time:.1f}s  best_val_mse={best_val:.4f}")

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_predictions(model, rna_data, adt_data, aml_t, nd, cc, mask):
    model.eval()
    adt_pred, aml_pred, embeddings = model(
        rna_data.x, rna_data.edge_index,
        node_degrees_rna=nd, clustering_coeffs_rna=cc,
    )
    m = mask.to(DEVICE)
    return {
        "adt_true":   adt_data.x[m].cpu().numpy(),
        "adt_pred":   adt_pred[m].cpu().numpy(),
        "aml_true":   aml_t[m].cpu().numpy(),
        "aml_logits": aml_pred[m].squeeze().cpu().numpy(),
        "embeddings": embeddings[m].cpu().numpy(),
    }


@torch.no_grad()
def get_all_embeddings(model, rna_data, nd, cc):
    model.eval()
    _, _, emb = model(
        rna_data.x, rna_data.edge_index,
        node_degrees_rna=nd, clustering_coeffs_rna=cc,
    )
    return emb.cpu().numpy()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

PALETTE = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#A65628", "#F781BF",
]


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {os.path.relpath(path)}")


def fig_training_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#2196F3", lw=1.5)
    ax.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["val_adt_loss"], color="#F44336", lw=1.5, label="Val MSE")
    ax.plot(epochs, history["val_pearson"],  color="#4CAF50", lw=1.5, label="Val Pearson r")
    ax.set_title("Validation ADT Metrics", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history["val_auc"], color="#9C27B0", lw=1.5)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random")
    ax.set_ylim(0, 1.05)
    ax.set_title("AML Classification AUC", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUC-ROC")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("DeepOMAPNet — Training Dynamics", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "01_training_curves.png")


def fig_umap_embeddings(embeddings_all, ctype_labels, aml_labels, adt_all, adt_names):
    if not HAS_UMAP:
        return
    print("  Running UMAP on fused embeddings …")
    reducer = UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.3)
    umap_coords = reducer.fit_transform(embeddings_all)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Cell type
    ax = axes[0]
    for i, name in enumerate(CELL_TYPE_NAMES):
        mask = ctype_labels == i
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=PALETTE[i % len(PALETTE)], s=12, alpha=0.7, label=name, rasterized=True)
    ax.set_title("Cell Type", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, markerscale=2, bbox_to_anchor=(1, 1))
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])

    # Panel B: AML status
    ax = axes[1]
    colors = ["#2196F3", "#F44336"]
    for label, name, c in zip([0, 1], ["Normal", "AML"], colors):
        mask = aml_labels == label
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=c, s=12, alpha=0.7, label=name, rasterized=True)
    ax.set_title("AML Status", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])

    # Panel C: CD34 expression (AML stem cell marker)
    ax = axes[2]
    cd34_idx = adt_names.index("CD34") if "CD34" in adt_names else 0
    expr = adt_all[:, cd34_idx]
    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                    c=expr, cmap="RdYlBu_r", s=12, alpha=0.8, rasterized=True)
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(f"{adt_names[cd34_idx]} Expression (AML marker)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("DeepOMAPNet — Fused Embedding Space (UMAP)", fontsize=15,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "02_umap_embeddings.png")


def fig_protein_scatter(adt_true, adt_pred, adt_names):
    n_proteins = min(12, adt_true.shape[1])
    ncols = 4
    nrows = math.ceil(n_proteins / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = axes.flatten()

    for i in range(n_proteins):
        ax = axes[i]
        true = adt_true[:, i]
        pred = adt_pred[:, i]
        r, p = pearsonr(true, pred)
        rho, _ = spearmanr(true, pred)

        ax.scatter(true, pred, s=8, alpha=0.5, color=PALETTE[i % len(PALETTE)], rasterized=True)

        # Best-fit line
        m, b = np.polyfit(true, pred, 1)
        xs = np.linspace(true.min(), true.max(), 50)
        ax.plot(xs, m * xs + b, "k--", lw=1, alpha=0.7)

        pname = adt_names[i] if i < len(adt_names) else f"ADT {i}"
        ax.set_title(f"{pname}  r={r:.2f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("True", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    for j in range(n_proteins, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("DeepOMAPNet — Per-Protein Predicted vs True (test set)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "03_protein_scatter.png")


def fig_pearson_per_protein(adt_true, adt_pred, adt_names):
    n_proteins = adt_true.shape[1]
    rs = []
    for i in range(n_proteins):
        r, _ = pearsonr(adt_true[:, i], adt_pred[:, i])
        rs.append(r if not math.isnan(r) else 0.0)

    names = [adt_names[i] if i < len(adt_names) else f"ADT {i}" for i in range(n_proteins)]
    order = np.argsort(rs)[::-1]
    rs_sorted    = [rs[i] for i in order]
    names_sorted = [names[i] for i in order]
    colors_sorted = [PALETTE[i % len(PALETTE)] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, n_proteins * 0.6), 5))
    bars = ax.bar(range(n_proteins), rs_sorted, color=colors_sorted, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(n_proteins))
    ax.set_xticklabels(names_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_ylim(-0.1, 1.0)
    ax.set_title(f"DeepOMAPNet — Per-Protein Pearson r (test set)  "
                 f"mean={np.mean(rs):.3f}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "04_pearson_per_protein.png")


def fig_aml_roc(aml_true, aml_logits):
    probs = 1.0 / (1.0 + np.exp(-aml_logits))
    fpr, tpr, thresholds = roc_curve(aml_true, probs)
    auc = roc_auc_score(aml_true, probs)

    # Youden's J for optimal threshold
    j_scores = tpr - fpr
    best_idx  = np.argmax(j_scores)
    best_thr  = thresholds[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(fpr, tpr, color="#F44336", lw=2, label=f"DeepOMAPNet (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], s=80, color="black", zorder=5,
               label=f"Best threshold={best_thr:.2f}")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("AML Classification — ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    pred_labels = (probs >= best_thr).astype(int)
    cm = confusion_matrix(aml_true, pred_labels)
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "AML"],
                    yticklabels=["Normal", "AML"], ax=ax)
    else:
        im = ax.imshow(cm, cmap="Blues")
        for (r, c), val in np.ndenumerate(cm):
            ax.text(c, r, str(val), ha="center", va="center", fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "AML"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "AML"])
        plt.colorbar(im, ax=ax)

    acc = accuracy_score(aml_true, pred_labels)
    f1  = f1_score(aml_true, pred_labels, zero_division=0)
    ax.set_title(f"Confusion Matrix  (Acc={acc:.2f}, F1={f1:.2f})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    fig.suptitle("DeepOMAPNet — AML Disease Classification", fontsize=15,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "05_aml_roc_and_confusion.png")


def fig_celltype_heatmap(adt_true, ctype_labels, adt_names):
    """Mean protein expression per cell type — shows biological structure."""
    n_types = max(ctype_labels) + 1
    n_prots = min(20, adt_true.shape[1])
    mean_expr = np.zeros((n_types, n_prots))
    for i in range(n_types):
        mask = ctype_labels == i
        if mask.any():
            mean_expr[i] = adt_true[mask, :n_prots].mean(0)

    # Row and column normalisation
    row_min = mean_expr.min(1, keepdims=True)
    row_max = mean_expr.max(1, keepdims=True)
    mean_expr_norm = (mean_expr - row_min) / (row_max - row_min + 1e-8)

    prot_names = [adt_names[i] if i < len(adt_names) else f"ADT {i}"
                  for i in range(n_prots)]
    ct_names_short = [CELL_TYPE_NAMES[i] if i < len(CELL_TYPE_NAMES) else f"CT{i}"
                      for i in range(n_types)]

    fig, ax = plt.subplots(figsize=(max(10, n_prots * 0.6), n_types * 0.9 + 1.5))
    cmap = "RdYlBu_r"
    im = ax.imshow(mean_expr_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Normalized mean expression")

    ax.set_xticks(range(n_prots))
    ax.set_xticklabels(prot_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(ct_names_short, fontsize=9)
    ax.set_title("Mean Surface Protein Expression per Cell Type (test set, row-normalized)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_celltype_protein_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  DeepOMAPNet — Synthetic CITE-seq Experiment")
    print("=" * 65)

    # 1. Data
    (rna_data, adt_data, aml_t, ctype_t, nd, cc,
     train_mask, val_mask, test_mask, ds) = prepare_data()

    n_genes = ds.n_genes
    n_adts  = ds.n_adts

    # 2. Training
    model, history = train(
        rna_data, adt_data, aml_t, ctype_t, nd, cc,
        train_mask, val_mask, n_genes, n_adts,
    )

    # 3. Test-set evaluation
    print("\nEvaluating on test set …")
    test_out = get_predictions(model, rna_data, adt_data, aml_t, nd, cc, test_mask)
    adt_true   = test_out["adt_true"]
    adt_pred   = test_out["adt_pred"]
    aml_true   = test_out["aml_true"]
    aml_logits = test_out["aml_logits"]

    rs   = [pearsonr(adt_true[:, p], adt_pred[:, p])[0] for p in range(n_adts)]
    rs   = [r if not math.isnan(r) else 0.0 for r in rs]
    rmse = math.sqrt(((adt_pred - adt_true) ** 2).mean())
    nrmse = rmse / (adt_true.std() + 1e-8)
    probs = 1.0 / (1.0 + np.exp(-aml_logits))
    try:
        auc = roc_auc_score(aml_true, probs)
    except ValueError:
        auc = 0.5
    acc = accuracy_score(aml_true, (probs >= 0.5).astype(int))
    f1  = f1_score(aml_true, (probs >= 0.5).astype(int), zero_division=0)

    print(f"\n{'─'*45}")
    print(f"  Test  NRMSE      : {nrmse:.4f}")
    print(f"  Test  Mean r     : {np.mean(rs):.4f}  (best protein: {max(rs):.4f})")
    print(f"  Test  AML AUC    : {auc:.4f}")
    print(f"  Test  AML Acc    : {acc:.4f}   F1: {f1:.4f}")
    print(f"{'─'*45}\n")

    # 4. Visualizations
    print("Generating figures …")

    fig_training_curves(history)

    # UMAP — full dataset
    all_emb      = get_all_embeddings(model, rna_data.to(DEVICE), nd, cc)
    ctype_np     = ctype_t.numpy()
    aml_np       = aml_t.numpy().astype(int)
    adt_all      = ds.adt                      # CLR-normalized, full dataset

    fig_umap_embeddings(all_emb, ctype_np, aml_np, adt_all, ds.adt_names)
    fig_protein_scatter(adt_true, adt_pred, ds.adt_names)
    fig_pearson_per_protein(adt_true, adt_pred, ds.adt_names)
    fig_aml_roc(aml_true, aml_logits)
    fig_celltype_heatmap(ds.adt[test_mask.numpy()], ctype_np[test_mask.numpy()], ds.adt_names)

    print(f"\nAll figures saved to  results/")
    print("Done.")


if __name__ == "__main__":
    main()
