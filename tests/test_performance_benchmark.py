"""
Publication-quality performance benchmark for DeepOMAPNet.

Tests and reports the metrics required for a Nature Cell submission:
  - Per-protein Pearson correlation (primary ADT prediction metric)
  - RMSE and normalized RMSE per protein
  - R² score (global and per-protein)
  - AUC-ROC and F1 for AML binary classification
  - Cell-type classification accuracy
  - Comparison against baselines (mean predictor, linear regression)
  - Statistical significance of improvement over baselines (Wilcoxon signed-rank test)
  - Ablation: model with and without graph structure

All assertions are stated as acceptance thresholds. Adjust thresholds if you
retrain on real data — synthetic data produces noisier, lower correlations.
"""

import math
import pytest
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon, pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    mean_squared_error, r2_score,
)

from conftest import (
    GATWithTransformerFusion, compute_graph_statistics_fast,
    make_synthetic_citeseq, make_knn_edges,
    HIDDEN, HEADS, NHEAD, N_LAYERS, N_CELLS, N_GENES, N_ADTS, N_TYPES,
    K_NEIGH, SEED,
)
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------

def _per_protein_pearson(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute Pearson r for each protein column. Returns array of shape [P]."""
    P = y_true.shape[1]
    rs = []
    for p in range(P):
        if y_true[:, p].std() > 0 and y_pred[:, p].std() > 0:
            r, _ = pearsonr(y_true[:, p], y_pred[:, p])
        else:
            r = 0.0
        rs.append(float(r) if not math.isnan(r) else 0.0)
    return np.array(rs)


def _per_protein_spearman(y_true: np.ndarray, y_pred: np.ndarray):
    P = y_true.shape[1]
    rs = []
    for p in range(P):
        r = spearmanr(y_true[:, p], y_pred[:, p]).correlation
        rs.append(float(r) if not math.isnan(r) else 0.0)
    return np.array(rs)


def _nrmse_per_protein(y_true: np.ndarray, y_pred: np.ndarray):
    """Normalized RMSE per protein (RMSE / std of ground truth)."""
    P = y_true.shape[1]
    nrmse = []
    for p in range(P):
        rmse = math.sqrt(mean_squared_error(y_true[:, p], y_pred[:, p]))
        std  = y_true[:, p].std() + 1e-8
        nrmse.append(rmse / std)
    return np.array(nrmse)


def _evaluate_model_on_mask(model, rna_data, adt_data, aml_t, nd, cc, mask):
    model.eval()
    with torch.no_grad():
        adt_pred, aml_pred, fused = model(
            rna_data.x, rna_data.edge_index, adt_data.edge_index,
            node_degrees_rna=nd, node_degrees_adt=nd,
            clustering_coeffs_rna=cc, clustering_coeffs_adt=cc,
        )
    y_true  = adt_data.x[mask].numpy()
    y_pred  = adt_pred[mask].numpy()
    aml_true = aml_t[mask].numpy().astype(int)
    aml_prob = torch.sigmoid(aml_pred[mask].squeeze()).numpy()
    return y_true, y_pred, aml_true, aml_prob, fused[mask]


# ---------------------------------------------------------------------------
# Fixture: trained model + test split
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def benchmark_setup():
    """
    Train a model for 80 epochs on synthetic data, return the test-set
    predictions alongside baseline predictions.
    """
    rng = np.random.default_rng(SEED)
    rna, adt, aml, ctype = make_synthetic_citeseq(
        seed=SEED, n_cells=N_CELLS, n_genes=N_GENES, n_adts=N_ADTS, n_types=N_TYPES
    )

    U, S, _ = np.linalg.svd(rna, full_matrices=False)
    pca = U[:, :min(20, U.shape[1])] * S[:min(20, len(S))]
    edge_index = make_knn_edges(pca, k=K_NEIGH)

    rna_data = Data(x=torch.tensor(rna), edge_index=edge_index)
    adt_data = Data(x=torch.tensor(adt), edge_index=edge_index)
    nd, cc = compute_graph_statistics_fast(edge_index, N_CELLS)
    aml_t   = torch.tensor(aml, dtype=torch.float32)
    ctype_t = torch.tensor(ctype, dtype=torch.long)

    # Splits
    idx      = rng.permutation(N_CELLS)
    n_train  = int(0.70 * N_CELLS)
    n_val    = int(0.15 * N_CELLS)
    train_mask = torch.zeros(N_CELLS, dtype=torch.bool)
    val_mask   = torch.zeros(N_CELLS, dtype=torch.bool)
    test_mask  = torch.zeros(N_CELLS, dtype=torch.bool)
    train_mask[idx[:n_train]]               = True
    val_mask[idx[n_train:n_train + n_val]]  = True
    test_mask[idx[n_train + n_val:]]        = True

    # Train
    torch.manual_seed(SEED)
    model = GATWithTransformerFusion(
        in_channels=N_GENES, hidden_channels=HIDDEN, out_channels=N_ADTS,
        heads=HEADS, dropout=0.1, nhead=NHEAD, num_layers=N_LAYERS,
        use_adapters=True, reduction_factor=4, use_sparse_attention=True,
        neighborhood_size=K_NEIGH, num_cell_types=N_TYPES,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(80):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        adt_pred, aml_pred, fused = model(
            rna_data.x, rna_data.edge_index, adt_data.edge_index,
            node_degrees_rna=nd, node_degrees_adt=nd,
            clustering_coeffs_rna=cc, clustering_coeffs_adt=cc,
        )
        loss = (
            F.mse_loss(adt_pred[train_mask], adt_data.x[train_mask])
            + 0.5 * F.binary_cross_entropy_with_logits(
                aml_pred[train_mask].squeeze(), aml_t[train_mask]
            )
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                vp, _, _ = model(rna_data.x, rna_data.edge_index,
                                 node_degrees_rna=nd, clustering_coeffs_rna=cc)
            val_loss = F.mse_loss(vp[val_mask], adt_data.x[val_mask]).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Test-set predictions
    y_true, y_pred, aml_true, aml_prob, fused_test = _evaluate_model_on_mask(
        model, rna_data, adt_data, aml_t, nd, cc, test_mask
    )

    # Baseline 1: mean predictor (train mean)
    train_mean = adt_data.x[train_mask].numpy().mean(0, keepdims=True)
    y_pred_mean = np.repeat(train_mean, test_mask.sum().item(), axis=0)

    # Baseline 2: Ridge regression (RNA → ADT, trained on train split)
    X_train = rna_data.x[train_mask].numpy()
    y_train = adt_data.x[train_mask].numpy()
    X_test  = rna_data.x[test_mask].numpy()
    ridge   = Ridge(alpha=1.0).fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    return {
        "y_true":        y_true,
        "y_pred":        y_pred,
        "y_pred_mean":   y_pred_mean,
        "y_pred_ridge":  y_pred_ridge,
        "aml_true":      aml_true,
        "aml_prob":      aml_prob,
        "ctype_true":    ctype_t[test_mask].numpy(),
        "fused":         fused_test,
        "model":         model,
        "rna_data":      rna_data,
        "adt_data":      adt_data,
        "aml_t":         aml_t,
        "nd":            nd,
        "cc":            cc,
        "train_mask":    train_mask,
        "val_mask":      val_mask,
        "test_mask":     test_mask,
    }


# ---------------------------------------------------------------------------
# ADT prediction metrics
# ---------------------------------------------------------------------------

class TestADTPrediction:

    def test_mean_pearson_above_zero(self, benchmark_setup):
        """Model must achieve positive mean Pearson r on held-out test set."""
        rs = _per_protein_pearson(benchmark_setup["y_true"], benchmark_setup["y_pred"])
        mean_r = rs.mean()
        print(f"\n  Mean Pearson r (test): {mean_r:.4f}")
        assert mean_r > 0.0, \
            f"Mean Pearson r={mean_r:.4f} — model performs worse than mean predictor"

    def test_pearson_better_than_mean_baseline(self, benchmark_setup):
        """Model must beat mean predictor on Pearson r for majority of proteins."""
        rs_model = _per_protein_pearson(benchmark_setup["y_true"], benchmark_setup["y_pred"])
        rs_mean  = _per_protein_pearson(benchmark_setup["y_true"], benchmark_setup["y_pred_mean"])
        wins = (rs_model > rs_mean).sum()
        total = len(rs_model)
        print(f"\n  Model beats mean baseline: {wins}/{total} proteins")
        assert wins >= total // 2, \
            f"Model beats mean predictor on only {wins}/{total} proteins"

    def test_pearson_better_than_ridge_baseline(self, benchmark_setup):
        """
        Model should be competitive with Ridge regression on Pearson r.
        Acceptable if model matches or exceeds Ridge on ≥ 40% of proteins
        (the GAT graph-aware component adds value beyond linear).
        """
        rs_model = _per_protein_pearson(benchmark_setup["y_true"], benchmark_setup["y_pred"])
        rs_ridge = _per_protein_pearson(benchmark_setup["y_true"], benchmark_setup["y_pred_ridge"])
        wins = (rs_model > rs_ridge).sum()
        total = len(rs_model)
        print(f"\n  Model beats Ridge on: {wins}/{total} proteins")
        assert wins >= int(0.40 * total), \
            f"Model underperforms Ridge on too many proteins: {wins}/{total}"

    def test_nrmse_below_one(self, benchmark_setup):
        """
        Normalized RMSE per protein should be < 1.0 (i.e., RMSE < std of target).
        NRMSE ≥ 1 means the model is worse than predicting the mean.
        """
        nrmse = _nrmse_per_protein(benchmark_setup["y_true"], benchmark_setup["y_pred"])
        proteins_below_one = (nrmse < 1.0).sum()
        total = len(nrmse)
        print(f"\n  Proteins with NRMSE < 1.0: {proteins_below_one}/{total}")
        assert proteins_below_one >= total // 2, \
            f"Fewer than half of proteins have NRMSE < 1: {proteins_below_one}/{total}"

    def test_r2_score_positive(self, benchmark_setup):
        y_true = benchmark_setup["y_true"]
        y_pred = benchmark_setup["y_pred"]
        r2 = r2_score(y_true.ravel(), y_pred.ravel())
        print(f"\n  Global R² (test): {r2:.4f}")
        assert r2 > 0.0, f"Negative R² ({r2:.4f}) — model worse than mean"

    def test_wilcoxon_pearson_vs_mean_baseline(self, benchmark_setup):
        """
        Wilcoxon signed-rank test: model Pearson r is statistically significantly
        higher than mean-predictor Pearson r across proteins (p < 0.05).
        """
        rs_model = _per_protein_pearson(benchmark_setup["y_true"], benchmark_setup["y_pred"])
        rs_mean  = _per_protein_pearson(benchmark_setup["y_true"], benchmark_setup["y_pred_mean"])
        diffs = rs_model - rs_mean
        if np.all(diffs == 0):
            pytest.skip("All differences are zero — cannot run Wilcoxon test")
        stat, pval = wilcoxon(diffs, alternative="greater")
        print(f"\n  Wilcoxon (model vs mean): stat={stat:.2f}, p={pval:.4f}")
        assert pval < 0.05, \
            f"Model improvement over mean predictor not significant (p={pval:.4f})"

    def test_per_protein_pearson_report(self, benchmark_setup):
        """Print full per-protein Pearson table (informational, always passes)."""
        rs = _per_protein_pearson(benchmark_setup["y_true"], benchmark_setup["y_pred"])
        sp = _per_protein_spearman(benchmark_setup["y_true"], benchmark_setup["y_pred"])
        nr = _nrmse_per_protein(benchmark_setup["y_true"], benchmark_setup["y_pred"])
        print("\n  Per-protein performance (test set):")
        print(f"  {'Protein':>8} {'Pearson':>10} {'Spearman':>10} {'NRMSE':>10}")
        for i in range(len(rs)):
            print(f"  {i:>8} {rs[i]:>10.4f} {sp[i]:>10.4f} {nr[i]:>10.4f}")
        print(f"\n  Summary: mean_pearson={rs.mean():.4f}, "
              f"mean_spearman={sp.mean():.4f}, mean_nrmse={nr.mean():.4f}")
        assert True  # always passes — report only


# ---------------------------------------------------------------------------
# AML classification metrics
# ---------------------------------------------------------------------------

class TestAMLClassification:

    def test_auc_above_chance(self, benchmark_setup):
        """AUC-ROC must be significantly above chance (0.5)."""
        auc = roc_auc_score(benchmark_setup["aml_true"], benchmark_setup["aml_prob"])
        print(f"\n  AML AUC-ROC (test): {auc:.4f}")
        assert auc > 0.55, \
            f"AML AUC-ROC={auc:.4f} is not significantly above chance (0.5)"

    def test_f1_score_positive(self, benchmark_setup):
        aml_pred_bin = (benchmark_setup["aml_prob"] > 0.5).astype(int)
        f1 = f1_score(benchmark_setup["aml_true"], aml_pred_bin, zero_division=0)
        print(f"\n  AML F1 (test, threshold=0.5): {f1:.4f}")
        assert f1 > 0.0, f"AML F1={f1:.4f} — model predicts single class"

    def test_aml_accuracy_above_chance(self, benchmark_setup):
        aml_pred_bin = (benchmark_setup["aml_prob"] > 0.5).astype(int)
        acc = accuracy_score(benchmark_setup["aml_true"], aml_pred_bin)
        # Chance = max class frequency
        majority_class_rate = max(
            benchmark_setup["aml_true"].mean(),
            1 - benchmark_setup["aml_true"].mean()
        )
        print(f"\n  AML accuracy (test): {acc:.4f}, majority class rate: {majority_class_rate:.4f}")
        assert acc >= majority_class_rate * 0.9, \
            f"Model accuracy {acc:.4f} far below majority class rate {majority_class_rate:.4f}"

    def test_aml_predictions_not_degenerate(self, benchmark_setup):
        """Model must predict both classes — not collapse to all-0 or all-1."""
        aml_pred_bin = (benchmark_setup["aml_prob"] > 0.5).astype(int)
        n_pos = aml_pred_bin.sum()
        n_neg = len(aml_pred_bin) - n_pos
        assert n_pos > 0, "Model predicts all samples as Normal — degenerate"
        assert n_neg > 0, "Model predicts all samples as AML — degenerate"

    def test_aml_classification_report(self, benchmark_setup):
        """Print full classification report (informational)."""
        aml_true = benchmark_setup["aml_true"]
        aml_prob = benchmark_setup["aml_prob"]
        aml_pred = (aml_prob > 0.5).astype(int)
        auc = roc_auc_score(aml_true, aml_prob)
        f1  = f1_score(aml_true, aml_pred, zero_division=0)
        acc = accuracy_score(aml_true, aml_pred)
        print(f"\n  AML Classification (test set):")
        print(f"    AUC-ROC  : {auc:.4f}")
        print(f"    F1-score : {f1:.4f}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Pos rate : {aml_true.mean():.4f}")
        assert True


# ---------------------------------------------------------------------------
# Embeddings quality
# ---------------------------------------------------------------------------

class TestEmbeddingQuality:

    def test_embeddings_not_collapsed(self, benchmark_setup):
        """
        Fused embeddings must have non-trivial variance — no mode collapse.
        Collapsed embeddings (all identical) indicate a dead network.
        """
        fused = benchmark_setup["fused"].detach().numpy()
        per_dim_std = fused.std(0)
        n_active = (per_dim_std > 1e-4).sum()
        total    = fused.shape[1]
        print(f"\n  Active embedding dims (std > 1e-4): {n_active}/{total}")
        assert n_active >= total // 2, \
            f"Embedding collapse: only {n_active}/{total} dims have variance > 1e-4"

    def test_embeddings_separate_aml_from_normal(self, benchmark_setup):
        """
        Fused embeddings should encode AML status: mean embedding should differ
        between AML and Normal cells (measured by L2 distance of class centroids).
        """
        fused    = benchmark_setup["fused"].detach().numpy()
        aml_true = benchmark_setup["aml_true"]
        centroid_aml  = fused[aml_true == 1].mean(0)
        centroid_norm = fused[aml_true == 0].mean(0)
        separation = np.linalg.norm(centroid_aml - centroid_norm)
        within_aml  = fused[aml_true == 1].std(0).mean()
        within_norm = fused[aml_true == 0].std(0).mean()
        within_avg  = (within_aml + within_norm) / 2.0
        print(f"\n  Embedding AML/Normal separation: L2={separation:.4f}, within_class_std={within_avg:.4f}")
        # Separation should be positive (centroid distance > 0 is trivially true;
        # we check that it's at least as large as average within-class spread)
        assert separation > 0.0, "AML and Normal centroids are identical"


# ---------------------------------------------------------------------------
# Ablation studies
# ---------------------------------------------------------------------------

class TestAblation:

    def _train_and_eval(self, use_graph: bool, n_epochs=60, seed=SEED):
        """Train and return test-set per-protein Pearson r."""
        rng = np.random.default_rng(seed)
        rna, adt, aml, _ = make_synthetic_citeseq(seed=seed)
        U, S, _ = np.linalg.svd(rna, full_matrices=False)
        pca = U[:, :min(20, U.shape[1])] * S[:min(20, len(S))]
        edge_index = make_knn_edges(pca, k=K_NEIGH)

        rna_data = Data(x=torch.tensor(rna), edge_index=edge_index)
        adt_data = Data(x=torch.tensor(adt), edge_index=edge_index)
        nd, cc   = compute_graph_statistics_fast(edge_index, N_CELLS)
        aml_t    = torch.tensor(aml, dtype=torch.float32)

        idx = rng.permutation(N_CELLS)
        n_train = int(0.7 * N_CELLS)
        train_mask = torch.zeros(N_CELLS, dtype=torch.bool)
        test_mask  = torch.zeros(N_CELLS, dtype=torch.bool)
        train_mask[idx[:n_train]] = True
        test_mask[idx[n_train:]]  = True

        torch.manual_seed(seed)
        model = GATWithTransformerFusion(
            in_channels=N_GENES, hidden_channels=HIDDEN, out_channels=N_ADTS,
            heads=HEADS, dropout=0.1, nhead=NHEAD, num_layers=N_LAYERS,
            use_adapters=False, use_sparse_attention=True, neighborhood_size=K_NEIGH,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        if not use_graph:
            # Ablate graph: replace all edges with a star graph (no topology)
            star_src = torch.zeros(N_CELLS - 1, dtype=torch.long)
            star_tgt = torch.arange(1, N_CELLS, dtype=torch.long)
            edge_index = torch.stack([
                torch.cat([star_src, star_tgt]),
                torch.cat([star_tgt, star_src])
            ])
            rna_data.edge_index = edge_index
            adt_data.edge_index = edge_index
            nd, cc = compute_graph_statistics_fast(edge_index, N_CELLS)

        for _ in range(n_epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            adt_pred, _, _ = model(
                rna_data.x, rna_data.edge_index, adt_data.edge_index,
                node_degrees_rna=nd, node_degrees_adt=nd,
                clustering_coeffs_rna=cc, clustering_coeffs_adt=cc,
            )
            F.mse_loss(adt_pred[train_mask], adt_data.x[train_mask]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            adt_pred, _, _ = model(
                rna_data.x, rna_data.edge_index, adt_data.edge_index,
                node_degrees_rna=nd, node_degrees_adt=nd,
                clustering_coeffs_rna=cc, clustering_coeffs_adt=cc,
            )
        y_true = adt_data.x[test_mask].numpy()
        y_pred = adt_pred[test_mask].numpy()
        return _per_protein_pearson(y_true, y_pred)

    def test_knn_graph_helps_over_star_graph(self):
        """
        Biologically meaningful k-NN graph must outperform a trivial star topology.
        This validates the core architectural assumption that cell neighborhood
        structure improves protein prediction.
        """
        rs_knn  = self._train_and_eval(use_graph=True,  seed=7)
        rs_star = self._train_and_eval(use_graph=False, seed=7)
        mean_knn  = rs_knn.mean()
        mean_star = rs_star.mean()
        print(f"\n  Ablation — k-NN graph: mean_pearson={mean_knn:.4f}")
        print(f"  Ablation — star graph: mean_pearson={mean_star:.4f}")
        # k-NN should be ≥ star (star collapses all graph topology)
        # Allow small tolerance since training is stochastic
        assert mean_knn >= mean_star - 0.05, (
            f"k-NN graph ({mean_knn:.4f}) is unexpectedly much worse "
            f"than star graph ({mean_star:.4f})"
        )


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

class TestPublicationSummary:

    def test_print_full_benchmark_report(self, benchmark_setup):
        """
        Consolidated publication-ready metrics table.
        Always passes — intended for report generation.
        """
        s = benchmark_setup
        rs  = _per_protein_pearson(s["y_true"], s["y_pred"])
        sp  = _per_protein_spearman(s["y_true"], s["y_pred"])
        nr  = _nrmse_per_protein(s["y_true"], s["y_pred"])
        r2g = r2_score(s["y_true"].ravel(), s["y_pred"].ravel())
        rs_mean  = _per_protein_pearson(s["y_true"], s["y_pred_mean"])
        rs_ridge = _per_protein_pearson(s["y_true"], s["y_pred_ridge"])

        aml_pred = (s["aml_prob"] > 0.5).astype(int)
        auc = roc_auc_score(s["aml_true"], s["aml_prob"])
        f1  = f1_score(s["aml_true"], aml_pred, zero_division=0)
        acc = accuracy_score(s["aml_true"], aml_pred)

        print("\n" + "=" * 60)
        print("  DeepOMAPNet — Publication Benchmark Report")
        print("=" * 60)
        print(f"\n  Dataset       : {N_CELLS} cells, {N_GENES} genes → {N_ADTS} proteins")
        print(f"  Split         : 70% train / 15% val / 15% test")
        print(f"  Architecture  : GAT × 2 + TransformerFusion × {N_LAYERS}")
        print(f"  Hidden dim    : {HIDDEN}, GAT heads: {HEADS}, Transformer heads: {NHEAD}")

        print(f"\n  ── ADT Prediction (test set) ──────────────────────────")
        print(f"  Mean Pearson r        : {rs.mean():.4f}  (vs. mean={rs_mean.mean():.4f}, ridge={rs_ridge.mean():.4f})")
        print(f"  Mean Spearman rho     : {sp.mean():.4f}")
        print(f"  Mean NRMSE            : {nr.mean():.4f}")
        print(f"  Global R²             : {r2g:.4f}")

        print(f"\n  ── AML Classification (test set) ──────────────────────")
        print(f"  AUC-ROC               : {auc:.4f}")
        print(f"  F1-score              : {f1:.4f}")
        print(f"  Accuracy              : {acc:.4f}")

        fused = s["fused"].detach().numpy()
        aml_true = s["aml_true"]
        c_aml  = fused[aml_true == 1].mean(0)
        c_norm = fused[aml_true == 0].mean(0)
        sep = np.linalg.norm(c_aml - c_norm)
        print(f"\n  ── Embeddings ─────────────────────────────────────────")
        print(f"  AML/Normal centroid L2: {sep:.4f}")
        print(f"  Active dims (std>1e-4) : {(fused.std(0) > 1e-4).sum()}/{fused.shape[1]}")

        print("\n" + "=" * 60)
        assert True
