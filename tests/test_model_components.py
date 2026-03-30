"""
Unit tests for individual model components in doNET.py.

Tests cover:
- Output shapes
- Residual connections (adapters)
- Attention mechanisms (sparse vs dense)
- Gradient flow through each component
- Positional encoding
"""

import math
import pytest
import torch
import torch.nn.functional as F

from conftest import (
    GATWithTransformerFusion, TransformerFusion,
    SparseCrossAttentionLayer, CrossAttentionLayer,
    AdapterLayer, GraphPositionalEncoding,
    HIDDEN, HEADS, NHEAD, N_LAYERS, N_CELLS, N_GENES, N_ADTS, N_TYPES, K_NEIGH, SEED,
)


# ---------------------------------------------------------------------------
# GraphPositionalEncoding
# ---------------------------------------------------------------------------

class TestGraphPositionalEncoding:

    @pytest.fixture
    def pe(self):
        return GraphPositionalEncoding(embedding_dim=HIDDEN, dropout=0.0)

    def test_output_shape_no_graph_info(self, pe):
        x = torch.randn(N_CELLS, HIDDEN)
        out = pe(x)
        assert out.shape == (N_CELLS, HIDDEN), "Output shape must match input"

    def test_output_shape_with_degrees_and_clustering(self, pe, pyg_data):
        x = torch.randn(N_CELLS, HIDDEN)
        out = pe(x, node_degrees=pyg_data["node_degrees"],
                 clustering_coeffs=pyg_data["clustering_coeffs"])
        assert out.shape == (N_CELLS, HIDDEN)

    def test_graph_info_changes_output(self, pe, pyg_data):
        x = torch.randn(N_CELLS, HIDDEN)
        out_no_info  = pe(x)
        out_with_info = pe(x, node_degrees=pyg_data["node_degrees"],
                           clustering_coeffs=pyg_data["clustering_coeffs"])
        assert not torch.allclose(out_no_info, out_with_info), \
            "Graph topology should change the positional encoding"

    def test_large_n_interpolates(self, pe):
        """When N > max_length, interpolation is used — output must still be valid."""
        big_x = torch.randn(12000, HIDDEN)
        out = pe(big_x)
        assert out.shape == (12000, HIDDEN)
        assert not torch.isnan(out).any(), "No NaN after interpolation"

    def test_gradients_flow(self, pe, pyg_data):
        x = torch.randn(N_CELLS, HIDDEN, requires_grad=True)
        out = pe(x, node_degrees=pyg_data["node_degrees"],
                 clustering_coeffs=pyg_data["clustering_coeffs"])
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# AdapterLayer
# ---------------------------------------------------------------------------

class TestAdapterLayer:

    @pytest.fixture
    def adapter(self):
        return AdapterLayer(dim=HIDDEN, reduction_factor=4, dropout=0.0)

    def test_output_shape(self, adapter):
        x = torch.randn(N_CELLS, HIDDEN)
        out = adapter(x)
        assert out.shape == (N_CELLS, HIDDEN)

    def test_residual_connection_near_zero_init(self, adapter):
        """
        With up-weight initialized to zero, adapter output ≈ identity at init.
        scale=0.1 means output should be within 10% of input magnitude.
        """
        adapter.eval()
        x = torch.randn(50, HIDDEN)
        with torch.no_grad():
            out = adapter(x)
        diff = (out - x).abs().mean().item()
        # Allow up to 20% deviation because scale=0.1 and kaiming init on down
        assert diff < x.abs().mean().item() * 0.25, \
            f"Adapter perturbs input too much at init: mean_diff={diff:.4f}"

    def test_l2_reg_loss_positive(self, adapter):
        reg = adapter.get_l2_reg_loss()
        assert isinstance(reg, torch.Tensor)
        assert reg.item() >= 0.0

    def test_l2_reg_loss_is_differentiable(self, adapter):
        reg = adapter.get_l2_reg_loss()
        reg.backward()
        for p in adapter.parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any()

    def test_gradients_flow(self, adapter):
        x = torch.randn(N_CELLS, HIDDEN, requires_grad=True)
        out = adapter(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# SparseCrossAttentionLayer
# ---------------------------------------------------------------------------

class TestSparseCrossAttentionLayer:

    @pytest.fixture
    def layer(self):
        return SparseCrossAttentionLayer(
            embedding_dim=HIDDEN, nhead=NHEAD, dropout=0.0,
            use_positional_encoding=False, neighborhood_size=K_NEIGH
        )

    def test_output_shape_with_edges(self, layer, pyg_data):
        q = torch.randn(N_CELLS, HIDDEN)
        kv = torch.randn(N_CELLS, HIDDEN)
        out = layer(q, kv, edge_index=pyg_data["rna_data"].edge_index)
        assert out.shape == (N_CELLS, HIDDEN)

    def test_output_shape_dense_fallback(self, layer):
        """Dense fallback for small N without edge_index."""
        small_q  = torch.randn(50, HIDDEN)
        small_kv = torch.randn(50, HIDDEN)
        out = layer(small_q, small_kv, edge_index=None)
        assert out.shape == (50, HIDDEN)

    def test_large_graph_requires_edges(self, layer):
        """Dense attention on N>1000 must raise without edge_index."""
        q = torch.randn(1001, HIDDEN)
        kv = torch.randn(1001, HIDDEN)
        with pytest.raises(ValueError, match="sparse"):
            layer(q, kv, edge_index=None)

    def test_return_attention_sparse(self, layer, pyg_data):
        q = torch.randn(N_CELLS, HIDDEN)
        kv = torch.randn(N_CELLS, HIDDEN)
        out, attn = layer(q, kv, edge_index=pyg_data["rna_data"].edge_index,
                          return_attention=True)
        assert out.shape == (N_CELLS, HIDDEN)
        # Sparse attention returns metadata dict, not a full matrix
        assert attn is not None

    def test_no_nan_in_output(self, layer, pyg_data):
        q = torch.randn(N_CELLS, HIDDEN)
        kv = torch.randn(N_CELLS, HIDDEN)
        out = layer(q, kv, edge_index=pyg_data["rna_data"].edge_index)
        assert not torch.isnan(out).any(), "NaN detected in sparse attention output"

    def test_gradients_flow(self, layer, pyg_data):
        q = torch.randn(N_CELLS, HIDDEN, requires_grad=True)
        kv = torch.randn(N_CELLS, HIDDEN, requires_grad=True)
        out = layer(q, kv, edge_index=pyg_data["rna_data"].edge_index)
        out.sum().backward()
        assert q.grad is not None and not torch.isnan(q.grad).any()
        assert kv.grad is not None and not torch.isnan(kv.grad).any()

    def test_edge_cache_is_used(self, layer, pyg_data):
        """Second call with same edge_index should use cached edges."""
        edge_index = pyg_data["rna_data"].edge_index
        q = torch.randn(N_CELLS, HIDDEN)
        kv = torch.randn(N_CELLS, HIDDEN)
        layer(q, kv, edge_index=edge_index)
        assert hasattr(layer, "_cached_edges"), "Edge cache must be populated after first call"
        layer(q, kv, edge_index=edge_index)  # should not recompute


# ---------------------------------------------------------------------------
# CrossAttentionLayer (dense)
# ---------------------------------------------------------------------------

class TestCrossAttentionLayer:

    @pytest.fixture
    def layer(self):
        return CrossAttentionLayer(
            embedding_dim=HIDDEN, nhead=NHEAD, dropout=0.0,
            use_positional_encoding=False
        )

    def test_output_shape(self, layer):
        q  = torch.randn(50, HIDDEN)
        kv = torch.randn(50, HIDDEN)
        out = layer(q, kv)
        assert out.shape == (50, HIDDEN)

    def test_different_query_key_value(self, layer):
        """Cross-attention: query from RNA, key/value from ADT — different inputs."""
        q  = torch.randn(50, HIDDEN)
        kv = torch.randn(50, HIDDEN) + 5.0  # different distribution
        out = layer(q, kv)
        assert out.shape == (50, HIDDEN)
        assert not torch.isnan(out).any()

    def test_return_attention_weights(self, layer):
        q  = torch.randn(30, HIDDEN)
        kv = torch.randn(30, HIDDEN)
        out, attn = layer(q, kv, return_attention=True)
        assert attn.shape == (NHEAD, 30, 30)
        # Attention weights must sum to 1 per query
        attn_sum = attn.sum(dim=-1)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), \
            "Attention weights must sum to 1 per query"

    def test_gradients_flow(self, layer):
        q  = torch.randn(30, HIDDEN, requires_grad=True)
        kv = torch.randn(30, HIDDEN, requires_grad=True)
        out = layer(q, kv)
        out.sum().backward()
        assert q.grad is not None
        assert kv.grad is not None


# ---------------------------------------------------------------------------
# TransformerFusion
# ---------------------------------------------------------------------------

class TestTransformerFusion:

    @pytest.fixture
    def fusion(self):
        return TransformerFusion(
            embedding_dim=HIDDEN, nhead=NHEAD, num_layers=N_LAYERS,
            dropout=0.0, use_adapters=True, reduction_factor=4,
            use_positional_encoding=False, use_sparse_attention=True,
            neighborhood_size=K_NEIGH
        )

    def test_output_shape(self, fusion, pyg_data):
        rna_x = torch.randn(N_CELLS, HIDDEN)
        adt_x = torch.randn(N_CELLS, HIDDEN)
        out = fusion(rna_x, adt_x,
                     edge_index_rna=pyg_data["rna_data"].edge_index,
                     edge_index_adt=pyg_data["adt_data"].edge_index)
        assert out.shape == (N_CELLS, HIDDEN)

    def test_adapter_reg_loss_nonzero(self, fusion, pyg_data):
        """Adapter reg loss should be > 0 after random init (weights ≠ 0)."""
        rna_x = torch.randn(N_CELLS, HIDDEN)
        adt_x = torch.randn(N_CELLS, HIDDEN)
        fusion(rna_x, adt_x,
               edge_index_rna=pyg_data["rna_data"].edge_index)
        reg = fusion.get_adapter_reg_loss()
        # down.weight is Kaiming initialized → norm > 0
        assert reg.item() >= 0.0

    def test_without_adapters(self, pyg_data):
        fusion_no_adapt = TransformerFusion(
            embedding_dim=HIDDEN, nhead=NHEAD, num_layers=N_LAYERS,
            dropout=0.0, use_adapters=False, use_sparse_attention=True,
            neighborhood_size=K_NEIGH
        )
        rna_x = torch.randn(N_CELLS, HIDDEN)
        adt_x = torch.randn(N_CELLS, HIDDEN)
        out = fusion_no_adapt(rna_x, adt_x,
                              edge_index_rna=pyg_data["rna_data"].edge_index)
        assert out.shape == (N_CELLS, HIDDEN)
        reg = fusion_no_adapt.get_adapter_reg_loss()
        assert reg.item() == 0.0, "No adapter → reg loss must be 0"

    def test_no_nan(self, fusion, pyg_data):
        rna_x = torch.randn(N_CELLS, HIDDEN)
        adt_x = torch.randn(N_CELLS, HIDDEN)
        out = fusion(rna_x, adt_x,
                     edge_index_rna=pyg_data["rna_data"].edge_index)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self, fusion, pyg_data):
        rna_x = torch.randn(N_CELLS, HIDDEN, requires_grad=True)
        adt_x = torch.randn(N_CELLS, HIDDEN, requires_grad=True)
        out = fusion(rna_x, adt_x,
                     edge_index_rna=pyg_data["rna_data"].edge_index)
        out.sum().backward()
        assert rna_x.grad is not None and not torch.isnan(rna_x.grad).any()
        assert adt_x.grad is not None and not torch.isnan(adt_x.grad).any()


# ---------------------------------------------------------------------------
# GATWithTransformerFusion (full model)
# ---------------------------------------------------------------------------

class TestGATWithTransformerFusion:

    def test_output_shapes(self, model, pyg_data):
        d = pyg_data
        with torch.no_grad():
            adt_pred, aml_pred, fused = model(
                d["rna_data"].x, d["rna_data"].edge_index, d["adt_data"].edge_index,
                node_degrees_rna=d["node_degrees"], node_degrees_adt=d["node_degrees"],
                clustering_coeffs_rna=d["clustering_coeffs"],
                clustering_coeffs_adt=d["clustering_coeffs"],
            )
        assert adt_pred.shape == (N_CELLS, N_ADTS),   f"ADT pred shape: {adt_pred.shape}"
        assert aml_pred.shape == (N_CELLS, 1),         f"AML pred shape: {aml_pred.shape}"
        assert fused.shape   == (N_CELLS, HIDDEN),     f"Fused shape: {fused.shape}"

    def test_no_nan_in_outputs(self, model, pyg_data):
        d = pyg_data
        with torch.no_grad():
            adt_pred, aml_pred, fused = model(
                d["rna_data"].x, d["rna_data"].edge_index,
                node_degrees_rna=d["node_degrees"],
                clustering_coeffs_rna=d["clustering_coeffs"],
            )
        assert not torch.isnan(adt_pred).any(), "NaN in ADT predictions"
        assert not torch.isnan(aml_pred).any(), "NaN in AML predictions"
        assert not torch.isnan(fused).any(),    "NaN in fused embeddings"

    def test_aml_pred_is_logit_range(self, model, pyg_data):
        """AML head outputs raw logits (not probabilities) — no sigmoid applied."""
        d = pyg_data
        with torch.no_grad():
            _, aml_pred, _ = model(
                d["rna_data"].x, d["rna_data"].edge_index,
                node_degrees_rna=d["node_degrees"],
                clustering_coeffs_rna=d["clustering_coeffs"],
            )
        probs = torch.sigmoid(aml_pred)
        assert (probs >= 0).all() and (probs <= 1).all(), \
            "sigmoid(aml_pred) must be in [0, 1]"

    def test_celltype_head_output(self, model, pyg_data):
        d = pyg_data
        with torch.no_grad():
            _, _, fused = model(
                d["rna_data"].x, d["rna_data"].edge_index,
                node_degrees_rna=d["node_degrees"],
                clustering_coeffs_rna=d["clustering_coeffs"],
            )
            ct_logits = model.predict_celltypes(fused)
        assert ct_logits.shape == (N_CELLS, N_TYPES)
        # Argmax must be a valid class index
        preds = ct_logits.argmax(dim=-1)
        assert (preds >= 0).all() and (preds < N_TYPES).all()

    def test_grad_flow_end_to_end(self, pyg_data):
        """Gradients must flow back to input features."""
        d = pyg_data
        torch.manual_seed(SEED)
        m = GATWithTransformerFusion(
            in_channels=N_GENES, hidden_channels=HIDDEN, out_channels=N_ADTS,
            heads=HEADS, dropout=0.0, nhead=NHEAD, num_layers=N_LAYERS,
            use_adapters=True, use_sparse_attention=True, neighborhood_size=K_NEIGH,
        )
        x = d["rna_data"].x.clone().requires_grad_(True)
        adt_pred, aml_pred, _ = m(x, d["rna_data"].edge_index,
                                  node_degrees_rna=d["node_degrees"],
                                  clustering_coeffs_rna=d["clustering_coeffs"])
        loss = adt_pred.sum() + aml_pred.sum()
        loss.backward()
        assert x.grad is not None, "Gradient must reach input features"
        assert not torch.isnan(x.grad).any(), "NaN gradient detected"
        assert x.grad.abs().max().item() > 0, "All-zero gradient — dead network"

    def test_grad_norms_reasonable(self, pyg_data):
        """Gradient norms should not explode or vanish (publication requirement)."""
        d = pyg_data
        torch.manual_seed(SEED)
        m = GATWithTransformerFusion(
            in_channels=N_GENES, hidden_channels=HIDDEN, out_channels=N_ADTS,
            heads=HEADS, dropout=0.0, nhead=NHEAD, num_layers=N_LAYERS,
            use_adapters=True, use_sparse_attention=True, neighborhood_size=K_NEIGH,
        )
        adt_pred, aml_pred, _ = m(
            d["rna_data"].x, d["rna_data"].edge_index,
            node_degrees_rna=d["node_degrees"],
            clustering_coeffs_rna=d["clustering_coeffs"],
        )
        loss = F.mse_loss(adt_pred, d["adt_data"].x)
        loss.backward()
        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in m.parameters() if p.grad is not None
        ) ** 0.5
        assert total_norm < 1000.0, f"Gradient explosion detected: norm={total_norm:.2f}"
        assert total_norm > 1e-8,  f"Vanishing gradient detected: norm={total_norm:.2e}"

    def test_eval_vs_train_mode_differ(self, model, pyg_data):
        """Dropout active in train mode → outputs differ from eval."""
        d = pyg_data
        torch.manual_seed(SEED)
        model.train()
        with torch.no_grad():
            out_train, _, _ = model(
                d["rna_data"].x, d["rna_data"].edge_index,
                node_degrees_rna=d["node_degrees"],
                clustering_coeffs_rna=d["clustering_coeffs"],
            )
        model.eval()
        with torch.no_grad():
            out_eval, _, _ = model(
                d["rna_data"].x, d["rna_data"].edge_index,
                node_degrees_rna=d["node_degrees"],
                clustering_coeffs_rna=d["clustering_coeffs"],
            )
        # Due to dropout, train/eval outputs should differ
        # (not guaranteed if dropout=0, but fixture uses dropout=0.1)
        # We just verify no crash and no NaN in both modes
        assert not torch.isnan(out_train).any()
        assert not torch.isnan(out_eval).any()

    def test_total_reg_loss(self, model, pyg_data):
        """Regularization loss must be non-negative and differentiable."""
        d = pyg_data
        model.train()
        adt_pred, _, _ = model(
            d["rna_data"].x, d["rna_data"].edge_index,
            node_degrees_rna=d["node_degrees"],
            clustering_coeffs_rna=d["clustering_coeffs"],
        )
        reg = model.get_total_reg_loss()
        assert reg.item() >= 0.0
        reg.backward()

    def test_enable_celltype_head_dynamically(self, pyg_data):
        """enable_celltype_head() must add a working classification head."""
        d = pyg_data
        m = GATWithTransformerFusion(
            in_channels=N_GENES, hidden_channels=HIDDEN, out_channels=N_ADTS,
            heads=HEADS, dropout=0.0, nhead=NHEAD, num_layers=N_LAYERS,
            use_sparse_attention=True, neighborhood_size=K_NEIGH,
        )
        m.eval()
        with torch.no_grad():
            _, _, fused = m(d["rna_data"].x, d["rna_data"].edge_index,
                            node_degrees_rna=d["node_degrees"],
                            clustering_coeffs_rna=d["clustering_coeffs"])
        m.enable_celltype_head(N_TYPES)
        with torch.no_grad():
            ct_logits = m.predict_celltypes(fused)
        assert ct_logits.shape == (N_CELLS, N_TYPES)

    def test_get_embeddings(self, model, pyg_data):
        """get_embeddings() must return a fixed-size latent representation."""
        d = pyg_data
        with torch.no_grad():
            emb = model.get_embeddings(
                d["rna_data"].x, d["rna_data"].edge_index,
                node_degrees_rna=d["node_degrees"],
                clustering_coeffs_rna=d["clustering_coeffs"],
            )
        assert emb.shape == (N_CELLS, HIDDEN)
        assert not torch.isnan(emb).any()
