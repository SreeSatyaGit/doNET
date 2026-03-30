import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.utils import softmax as segment_softmax
from torch_scatter import scatter_add
from torch import nn
from typing import Optional
import math

class GraphPositionalEncoding(nn.Module):
    """Graph-aware positional encoding based on node connectivity"""
    def __init__(self, embedding_dim, max_length=10000, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        
        self.pos_embedding = nn.Parameter(torch.randn(max_length, embedding_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        self.degree_embedding = nn.Linear(1, embedding_dim // 4)
        self.clustering_embedding = nn.Linear(1, embedding_dim // 4)
        
    def forward(self, x, edge_index=None, node_degrees=None, clustering_coeffs=None):
        """
        Args:
            x: Node features [N, embedding_dim]
            edge_index: Graph edges [2, E] (optional)
            node_degrees: Node degrees [N] (optional)
            clustering_coeffs: Clustering coefficients [N] (optional)
        """
        N = x.size(0)
        
        if N <= self.pos_embedding.size(0):
            pos_enc = self.pos_embedding[:N]
        else:
            pos_enc = F.interpolate(
                self.pos_embedding.unsqueeze(0).transpose(1, 2),
                size=N, mode='linear', align_corners=False
            ).squeeze(0).transpose(0, 1)
        
        base_dim = self.embedding_dim // 2
        quarter_dim = self.embedding_dim // 4
        base_enc = pos_enc[:, :base_dim]
        
        degree_enc = None
        clustering_enc = None
        if clustering_coeffs is not None:
            clustering_enc = self.clustering_embedding(clustering_coeffs.unsqueeze(-1).float())
        else:
            clustering_enc = torch.zeros(x.size(0), quarter_dim, device=x.device, dtype=x.dtype)
        if node_degrees is not None:
            degree_enc = self.degree_embedding(node_degrees.unsqueeze(-1).float())
        else:
            degree_enc = torch.zeros(x.size(0), quarter_dim, device=x.device, dtype=x.dtype)
        
        pos_enc = torch.cat([base_enc, clustering_enc, degree_enc], dim=-1)
        
        return self.dropout(x + pos_enc)

class SparseCrossAttentionLayer(nn.Module):
    """Truly sparse cross-attention layer using edge lists (no dense masks)"""
    def __init__(self, embedding_dim, nhead=8, dropout=0.1, use_positional_encoding=True, 
                 neighborhood_size=50):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nhead = nhead
        self.head_dim = embedding_dim // nhead
        self.use_positional_encoding = use_positional_encoding
        self.neighborhood_size = neighborhood_size
        
        assert embedding_dim % nhead == 0, "embedding_dim must be divisible by nhead"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization (pre-LN)
        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(embedding_dim)
        self.norm_out = nn.LayerNorm(embedding_dim)
        
        # Positional encoding for biological topology
        if use_positional_encoding:
            self.pos_encoding = GraphPositionalEncoding(embedding_dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _preprocess_edges(self, edge_index, num_nodes, device):
        """Create symmetric, self-loop-augmented, optionally pruned edge list.
        Cached per-instance to avoid recomputation each forward.
        """
        if hasattr(self, '_cached_edges'):
            cached_num_nodes, cached_device, cached_ptr, cached_version, cached_shape, cached_edges = self._cached_edges
            if (
                cached_num_nodes == num_nodes
                and cached_device == device
                and cached_ptr == edge_index.data_ptr()
                and cached_version == edge_index._version
                and cached_shape == tuple(edge_index.shape)
            ):
                return cached_edges
        
        row, col = edge_index[0], edge_index[1]
        
        row_sym = torch.cat([row, col], dim=0)
        col_sym = torch.cat([col, row], dim=0)
        edge_index_sym = torch.stack([row_sym, col_sym], dim=0)
        
        self_loops = torch.arange(num_nodes, device=device)
        loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index_sym = torch.cat([edge_index_sym, loops], dim=1)
        
        if self.neighborhood_size < num_nodes:
            row, col = edge_index_sym[0], edge_index_sym[1]
            sort_idx = torch.argsort(row)
            row = row[sort_idx]
            col = col[sort_idx]
            
            deg = torch.bincount(row, minlength=num_nodes)
            starts = torch.zeros(num_nodes + 1, device=device, dtype=torch.long)
            starts[1:] = torch.cumsum(deg, dim=0)
            
            keep_mask = torch.ones(row.numel(), device=device, dtype=torch.bool)
            overfull = torch.where(deg > self.neighborhood_size)[0]
            if overfull.numel() > 0:
                idx_ranges = torch.stack([starts[overfull], starts[overfull] + deg[overfull]], dim=1)
                for s, e in idx_ranges.tolist():
                    keep_mask[s + self.neighborhood_size:e] = False
            row = row[keep_mask]; col = col[keep_mask]
            edge_index_sym = torch.stack([row, col], dim=0)
        
        self._cached_edges = (
            num_nodes,
            device,
            edge_index.data_ptr(),
            edge_index._version,
            tuple(edge_index.shape),
            edge_index_sym,
        )
        return edge_index_sym
    
    def _sparse_attention_vectorized(self, q, k, v, edge_index):
        """
        Compute truly sparse attention using edge lists (vectorized, no dense masks)
        Args:
            q, k, v: [nhead, N, head_dim]
            edge_index: [2, E] edge list
        Returns:
            out: [nhead, N, head_dim]
            attn_weights: None (to save memory)
        """
        nhead, N, head_dim = q.shape
        device = q.device
        
        q = q.view(nhead, N, head_dim)
        k = k.view(nhead, N, head_dim)
        v = v.view(nhead, N, head_dim)
        
        src = edge_index[0]
        tgt = edge_index[1]
        
        q_src = q[:, src, :]
        k_tgt = k[:, tgt, :]
        v_tgt = v[:, tgt, :]
        
        scores = (q_src * k_tgt).sum(dim=-1) * self.scale
        head_offsets = (torch.arange(nhead, device=device) * N).unsqueeze(1)
        segment_ids = head_offsets + src.unsqueeze(0)
        attn = segment_softmax(scores.flatten(), segment_ids.flatten())
        attn = attn.view(nhead, -1)
        attn = self.dropout(attn)

        attn_exp = attn.unsqueeze(-1)
        contrib = attn_exp * v_tgt

        flat_src = (head_offsets + src.unsqueeze(0)).reshape(-1)
        contrib_flat = contrib.reshape(-1, head_dim)
        out_flat = scatter_add(contrib_flat, flat_src, dim=0, dim_size=nhead * N)
        out = out_flat.view(nhead, N, head_dim)

        return out, None
    
    def forward(self, query, key_value, edge_index=None, node_degrees=None, 
                clustering_coeffs=None, return_attention=False):
        """
        Args:
            query: RNA embeddings [N, embedding_dim]
            key_value: ADT embeddings [N, embedding_dim]
            edge_index: Graph edges [2, E] (optional)
            node_degrees: Node degrees [N] (optional)
            clustering_coeffs: Clustering coefficients [N] (optional)
            return_attention: Whether to return attention weights
        Returns:
            fused_embeddings: [N, embedding_dim]
            attention_weights: [nhead, N, N] (if return_attention=True)
        """
        if self.use_positional_encoding:
            query = self.pos_encoding(query, edge_index, node_degrees, clustering_coeffs)
            key_value = self.pos_encoding(key_value, edge_index, node_degrees, clustering_coeffs)
        
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        
        N = q.size(0)
        q = q.view(N, self.nhead, self.head_dim).transpose(0, 1)
        k = k.view(N, self.nhead, self.head_dim).transpose(0, 1)
        v = v.view(N, self.nhead, self.head_dim).transpose(0, 1)
        
        if edge_index is not None:
            edge_list = self._preprocess_edges(edge_index, N, q.device)
            out, attn_weights = self._sparse_attention_vectorized(q, k, v, edge_list)
        else:
            if N < 1000:
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                out = torch.matmul(attn_weights, v)
            else:
                raise ValueError(f"Graph too large ({N} nodes) for dense attention. Provide edge_index for sparse attention.")
        
        out = out.transpose(0, 1).contiguous().view(N, self.embedding_dim)
        
        out = self.out_proj(out)
        out = self.norm_out(out)
        
        if return_attention:
            if attn_weights is None:
                return out, {"sparse_attention": True, "message": "Sparse attention used - no full attention matrix stored"}
            return out, attn_weights
        return out

class CrossAttentionLayer(nn.Module):
    """Original dense cross-attention layer (kept for backward compatibility)"""
    def __init__(self, embedding_dim, nhead=8, dropout=0.1, use_positional_encoding=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nhead = nhead
        self.head_dim = embedding_dim // nhead
        self.use_positional_encoding = use_positional_encoding
        
        assert embedding_dim % nhead == 0, "embedding_dim must be divisible by nhead"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization (pre-LN)
        self.norm_q = nn.LayerNorm(embedding_dim)
        self.norm_kv = nn.LayerNorm(embedding_dim)
        self.norm_out = nn.LayerNorm(embedding_dim)
        
        # Positional encoding for biological topology
        if use_positional_encoding:
            self.pos_encoding = GraphPositionalEncoding(embedding_dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, query, key_value, edge_index=None, node_degrees=None, 
                clustering_coeffs=None, return_attention=False):
        """
        Args:
            query: RNA embeddings [N, embedding_dim]
            key_value: ADT embeddings [N, embedding_dim]
            edge_index: Graph edges [2, E] (optional)
            node_degrees: Node degrees [N] (optional)
            clustering_coeffs: Clustering coefficients [N] (optional)
            return_attention: Whether to return attention weights
        Returns:
            fused_embeddings: [N, embedding_dim]
            attention_weights: [nhead, N, N] (if return_attention=True)
        """
        if self.use_positional_encoding:
            query = self.pos_encoding(query, edge_index, node_degrees, clustering_coeffs)
            key_value = self.pos_encoding(key_value, edge_index, node_degrees, clustering_coeffs)
        
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        
        N = q.size(0)
        q = q.view(N, self.nhead, self.head_dim).transpose(0, 1)
        k = k.view(N, self.nhead, self.head_dim).transpose(0, 1)
        v = v.view(N, self.nhead, self.head_dim).transpose(0, 1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights_dropped, v)
        
        out = out.transpose(0, 1).contiguous().view(N, self.embedding_dim)
        
        out = self.out_proj(out)
        out = self.norm_out(out)
        
        if return_attention:
            return out, attn_weights
        return out

class AdapterLayer(nn.Module):
    """Enhanced adapter layer with regularization and improved initialization"""
    def __init__(self, dim, reduction_factor=4, dropout=0.1, use_layernorm=True, 
                 adapter_l2_reg=1e-4, init_scale=0.1):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.adapter_l2_reg = adapter_l2_reg
        hidden_dim = max(dim // reduction_factor, 16)
        
        if use_layernorm:
            self.norm = nn.LayerNorm(dim)
        
        self.down = nn.Linear(dim, hidden_dim)
        nn.init.kaiming_normal_(self.down.weight, nonlinearity='relu')
        nn.init.zeros_(self.down.bias)
        
        self.up = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
        
        self.activation_dropout = nn.Dropout(dropout * 0.5)
        
    def forward(self, x):
        identity = x
        
        if self.use_layernorm:
            x = self.norm(x)
        
        x = self.down(x)
        x = F.gelu(x)
        x = self.activation_dropout(x)
        
        x = self.up(x)
        x = self.dropout(x)
        
        return identity + self.scale * x
    
    def get_l2_reg_loss(self):
        """Compute L2 regularization loss for adapter parameters"""
        l2_loss = torch.tensor(0.0, device=self.down.weight.device)
        for param in [self.down.weight, self.up.weight]:
            l2_loss += torch.norm(param, p=2) ** 2
        return self.adapter_l2_reg * l2_loss

class TransformerFusion(nn.Module):
    def __init__(self, embedding_dim, nhead=8, num_layers=3, dropout=0.1, use_adapters=True,
                 reduction_factor=4, adapter_l2_reg=1e-4, use_positional_encoding=True,
                 use_sparse_attention=True, neighborhood_size=50):
        super().__init__()
        self.num_layers = num_layers
        self.use_adapters = use_adapters
        self.adapter_l2_reg = adapter_l2_reg
        self.use_positional_encoding = use_positional_encoding
        self.use_sparse_attention = use_sparse_attention
        
        self.rna_proj = nn.Linear(embedding_dim, embedding_dim)
        self.adt_proj = nn.Linear(embedding_dim, embedding_dim)
        
        nn.init.xavier_uniform_(self.rna_proj.weight)
        nn.init.xavier_uniform_(self.adt_proj.weight)
        nn.init.zeros_(self.rna_proj.bias)
        nn.init.zeros_(self.adt_proj.bias)
        
        if use_sparse_attention:
            self.cross_attention_layers = nn.ModuleList([
                SparseCrossAttentionLayer(embedding_dim, nhead=nhead, dropout=dropout, 
                                         use_positional_encoding=use_positional_encoding,
                                         neighborhood_size=neighborhood_size)
                for _ in range(num_layers)
            ])
        else:
            self.cross_attention_layers = nn.ModuleList([
                CrossAttentionLayer(embedding_dim, nhead=nhead, dropout=dropout, 
                                   use_positional_encoding=use_positional_encoding)
                for _ in range(num_layers)
            ])
        
        self.rna_transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=embedding_dim,
                out_channels=embedding_dim // nhead,
                heads=nhead,
                dropout=dropout,
                edge_dim=None,
                concat=True
            ) for _ in range(num_layers)
        ])
        
        self.adt_transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=embedding_dim,
                out_channels=embedding_dim // nhead,
                heads=nhead,
                dropout=dropout,
                edge_dim=None,
                concat=True
            ) for _ in range(num_layers)
        ])
        
        if use_adapters:
            self.adapters = nn.ModuleList([
                AdapterLayer(embedding_dim, reduction_factor=reduction_factor, 
                           dropout=dropout, adapter_l2_reg=adapter_l2_reg)
                for _ in range(num_layers)
            ])
        
        self.rna_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])
        self.adt_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.final_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, rna_x, adt_x, edge_index_rna, edge_index_adt=None, 
                node_degrees_rna=None, node_degrees_adt=None,
                clustering_coeffs_rna=None, clustering_coeffs_adt=None,
                return_attention=False):
        """
        Args:
            rna_x: RNA embeddings [N, embedding_dim]
            adt_x: ADT embeddings [N, embedding_dim]
            edge_index_rna: RNA graph edges [2, E_rna]
            edge_index_adt: ADT graph edges [2, E_adt] (optional)
            node_degrees_rna: RNA node degrees [N] (optional)
            node_degrees_adt: ADT node degrees [N] (optional)
            clustering_coeffs_rna: RNA clustering coefficients [N] (optional)
            clustering_coeffs_adt: ADT clustering coefficients [N] (optional)
            return_attention: Whether to return attention weights
        Returns:
            fused_embeddings: [N, embedding_dim]
            attention_weights: List of attention weights (if return_attention=True)
        """
        if edge_index_adt is None:
            edge_index_adt = edge_index_rna
            
        rna_proj = self.rna_proj(rna_x)
        adt_proj = self.adt_proj(adt_x)
        
        attention_weights = [] if return_attention else None
        
        for i in range(self.num_layers):
            rna_res = self.rna_transformer_layers[i](rna_proj, edge_index_rna)
            adt_res = self.adt_transformer_layers[i](adt_proj, edge_index_adt)
            
            if self.use_adapters:
                rna_res = self.adapters[i](rna_res)
                adt_res = self.adapters[i](adt_res)
            
            rna_proj = self.rna_norms[i](rna_proj + rna_res)
            adt_proj = self.adt_norms[i](adt_proj + adt_res)
            
            if return_attention:
                rna_fused, rna_attn = self.cross_attention_layers[i](
                    rna_proj, adt_proj, edge_index_rna, node_degrees_rna, 
                    clustering_coeffs_rna, return_attention=True
                )
                adt_fused, adt_attn = self.cross_attention_layers[i](
                    adt_proj, rna_proj, edge_index_adt, node_degrees_adt,
                    clustering_coeffs_adt, return_attention=True
                )
                attention_weights.append({
                    'rna_to_adt': rna_attn,
                    'adt_to_rna': adt_attn,
                    'layer': i
                })
            else:
                rna_fused = self.cross_attention_layers[i](
                    rna_proj, adt_proj, edge_index_rna, node_degrees_rna, 
                    clustering_coeffs_rna, return_attention=False
                )
                adt_fused = self.cross_attention_layers[i](
                    adt_proj, rna_proj, edge_index_adt, node_degrees_adt,
                    clustering_coeffs_adt, return_attention=False
                )
            
            rna_proj = rna_proj + self.dropout(rna_fused)
            adt_proj = adt_proj + self.dropout(adt_fused)
        
        fused_embeddings = self.final_fusion(torch.cat([rna_proj, adt_proj], dim=-1))
        
        if return_attention:
            return fused_embeddings, attention_weights
        return fused_embeddings
    
    def get_adapter_reg_loss(self):
        """Compute total adapter regularization loss"""
        if not self.use_adapters:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        total_reg_loss = 0.0
        for adapter in self.adapters:
            total_reg_loss += adapter.get_l2_reg_loss()
        return total_reg_loss

class GATWithTransformerFusion(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6, 
                 nhead=8, num_layers=3, use_adapters=True, reduction_factor=4, 
                 adapter_l2_reg=1e-4, use_positional_encoding=True, 
                 use_sparse_attention=True, neighborhood_size=50,
                 num_cell_types: Optional[int] = None):
        super().__init__()
        self.dropout = dropout
        self.use_adapters = use_adapters
        self.adapter_l2_reg = adapter_l2_reg
        self.use_positional_encoding = use_positional_encoding
        self.use_sparse_attention = use_sparse_attention
        self.num_cell_types = num_cell_types
        self.gat_rna1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat_rna2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        
        self.gat_adt_init = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        
        self.transformer_fusion = TransformerFusion(
            embedding_dim=hidden_channels,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_adapters=use_adapters,
            reduction_factor=reduction_factor,
            adapter_l2_reg=adapter_l2_reg,
            use_positional_encoding=use_positional_encoding,
            use_sparse_attention=use_sparse_attention,
            neighborhood_size=neighborhood_size
        )
        
        self.gat_adt = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        if self.num_cell_types is not None and self.num_cell_types > 1:
            self.celltype_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.LayerNorm(hidden_channels // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, self.num_cell_types)
            )
        else:
            self.celltype_head = None
        
        self.batch_norm_rna = nn.BatchNorm1d(hidden_channels)
        self.batch_norm_adt = nn.BatchNorm1d(hidden_channels)
        
        self._init_final_layers()
        
    def _init_final_layers(self):
        """Initialize final projection layers with proper scaling"""
        for module in self.final_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        for module in self.classification_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        if hasattr(self, 'celltype_head') and self.celltype_head is not None:
            for module in self.celltype_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
    def forward(self, x, edge_index_rna, edge_index_adt=None, return_attention=False,
                node_degrees_rna=None, node_degrees_adt=None,
                clustering_coeffs_rna=None, clustering_coeffs_adt=None):
        """
        Args:
            x: Input features [N, in_channels]
            edge_index_rna: RNA graph edges [2, E_rna]
            edge_index_adt: ADT graph edges [2, E_adt] (optional, uses RNA edges if None)
            return_attention: Whether to return attention weights
            node_degrees_rna: RNA node degrees [N] (optional)
            node_degrees_adt: ADT node degrees [N] (optional)
            clustering_coeffs_rna: RNA clustering coefficients [N] (optional)
            clustering_coeffs_adt: ADT clustering coefficients [N] (optional)
        Returns:
            adt_pred: ADT predictions [N, out_channels]
            aml_pred: AML classification predictions [N, 1]
            fused_embeddings: Fused embeddings [N, hidden_channels]
            (optional) If return_attention=True: attention weights structure
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat_rna1(x, edge_index_rna)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        rna_embeddings = self.gat_rna2(x, edge_index_rna)
        rna_embeddings = F.elu(rna_embeddings)
        rna_embeddings = self.batch_norm_rna(rna_embeddings)
        
        edge_index_adt = edge_index_adt if edge_index_adt is not None else edge_index_rna
        initial_adt = self.gat_adt_init(rna_embeddings, edge_index_adt)
        initial_adt = F.elu(initial_adt)
        initial_adt = self.batch_norm_adt(initial_adt)
        
        if return_attention:
            fused_embeddings, attention_weights = self.transformer_fusion(
                rna_embeddings, initial_adt, edge_index_rna, edge_index_adt,
                node_degrees_rna, node_degrees_adt,
                clustering_coeffs_rna, clustering_coeffs_adt,
                return_attention=True
            )
        else:
            fused_embeddings = self.transformer_fusion(
                rna_embeddings, initial_adt, edge_index_rna, edge_index_adt,
                node_degrees_rna, node_degrees_adt,
                clustering_coeffs_rna, clustering_coeffs_adt,
                return_attention=False
            )
        
        adt_features = self.gat_adt(fused_embeddings, edge_index_adt)
        
        adt_pred = self.final_proj(adt_features)
        
        aml_pred = self.classification_head(adt_features)
        
        if return_attention:
            return adt_pred, aml_pred, fused_embeddings, attention_weights
        return adt_pred, aml_pred, fused_embeddings

    def predict_celltypes(self, fused_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict multi-class cell types from fused embeddings.
        Args:
            fused_embeddings: [N, hidden_channels]
        Returns:
            logits: [N, num_cell_types]
        """
        if self.celltype_head is None:
            raise RuntimeError("Cell type head not initialized. Recreate model with num_cell_types set or call enable_celltype_head(num_classes).")
        return self.celltype_head(fused_embeddings)

    def enable_celltype_head(self, num_cell_types: int, dropout: float | None = None) -> None:
        """Dynamically add/enable the cell type head after initialization."""
        self.num_cell_types = int(num_cell_types)
        hidden_channels = self.classification_head[0].in_features
        p = dropout if dropout is not None else float(self.dropout)
        self.celltype_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_channels // 2, self.num_cell_types)
        )
        for module in self.celltype_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_embeddings(self, x, edge_index_rna, edge_index_adt=None,
                       node_degrees_rna=None, node_degrees_adt=None,
                       clustering_coeffs_rna=None, clustering_coeffs_adt=None):
        """Get embeddings for downstream tasks"""
        with torch.no_grad():
            x = F.dropout(x, p=self.dropout, training=False)
            x = self.gat_rna1(x, edge_index_rna)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=False)
            rna_embeddings = self.gat_rna2(x, edge_index_rna)
            rna_embeddings = F.elu(rna_embeddings)
            
            edge_index_adt = edge_index_adt if edge_index_adt is not None else edge_index_rna
            initial_adt = self.gat_adt_init(rna_embeddings, edge_index_adt)
            initial_adt = F.elu(initial_adt)
            
            fused_embeddings = self.transformer_fusion(
                rna_embeddings, initial_adt, edge_index_rna, edge_index_adt,
                node_degrees_rna, node_degrees_adt,
                clustering_coeffs_rna, clustering_coeffs_adt,
                return_attention=False
            )
            return fused_embeddings
    
    def get_attention_weights(self, x, edge_index_rna, edge_index_adt=None,
                            node_degrees_rna=None, node_degrees_adt=None,
                            clustering_coeffs_rna=None, clustering_coeffs_adt=None):
        """
        Extract attention weights from all layers for visualization.
        
        Args:
            x: Input features [N, in_channels]
            edge_index_rna: RNA graph edges [2, E_rna]
            edge_index_adt: ADT graph edges [2, E_adt] (optional)
            node_degrees_rna: RNA node degrees [N] (optional)
            node_degrees_adt: ADT node degrees [N] (optional)
            clustering_coeffs_rna: RNA clustering coefficients [N] (optional)
            clustering_coeffs_adt: ADT clustering coefficients [N] (optional)
        
        Returns:
            attention_dict: Dictionary containing attention weights from each layer
        """
        attention_dict = {}
        
        with torch.no_grad():
            x = F.dropout(x, p=self.dropout, training=False)
            
            x = self.gat_rna1(x, edge_index_rna)
            x = F.elu(x)
            attention_dict['gat_rna1'] = None
            
            x = F.dropout(x, p=self.dropout, training=False)
            
            rna_embeddings = self.gat_rna2(x, edge_index_rna)
            rna_embeddings = F.elu(rna_embeddings)
            attention_dict['gat_rna2'] = None
            
            rna_embeddings = self.batch_norm_rna(rna_embeddings)
            
            edge_index_adt = edge_index_adt if edge_index_adt is not None else edge_index_rna
            initial_adt = self.gat_adt_init(rna_embeddings, edge_index_adt)
            initial_adt = F.elu(initial_adt)
            attention_dict['gat_adt_init'] = None
            
            initial_adt = self.batch_norm_adt(initial_adt)
            
            fused_embeddings, transformer_attn = self.transformer_fusion(
                rna_embeddings, initial_adt, edge_index_rna, edge_index_adt,
                node_degrees_rna, node_degrees_adt,
                clustering_coeffs_rna, clustering_coeffs_adt,
                return_attention=True
            )
            attention_dict['transformer'] = transformer_attn
            
            adt_features = self.gat_adt(fused_embeddings, edge_index_adt)
            attention_dict['gat_adt'] = None
        
        return attention_dict

    def get_total_reg_loss(self):
        """Get total regularization loss from adapters and projection layers"""
        reg_loss = self.transformer_fusion.get_adapter_reg_loss()
        
        for name, param in self.named_parameters():
            if 'proj' in name and 'weight' in name:
                reg_loss += self.adapter_l2_reg * torch.norm(param, p=2) ** 2
                
        return reg_loss

    
def compute_graph_statistics_fast(edge_index, num_nodes):
    """
    Simple and fast graph statistics computation
    
    Args:
        edge_index: Graph edges [2, E]
        num_nodes: Number of nodes
    
    Returns:
        node_degrees: Node degrees [N]
        clustering_coeffs: Simple degree-based clustering approximation [N]
    """
    device = edge_index.device
    
    node_degrees = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
    node_degrees[unique_nodes] = counts.float()
    
    unique_nodes_in, counts_in = torch.unique(edge_index[1], return_counts=True)
    node_degrees[unique_nodes_in] += counts_in.float()
    
    max_degree = node_degrees.max()
    if max_degree > 0:
        normalized_degrees = node_degrees / max_degree
        clustering_coeffs = 0.5 * (1.0 - normalized_degrees) + 0.1
    else:
        clustering_coeffs = torch.full((num_nodes,), 0.3, device=device, dtype=torch.float32)
    
    return node_degrees, clustering_coeffs
