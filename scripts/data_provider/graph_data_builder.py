import torch
import numpy as np
import scanpy as sc
from scipy import sparse
from torch_geometric.data import Data

def sparsify_graph(adata, max_edges_per_node=50):
    if "connectivities" not in adata.obsp:
        print("No connectivity graph found. Computing neighbors first...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

    adjacency_matrix = adata.obsp["connectivities"].tocsr()
    num_nodes = adjacency_matrix.shape[0]

    average_degree = adjacency_matrix.nnz / num_nodes
    if average_degree <= max_edges_per_node:
        print(f"Graph already sparse enough (avg degree: {average_degree:.1f})")
        return adata

    print(f"Sparsifying graph from avg degree {average_degree:.1f} to max {max_edges_per_node}")

    row_indices = []
    col_indices = []
    edge_weights = []

    for node_idx in range(num_nodes):
        start_idx = adjacency_matrix.indptr[node_idx]
        end_idx = adjacency_matrix.indptr[node_idx + 1]
        neighbor_indices = adjacency_matrix.indices[start_idx:end_idx]
        neighbor_weights = adjacency_matrix.data[start_idx:end_idx]

        if len(neighbor_indices) > max_edges_per_node:
            top_k_indices = np.argpartition(neighbor_weights, -max_edges_per_node)[-max_edges_per_node:]
            neighbor_indices = neighbor_indices[top_k_indices]
            neighbor_weights = neighbor_weights[top_k_indices]

        row_indices.extend([node_idx] * len(neighbor_indices))
        col_indices.extend(neighbor_indices)
        edge_weights.extend(neighbor_weights)

    sparse_adjacency = sparse.csr_matrix(
        (edge_weights, (row_indices, col_indices)),
        shape=(num_nodes, num_nodes)
    )

    sparse_adjacency = (sparse_adjacency + sparse_adjacency.T) / 2
    adata.obsp["connectivities"] = sparse_adjacency

    new_average_degree = sparse_adjacency.nnz / num_nodes
    print(f"New average degree: {new_average_degree:.1f}")

    return adata

def build_pyg_data(adata, use_pca=True, sparsify_large_graphs=True, max_edges_per_node=50):
    print(f"build_pyg_data called with use_pca={use_pca}")
    print(f"Input adata shape: {adata.shape}")
    print(f"Available obsm keys: {list(adata.obsm.keys())}")
    
    if use_pca:
        num_features = adata.shape[1]
        if num_features <= 50:
            print(f"Feature count ({num_features}) is low. Skipping PCA and using raw features for graph construction.")
            # Mock X_pca with raw data so follow-up steps (neighbors) work consistently
            if sparse.issparse(adata.X):
                adata.obsm["X_pca"] = adata.X.toarray()
            else:
                adata.obsm["X_pca"] = np.array(adata.X)
        else:
            # Always compute PCA with exactly 50 components for consistency when feature count allows
            print("Computing PCA with 50 components...")
            sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
            print(f"PCA computed, shape: {adata.obsm['X_pca'].shape}")

    if "connectivities" not in adata.obsp:
        print("Computing neighbor graph first...")
        # Use X_pca (which we ensured exists) if use_pca is True
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca" if use_pca else None)

    if "leiden" not in adata.obs:
        # Check if we have neighbors computed
        if "connectivities" not in adata.obsp:
             sc.pp.neighbors(adata, n_neighbors=15)
        print("Computing leiden clusters first...")
        sc.tl.leiden(adata, resolution=1.0)

    if sparsify_large_graphs:
        adjacency_matrix = adata.obsp["connectivities"]
        average_degree = adjacency_matrix.nnz / adjacency_matrix.shape[0]
        if average_degree > max_edges_per_node:
            print(f"Large graph detected (avg degree: {average_degree:.1f}), applying sparsification...")
            adata = sparsify_graph(adata, max_edges_per_node)

    if use_pca:
        # We ensured X_pca exists above
        node_features = adata.obsm["X_pca"]
        print(f"Using representation for nodes, shape: {node_features.shape}")
    else:
        # Handle both sparse and dense matrices
        if sparse.issparse(adata.X):
            node_features = adata.X.toarray()
        else:
            node_features = adata.X
        print(f"Using raw features, shape: {node_features.shape}")
    node_labels = adata.obs["leiden"].astype(int).to_numpy()

    adjacency_matrix = adata.obsp["connectivities"].tocsr()
    upper_triangle = sparse.triu(adjacency_matrix, k=1)
    source_nodes, target_nodes = upper_triangle.nonzero()
    edge_index = torch.tensor(np.vstack([source_nodes, target_nodes]), dtype=torch.long)

    pyg_data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(node_labels, dtype=torch.long),
    )

    return pyg_data

def extract_embeddings(model, pyg_data):
    model.eval()

    device = next(model.parameters()).device
    if pyg_data.x.device != device:
        print(f"Moving data from {pyg_data.x.device} to {device}")
        pyg_data = pyg_data.to(device)

    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        node_embeddings = model.get_embeddings(pyg_data.x, pyg_data.edge_index)
        node_embeddings = node_embeddings.cpu()

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return node_embeddings

def setup_graph_processing(rna_adata, adt_adata,
                          n_neighbors=15, n_pcs=50,
                          rna_sparse_threshold=5000000, adt_sparse_threshold=10000000,
                          rna_max_edges_dense=200, rna_max_edges_sparse=100,
                          adt_max_edges_dense=100, adt_max_edges_sparse=50):
    config = {}

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available GPU memory: {gpu_memory:.1f} GB")
        config['gpu_memory_gb'] = gpu_memory
        config['use_gpu'] = True

        if "connectivities" not in rna_adata.obsp:
            print("Computing neighbors for RNA data...")
            sc.pp.neighbors(rna_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

        if "connectivities" not in adt_adata.obsp:
            print("Computing neighbors for ADT data...")
            sc.pp.neighbors(adt_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

        rna_edges = rna_adata.obsp["connectivities"].nnz
        adt_edges = adt_adata.obsp["connectivities"].nnz

        print(f"RNA graph edges: {rna_edges:,}")
        print(f"ADT graph edges: {adt_edges:,}")

        max_edges_rna = rna_max_edges_sparse if rna_edges > rna_sparse_threshold else rna_max_edges_dense
        max_edges_adt = adt_max_edges_sparse if adt_edges > adt_sparse_threshold else adt_max_edges_dense

        print(f"Using max edges per node - RNA: {max_edges_rna}, ADT: {max_edges_adt}")

    else:
        print("Using CPU - no memory constraints")
        max_edges_rna = rna_max_edges_dense
        max_edges_adt = adt_max_edges_dense
        config['gpu_memory_gb'] = 0
        config['use_gpu'] = False

    config.update({
        'max_edges_rna': max_edges_rna,
        'max_edges_adt': max_edges_adt,
        'rna_edges': rna_adata.obsp["connectivities"].nnz if "connectivities" in rna_adata.obsp else 0,
        'adt_edges': adt_adata.obsp["connectivities"].nnz if "connectivities" in adt_adata.obsp else 0
    })

    return config

def process_data_with_graphs(rna_adata, adt_adata, **kwargs):
    config = setup_graph_processing(rna_adata, adt_adata, **kwargs)

    print("Building PyG data objects...")
    rna_pyg_data = build_pyg_data(rna_adata, use_pca=True, sparsify_large_graphs=True,
                                  max_edges_per_node=config['max_edges_rna'])
    adt_pyg_data = build_pyg_data(adt_adata, use_pca=True, sparsify_large_graphs=True,
                                  max_edges_per_node=config['max_edges_adt'])

    print(f"RNA PyG data - Nodes: {rna_pyg_data.num_nodes}, Edges: {rna_pyg_data.num_edges}, Features: {rna_pyg_data.num_node_features}")
    print(f"ADT PyG data - Nodes: {adt_pyg_data.num_nodes}, Edges: {adt_pyg_data.num_edges}, Features: {adt_pyg_data.num_node_features}")

    return rna_pyg_data, adt_pyg_data, config