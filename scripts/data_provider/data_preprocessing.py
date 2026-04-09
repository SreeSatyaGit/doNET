# data_preprocessing.py
import scanpy as sc
import anndata as ad
import numpy as np
from typing import Dict, Tuple
import anndata
from scipy import sparse


def clr_normalize(adata: ad.AnnData, axis: int = 1) -> ad.AnnData:
    """
    Apply CLR (Centered Log-Ratio) normalization to AnnData object.
    
    Args:
        adata: AnnData object to normalize
        axis: Axis along which to normalize (0=features, 1=cells/rows)
              Use 1 for per-cell normalization (standard for ADT data)
    
    Returns:
        CLR-normalized AnnData object
    """
    # Make a copy to avoid modifying original
    adata_clr = adata.copy()
    
    # Convert to dense if sparse
    X = adata_clr.X.toarray() if sparse.issparse(adata_clr.X) else adata_clr.X.copy()
    
    # Check for negative values and handle them
    if np.any(X < 0):
        print(f"Warning: Found {np.sum(X < 0)} negative values in data. Adding offset.")
        X = X - X.min() + 1.0
    else:
        # Add pseudocount (avoid log(0))
        X += 1.0
    
    # CLR transformation: log(X / geometric_mean(X))
    if axis == 1:  # Normalize across features (per cell)
        # Calculate geometric mean for each cell (row)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_X = np.log(X)
            geometric_means = np.exp(np.mean(log_X, axis=1, keepdims=True))
            # Replace any invalid values with 1
            geometric_means = np.where(np.isfinite(geometric_means), geometric_means, 1.0)
            X_clr = np.log(X / geometric_means)
    else:  # Normalize across cells (per feature)
        # Calculate geometric mean for each feature (column)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_X = np.log(X)
            geometric_means = np.exp(np.mean(log_X, axis=0, keepdims=True))
            # Replace any invalid values with 1
            geometric_means = np.where(np.isfinite(geometric_means), geometric_means, 1.0)
            X_clr = np.log(X / geometric_means)
    
    # Replace any remaining NaN or inf values with 0
    X_clr = np.nan_to_num(X_clr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Update AnnData
    adata_clr.X = X_clr
    
    return adata_clr


def zscore_normalize(adata: ad.AnnData) -> Tuple[ad.AnnData, np.ndarray, np.ndarray]:
    """
    Apply z-score normalization to AnnData object (per feature).
    
    Args:
        adata: AnnData object to normalize
    
    Returns:
        Tuple of (z-score normalized AnnData, means, stds)
    """
    # Make a copy
    adata_zscore = adata.copy()
    
    # Get data (dense)
    X = adata_zscore.X.toarray() if sparse.issparse(adata_zscore.X) else adata_zscore.X.copy()
    
    # Check for NaN values
    if np.any(np.isnan(X)):
        print(f"Warning: Found {np.sum(np.isnan(X))} NaN values before z-score normalization.")
        # Replace NaN with 0
        X = np.nan_to_num(X, nan=0.0)
    
    # Calculate mean and std for each feature (column)
    means = np.nanmean(X, axis=0, keepdims=True)
    stds = np.nanstd(X, axis=0, keepdims=True) + 1e-8  # Add small epsilon to avoid division by zero
    
    # Handle edge cases where std is 0
    stds = np.where(stds == 0, 1.0, stds)
    
    # Apply z-score normalization
    X_zscore = (X - means) / stds
    
    # Replace any remaining NaN or inf values with 0
    X_zscore = np.nan_to_num(X_zscore, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Update AnnData
    adata_zscore.X = X_zscore
    
    return adata_zscore, means, stds


from typing import Dict, Tuple, List, Optional


def prepare_train_test_anndata(
    GSM_Controls_RNA=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/GSMControlRNA.h5ad"),
    GSM_Controls_ADT=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/ControlADT.h5ad"),
    GSM_AML_RNA_A=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/AMLARNA.h5ad"),
    GSM_AML_ADT_A=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/AMLAADT.h5ad"),
    GSM_AML_RNA_B=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/AMLBRNA.h5ad"),
    GSM_AML_ADT_B=sc.read_h5ad("/projects/vanaja_lab/satya/Datasets/AMLBADT.h5ad"),
    marker_list: Optional[List[str]] = None,
):
    adata_gene = anndata.concat(
        [GSM_AML_RNA_B, GSM_AML_RNA_A, GSM_Controls_RNA],
        join="outer",
        label="source",
        keys=["GSM_AML_RNA_B", "GSM_AML_RNA_A", "GSM_Controls_RNA"],
    )

    adata_protein = anndata.concat(
        [GSM_Controls_ADT, GSM_AML_ADT_A, GSM_AML_ADT_B],
        join="outer",
        label="source",
        keys=["GSM_Controls_ADT", "GSM_AML_ADT_A", "GSM_AML_ADT_B"],
    )

    if marker_list is not None:
        available_markers = [m for m in marker_list if m in adata_protein.var_names]
        if len(available_markers) == 0:
            print("Warning: None of the requested markers were found. Using all markers.")
        else:
            if len(available_markers) < len(marker_list):
                missing = set(marker_list) - set(available_markers)
                print(f"Warning: Markers {missing} not found in dataset. Predicting {len(available_markers)} markers.")
            adata_protein = adata_protein[:, available_markers].copy()
            print(f"Successfully restricted to {len(available_markers)} surface markers.")

    print("All sample IDs in gene data:", adata_gene.obs["samples"].unique())

    samples = list(adata_gene.obs["samples"].unique())

    aml_samples = [s for s in samples if s.startswith("AML")]
    control_samples = [s for s in samples if s.startswith("Control")]

    np.random.seed(42)
    aml_samples = np.random.permutation(aml_samples)
    control_samples = np.random.permutation(control_samples)

    def split_indices(n, frac=0.8):
        split_at = int(np.ceil(frac * n))
        return split_at

    aml_split = split_indices(len(aml_samples), 0.8)
    control_split = split_indices(len(control_samples), 0.8)

    aml_train = aml_samples[:aml_split].tolist()
    aml_test = aml_samples[aml_split:].tolist()
    control_train = control_samples[:control_split].tolist()
    control_test = control_samples[control_split:].tolist()

    print("AML 80% train:", aml_train)
    print("AML 20% test:", aml_test)
    print("Control 80% train:", control_train)
    print("Control 20% test:", control_test)

    train_samples = aml_train + control_train

    train_mask_gene = adata_gene.obs["samples"].isin(train_samples)
    train_mask_protein = adata_protein.obs["samples"].isin(train_samples)

    adata_gene_train = adata_gene[train_mask_gene].copy()
    adata_protein_train = adata_protein[train_mask_protein].copy()
    adata_gene_test = adata_gene[~train_mask_gene].copy()
    adata_protein_test = adata_protein[~train_mask_protein].copy()

    def align_obs(gene_data, protein_data):
        common_cells = sorted(
            set(gene_data.obs_names).intersection(set(protein_data.obs_names))
        )
        gene_data_aligned = gene_data[common_cells].copy()
        protein_data_aligned = protein_data[common_cells].copy()
        assert all(
            gene_data_aligned.obs_names == protein_data_aligned.obs_names
        ), "Cell IDs do not match after alignment!"
        return gene_data_aligned, protein_data_aligned

    adata_gene_train, adata_protein_train = align_obs(
        adata_gene_train, adata_protein_train
    )
    adata_gene_test, adata_protein_test = align_obs(
        adata_gene_test, adata_protein_test
    )

    print("Train cells:", adata_gene_train.n_obs, "| Test cells:", adata_gene_test.n_obs)

    adata_gene_train.X = adata_gene_train.X.astype("float32")
    adata_gene_test.X = adata_gene_test.X.astype("float32")
    adata_protein_train.X = adata_protein_train.X.astype("float32")
    adata_protein_test.X = adata_protein_test.X.astype("float32")
    
    # Apply CLR + z-score normalization to protein (ADT) data
    print("\n" + "="*60)
    print("NORMALIZING PROTEIN (ADT) DATA")
    print("="*60)
    
    # Normalize training data: CLR first, then z-score
    print("\nStep 1: Applying CLR normalization to training protein data...")
    adata_protein_train_clr = clr_normalize(adata_protein_train, axis=1)
    print(f"  CLR normalization complete. Shape: {adata_protein_train_clr.shape}")
    
    print("\nStep 2: Applying z-score normalization to training protein data...")
    adata_protein_train, train_means, train_stds = zscore_normalize(adata_protein_train_clr)
    
    # Check for NaN values after normalization
    X_train = adata_protein_train.X.toarray() if sparse.issparse(adata_protein_train.X) else adata_protein_train.X
    n_nan = np.sum(np.isnan(X_train))
    if n_nan > 0:
        print(f"  Warning: Found {n_nan} NaN values after normalization, replacing with 0.")
        adata_protein_train.X = np.nan_to_num(X_train, nan=0.0)
    
    print(f"  Z-score normalization complete.")
    print(f"  Mean of feature means: {train_means.mean():.4f}")
    print(f"  Mean of feature stds: {train_stds.mean():.4f}")
    
    # Normalize test data: CLR first, then z-score using training statistics
    print("\nStep 3: Applying CLR normalization to test protein data...")
    adata_protein_test_clr = clr_normalize(adata_protein_test, axis=1)
    print(f"  CLR normalization complete. Shape: {adata_protein_test_clr.shape}")
    
    print("\nStep 4: Applying z-score normalization to test protein data (using training statistics)...")
    # Apply z-score using training statistics for consistency
    X_test = adata_protein_test_clr.X.toarray() if sparse.issparse(adata_protein_test_clr.X) else adata_protein_test_clr.X.copy()
    X_test_zscore = (X_test - train_means) / train_stds
    
    # Check for NaN values after normalization
    n_nan = np.sum(np.isnan(X_test_zscore))
    if n_nan > 0:
        print(f"  Warning: Found {n_nan} NaN values after normalization, replacing with 0.")
        X_test_zscore = np.nan_to_num(X_test_zscore, nan=0.0)
    
    adata_protein_test.X = X_test_zscore
    
    print("  Z-score normalization complete.")
    print("  Test data normalized using training statistics for consistency.")
    print("="*60)

    return (
        adata_gene_train,
        adata_gene_test,
        adata_protein_train,
        adata_protein_test,
    )
