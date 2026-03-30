"""
Realistic synthetic CITE-seq data generator for DeepOMAPNet.

Models a PBMC + AML dataset with:
- 7 biologically-defined cell types (CD4 T, CD8 T, B cells, NK, Monocytes, pDC, AML blasts)
- 30 surface protein markers (ADT) with bimodal on/off expression per cell type
- 500 RNA features with negative-binomial overdispersion
- AML condition that shifts cell-type proportions and upregulates stem/myeloid markers
- Realistic RNA–ADT correlation through shared latent factors

Usage:
    from scripts.data_provider.synthetic_citeseq import generate_citeseq_dataset
    dataset = generate_citeseq_dataset(n_normal=1000, n_aml=1000)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Protein and gene panel definitions
# ---------------------------------------------------------------------------

ADT_PANEL = [
    # Lineage
    "CD3",  "CD4",  "CD8",  "CD14", "CD16", "CD19", "CD20", "CD56", "CD45",
    # T-cell subsets
    "CD25", "CD27", "CD44", "CD45RA", "CD45RO",
    # Myeloid / activation
    "HLA-DR", "CD11b", "CD11c", "CD64", "CD123",
    # B-cell / NK
    "CD38",  "CD138", "CD57",
    # AML-associated
    "CD34",  "CD117", "CD33",  "CD13",  "CD135", "CD99",
    # General
    "CD31",  "CD44",
]
assert len(ADT_PANEL) == 30, f"ADT panel must be 30 proteins, got {len(ADT_PANEL)}"

N_RNA_FEATURES = 500   # highly variable gene count

# ---------------------------------------------------------------------------
# Cell-type definitions
# ---------------------------------------------------------------------------
# Each cell type has:
#   'adt_pos'  : proteins highly expressed (bimodal positive)
#   'adt_neg'  : proteins absent / low
#   'rna_dims' : which RNA feature indices are upregulated (relative to N_RNA_FEATURES)

_CELL_TYPES = {
    "CD4_T": {
        "adt_pos":  ["CD3", "CD4", "CD45", "CD44", "CD27"],
        "adt_var":  ["CD45RA", "CD45RO", "CD25"],          # variable (activation state)
        "rna_dims": list(range(0, 60)),
    },
    "CD8_T": {
        "adt_pos":  ["CD3", "CD8", "CD45", "CD44"],
        "adt_var":  ["CD45RA", "CD45RO", "CD57"],
        "rna_dims": list(range(50, 110)),
    },
    "NK": {
        "adt_pos":  ["CD56", "CD16", "CD45", "CD57"],
        "adt_var":  ["CD11b", "CD27"],
        "rna_dims": list(range(100, 160)),
    },
    "B_cell": {
        "adt_pos":  ["CD19", "CD20", "CD45", "HLA-DR"],
        "adt_var":  ["CD27", "CD38", "CD138"],
        "rna_dims": list(range(150, 220)),
    },
    "Monocyte": {
        "adt_pos":  ["CD14", "CD11b", "CD64", "HLA-DR", "CD45"],
        "adt_var":  ["CD16", "CD11c"],
        "rna_dims": list(range(210, 290)),
    },
    "pDC": {
        "adt_pos":  ["CD123", "HLA-DR", "CD45"],
        "adt_var":  ["CD11c"],
        "rna_dims": list(range(280, 340)),
    },
    "AML_blast": {
        "adt_pos":  ["CD34", "CD117", "CD33", "CD13", "CD99", "CD135", "CD45"],
        "adt_var":  ["CD38", "CD11b", "HLA-DR"],
        "rna_dims": list(range(330, 420)),
    },
}

CELL_TYPE_NAMES = list(_CELL_TYPES.keys())

# Proportions in Normal and AML samples
_NORMAL_PROPORTIONS = {
    "CD4_T":     0.30,
    "CD8_T":     0.20,
    "NK":        0.10,
    "B_cell":    0.15,
    "Monocyte":  0.18,
    "pDC":       0.07,
    "AML_blast": 0.00,
}
_AML_PROPORTIONS = {
    "CD4_T":     0.15,
    "CD8_T":     0.12,
    "NK":        0.08,
    "B_cell":    0.07,
    "Monocyte":  0.08,
    "pDC":       0.03,
    "AML_blast": 0.47,
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclass
class CITEseqDataset:
    rna:             np.ndarray   # [N, G]  log-normalized, z-scored
    adt:             np.ndarray   # [N, P]  CLR-normalized
    rna_raw:         np.ndarray   # [N, G]  raw counts
    adt_raw:         np.ndarray   # [N, P]  raw counts
    aml_label:       np.ndarray   # [N]     0=Normal, 1=AML
    celltype_label:  np.ndarray   # [N]     int index into CELL_TYPE_NAMES
    celltype_names:  List[str]    = field(default_factory=lambda: CELL_TYPE_NAMES)
    adt_names:       List[str]    = field(default_factory=lambda: ADT_PANEL)
    n_cells:         int          = 0
    n_genes:         int          = N_RNA_FEATURES
    n_adts:          int          = 30


def _neg_binomial_counts(mu: np.ndarray, dispersion: float, rng) -> np.ndarray:
    """
    Negative binomial via Gamma-Poisson mixture.
    dispersion (r): smaller = more overdispersed (r → ∞ → Poisson).
    """
    r = dispersion
    p = r / (r + mu)
    gamma_sample = rng.gamma(shape=r, scale=(1 - p) / p, size=mu.shape)
    return rng.poisson(gamma_sample).astype(np.float32)


def _bimodal_adt(n: int, p_positive: float, mu_pos: float, mu_neg: float, rng) -> np.ndarray:
    """
    Bimodal ADT counts: each cell is 'on' with probability p_positive.
    'On' cells ~ NB(mu_pos, r=3); 'Off' cells ~ NB(mu_neg, r=5).
    """
    is_pos = rng.uniform(size=n) < p_positive
    counts = np.zeros(n, dtype=np.float32)
    n_pos = is_pos.sum()
    n_neg = n - n_pos
    if n_pos > 0:
        counts[is_pos]  = _neg_binomial_counts(
            np.full(n_pos, mu_pos), dispersion=3.0, rng=rng
        )
    if n_neg > 0:
        counts[~is_pos] = _neg_binomial_counts(
            np.full(n_neg, mu_neg), dispersion=5.0, rng=rng
        )
    return counts


def _clr_normalize(X: np.ndarray) -> np.ndarray:
    """CLR normalization: log(X+1) - mean(log(X+1)) per cell."""
    log_X = np.log(X + 1.0)
    return log_X - log_X.mean(axis=1, keepdims=True)


def _log_normalize_rna(counts: np.ndarray, scale: float = 1e4) -> np.ndarray:
    """Library-size normalize then log1p."""
    lib = counts.sum(axis=1, keepdims=True) + 1e-8
    return np.log1p(counts / lib * scale)


def _zscore(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(0, keepdims=True)
    std  = X.std(0, keepdims=True) + 1e-8
    return (X - mean) / std, mean.squeeze(), std.squeeze()


def generate_citeseq_dataset(
    n_normal: int = 1000,
    n_aml:    int = 1000,
    seed:     int = 42,
) -> CITEseqDataset:
    """
    Generate a realistic paired RNA + ADT dataset.

    Parameters
    ----------
    n_normal : cells from healthy donors
    n_aml    : cells from AML patients
    seed     : random seed (fully reproducible)

    Returns
    -------
    CITEseqDataset with normalized RNA, CLR-normalized ADT, labels.
    """
    rng = np.random.default_rng(seed)
    N   = n_normal + n_aml
    P   = len(ADT_PANEL)
    G   = N_RNA_FEATURES

    adt_idx = {name: i for i, name in enumerate(ADT_PANEL)}

    # ---- Assign cell types ------------------------------------------------
    ct_names = CELL_TYPE_NAMES
    ct_idx   = {n: i for i, n in enumerate(ct_names)}

    def _sample_celltypes(n, proportions):
        props = np.array([proportions[t] for t in ct_names])
        props /= props.sum()
        return rng.choice(len(ct_names), size=n, p=props)

    ct_normal = _sample_celltypes(n_normal, _NORMAL_PROPORTIONS)
    ct_aml    = _sample_celltypes(n_aml,    _AML_PROPORTIONS)
    celltype_label = np.concatenate([ct_normal, ct_aml])
    aml_label      = np.array([0] * n_normal + [1] * n_aml, dtype=np.int32)

    # ---- Shared latent factors (for RNA-ADT correlation) ------------------
    # Each cell type has a "base" latent factor plus within-type variation
    n_factors = len(ct_names) + 4   # cell-type factors + cross-cutting axes
    latent = np.zeros((N, n_factors), dtype=np.float32)
    for i, ctype in enumerate(celltype_label):
        latent[i, ctype] = rng.beta(5, 2)           # strong type signal
        latent[i, len(ct_names):] = rng.dirichlet(np.ones(4)) * 0.5

    # Normalize rows
    latent = latent / (latent.sum(1, keepdims=True) + 1e-8)

    # ---- Generate RNA counts ----------------------------------------------
    W_rna = np.zeros((n_factors, G), dtype=np.float32)
    for i, ctype_name in enumerate(ct_names):
        dims = _CELL_TYPES[ctype_name]["rna_dims"]
        W_rna[i, dims] = rng.exponential(scale=3.0, size=len(dims))

    # Cross-cutting factors affect random genes
    for j in range(len(ct_names), n_factors):
        random_dims = rng.choice(G, size=30, replace=False)
        W_rna[j, random_dims] = rng.exponential(scale=1.5, size=30)

    mu_rna = np.exp(latent @ W_rna + 0.5)   # [N, G] positive mean
    rna_raw = _neg_binomial_counts(mu_rna, dispersion=2.0, rng=rng)
    rna_norm = _log_normalize_rna(rna_raw)
    rna_norm, _, _ = _zscore(rna_norm)

    # ---- Generate ADT counts (bimodal per cell type) ---------------------
    adt_raw = np.zeros((N, P), dtype=np.float32)

    # Base expression parameters per protein per cell type
    for c_i, ctype_name in enumerate(ct_names):
        ct_def  = _CELL_TYPES[ctype_name]
        pos_set = set(ct_def["adt_pos"])
        var_set = set(ct_def.get("adt_var", []))
        cell_mask = (celltype_label == c_i)
        n_c = cell_mask.sum()
        if n_c == 0:
            continue

        for p_i, pname in enumerate(ADT_PANEL):
            if pname in pos_set:
                # Strongly positive cells for this type
                adt_raw[cell_mask, p_i] = _bimodal_adt(
                    n_c, p_positive=0.90,
                    mu_pos=rng.uniform(150, 400),
                    mu_neg=rng.uniform(3, 15),
                    rng=rng
                )
            elif pname in var_set:
                # Variable expression (activation markers etc.)
                adt_raw[cell_mask, p_i] = _bimodal_adt(
                    n_c, p_positive=rng.uniform(0.2, 0.7),
                    mu_pos=rng.uniform(80, 250),
                    mu_neg=rng.uniform(3, 20),
                    rng=rng
                )
            else:
                # Negative / background
                adt_raw[cell_mask, p_i] = _neg_binomial_counts(
                    np.full(n_c, rng.uniform(2, 10)), dispersion=5.0, rng=rng
                )

    # Add RNA-correlated noise to ADT (makes prediction task non-trivial but learnable)
    W_adt = np.zeros((n_factors, P), dtype=np.float32)
    for i, ctype_name in enumerate(ct_names):
        pos_names = _CELL_TYPES[ctype_name]["adt_pos"]
        for pname in pos_names:
            p_i = adt_idx[pname]
            W_adt[i, p_i] = rng.uniform(0.3, 1.0)

    adt_latent_signal = np.exp(latent @ W_adt * 0.5)
    adt_raw = adt_raw * (1.0 + adt_latent_signal * 0.3)   # blend signal in

    adt_norm = _clr_normalize(adt_raw)
    adt_norm, _, _ = _zscore(adt_norm)

    return CITEseqDataset(
        rna=rna_norm.astype(np.float32),
        adt=adt_norm.astype(np.float32),
        rna_raw=rna_raw.astype(np.float32),
        adt_raw=adt_raw.astype(np.float32),
        aml_label=aml_label,
        celltype_label=celltype_label,
        celltype_names=CELL_TYPE_NAMES,
        adt_names=ADT_PANEL,
        n_cells=N,
        n_genes=G,
        n_adts=P,
    )


if __name__ == "__main__":
    import time
    t0 = time.time()
    ds = generate_citeseq_dataset(n_normal=1000, n_aml=1000)
    print(f"Generated {ds.n_cells} cells in {time.time()-t0:.2f}s")
    print(f"  RNA  : {ds.rna.shape}  range=[{ds.rna.min():.2f}, {ds.rna.max():.2f}]")
    print(f"  ADT  : {ds.adt.shape}  range=[{ds.adt.min():.2f}, {ds.adt.max():.2f}]")
    print(f"  AML  : {ds.aml_label.sum()} / {ds.n_cells} AML cells")
    for i, name in enumerate(CELL_TYPE_NAMES):
        count = (ds.celltype_label == i).sum()
        print(f"  {name:12s}: {count:4d} cells ({count/ds.n_cells*100:.1f}%)")
