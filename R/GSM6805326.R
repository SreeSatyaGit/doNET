# GSM6805326.R
library(Seurat)
library(dplyr)
library(Matrix)
library(readr)

# Increase connection buffer size for large files
Sys.setenv("VROOM_CONNECTION_SIZE" = 1048576)  # 1MB buffer

# Function to inspect file structure before reading
inspect_file_structure <- function(data_dir = ".") {
  
  files <- list.files(data_dir, pattern = "\\.csv\\.gz$", full.names = TRUE)
  
  cat("=== File Structure Inspection ===\n")
  
  # Check first few lines of each file type
  rna_files <- files[grepl("RNA_counts", files)]
  adt_files <- files[grepl("ADT_counts", files)]
  cell_assign_files <- files[grepl("cell_sample_assign", files)]
  
  if (length(rna_files) > 0) {
    cat("\n--- RNA Files Structure ---\n")
    sample_file <- rna_files[1]
    cat("Sample file:", basename(sample_file), "\n")
    
    # Read first few lines
    sample_data <- readLines(sample_file, n = 5)
    cat("First 5 lines:\n")
    for (i in 1:length(sample_data)) {
      cat("Line", i, ":", sample_data[i], "\n")
    }
    
    # Check dimensions
    temp_data <- read_csv(sample_file, n_max = 10, show_col_types = FALSE)
    cat("Columns:", ncol(temp_data), "\n")
    cat("First column name:", names(temp_data)[1], "\n")
    cat("Sample column names:", paste(names(temp_data)[2:min(6, ncol(temp_data))], collapse = ", "), "\n")
  }
  
  if (length(adt_files) > 0) {
    cat("\n--- ADT Files Structure ---\n")
    sample_file <- adt_files[1]
    cat("Sample file:", basename(sample_file), "\n")
    
    # Read first few lines
    sample_data <- readLines(sample_file, n = 5)
    cat("First 5 lines:\n")
    for (i in 1:length(sample_data)) {
      cat("Line", i, ":", sample_data[i], "\n")
    }
    
    # Check dimensions
    temp_data <- read_csv(sample_file, n_max = 10, show_col_types = FALSE)
    cat("Columns:", ncol(temp_data), "\n")
    cat("First column name:", names(temp_data)[1], "\n")
    cat("Sample column names:", paste(names(temp_data)[2:min(6, ncol(temp_data))], collapse = ", "), "\n")
  }
  
  if (length(cell_assign_files) > 0) {
    cat("\n--- Cell Assignment Files Structure ---\n")
    sample_file <- cell_assign_files[1]
    cat("Sample file:", basename(sample_file), "\n")
    
    # Read first few lines
    sample_data <- readLines(sample_file, n = 5)
    cat("First 5 lines:\n")
    for (i in 1:length(sample_data)) {
      cat("Line", i, ":", sample_data[i], "\n")
    }
    
    # Check dimensions
    temp_data <- read_csv(sample_file, n_max = 10, show_col_types = FALSE)
    cat("Columns:", ncol(temp_data), "\n")
    cat("Column names:", paste(names(temp_data), collapse = ", "), "\n")
  }
  
  cat("\n=== Inspection Complete ===\n")
}

# Function to read CITE-seq files and create Seurat objects
read_cite_seq_files <- function(data_dir = ".") {
  
  # Get all files in the directory
  files <- list.files(data_dir, pattern = "\\.csv\\.gz$", full.names = TRUE)
  
  # Separate files by type
  rna_files <- files[grepl("RNA_counts", files)]
  adt_files <- files[grepl("ADT_counts", files)]
  cell_assign_files <- files[grepl("cell_sample_assign", files)]
  
  cat("Found", length(rna_files), "RNA files\n")
  cat("Found", length(adt_files), "ADT files\n")
  cat("Found", length(cell_assign_files), "cell assignment files\n")
  
  # Extract sample IDs from file names
  extract_sample_id <- function(filename) {
    basename(filename) %>%
      gsub("_.*", "", .)
  }
  
  # Read RNA data
  rna_data <- list()
  for (file in rna_files) {
    sample_id <- extract_sample_id(file)
    cat("Reading RNA data for", sample_id, "\n")
    
    # Read the CSV file with better error handling
    tryCatch({
      rna_counts <- read_csv(file, show_col_types = FALSE, guess_max = 10000)
      
      # Check for parsing issues
      if (nrow(problems(rna_counts)) > 0) {
        cat("Warning: Parsing issues detected for", sample_id, "\n")
        print(problems(rna_counts))
      }
      
      # Convert to matrix (assuming first column is gene names)
      gene_names <- rna_counts[[1]]
      count_matrix <- as.matrix(rna_counts[, -1])
      
      # Remove any non-numeric columns and handle missing values
      count_matrix <- apply(count_matrix, 2, function(x) {
        as.numeric(as.character(x))
      })
      
      # Handle any remaining NA values
      count_matrix[is.na(count_matrix)] <- 0
      
      # Handle duplicate gene names
      if (any(duplicated(gene_names))) {
        cat("Warning: Found", sum(duplicated(gene_names)), "duplicate gene names in", sample_id, "\n")
        # Aggregate duplicate genes by summing their counts
        unique_genes <- unique(gene_names)
        aggregated_matrix <- matrix(0, nrow = length(unique_genes), ncol = ncol(count_matrix))
        
        for (i in seq_along(unique_genes)) {
          gene_idx <- which(gene_names == unique_genes[i])
          if (length(gene_idx) > 1) {
            aggregated_matrix[i, ] <- colSums(count_matrix[gene_idx, , drop = FALSE])
          } else {
            aggregated_matrix[i, ] <- count_matrix[gene_idx, ]
          }
        }
        
        count_matrix <- aggregated_matrix
        gene_names <- unique_genes
      }
      
      rownames(count_matrix) <- gene_names
      
      # Convert to sparse matrix
      rna_data[[sample_id]] <- Matrix(count_matrix, sparse = TRUE)
      
    }, error = function(e) {
      cat("Error reading", file, ":", e$message, "\n")
      # Try alternative reading method
      tryCatch({
        rna_counts <- read.csv(file, stringsAsFactors = FALSE, check.names = FALSE)
        gene_names <- rna_counts[[1]]
        count_matrix <- as.matrix(rna_counts[, -1])
        count_matrix <- apply(count_matrix, 2, as.numeric)
        count_matrix[is.na(count_matrix)] <- 0
        
        # Handle duplicate gene names
        if (any(duplicated(gene_names))) {
          cat("Warning: Found", sum(duplicated(gene_names)), "duplicate gene names in", sample_id, "\n")
          unique_genes <- unique(gene_names)
          aggregated_matrix <- matrix(0, nrow = length(unique_genes), ncol = ncol(count_matrix))
          
          for (i in seq_along(unique_genes)) {
            gene_idx <- which(gene_names == unique_genes[i])
            if (length(gene_idx) > 1) {
              aggregated_matrix[i, ] <- colSums(count_matrix[gene_idx, , drop = FALSE])
            } else {
              aggregated_matrix[i, ] <- count_matrix[gene_idx, ]
            }
          }
          
          count_matrix <- aggregated_matrix
          gene_names <- unique_genes
        }
        
        rownames(count_matrix) <- gene_names
        rna_data[[sample_id]] <<- Matrix(count_matrix, sparse = TRUE)
        cat("Successfully read", sample_id, "using alternative method\n")
      }, error = function(e2) {
        cat("Failed to read", file, "with alternative method:", e2$message, "\n")
      })
    })
  }
  
  # Read ADT data
  adt_data <- list()
  for (file in adt_files) {
    sample_id <- extract_sample_id(file)
    cat("Reading ADT data for", sample_id, "\n")
    
    # Read the CSV file with better error handling
    tryCatch({
      adt_counts <- read_csv(file, show_col_types = FALSE, guess_max = 10000)
      
      # Check for parsing issues
      if (nrow(problems(adt_counts)) > 0) {
        cat("Warning: Parsing issues detected for", sample_id, "\n")
        print(problems(adt_counts))
      }
      
      # Convert to matrix (assuming first column is ADT names)
      adt_names <- adt_counts[[1]]
      count_matrix <- as.matrix(adt_counts[, -1])
      
      # Remove any non-numeric columns and handle missing values
      count_matrix <- apply(count_matrix, 2, function(x) {
        as.numeric(as.character(x))
      })
      
      # Handle any remaining NA values
      count_matrix[is.na(count_matrix)] <- 0
      
      # Handle duplicate ADT names
      if (any(duplicated(adt_names))) {
        cat("Warning: Found", sum(duplicated(adt_names)), "duplicate ADT names in", sample_id, "\n")
        # Aggregate duplicate ADTs by summing their counts
        unique_adts <- unique(adt_names)
        aggregated_matrix <- matrix(0, nrow = length(unique_adts), ncol = ncol(count_matrix))
        
        for (i in seq_along(unique_adts)) {
          adt_idx <- which(adt_names == unique_adts[i])
          if (length(adt_idx) > 1) {
            aggregated_matrix[i, ] <- colSums(count_matrix[adt_idx, , drop = FALSE])
          } else {
            aggregated_matrix[i, ] <- count_matrix[adt_idx, ]
          }
        }
        
        count_matrix <- aggregated_matrix
        adt_names <- unique_adts
      }
      
      rownames(count_matrix) <- adt_names
      
      # Convert to sparse matrix
      adt_data[[sample_id]] <- Matrix(count_matrix, sparse = TRUE)
      
    }, error = function(e) {
      cat("Error reading", file, ":", e$message, "\n")
      # Try alternative reading method
      tryCatch({
        adt_counts <- read.csv(file, stringsAsFactors = FALSE, check.names = FALSE)
        adt_names <- adt_counts[[1]]
        count_matrix <- as.matrix(adt_counts[, -1])
        count_matrix <- apply(count_matrix, 2, as.numeric)
        count_matrix[is.na(count_matrix)] <- 0
        
        # Handle duplicate ADT names
        if (any(duplicated(adt_names))) {
          cat("Warning: Found", sum(duplicated(adt_names)), "duplicate ADT names in", sample_id, "\n")
          unique_adts <- unique(adt_names)
          aggregated_matrix <- matrix(0, nrow = length(unique_adts), ncol = ncol(count_matrix))
          
          for (i in seq_along(unique_adts)) {
            adt_idx <- which(adt_names == unique_adts[i])
            if (length(adt_idx) > 1) {
              aggregated_matrix[i, ] <- colSums(count_matrix[adt_idx, , drop = FALSE])
            } else {
              aggregated_matrix[i, ] <- count_matrix[adt_idx, ]
            }
          }
          
          count_matrix <- aggregated_matrix
          adt_names <- unique_adts
        }
        
        rownames(count_matrix) <- adt_names
        adt_data[[sample_id]] <<- Matrix(count_matrix, sparse = TRUE)
        cat("Successfully read", sample_id, "using alternative method\n")
      }, error = function(e2) {
        cat("Failed to read", file, "with alternative method:", e2$message, "\n")
      })
    })
  }
  
  # Read cell assignments
  cell_assignments <- list()
  for (file in cell_assign_files) {
    sample_id <- extract_sample_id(file)
    cat("Reading cell assignments for", sample_id, "\n")
    
    # Read the CSV file with better error handling
    tryCatch({
      cell_assign <- read_csv(file, show_col_types = FALSE, guess_max = 10000)
      
      # Check for parsing issues
      if (nrow(problems(cell_assign)) > 0) {
        cat("Warning: Parsing issues detected for", sample_id, "\n")
        print(problems(cell_assign))
      }
      
      cell_assignments[[sample_id]] <- cell_assign
      
    }, error = function(e) {
      cat("Error reading", file, ":", e$message, "\n")
      # Try alternative reading method
      tryCatch({
        cell_assign <- read.csv(file, stringsAsFactors = FALSE, check.names = FALSE)
        cell_assignments[[sample_id]] <<- cell_assign
        cat("Successfully read", sample_id, "using alternative method\n")
      }, error = function(e2) {
        cat("Failed to read", file, "with alternative method:", e2$message, "\n")
      })
    })
  }
  
  # Create Seurat objects for each sample
  seurat_objects <- list()
  
  for (sample_id in names(rna_data)) {
    cat("Creating Seurat object for", sample_id, "\n")
    
    # Create Seurat object with RNA data
    seurat_obj <- CreateSeuratObject(
      counts = rna_data[[sample_id]],
      project = sample_id,
      assay = "RNA"
    )
    
    # Add ADT data if available
    if (sample_id %in% names(adt_data)) {
      # Transpose ADT matrix (cells x features)
      adt_matrix <- t(adt_data[[sample_id]])
      
      # Create ADT assay
      seurat_obj[["ADT"]] <- CreateAssayObject(counts = adt_matrix)
    }
    
    # Add cell assignments if available
    if (sample_id %in% names(cell_assignments)) {
      cell_assign <- cell_assignments[[sample_id]]
      
      # Add cell assignment metadata
      for (col_name in colnames(cell_assign)) {
        if (col_name %in% rownames(seurat_obj)) {
          seurat_obj@meta.data[[col_name]] <- cell_assign[[col_name]]
        }
      }
    }
    
    # Add sample metadata
    seurat_obj@meta.data$sample_id <- sample_id
    
    seurat_objects[[sample_id]] <- seurat_obj
  }
  
  return(list(
    seurat_objects = seurat_objects,
    rna_data = rna_data,
    adt_data = adt_data,
    cell_assignments = cell_assignments
  ))
}

# Function to merge all samples into a single Seurat object
merge_cite_seq_samples <- function(cite_seq_data) {
  
  cat("Merging", length(cite_seq_data$seurat_objects), "samples...\n")
  
  # Merge all Seurat objects
  merged_obj <- merge(
    cite_seq_data$seurat_objects[[1]], 
    y = cite_seq_data$seurat_objects[-1],
    add.cell.ids = names(cite_seq_data$seurat_objects),
    project = "CITE_seq_merged"
  )
  
  # Add sample information to metadata
  merged_obj@meta.data$sample_id <- gsub("_.*", "", rownames(merged_obj@meta.data))
  
  cat("Merged object contains", ncol(merged_obj), "cells and", nrow(merged_obj), "genes\n")
  
  return(merged_obj)
}

# Function to perform basic preprocessing
preprocess_cite_seq <- function(seurat_obj) {
  
  cat("Performing basic preprocessing...\n")
  
  # Normalize RNA data
  seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
  
  # Find variable features
  seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
  
  # Scale data
  seurat_obj <- ScaleData(seurat_obj, features = VariableFeatures(seurat_obj))
  
  # Run PCA
  seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(seurat_obj))
  
  # Find neighbors
  seurat_obj <- FindNeighbors(seurat_obj, dims = 1:30)
  
  # Cluster cells
  seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)
  
  # Run UMAP
  seurat_obj <- RunUMAP(seurat_obj, dims = 1:30)
  
  # Process ADT data if available
  if ("ADT" %in% names(seurat_obj@assays)) {
    cat("Processing ADT data...\n")
    
    # Normalize ADT data
    seurat_obj <- NormalizeData(seurat_obj, assay = "ADT", normalization.method = "CLR")
    
    # Scale ADT data
    seurat_obj <- ScaleData(seurat_obj, assay = "ADT")
  }
  
  return(seurat_obj)
}

# Main execution
main <- function(data_dir = ".") {
  
  cat("=== CITE-seq Data Loading Pipeline ===\n")
  
  # Read all files
  cite_seq_data <- read_cite_seq_files(data_dir)
  
  # Merge samples
  merged_obj <- merge_cite_seq_samples(cite_seq_data)
  
  # Preprocess
  processed_obj <- preprocess_cite_seq(merged_obj)
  
  cat("=== Pipeline Complete ===\n")
  cat("Final object:", ncol(processed_obj), "cells,", nrow(processed_obj), "genes\n")
  
  if ("ADT" %in% names(processed_obj@assays)) {
    cat("ADT features:", nrow(processed_obj@assays$ADT), "\n")
  }
  
  return(list(
    raw_data = cite_seq_data,
    merged_object = merged_obj,
    processed_object = processed_obj
  ))
}

result <- main("/projects/vanaja_lab/satya/Datasets/GSE220474")

GSM6805319 <- result$processed_object

GSM6805319 <- FindNeighbors(GSM6805319, dims = 1:30, reduction = "pca")
GSM6805319 <- FindClusters(GSM6805319, resolution = 2, cluster.name = "unintegrated_clusters")


GSM6805319 <- RunUMAP(GSM6805319, dims = 1:30, reduction = "pca", reduction.name = "umap.unintegrated")
suppressWarnings({
  DimPlot(GSM6805319, reduction = "umap.unintegrated", group.by = c("sample_id"))
})

# Save Seurat object as H5AD file
library(SeuratDisk)

# Check assay structure and fix if needed
cat("Checking RNA assay structure...\n")
cat("Available layers:", names(GSM6805319@assays$RNA@layers), "\n")

# Ensure data layer exists (required for H5AD conversion)
if (!"data" %in% names(GSM6805319@assays$RNA@layers)) {
  cat("Adding data layer to RNA assay...\n")
  GSM6805319@assays$RNA@layers$data <- GSM6805319@assays$RNA@counts
}

# Method 1: Try SeuratDisk conversion
tryCatch({
  cat("Attempting SeuratDisk conversion...\n")
  SaveH5Seurat(GSM6805319, filename = "/projects/vanaja_lab/satya/GSM6805319.h5Seurat", overwrite = TRUE)
  Convert("/projects/vanaja_lab/satya/GSM6805319.h5Seurat", dest = "h5ad", overwrite = TRUE)
  cat("Successfully converted to H5AD using SeuratDisk!\n")
}, error = function(e) {
  cat("SeuratDisk conversion failed:", e$message, "\n")
  
  # Method 2: Alternative approach using reticulate and scanpy
  cat("Trying alternative method with reticulate...\n")
  tryCatch({
    library(reticulate)
    
    # Check if scanpy is available
    if (!py_module_available("scanpy")) {
      cat("Installing scanpy...\n")
      py_install("scanpy")
    }
    
    sc <- import("scanpy")
    np <- import("numpy")
    pd <- import("pandas")
    
    # Extract count matrix (transpose for scanpy format: cells x genes)
    count_matrix <- t(as.matrix(GSM6805319@assays$RNA@counts))
    
    # Create AnnData object
    adata <- sc$AnnData(
      X = count_matrix,
      obs = GSM6805319@meta.data,
      var = data.frame(gene_names = rownames(GSM6805319))
    )
    
    # Add reductions if available
    if ("pca" %in% names(GSM6805319@reductions)) {
      adata$obsm$update(list(X_pca = GSM6805319@reductions$pca@cell.embeddings))
    }
    if ("umap" %in% names(GSM6805319@reductions)) {
      adata$obsm$update(list(X_umap = GSM6805319@reductions$umap@cell.embeddings))
    }
    if ("umap.unintegrated" %in% names(GSM6805319@reductions)) {
      adata$obsm$update(list(X_umap_unintegrated = GSM6805319@reductions$umap.unintegrated@cell.embeddings))
    }
    if ("integrated.cca" %in% names(GSM6805319@reductions)) {
      adata$obsm$update(list(X_integrated_cca = GSM6805319@reductions$integrated.cca@cell.embeddings))
    }
    
    # Save as H5AD
    adata$write_h5ad("/projects/vanaja_lab/satya/GSM6805319.h5ad")
    cat("Successfully converted to H5AD using reticulate!\n")
    
  }, error = function(e2) {
    cat("Reticulate method also failed:", e2$message, "\n")
    
    # Method 3: Manual H5AD creation using hdf5r
    cat("Trying manual H5AD creation...\n")
    tryCatch({
      library(hdf5r)
      
      # Create H5AD file structure manually
      h5_file <- H5File$new("/projects/vanaja_lab/satya/GSM6805319_manual.h5ad", mode = "w")
      
      # Create basic structure
      h5_file$create_group("obs")
      h5_file$create_group("var")
      h5_file$create_group("obsm")
      h5_file$create_group("uns")
      
      # Add count matrix as X
      count_matrix <- as.matrix(GSM6805319@assays$RNA@counts)
      h5_file$create_dataset("X", count_matrix, chunk_dims = c(1000, 1000))
      
      # Add metadata
      h5_file[["obs"]]$create_dataset("index", as.character(rownames(GSM6805319@meta.data)))
      h5_file[["var"]]$create_dataset("index", as.character(rownames(GSM6805319)))
      
      # Add reductions
      if ("pca" %in% names(GSM6805319@reductions)) {
        h5_file[["obsm"]]$create_dataset("X_pca", GSM6805319@reductions$pca@cell.embeddings)
      }
      if ("umap" %in% names(GSM6805319@reductions)) {
        h5_file[["obsm"]]$create_dataset("X_umap", GSM6805319@reductions$umap@cell.embeddings)
      }
      
      h5_file$close_all()
      cat("Successfully created H5AD file manually!\n")
      
    }, error = function(e3) {
      cat("Manual H5AD creation failed:", e3$message, "\n")
      cat("All conversion methods failed. Please check your data and try alternative approaches.\n")
    })
  })
})

cat("Conversion attempt completed.\n")