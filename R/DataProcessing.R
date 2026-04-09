# DataProcessing.R



library(Seurat)
library(SeuratData)
library(SeuratDisk)

library(data.table)
library(ggplot2)

# Define directories for each file type
rna_dir <- "/projects/vanaja_lab/satya/Datasets/GSE185381"
adt_dir <- file.path(rna_dir)
meta_dir <- file.path(rna_dir)

# List all RNA_soupx_processed files (full paths)
rna_files <- list.files(path = rna_dir, pattern = "RNA_soupx_processed\\.csv\\.gz$", full.names = TRUE)

# Create two empty lists to store Seurat objects for RNA and ADT assays separately
GSM_Proc_RNA <- list()
GSM_Proc_ADT <- list()
mergeALL <- list()

# Loop over each RNA file and process corresponding ADT and metadata files
for (rna_file in rna_files) {
  # Extract sample name by removing the RNA file suffix
  sample_name <- sub("_RNA_soupx_processed\\.csv\\.gz$", "", basename(rna_file))
  
  # --- Read RNA counts ---
  rna_counts <- fread(rna_file)
  rna_counts <- as.data.frame(rna_counts)
  rownames(rna_counts) <- rna_counts[, 1]  # Set the first column as row names
  rna_counts <- rna_counts[, -1]           # Remove the first column
  
  # --- Read ADT counts ---
  adt_file <- file.path(adt_dir, paste0(sample_name, "_ADT_processed.csv.gz"))
  if (!file.exists(adt_file)) {
    message("Skipping sample ", sample_name, ": ADT file does not exist: ", adt_file)
    next
  }
  adt_counts <- fread(adt_file)
  adt_counts <- as.data.frame(adt_counts)
  rownames(adt_counts) <- adt_counts[, 1]
  adt_counts <- adt_counts[, -1]
  
  # --- Read Metadata ---
  meta_file <- file.path(meta_dir, paste0(sample_name, "_metadata.csv.gz"))
  metadata <- fread(meta_file)
  metadata <- as.data.frame(metadata)
  rownames(metadata) <- metadata[, 1]
  metadata <- metadata[, -1]
  
  # --- Create Seurat objects separately ---
  # Create a Seurat object for RNA counts
  seurat_obj_rna <- CreateSeuratObject(counts = rna_counts, meta.data = metadata)
  seurat_obj_rna$sample <- sample_name
  
  # Create a Seurat object for ADT counts; specify the assay as "ADT"
  seurat_obj_adt <- CreateSeuratObject(counts = adt_counts, meta.data = metadata, assay = "ADT")
  seurat_obj_adt$sample <- sample_name
  
  # Store the Seurat objects in their respective lists
  GSM_Proc_RNA[[sample_name]] <- seurat_obj_rna
  GSM_Proc_ADT[[sample_name]] <- seurat_obj_adt
  
  
  RnaAndAdt <- CreateSeuratObject(counts = rna_counts, meta.data = metadata)
  RnaAndAdt$sample <- sample_name
  RnaAndAdt[["ADT"]] <- CreateAssayObject(counts = adt_counts)
  
  mergeALL[[sample_name]] <- RnaAndAdt
  
}

#GSM_Proc_Merge <- readRDS("/work/vanaja_lab/satya/GSM185381MergeRNA.rds")

GSMRNA_Merge <- merge(GSM_Proc_RNA[[1]], y = GSM_Proc_RNA[-1],
                      add.cell.ids = names(GSM_Proc_RNA),
                      project = "Merged_GSM_Proc")

GSMADT_Merge <- merge(GSM_Proc_ADT[[1]], y = GSM_Proc_ADT[-1],
                    add.cell.ids = names(GSM_Proc_ADT),
                    project = "Merged_GSM_Proc")

#GSMRNA_ADT <- merge(mergeALL[[1]], y = mergeALL[-1],
        #         add.cell.ids = names(mergeALL),
              #   project = "GSMRNA_ADT")



# Step 1: Ensure correct default assays
#DefaultAssay(GSMRNA_Merge) <- "RNA"
DefaultAssay(GSMADT_Merge) <- "ADT"


# ADT Normalization
GSMADT_Merge <- NormalizeData(GSMADT_Merge, normalization.method = "CLR", margin = 2, assay = "ADT")
# Gene Normalization
GSMADT_Merge <- NormalizeData(GSMADT_Merge)

GSMADT_Merge <- FindVariableFeatures(GSMADT_Merge)
GSMADT_Merge <- ScaleData(GSMADT_Merge)
GSMADT_Merge <- RunPCA(GSMADT_Merge)

GSMADT_Merge <- FindNeighbors(GSMADT_Merge, dims = 1:30, reduction = "pca")
GSMADT_Merge <- FindClusters(GSMADT_Merge, resolution = 2, cluster.name = "unintegrated_clusters")

GSMADT_Merge <- RunUMAP(GSMADT_Merge, dims = 1:30, reduction = "pca", reduction.name = "umap.unintegrated")
DimPlot(GSMADT_Merge, reduction = "umap.unintegrated", group.by = c("sample"))


GSMADT_Merge <- IntegrateLayers(object = GSMADT_Merge, method = CCAIntegration, orig.reduction = "pca", new.reduction = "integrated.cca",
                             verbose = FALSE)

# re-join layers after integration
GSMADT_Merge[["ADT"]] <- JoinLayers(GSMADT_Merge[["ADT"]])

GSMADT_Merge <- FindNeighbors(GSMADT_Merge, reduction = "integrated.cca", dims = 1:30)
GSMADT_Merge <- FindClusters(GSMADT_Merge, resolution = 1)

GSMADT_Merge <- RunUMAP(GSMADT_Merge, dims = 1:30, reduction = "integrated.cca")
# Visualization
DimPlot(GSMADT_Merge, reduction = "umap", group.by = c("sample", "seurat_annotations"))





output_file <- "/projects/vanaja_lab/satya/Datasets/GSE120221/GSE120221.h5ad"

# Write the Seurat object to an .h5ad file
WriteH5AD(GSMADT_Merge, file = output_file, assay = "ADT")

# Step 2: Create new Seurat object from RNA data
GSMControlAdt <- GSMADT_Merge

# Step 3: Add ADT assay to the RNA-based Seurat object
# Make sure cell names match between the RNA and ADT objects
common_cells <- intersect(colnames(GSMRNA_Merge), colnames(GSMADT_Merge))
GSMControlAdt <- subset(GSMControlAdt, cells = common_cells)

adt_assay <- GetAssay(GSMADT_Merge, assay = "ADT")
adt_assay <- subset(adt_assay, cells = common_cells)

GSMControlAdt[["ADT"]] <- adt_assay

# Optional: Set default assay back to RNA (or ADT, depending on what you want to work on)
DefaultAssay(GSMControlAdt) <- "ADT"

unique(GSMControlAdt@meta.data$samples)
# Subset the Seurat object for AML samples
aml_cells <- rownames(GSMControlAdt@meta.data)[grepl("^AML", GSMControlAdt@meta.data$samples)]
AML <- subset(GSMControlAdt, cells = aml_cells)

# Get sorted unique sample names from the AML object's metadata
unique_samples <- sort(unique(AML@meta.data$samples))

# Determine the halfway point
half_index <- floor(length(unique_samples) / 2)

# Divide the sample names into two groups
group1_samples <- unique_samples[1:half_index]
group2_samples <- unique_samples[(half_index + 1):length(unique_samples)]

# Identify cells belonging to each group based on the sample names
cells_group1 <- rownames(AML@meta.data)[AML@meta.data$samples %in% group1_samples]
cells_group2 <- rownames(AML@meta.data)[AML@meta.data$samples %in% group2_samples]

# Create the two AML subsets
AMLA <- subset(AML, cells = cells_group1)
AMLB <- subset(AML, cells = cells_group2)


# Define the samples to include
samples_to_include <- c('AML003', 'AML0048', 'AML005', 'AML006', 'AML009','AML012','AML0160','AML022','AML025','AML028','AML0361',
                        'AML038','AML043','AML048','AML052','AML073','AML2910','AML3050','AML3730','AML3762','AML3948')

library(MuDataSeurat)
# Subset the Seurat object
GSM_AML <- subset(GSM_Proc_Merge, subset = samples %in% samples_to_include)
GSM_AML <- JoinLayers(GSM_AML)
# Specify the output filename

DefaultAssay(AMLA) <- "ADT"

AMLA[["ADT"]] <- JoinLayers(AMLA[["ADT"]])

output_file <- "/projects/vanaja_lab/satya/Datasets/GSE120221/GSE120221.h5ad"

# Write the Seurat object to an .h5ad file
WriteH5AD(CONTROL, file = output_file, assay = "ADT")









