# AMLTirated.R
library(vegan)
library(dplyr)
library(devtools)
library(oposSOM)
library(Seurat)
library(SeuratData)
library(patchwork)
library(biomaRt)
library(ggplot2)
library(googledrive)

options(SeuratData.repo.use = "http://seurat.nygenome.org")

# Load the dataset
reference <- readRDS("/brahms/hartmana/vignette_data/pbmc_multimodal_2023.rds")
reference <- LoadData("pbmcMultiome","pbmc.rna")
reference <- UpdateSeuratObject(reference)
pbmc3k <- LoadData("pbmc3k")

file <- drive_get("AML_Datasets/titrated-seurat.rds")
temp_file <- tempfile(fileext = ".rds")
drive_download(file, path = temp_file, overwrite = TRUE)
Titrated_AML <- readRDS(temp_file)



Titrated_AML <- loadData()
AMLPatients <- merge(
  x = AML1,
  y = list(AML2,AML3,AML3B,AML4,AML5),
  add.cell.ids = c("AML1", "AML2","AML3","AML3B","AML4","AML5"),
  project = "AMLPatients"
)

# Combined Seurat object setup
HealthyPatients <- merge(
  x = h2,
  y = list(h3,h5,h6,h7),
  add.cell.ids = c("h2", "h3","h5","h6","h7"),
  project = "HealthyPatients"
)
combined_AML_Healthy <- UpdateSeuratObject(combined_AML_Healthy)
combined_AML_Healthy <- SCTransform(combined_AML_Healthy, verbose = FALSE)

if (!"SCTModel.list" %in% slotNames(reference[["SCT"]])) {
  stop("SCT model is missing in the reference object.")
}


anchors <- FindTransferAnchors(
  reference = reference,
  query = combined_AML_Healthy,
  normalization.method = "SCT",
  reference.reduction = "spca",
  dims = 1:50,
  recompute.residuals = TRUE
)

combined_AML_Healthy <- MapQuery(
  anchorset = anchors,
  query = combined_AML_Healthy,
  reference = reference,
  refdata = list(
    celltype.l1 = "celltype.l1",
    celltype.l2 = "celltype.l2",
    predicted_ADT = "ADT"
  ),
  reference.reduction = "spca", 
  reduction.model = "wnn.umap"
)
p1 = DimPlot(combined_AML_Healthy, reduction = "ref.umap", group.by = "predicted.celltype.l1", label = TRUE, label.size = 3, repel = TRUE) + NoLegend()
p2 = DimPlot(combined_AML_Healthy, reduction = "ref.umap", group.by = "predicted.celltype.l2", label = TRUE, label.size = 3 ,repel = TRUE) + NoLegend()
p1 + p2

combined_AML_Healthy <- NormalizeData(combined_AML_Healthy)
combined_AML_Healthy <- FindVariableFeatures(combined_AML_Healthy)
combined_AML_Healthy <- ScaleData(combined_AML_Healthy)
combined_AML_Healthy <- RunPCA(combined_AML_Healthy)

combined_AML_Healthy <- IntegrateLayers(
  object = combined_AML_Healthy,
  method = CCAIntegration,
  orig.reduction = "pca",
  new.reduction = "integrated.cca",
  verbose = FALSE
)

combined_AML_Healthy <- FindNeighbors(combined_AML_Healthy, reduction = "integrated.cca", dims = 1:30)
combined_AML_Healthy <- FindClusters(combined_AML_Healthy, resolution = 0.1, cluster.name = "cca_clusters")
combined_AML_Healthy <- RunUMAP(combined_AML_Healthy, reduction = "integrated.cca", dims = 1:30, reduction.name = "umap.cca")

p1 <- DimPlot(combined_AML_Healthy, reduction = "umap.cca", group.by = c("orig.ident", "cca_clusters"))
p2 <- DimPlot(combined_AML_Healthy, reduction = "umap.cca", split.by = "cca_clusters", group.by = "orig.ident") 
p3 <- DimPlot(combined_AML_Healthy, reduction = "umap.cca", split.by = "orig.ident", group.by = "cca_clusters") 

p1
# Identify most common and highly expressed genes in each cluster
# Ensure clustering has been performed
combined_AML_Healthy <- FindClusters(combined_AML_Healthy, resolution = 0.1)

# Calculate average expression per cluster

average_expression <- AverageExpression(combined_AML_Healthy, group.by = "orig.ident")

# Extract average RNA expression data
average_rna <- average_expression$RNA

# Convert to a data frame
average_rna_df <- as.data.frame(as.matrix(average_rna))

# Find top 10 highly expressed genes per cluster
top_genes_per_cluster <- lapply(
  colnames(average_rna_df),
  function(cluster) {
    cluster_data <- average_rna_df[, cluster, drop = FALSE]  # Select data for the cluster
    cluster_data <- data.frame(gene = rownames(cluster_data), expression = cluster_data[, 1])  # Add gene names
    sorted_genes <- cluster_data[order(-cluster_data$expression), ]  # Sort by expression
    head(sorted_genes, 10)  # Return top 10 genes
  }
)
names(top_genes_per_cluster) <- colnames(average_rna_df)

# View results
for (cluster_id in names(top_genes_per_cluster)) {
  cat("Cluster", cluster_id, "Top Genes:\n")
  print(top_genes_per_cluster[[cluster_id]])
  cat("\n\n")
}

#oposSOM Analysis

# Create a new opossom environment
env <- opossom.new(list(
  dataset.name = "SandRVResutls",
  dim.1stLvlSom = 40
))

# Example: Assign normalized data to the environment
env$indata <- as.matrix(log(average_rna + 1))  # Add 1 to avoid log(0)




group.info <- data.frame( 
  group.labels = c("Senstive",
                   "RV"),
  
  group.colors = c(
    "red2",
    "brown"),
  
  row.names=colnames(average_rna_df))
# Run the oposSOM pipeline
opossom.run(env)




# Alpha diversity measures
metadata <- combined_AML_Healthy@meta.data
cluster_counts <- table(metadata$seurat_clusters)
species_richness <- length(cluster_counts)
proportions <- cluster_counts / sum(cluster_counts)
shannon_index <- -sum(proportions * log(proportions))
simpson_index <- 1 - sum(proportions^2)
abundance_matrix <- as.matrix(cluster_counts)
shannon_index_vegan <- diversity(abundance_matrix, index = "shannon")
simpson_index_vegan <- diversity(abundance_matrix, index = "simpson")
