# RefMapping.R
library(Seurat)
library(SeuratData)
library(ggplot2)
library(reticulate)
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(c(
  "basilisk", "DelayedArray", "S4Vectors", 
  "SingleCellExperiment", "SummarizedExperiment"
))
BiocManager::install("zellkonverter")
library(zellkonverter)

# we will use data from 2 technologies for the reference
#BM.ref <- subset(GSMRNA_Merge, samples %in% c("AML0612", "Control0082"))

with_progress({
  p <- progressor(10)
  result <- future.apply::future_lapply(1:10, function(x) {
    Sys.sleep(1)
    p()
    sqrt(x)
  })
})

CONTROL$sample <- "CONTROL"
AMLA$sample    <- "AML"

CONTROL <- RenameCells(CONTROL, add.cell.id = "CTRL")
AMLA    <- RenameCells(AMLA,    add.cell.id = "AML")



BM.ref<- bm

#BM.ref[["RNA"]] <- split(BM.ref[["RNA"]], f = BM.ref$samples)

# pre-process dataset (without integration)
BM.ref <- NormalizeData(BM.ref)
BM.ref <- FindVariableFeatures(BM.ref)
BM.ref <- ScaleData(BM.ref)
BM.ref <- RunPCA(BM.ref)
BM.ref <- FindNeighbors(BM.ref, dims = 1:30)
BM.ref <- FindClusters(BM.ref)

BM.ref <- RunUMAP(BM.ref, dims = 1:30)
DimPlot(BM.ref, group.by = c("Broad_cell_identity", "samples"))


BM.ref <- IntegrateLayers(object = BM.ref, method = CCAIntegration, orig.reduction = "pca", new.reduction = "integrated.cca",
                        verbose = FALSE)

# re-join layers after integration
BM.ref[["ADT"]] <- JoinLayers(BM.ref[["ADT"]])

BM.ref <- FindNeighbors(BM.ref, reduction = "integrated.cca", dims = 1:30)
BM.ref <- FindClusters(BM.ref, resolution = 1)

# Set the path to the directory containing the three files
data_dir <- "/home/nandivada.s/Datasets/GSM3872434_ETV6-RUNX1_1/"
cell_anno <- fread("/home/nandivada.s/Datasets/GSE132509_cell_annotations/GSE132509_cell_annotations.tsv.gz")

# Read the data
etv6_runx1_data <- Read10X(data.dir = data_dir)

# Optionally create a Seurat object directly
etv6_runx1 <- CreateSeuratObject(counts = etv6_runx1_data, project = "ETV6_RUNX1")


# select two technologies for the query datasets

DefaultAssay(adata_gene_test) <- 'ADT'
BM.query <- adata_gene_test
BM.query <- NormalizeData(BM.query, normalization.method = 'CLR')
BM.anchors <- FindTransferAnchors(reference = BM.ref, query = BM.query, dims = 1:30,
                                        reference.reduction = "pca")
predictions <- TransferData(anchorset = BM.anchors, refdata = BM.ref$Broad_cell_identity, dims = 1:30)


# Add as metadata
BM.query <- AddMetaData(BM.query, metadata = predictions)


BM.ref <- RunUMAP(BM.ref, dims = 1:30, return.model = TRUE)
BM.query <- MapQuery(anchorset = BM.anchors, reference = BM.ref, query = BM.query,
                           refdata = list(celltype = "Broad_cell_identity"), reference.reduction = "pca", reduction.model = "umap")



p1 <- DimPlot(BM.ref, reduction = "umap", group.by = "Broad_cell_identity", label = TRUE, label.size = 3,
              repel = TRUE) + NoLegend() + ggtitle("Reference annotations")
p2 <- DimPlot(BM.query, reduction = "ref.umap", group.by = "predicted.id", label = TRUE,
              label.size = 3, repel = TRUE) + NoLegend() + ggtitle("Query transferred labels")
p1 + p2
# If you already removed the extra assay:


score_mat <- GetAssayData(BM.query,
                          assay = "prediction.score.celltype",
                          slot  = "data")           # or "counts" if preferred

# 2. Transpose so each row = a cell; add a column with the cell ID
df_scores <- as.data.frame(t(as.matrix(score_mat)))
df_scores <- cbind(cell_id = rownames(df_scores), df_scores)

# 3. Write to a TSV
write.table(df_scores,
            file      = "BM_query_ADT_prediction_scores.tsv",
            sep       = "\t",
            row.names = FALSE,
            quote     = FALSE)

cat("Saved", nrow(df_scores), "cells ×", ncol(df_scores)-1, "cell-type based on Protein scores to BM_query_prediction_scores.tsv\n")
