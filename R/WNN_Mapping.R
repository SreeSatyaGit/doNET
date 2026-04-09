# WNN_Mapping.R
library(Seurat)
library(SeuratData)
library(cowplot)
library(dplyr)

adata_gene_train[["ADT"]] <- adata_protein_train[["ADT"]]
adata_gene_test[["ADT"]] <- adata_protein_test[["ADT"]]
bm = adata_gene_train
DefaultAssay(bm) <- 'RNA'
bm <- NormalizeData(bm) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()


DefaultAssay(bm) <- 'ADT'
# we will use all ADT features for dimensional reduction
# we set a dimensional reduction name to avoid overwriting the 
VariableFeatures(bm) <- rownames(bm[["ADT"]])
bm <- NormalizeData(bm, normalization.method = 'CLR', margin = 2) %>% 
  ScaleData() %>% RunPCA(reduction.name = 'apca')

# Identify multimodal neighbors. These will be stored in the neighbors slot, 
# and can be accessed using bm[['weighted.nn']]
# The WNN graph can be accessed at bm[["wknn"]], 
# and the SNN graph used for clustering at bm[["wsnn"]]
# Cell-specific modality weights can be accessed at bm$RNA.weight
bm <- FindMultiModalNeighbors(
  bm, reduction.list = list("pca", "apca"), 
  dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
)

bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, resolution = 2, verbose = FALSE)

p1 <- DimPlot(bm, reduction = 'wnn.umap', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
p2 <- DimPlot(bm, reduction = 'wnn.umap', group.by = 'Broad_cell_identity', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
p1 + p2

bm <- RunUMAP(bm, reduction = 'pca', dims = 1:30, assay = 'RNA', 
              reduction.name = 'rna.umap', reduction.key = 'rnaUMAP_')
bm <- RunUMAP(bm, reduction = 'apca', dims = 1:18, assay = 'ADT', 
              reduction.name = 'adt.umap', reduction.key = 'adtUMAP_')

p3 <- DimPlot(bm, reduction = 'rna.umap', group.by = 'Broad_cell_identity', label = TRUE, 
              repel = TRUE, label.size = 2.5) + NoLegend()
p4 <- DimPlot(bm, reduction = 'adt.umap', group.by = 'Broad_cell_identity', label = TRUE, 
              repel = TRUE, label.size = 2.5) + NoLegend()
p3 + p4

DefaultAssay(bm) <- 'RNA'
bm <- RunSPCA(bm, assay = 'RNA',graph = 'wsnn')

bm <- FindNeighbors(
  object = bm,
  reduction = "spca",
  dims = 1:30,
  graph.name = "spca.annoy.neighbors", 
  k.param = 50,
  cache.index = TRUE,
  return.neighbor = TRUE,
  l2.norm = TRUE
)


DefaultAssay(adata_gene_test) <- 'RNA'
BM.query <- adata_gene_test
BM.query <- NormalizeData(BM.query)
BM.query<- FindVariableFeatures(BM.query)
BM.query <- ScaleData(BM.query, assay = 'RNA')


anchors <- list()

  anchors <- FindTransferAnchors(
    reference = bm,
    query = BM.query,
    k.filter = NA,
    reference.reduction = "spca", 
    reference.neighbors = "spca.annoy.neighbors", 
    dims = 1:30
  )
  predictions <- TransferData(anchorset = anchors, refdata = bm$Broad_cell_identity, dims = 1:30)
  BM.query <- AddMetaData(BM.query, metadata = predictions)
  
  
  bm <- RunUMAP(bm, dims = 1:30, return.model = TRUE)

  BM.query <- tryCatch({
    MapQuery(
      anchorset = anchors, 
      query = BM.query,
      reference = bm, 
      refdata = list(celltype = "Broad_cell_identity"),
      reference.reduction = "spca",
      reduction.model = "umap"
    )
  }, error = function(e) {
    message(paste(e$message))
    return(NULL)  # or return the original object: BM.query.batches[[i]]
  })

  p1 <- DimPlot(bm, reduction = "wnn.umap", group.by = "Broad_cell_identity", label = TRUE, label.size = 3,
                repel = TRUE) + NoLegend() + ggtitle("Reference annotations")
  p2 <- DimPlot(BM.query, reduction = "ref.spca", group.by = "predicted.id", label = TRUE,
                label.size = 3, repel = TRUE) + NoLegend() + ggtitle("Query transferred labels SPCA")
  p3 <- DimPlot(BM.query, reduction = "ref.umap", group.by = "predicted.id", label = TRUE,
                label.size = 3, repel = TRUE) + NoLegend() + ggtitle("Query transferred labels UMAP")
  p1 + p2 + p3
  # If you already removed the extra assay:
  
  
  score_mat <- GetAssayData(BM.query,
                            assay = "prediction.score.celltype",
                            slot  = "data")           # or "counts" if preferred
  
  # 2. Transpose so each row = a cell; add a column with the cell ID
  df_scores <- as.data.frame(t(as.matrix(score_mat)))
  df_scores <- cbind(cell_id = rownames(df_scores), df_scores)
  
  # 3. Write to a TSV
  write.table(df_scores,
              file      = "BM_query_ADT_prediction_spca_scores.tsv",
              sep       = "\t",
              row.names = FALSE,
              quote     = FALSE)
  
  cat("Saved", nrow(df_scores), "cells ×", ncol(df_scores)-1, "cell-type based on Protein scores to BM_query_prediction_spca_scores.tsv\n")
  
