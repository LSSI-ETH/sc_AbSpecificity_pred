# 
# R script for doing gene expression ananlysis and classification based on that! 
#
# 1. Pseudobulk cells
# 2. split off 30% of the cells
# 3. classify cells based on gene expression program (define upregulated gene program)
# Date: 16.07.2024
# Author: Lena Erlach



library(ggplot2)
library(Seurat)  
library(dplyr)
library(tidyverse)


library(scater)
library(cowplot)
library(Matrix.utils)
library(edgeR)
library(magrittr)
library(Matrix)
library(purrr)
library(reshape2)
library(S4Vectors)
library(tibble)
library(SingleCellExperiment)
library(pheatmap)
library(apeglm)
library(png)
library(DESeq2)
library(RColorBrewer)
library(ashr)
library(EnhancedVolcano)
library(caret)

###### Custom function for DE analysis
DE_function <- function(Seurat3, DE_outPath, save_stuff, antigens, control_for_antigens) {

  Idents(Seurat3) <- "seurat_clusters"
  
  ### Rename clusters
  new_ident <- setNames(c("B cells",
                          "B cells",
                          "B cells",
                          "B cells",
                          "B cells",
                          "B cells",
                          "B cells",
                          "B cells",
                          "B cells",
                          "B cells",
                          "B cells",
                          "B cells",
                          "B cells"),
  levels(Seurat3))
  
  Seurat3 <- RenameIdents(Seurat3, new_ident)
  
  
  # Now we access count matrix and metadata
  counts <- Seurat3@assays$RNA@counts
  metadata <- Seurat3@meta.data
  
  
  # Add cluster identity to metadata:
  metadata$cluster_id <- factor(Seurat3@active.ident)
  
  
  # Create single cell experiment object
  sce <- SingleCellExperiment(assays = list(counts = counts), 
                              colData = metadata)
  
  
  # Finally, we identify groups for count data aggregation
  groups <- colData(sce)[, c("cluster_id", "sample_id", "ag_specificity")]
  
  
  
  ##Acquiring metrics for aggregation of cells within samples
  # named vector of cluster names - in this case, they are all B cells.
  cids <- purrr::set_names(levels(sce$cluster_id))
  
  
  # Now we get the number of clusters 
  nc <- length(cids)
  
  
  # Similarly, a named vector of sample names and the number of samples:
  sids <- purrr::set_names(levels(as.factor(sce$sample_id)))
  ns <- length(sids)
  
  n_cells <-as.numeric(table(sce$sample_id))
  
  # Next, match the sids vector to the rows by sample_id column for reordering:
  m <- match(sids, sce$sample_id)
  #The output of match() is a vector of the positions of the first matches of sids in sample_id
  
  
  # Now use this to create metadata, by combining the reordered metadata with the number of cells corresponding to each sample:
  ei <- data.frame(colData(sce)[m, ], 
                   n_cells, row.names = NULL) %>% 
    dplyr::select(-"cluster_id")
  
  # Now, we can aggregate counts to sample level: I.e., we sum counts within samples and clusters (of which we only have one now). 
  
  # First, we subset the metadata to only include the cluster and sample IDs.
  groups <- colData(sce)[, c("cluster_id", "sample_id")]
  
  
  # Now, we can finally aggregate across cluster-sample groups:
  pb <- aggregate.Matrix(t(counts(sce)), 
                         groupings = groups, fun = "sum") 
  
  
  # We still split into individual matrices (even though we only had one to start with and end with the same one) just to make sure we also swap the rows and columns.
  splitf <- sapply(stringr::str_split(rownames(pb), 
                                      pattern = "_",  
                                      n = 2),'[', 1)
  
  
  
  pb <- split.data.frame(pb, 
                         factor(splitf)) %>%
    lapply(function(u) 
      set_colnames(t(u), 
                   rownames(u)))
                   # stringr::str_extract(rownames(u), "(?<=_)[:alnum:]+")))
  
  
  
  
  
  
  
  ##Sample-level metadata
  # We have the 'ei' dataframe, but we need to combine this with cluster IDs
  
  # First, create a vector of sample names combined for each cluster:
  get_sample_ids <- function(x){
    pb[[x]] %>%
      colnames()
  }
  
  de_samples <- map(1:length(cids), get_sample_ids) %>%
    unlist()
  
  samples_list <- map(1:length(cids), get_sample_ids)
  
  
  
  get_cluster_ids <- function(x){
    rep(names(pb)[x], 
        each = length(samples_list[[x]]))
  }
  
  
  de_cluster_ids <- map(1:length(cids), get_cluster_ids) %>%
    unlist()
  
  
  # Now, we create a dataframe with the cluster IDs and the corresponding sample IDs by merging.
  gg_df <- data.frame(cluster_id = de_cluster_ids,
                      sample_id = de_samples)
  
  # edit gg_df manually
  gg_df[, "sample_id"] = ei[, "sample_id"]
  ei[, "ag_specificity"] <- factor(ei[, "ag_specificity"])
  ei[, "group_id"] <- factor(ei[, "group_id"])
  
  
  gg_df <- left_join(gg_df, ei[, c("sample_id", "ag_specificity", "group_id")]) 
  
  
  metadata <- gg_df %>%
    dplyr::select(cluster_id, sample_id, ag_specificity, group_id) 
  rownames(metadata) <- metadata$sample_id
  
  # add antigen to metadata
  metadata$antigen <- factor(antigens)
  

  
  # First, we select the single B cell cluster:
  clusters <- levels(as.factor(metadata$cluster_id))
  
  # subset the counts to this cluster:
  counts_subset <- pb[[clusters[1]]]
  colnames(counts_subset) <- ei[, "sample_id"]
  cluster_counts <- data.frame(counts_subset[, which(colnames(counts_subset) %in% rownames(metadata))])
  
  
  
  
  
  
  
  
  ###################################################### 
  #Differential expression analysis with DESeq2
  ######################################################
  
  ######### Create DESeq2 object:
  if (control_for_antigens == TRUE) {
    dds <- DESeqDataSetFromMatrix(cluster_counts, 
                                  colData = metadata, 
                                  design = ~ ag_specificity + antigen)
  } else {
    dds <- DESeqDataSetFromMatrix(cluster_counts, 
                                  colData = metadata, 
                                  design = ~ ag_specificity )
  }
  
  
  # We start with PCA, but before that, we normalize and log-transform the counts:
  rld <- rlog(dds, blind=TRUE)
  p <- DESeq2::plotPCA(rld, intgroup = "sample_id") + scale_color_brewer(palette="Set1") + theme_light()
  p
  if (save_stuff == TRUE) {ggsave(paste0(DE_outPath, "PCA_sample_id.png"), p, width=5, height = 5)}
  
  p <- DESeq2::plotPCA(rld, intgroup = "ag_specificity") + scale_color_brewer(palette="Set1") + theme_light()
  p
  if (save_stuff == TRUE) {ggsave(paste0(DE_outPath, "PCA_ag_specificity.png"), p, width=5, height = 5)}
  
  
  
  
  ######################################################
  # Correlation analysis
  ######################################################
  
  
  # We also do hierarchical clustering. 
  # Extract the rlog matrix from the object and compute pairwise correlation values
  rld_mat <- assay(rld)
  rld_cor <- cor(rld_mat)
  
  # Plot heatmap
  p <- pheatmap(rld_cor, annotation = metadata[, c("sample_id", "ag_specificity", "group_id"), drop=F]) #+ theme_light()
  if (save_stuff == TRUE) {ggsave(paste0(DE_outPath, "Correlation_heatmap.png"), p, width=5, height = 4)}
  
  # We can proceed with DGEA:
  dds <- DESeq(dds)
  
  # Plot dispersion estimates to check fit of negative binomial model:
  plotDispEsts(dds)
  
  
  ##Exploring results
  contrast <- c("ag_specificity", levels(metadata$ag_specificity)[1], levels(metadata$ag_specificity)[2])
  
  # resultsNames(dds) - ignore this. I case of using coef, this command must be consulted.
  res <- results(dds, 
                 contrast = contrast,
                 alpha = 0.05)
  
  res <- lfcShrink(dds, 
                   type = "ashr",
                   contrast = contrast,
                   res=res)
  
  # Turn the results object into a tibble for use with tidyverse functions
  res_tbl <- res %>%
    data.frame() %>%
    rownames_to_column(var="gene") %>%
    as_tibble()
  
  
  write.csv(res_tbl,
            paste0(DE_outPath, levels(metadata$ag_specificity)[1], "_vs_", levels(metadata$ag_specificity)[2], "_all_genes.csv"),
            quote = FALSE, 
            row.names = FALSE)
  
  
  # Set thresholds
  padj_cutoff <- 0.05
  
  # Subset the significant results
  sig_res <- dplyr::filter(res_tbl, padj < padj_cutoff) %>%
    dplyr::arrange(padj)
  
  # Write significant results to file
  if (save_stuff == TRUE) {write.csv(sig_res,
            paste0(DE_outPath, levels(metadata$ag_specificity)[1], "_vs_", levels(metadata$ag_specificity)[2], "_sig_genes.csv"),
            quote = FALSE, 
            row.names = FALSE)}
  
  
  
  ######################################################
  # Norm count of sign. genes analysis
  ######################################################
  
  # Scatterplot of normalized expression of top 20 most significant genes:
  ## ggplot of top genes
  normalized_counts <- counts(dds, 
                              normalized = TRUE)
  
  ## Order results by padj values
  top20_sig_genes <- sig_res %>%
    dplyr::arrange(padj) %>%
    dplyr::pull(gene) %>%
    head(n=20)
  
  
  top20_sig_norm <- data.frame(normalized_counts) %>%
    rownames_to_column(var = "gene") %>%
    dplyr::filter(gene %in% top20_sig_genes)
  
  gathered_top20_sig <- top20_sig_norm %>%
    gather(colnames(top20_sig_norm)[2:length(colnames(top20_sig_norm))], key = "samplename", value = "normalized_counts")
  
  gathered_top20_sig <- inner_join(ei[, c("sample_id", "ag_specificity" )], gathered_top20_sig, by = c("sample_id" = "samplename"))
  
  
  ## plot using ggplot2
  p <- ggplot(gathered_top20_sig) +
    geom_point(aes(x = gene, 
                   y = normalized_counts, 
                   color = ag_specificity), 
               position=position_jitter(w=0.1,h=0)) +
    scale_y_log10() +
    xlab("Genes") +
    ylab("log10 Normalized Counts") +
    ggtitle("Top 20 Significant DE Genes") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    theme(plot.title = element_text(hjust = 0.5))
  
  if (save_stuff == TRUE) {ggsave(paste0(DE_outPath, "Norm_counts_ag_specificity.png"), p, width=8, height = 5)}
  
  
  ## plot using ggplot2
  p <- ggplot(gathered_top20_sig) +
    geom_point(aes(x = gene, 
                   y = normalized_counts, 
                   color = sample_id), 
               position=position_jitter(w=0.1,h=0)) +
    scale_y_log10() +
    xlab("Genes") +
    ylab("log10 Normalized Counts") +
    ggtitle("Top 20 Significant DE Genes") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    theme(plot.title = element_text(hjust = 0.5))
  
  if (save_stuff == TRUE) {ggsave(paste0(DE_outPath, "Norm_counts_sample_id.png"), p, width=8, height = 5)}
  
  
  
  
  ######################################################
  # Heatmap of sign. genes
  ######################################################
  
  # We can also make a clustering heatmap of the significant genes:
  # Extract normalized counts for only the significant genes
  sig_norm <- data.frame(normalized_counts) %>%
    rownames_to_column(var = "gene") %>%
    dplyr::filter(gene %in% sig_res$gene)
  
  # Set a color palette
  heat_colors <- brewer.pal(6, "RdBu")
  
  # Run pheatmap using the metadata data frame for the annotation
  p <- pheatmap(sig_norm[ , 2:length(colnames(sig_norm))], 
           color = heat_colors, 
           cluster_rows = T, 
           show_rownames = F,
           annotation = metadata[, c("ag_specificity", "sample_id", "group_id")], 
           border_color = NA, 
           fontsize = 10, 
           scale = "row", 
           fontsize_row = 10, 
           height = 20)        
  
  if (save_stuff == TRUE) {ggsave(paste0(DE_outPath, "Sig_genes_heatmap.png"), p, width=6, height = 6)}
  
  
  
  
  ######################################################
  # Volcano plot
  ######################################################
  
  
  ## Obtain logical vector where TRUE values denote padj values < 0.05 and fold change > 1.5 in either direction
  res_table_thres <- res_tbl %>% 
    mutate(threshold = padj < 0.05 & abs(log2FoldChange) >= 0.58)
  
  ## Volcano plot
  ggplot(res_table_thres) +
    geom_point(aes(x = log2FoldChange, y = -log10(padj), colour = threshold)) +
    ggtitle("Differential expression between specific and non-specific B cells") +
    xlab("log2 fold change") + 
    ylab("-log10 adjusted p-value") +
    scale_y_continuous(limits = c(0,20)) +
    theme(legend.position = "none",
          plot.title = element_text(size = rel(1.5), hjust = 0.5),
          axis.title = element_text(size = rel(1.25)))               
  
  res_tbl <- column_to_rownames(res_tbl, "gene")
  
  
  p <- EnhancedVolcano(res_tbl,
                  lab = rownames(res_tbl),
                  x = 'log2FoldChange',
                  y = 'padj',
                  title = "Specific vs non-specific (merged B cells)",
                  xlab = bquote(~Log[2]~ 'fold change'),
                  pCutoff = 10e-11,
                  FCcutoff = 1.0,
                  pointSize = 2.5,
                  labSize = 4.0,
                  colAlpha = 1,
                  legendLabSize = 10,
                  legendIconSize = 4.0,
                  drawConnectors = TRUE,
                  widthConnectors = 0.75)
  
  if (save_stuff == TRUE) {ggsave(paste0(DE_outPath, "Labelled_Volcano.png"), p, width=6, height = 6)}
  
  return(sig_res)
}
  
DE_clf_function <- function(reference_train, reference_test, sig_res) {
  Idents(reference_test) <- "sample_id"
  Idents(reference_train) <- "sample_id"
  
  
  ## Order results by padj values & log2FoldChange
  top10_up <- sig_res %>%
    dplyr::arrange(padj) %>%
    head(n=50) %>%
    dplyr::arrange(-log2FoldChange) %>%
    dplyr::pull(gene) %>%
    head(n=10)
  
  ## Order results by padj values & log2FoldChange
  top10_down <- sig_res %>%
    dplyr::arrange(padj) %>%
    head(n=50) %>%
    dplyr::arrange(log2FoldChange) %>%
    dplyr::pull(gene) %>%
    head(n=10)
  
  
  
  
  #### Add module score to train and test datasets
  reference_train <- AddModuleScore(
    object = reference_train,
    features = list(top10_up),
    nbin = 1,
    ctrl = 5,
    name = 'top10_up'
  )
  #### Add module score to train and test datasets
  reference_train <- AddModuleScore(
    object = reference_train,
    features = list(top10_down),
    nbin = 1,
    ctrl = 5,
    name = 'top10_down'
  )
  
  
  reference_test <- AddModuleScore(
    object = reference_train,
    features = list(top10_up),
    ctrl = 5,
    name = 'top10_up'
  )
  reference_test <- AddModuleScore(
    object = reference_train,
    features = list(top10_down),
    ctrl = 5,
    name = 'top10_down'
  )
  print("Top10_up: ")
  print(top10_up)
  print("Top10_down: ")
  print(top10_down)
  
  
  FeaturePlot(reference_train, features = "top10_up1") + FeaturePlot(reference_train, features = "top10_down1")
  FeaturePlot(reference_test, features = "top10_up1") + FeaturePlot(reference_test, features = "top10_down1")
  
  return(list(reference_train, reference_test, top10_up, top10_down))
  
}  

group_names <- c("specific", "nonspecific")

# RBD_path <- "/Users/lerlach/Documents/current_work/OVA_seq/RBD_Wuhan_baited/Analysis/Modelling/data/VDJ_GEX_object_scPredTrained_RBD_Wuhan_harmony_overlapping_clones_rm.rds"
# OVA_path <- "/Users/lerlach/Documents/current_work/OVA_seq/Analysis/Modelling/data/VDJ_GEX_object_scPredTrained_OverlappingCDRsExcl_group_id_OVA_harmony.rds"
# INT_path <- "/Users/lerlach/Documents/current_work/OVA_seq/OVA_RBD_Wuhan_integrated/Analysis/Modelling/data/VDJ_GEX_object_OVA_RBD_harmony_overlapping_clones_rm.rds"
# 
# DE_outPath <- "/Users/lerlach/Documents/current_work/OVA_seq/Analysis/DE_analysis/OVA_RBD_integrated_DGEA/RBD/Traindata_only/"

vgm_out_path <- "/Users/lerlach/Documents/current_work/Publications/2024_single-cell_Ab_specificity_predictions/CellSystems_sc_Ab_specificity/Rcode/Analysis_test/"
# /Analysis_test/data/VDJ_GEX_object_OVA_RBD_integrated_harmony_overlapping_clones_rm.rds
RBD_path <- paste0(vgm_out_path, "data/VDJ_GEX_object_RBD_harmony_overlapping_clones_rm.rds")
OVA_path <- paste0(vgm_out_path, "data/VDJ_GEX_object_OVA_harmony_overlapping_clones_rm.rds")
INT_path <- paste0(vgm_out_path, "data/VDJ_GEX_object_OVA_RBD_integrated_harmony_overlapping_clones_rm.rds")

DE_outPath <- "/Users/lerlach/Documents/current_work/Publications/2024_single-cell_Ab_specificity_predictions/CellSystems_sc_Ab_specificity/Rcode/Analysis_test/DE_Analysis/"

set.seed(111)

### Load original file
VDJ_GEX_matrix <- readRDS(INT_path)
reference <- VDJ_GEX_matrix[["GEX"]]
VDJ_mat <- VDJ_GEX_matrix[["VDJ"]]


#### Set TRUE if plots are saved
save_stuff <- TRUE


##########################################################################
####### TRAIN TEST SPLIT
##########################################################################

##### Split off 30% of the cells randomly 
# set another reference barcode (unique for whole dataset)
reference$barcode <- names(reference$orig.ident)
reference@meta.data$ag_specificity <- factor(reference_train@meta.data$ag_specificity, levels = c("specific", "nonspecific") )


bcs <- names(reference$barcode)
num_70 <- round(length(bcs)*0.7)

bc_70 <- sample(bcs, num_70, replace = F)
bc_30 <- subset(bcs, !(bcs %in% bc_70))

# check the barcodes
unique(bc_70 %in% bc_30)
length(bcs)
length(unique(bcs))
length(bc_70) + length(bc_30)

reference_train <- subset(reference, subset = barcode %in% bc_70)
reference_test <- subset(reference, subset = barcode %in% bc_30)




##########################################################################
####### PSEUDO BULKING AND DE TESTING
##########################################################################




Seurat3 <- reference
Seurat3$ag_specificity <- factor(Seurat3$ag_specificity, levels = c("specific", "nonspecific") )


# run for INT
antigens <- c("OVA", "RBD", "OVA", "RBD", "OVA", "OVA")
sig_res <- DE_function(Seurat3, DE_outPath, save_stuff = save_stuff, antigens, control_for_antigens=TRUE)

















