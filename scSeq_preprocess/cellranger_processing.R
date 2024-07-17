#!/usr/bin/env Rscript


############ 
# R analysis of "Evaluating predictive patterns of antigen-specific B cells by single-cell transcriptome and antibody repertoire sequencing.
# Author: Lena Erlach 
# 17th June 2024

# edit 12.12.2022 : VGM was run additionally with harmony

#####


library(ggplot2)
library(plotly)
library(htmlwidgets)
library(Seurat)  
library(dplyr)
library(cowplot)
library(gridExtra)
library(reshape2)
library(tidyverse)
library(RColorBrewer)
library(ggpubr)
library(Platypus)


setwd('/Users/lerlach/Documents/current_work/Publications/2024_single-cell_Ab_specificity_predictions/CellSystems_sc_Ab_specificity/Rcode/')
source("call_MIXCR_fct.R")

source("utils.R")
# Set working directory

# set path to MiXCR
mixcr.directory <- "/usr/local/Cellar/mixcr/4.2.0-1/mixcr"

# cellranger output files
cellranger_out_path_OVA <- paste0("/Users/lerlach/Documents/current_work/OVA_seq/aligned_data/OVA_specific/")
cellranger_out_path_RBD <- paste0("/Users/lerlach/Documents/current_work/OVA_seq/RBD_Wuhan_baited/Aligned/")
vgm_out_path <-  paste0(getwd(), "/Analysis_test/")

ifelse(!dir.exists(file.path(vgm_out_path)), dir.create(file.path(vgm_out_path), recursive = TRUE), FALSE)


# Setups
species <- "mmu"
set.seed(123)





################################################################################
# PLATYPUS: CELLRANGER OUTPUT PROCESSING 
################################################################################

### Process OVA files
  
VDJ.out.directory.list <- list()
VDJ.out.directory.list[[1]] <- c(paste0(cellranger_out_path_OVA, "vdj/S9/"))
VDJ.out.directory.list[[2]] <- c(paste0(cellranger_out_path_OVA, "vdj/S10/"))
VDJ.out.directory.list[[3]] <- c(paste0(cellranger_out_path_OVA, "vdj/S11/"))     
VDJ.out.directory.list[[4]] <- c(paste0(cellranger_out_path_OVA, "vdj/S12/"))

GEX.out.directory.list <- list()
GEX.out.directory.list[[1]] <- c(paste0(cellranger_out_path_OVA, "gex/S1/"))
GEX.out.directory.list[[2]] <- c(paste0(cellranger_out_path_OVA, "gex/S2/"))
GEX.out.directory.list[[3]] <- c(paste0(cellranger_out_path_OVA, "gex/S3/")) 
GEX.out.directory.list[[4]] <- c(paste0(cellranger_out_path_OVA, "gex/S4/")) 

# settings for VGM
group_id <- c(1,2,1,2)
VDJ_GEX_matrix <- list()

VDJ_GEX_matrix[["OVA"]] <- VDJ_GEX_matrix(VDJ.out.directory.list = VDJ.out.directory.list,
                                 GEX.out.directory.list = GEX.out.directory.list,
                                 GEX.integrate = T, VDJ.combine = T,
                                 integrate.GEX.to.VDJ = T,
                                 integrate.VDJ.to.GEX = T,
                                 exclude.GEX.not.in.VDJ = F,
                                 filter.overlapping.barcodes.GEX = T,
                                 filter.overlapping.barcodes.VDJ = T,
                                 get.VDJ.stats = T,
                                 exclude.on.cell.state.markers = c("CD3E+;CD3G;CD3D"),
                                 parallel.processing = "none",
                                 subsample.barcodes = F,
                                 trim.and.align = T,
                                 integration.method = "harmony",
                                 group.id = group_id,
                                 neighbor.dim = 1:20,
                                 mds.dim=1:20,)





### Process RBD files
VDJ.out.directory.list <- list()
VDJ.out.directory.list[[1]] <- c(paste0(cellranger_out_path_RBD, "vdj/LN_Wuhan_pos/outs/"))
VDJ.out.directory.list[[2]] <- c(paste0(cellranger_out_path_RBD, "vdj/LN_Wuhan_neg/outs/"))


GEX.out.directory.list <- list()
GEX.out.directory.list[[1]] <- c(paste0(cellranger_out_path_RBD, "gex/LN_Wuhan_pos/outs/"))
GEX.out.directory.list[[2]] <- c(paste0(cellranger_out_path_RBD, "gex/LN_Wuhan_neg/outs/"))



# settings for VGM
group_id <- c("LN_Wuhan_pos","LN_Wuhan_neg")
VDJ_GEX_matrix[["RBD"]] <- VDJ_GEX_matrix(VDJ.out.directory.list = VDJ.out.directory.list,
                                 GEX.out.directory.list = GEX.out.directory.list,
                                 GEX.integrate = T, VDJ.combine = T,
                                 integrate.GEX.to.VDJ = T,
                                 integrate.VDJ.to.GEX = T,
                                 exclude.GEX.not.in.VDJ = F,
                                 filter.overlapping.barcodes.GEX = T,
                                 filter.overlapping.barcodes.VDJ = T,
                                 VDJ.gene.filter = T,   #########
                                 get.VDJ.stats = T,
                                 exclude.on.cell.state.markers = "none", #####
                                 parallel.processing = "none",
                                 subsample.barcodes = F,
                                 trim.and.align = T,
                                 integration.method = "harmony",
                                 group.id = group_id,
                                 neighbor.dim = 1:10,
                                 mds.dim=1:10)




# DimPlot(VDJ_GEX_matrix[["OVA"]][["GEX"]]) + DimPlot(VDJ_GEX_matrix[["RBD"]][["GEX"]])




########################################################
# CALL MIXCR 
# ########################################################
### OVA
VDJ_mat_mixcr <- VDJ_call_MIXCR(VDJ = VDJ_GEX_matrix[["OVA"]][["VDJ"]],
                                mixcr.directory = mixcr.directory,
                                species = species, platypus.version = "v3", simplify = TRUE)

VDJ_GEX_matrix[["OVA"]][['VDJ']] <- VDJ_mat_mixcr


### RBD
VDJ_mat_mixcr <- VDJ_call_MIXCR(VDJ = VDJ_GEX_matrix[["RBD"]][["VDJ"]],
                                mixcr.directory = mixcr.directory,
                                species = species, platypus.version = "v3", simplify = TRUE)

VDJ_GEX_matrix[["RBD"]][['VDJ']] <- VDJ_mat_mixcr



# # Save intermediate file
# saveRDS(VDJ_GEX_matrix[["OVA"]], paste0(vgm_out_path, "proc0VDJ_GEX_OVA_intermediate.rds"))
# saveRDS(VDJ_GEX_matrix[["RBD"]], paste0(vgm_out_path, "proc0VDJ_GEX_RBD_intermediate.rds"))
# 




########################################################
# ADDITIONAL DATA PREP
#########################################################

##### Preprocess OVA file; 

VDJ_GEX_matrix[["OVA"]] <- data_prep(VDJ_GEX_matrix[["OVA"]], rename_group_id_OVA=FALSE, filter_1HCLC = FALSE)
VDJ_GEX_matrix[["OVA"]][["GEX"]] <- OVA_RBD_renaming(GEX = VDJ_GEX_matrix[["OVA"]][["GEX"]])
VDJ_mat_OVA <- VDJ_GEX_matrix[["OVA"]][["VDJ"]]


for (i in c(1:4)) {
  VDJ_mat_OVA$sample_id[VDJ_mat_OVA$sample_id == paste0("s", i)] <- paste0("s", i, "_OVA")
}
VDJ_mat_OVA$group_id[VDJ_mat_OVA$group_id == "1"] <- "OVA_pos"
VDJ_mat_OVA$group_id[VDJ_mat_OVA$group_id == "2"] <- "OVA_neg"

VDJ_GEX_matrix[["OVA"]][["VDJ"]] <- VDJ_mat_OVA



##### Preprocess file; 
VDJ_GEX_matrix[["RBD"]] <- data_prep(VDJ_GEX_matrix[["RBD"]], rename_group_id_OVA=FALSE, filter_1HCLC = FALSE)
VDJ_GEX_matrix[["RBD"]][["GEX"]] <- OVA_RBD_renaming(GEX = VDJ_GEX_matrix[["RBD"]][["GEX"]])
VDJ_mat_RBD <- VDJ_GEX_matrix[["RBD"]][["VDJ"]]

for (i in c(1:2)) {
  VDJ_mat_RBD$sample_id[VDJ_mat_RBD$sample_id == paste0("s", i)] <- paste0("s", i, "_RBD")
}


VDJ_GEX_matrix[["RBD"]][["VDJ"]] <- VDJ_mat_RBD


# 
# saveRDS(VDJ_GEX_matrix[["OVA"]], paste0(vgm_out_path, "proc1VDJ_GEX_OVA_intermediate.rds"))
# saveRDS(VDJ_GEX_matrix[["RBD"]], paste0(vgm_out_path, "proc1VDJ_GEX_RBD_intermediate.rds"))
# 
# 
#
# # VDJ_GEX_matrix <- list()
# VDJ_GEX_matrix[["RBD"]] <- readRDS(paste0(vgm_out_path, "proc1VDJ_GEX_RBD_intermediate.rds"))
# VDJ_GEX_matrix[["OVA"]] <- readRDS(paste0(vgm_out_path, "proc1VDJ_GEX_OVA_intermediate.rds"))



########################################################
# SAVE VDJ FILES
#########################################################

### Preprocess VDJ add VDJ and VJ sequences

for (dataset in c("OVA", "RBD")) {
  VDJ_mixcr <- VDJ_GEX_matrix[[dataset]][["VDJ"]]
  
  VDJ_mixcr$cdr_comb <- paste0(VDJ_mixcr$VDJ_aaSeqCDR3, "_", VDJ_mixcr$VJ_aaSeqCDR3)
  VDJ_mixcr$VDJ_aaSeq <- paste0(VDJ_mixcr$VDJ_aaSeqFR1, VDJ_mixcr$VDJ_aaSeqCDR1, VDJ_mixcr$VDJ_aaSeqFR2, VDJ_mixcr$VDJ_aaSeqCDR2, VDJ_mixcr$VDJ_aaSeqFR3, VDJ_mixcr$VDJ_aaSeqCDR3, VDJ_mixcr$VDJ_aaSeqFR4)
  VDJ_mixcr$VJ_aaSeq <- paste0(VDJ_mixcr$VJ_aaSeqFR1, VDJ_mixcr$VJ_aaSeqCDR1, VDJ_mixcr$VJ_aaSeqFR2, VDJ_mixcr$VJ_aaSeqCDR2, VDJ_mixcr$VJ_aaSeqFR3, VDJ_mixcr$VJ_aaSeqCDR3, VDJ_mixcr$VJ_aaSeqFR4)
  VDJ_mixcr$VDJ_VJ_aaSeq <- paste0(VDJ_mixcr$VDJ_aaSeq, "_", VDJ_mixcr$VJ_aaSeq)
  
  VDJ_mixcr$VDJ_nSeq <- paste0(VDJ_mixcr$VDJ_nSeqFR1, VDJ_mixcr$VDJ_nSeqCDR1, VDJ_mixcr$VDJ_nSeqFR2, VDJ_mixcr$VDJ_nSeqCDR2, VDJ_mixcr$VDJ_nSeqFR3, VDJ_mixcr$VDJ_nSeqCDR3, VDJ_mixcr$VDJ_nSeqFR4)
  VDJ_mixcr$VJ_nSeq <- paste0(VDJ_mixcr$VJ_nSeqFR1, VDJ_mixcr$VJ_nSeqCDR1, VDJ_mixcr$VJ_nSeqFR2, VDJ_mixcr$VJ_nSeqCDR2, VDJ_mixcr$VJ_nSeqFR3, VDJ_mixcr$VJ_nSeqCDR3, VDJ_mixcr$VJ_nSeqFR4)
  
  VDJ_mixcr$seq_complete <- NA
  # add column specifying if sequence is complete
  for (i in 1:nrow(VDJ_mixcr)) {
    if (all(VDJ_mixcr[i,c("VDJ_aaSeqFR1", "VDJ_aaSeqCDR1", "VDJ_aaSeqFR2", "VDJ_aaSeqCDR2", "VDJ_aaSeqFR3", "VDJ_aaSeqCDR3", "VDJ_aaSeqFR4", "VJ_aaSeqFR1", "VJ_aaSeqCDR1", "VJ_aaSeqFR2", "VJ_aaSeqCDR2", "VJ_aaSeqFR3", "VJ_aaSeqCDR3", "VJ_aaSeqFR4")] != "")) {
      VDJ_mixcr$seq_complete[i] <- TRUE}
    else {
      VDJ_mixcr$seq_complete[i] <- FALSE
      #print(paste0("cell ",  i,": not complete"))
    }
  }

  VDJ_GEX_matrix[[dataset]][["VDJ"]]<- VDJ_mixcr
}



### Save csv file
write.csv(VDJ_GEX_matrix[["OVA"]][["VDJ"]], paste0(vgm_out_path, "VDJ_data_OVA_mixcr.csv"))
write.csv(VDJ_GEX_matrix[["RBD"]][["VDJ"]], paste0(vgm_out_path, "VDJ_data_RBD_mixcr.csv"))



########################################################
# DATASET INTEGRATION
#########################################################


VDJ_GEX_matrix[["OVA"]][["GEX"]] <- subset(VDJ_GEX_matrix[["OVA"]][["GEX"]], subset = Nr_of_VDJ_chains < 2 & Nr_of_VJ_chains < 2 | is.na(Nr_of_VDJ_chains) | is.na(Nr_of_VJ_chains))
VDJ_GEX_matrix[["RBD"]][["GEX"]] <- subset(VDJ_GEX_matrix[["RBD"]][["GEX"]], subset = Nr_of_VDJ_chains < 2 & Nr_of_VJ_chains < 2 | is.na(Nr_of_VDJ_chains) | is.na(Nr_of_VJ_chains))



### Combine the VDJ_matrices for OVA and RBD
VDJ_mat <- rbind(VDJ_mat_OVA, VDJ_mat_RBD)


####### run harmony

# Combine to list
ifnb.list <- list(VDJ_GEX_matrix[["OVA"]][["GEX"]], VDJ_GEX_matrix[["RBD"]][["GEX"]] )

# select features that are repeatedly variable across datasets for integration
features <- SelectIntegrationFeatures(object.list = ifnb.list)

# Identify anchors and use these anchors to integrate the two datasets together
immune.anchors <- FindIntegrationAnchors(object.list = ifnb.list, anchor.features = features)

# create an 'integrated' data assay
immune_combined <- IntegrateData(anchorset = immune.anchors)

DefaultAssay(immune_combined) <- "integrated"

# # Run the standard workflow for visualization and clustering
immune_combined <- ScaleData(immune_combined, verbose = FALSE) 
immune_combined <- RunPCA(immune_combined, npcs = 50) 
immune_combined <- harmony::RunHarmony(immune_combined, group.by.vars="sample_id", project.dim = F) 
immune_combined <- RunUMAP(immune_combined, dims =  c(1:10), reduction = "harmony")
immune_combined <- FindNeighbors(immune_combined, dims = c(1:10), verbose = T,reduction = "harmony") 
immune_combined <- FindClusters(immune_combined, resolution = 0.5)# ,reduction = "harmony") 




########################################################
# Filter T cells
########################################################


###### OVA
DimPlot(VDJ_GEX_matrix[["OVA"]][["GEX"]], reduction = "umap", label = TRUE, pt.size = 0.7) + 
  FeaturePlot(VDJ_GEX_matrix[["OVA"]][["GEX"]], features=c("CD3E", "XBP1"))


GEX <- subset(VDJ_GEX_matrix[["OVA"]][["GEX"]], subset = seurat_clusters %in% c(0:9, 12) | is.na(seurat_clusters) )
VDJ_mat <- subset(VDJ_GEX_matrix[["OVA"]][["VDJ"]], subset= seurat_clusters %in% c(0:9,12) | is.na(seurat_clusters) )

GEX <- GEX %>%
  ScaleData() %>%
  RunPCA() %>%
  RunUMAP(dims = 1:30)

# p1 <- DimPlot(GEX, reduction = "umap", group.by = "group_id", pt.size = 0.7)
# p2 <- DimPlot(GEX, reduction = "umap", label = TRUE, repel = TRUE)
# p1+ p2

VDJ_GEX_matrix[["OVA"]][["GEX"]] <- GEX
VDJ_GEX_matrix[["OVA"]][["VDJ"]] <- VDJ_mat


###### RBD
DimPlot(VDJ_GEX_matrix[["RBD"]][["GEX"]], reduction = "umap", label = TRUE, pt.size = 0.7) + 
  FeaturePlot(VDJ_GEX_matrix[["RBD"]][["GEX"]], features=c("CD3E", "XBP1"))


GEX <- subset(VDJ_GEX_matrix[["RBD"]][["GEX"]], subset = seurat_clusters %in% c(0:6, 8,9,12))
VDJ_mat <- subset(VDJ_GEX_matrix[["RBD"]][["VDJ"]], subset= seurat_clusters %in% c(0:6, 8,9,12))

GEX <- GEX %>%
  ScaleData() %>%
  RunPCA() %>%
  RunUMAP(dims = 1:30)

# p1 <- DimPlot(GEX, reduction = "umap", group.by = "group_id", pt.size = 0.7)
# p2 <- DimPlot(GEX, reduction = "umap", label = TRUE, repel = TRUE)
# p1+ p2


VDJ_GEX_matrix[["RBD"]][["GEX"]] <- GEX
VDJ_GEX_matrix[["RBD"]][["VDJ"]] <- VDJ_mat



###### INTEGRATED

immune.combined_tex <- subset(immune_combined, subset = seurat_clusters %in% c(0:7,10,11))
# VDJ_mat <- subset(VDJ_mat, subset= seurat_clusters %in% c(0:10, 13, 14))

immune.combined_tex <- immune.combined_tex %>%
  ScaleData() %>%
  RunPCA() %>%
  RunUMAP(dims = 1:30)





####################################################################################
# Filter cells of overlapping clones 
####################################################################################

###### OVA

GEX <- VDJ_GEX_matrix[["OVA"]][["GEX"]]
VDJ_mat <- VDJ_GEX_matrix[["OVA"]][["VDJ"]]

# remove cells from overlapping clones (CDR3s)
bc_rm <- list()
for (i in list(list('s1_OVA', 's2_OVA'), list('s3_OVA', 's4_OVA'))) {
  cdrs_s1 <- unique(VDJ_mat$cdr_comb[VDJ_mat$sample_id == i[[1]]])
  cdrs_s2 <- unique(VDJ_mat$cdr_comb[VDJ_mat$sample_id == i[[2]]])
  cdrs_overlap <- intersect(cdrs_s1, cdrs_s2)
  # barcodes to remove
  bc_rm <- append(bc_rm, VDJ_mat$orig_barcode[VDJ_mat$cdr_comb %in% cdrs_overlap])
}
print(paste0('number of barcodes to remove: ', length(bc_rm)))

GEX <- subset(GEX, subset = orig_barcode %in% bc_rm, invert=TRUE)


GEX <- GEX %>% ScaleData() %>%
  RunPCA() %>%
  RunUMAP(dims = 1:30)


VDJ_GEX_matrix[["OVA"]][["GEX"]] <- GEX

###### RBD

GEX <- VDJ_GEX_matrix[["RBD"]][["GEX"]]
VDJ_mat <- VDJ_GEX_matrix[["RBD"]][["VDJ"]]

# remove cells from overlapping clones (CDR3s)
bc_rm <- list()
for (i in list(list('s1_RBD', 's2_RBD'))) {
  cdrs_s1 <- unique(VDJ_mat$cdr_comb[VDJ_mat$sample_id == i[[1]]])
  cdrs_s2 <- unique(VDJ_mat$cdr_comb[VDJ_mat$sample_id == i[[2]]])
  cdrs_overlap <- intersect(cdrs_s1, cdrs_s2)
  # barcodes to remove
  bc_rm <- append(bc_rm, VDJ_mat$orig_barcode[VDJ_mat$cdr_comb %in% cdrs_overlap])
}
print(paste0('number of barcodes to remove: ', length(bc_rm)))


GEX <- subset(GEX, subset = orig_barcode %in% bc_rm, invert=TRUE)


GEX <- GEX %>% ScaleData() %>%
  RunPCA() %>%
  RunUMAP(dims = 1:30)


VDJ_GEX_matrix[["RBD"]][["GEX"]] <- GEX




###### INTEGRATED


VDJ_mat <- rbind(VDJ_GEX_matrix[["OVA"]][["VDJ"]], VDJ_GEX_matrix[["RBD"]][["VDJ"]])
# VDJ_mat <- subset(VDJ_mat, subset = seurat_clusters %in% c(0:10, 13, 14))

# remove cells from overlapping clones (CDR3s)
bc_rm <- list()
for (i in list(list('s1_OVA', 's2_OVA'), list('s3_OVA', 's4_OVA'), list('s1_RBD', 's2_RBD'))) {
  cdrs_s1 <- unique(VDJ_mat$cdr_comb[VDJ_mat$sample_id == i[[1]]])
  cdrs_s2 <- unique(VDJ_mat$cdr_comb[VDJ_mat$sample_id == i[[2]]])
  cdrs_overlap <- intersect(cdrs_s1, cdrs_s2)
  # barcodes to remove
  bc_rm <- append(bc_rm, VDJ_mat$orig_barcode[VDJ_mat$cdr_comb %in% cdrs_overlap])
}
print(paste0('number of barcodes to remove: ', length(bc_rm)))


# ### Prepare dataset remove overlapping clones
GEX_integr <- subset(immune.combined_tex, subset = orig_barcode %in% bc_rm, invert=TRUE)
## remove from VDJ_mat
VDJ_mat <- subset(VDJ_mat, subset= !(orig_barcode %in% bc_rm))

DefaultAssay(GEX_integr) <- "integrated"
GEX_integr <- GEX_integr %>% ScaleData() %>%
  RunPCA() %>%
  RunUMAP(dims = 1:30)


VDJ_GEX_matrix[["INT"]][["GEX"]] <- GEX_integr








####################################################################################
# SAVE SCALED GEX DATAFRAMES  
####################################################################################


######### REPEAT WITH OVA
GEX_OVA <- VDJ_GEX_matrix[["OVA"]][["GEX"]]
saveRDS(VDJ_GEX_matrix[["OVA"]], paste0(vgm_out_path, "VDJ_GEX_object_OVA_Wuhan_harmony_overlapping_clones_rm.rds"))
write.csv(GEX_OVA@assays$RNA@scale.data, paste0(vgm_out_path, "scaled_GEX_OVA_harmony_overlappingclonesexcl.csv"),  row.names = TRUE, col.names = TRUE)


######### REPEAT WITH RBD
GEX_RBD <- VDJ_GEX_matrix[["RBD"]][["GEX"]]
saveRDS(VDJ_GEX_matrix[["RBD"]], paste0(vgm_out_path, "VDJ_GEX_object_RBD_Wuhan_harmony_overlapping_clones_rm.rds"))
write.csv(GEX_RBD@assays$RNA@scale.data, paste0(vgm_out_path, "scaled_GEX_RBD_harmony_overlappingclonesexcl.csv"), row.names = TRUE, col.names = TRUE)



######### SAVE INTEGRATED DATAFRAME
GEX_integr <- VDJ_GEX_matrix[["INT"]][["GEX"]]
saveRDS(VDJ_GEX_matrix[["INT"]], paste0(vgm_out_path, "VDJ_GEX_object_OVA_RBD_integrated_harmony_overlapping_clones_rm.rds"))
write.csv(GEX_integr@assays$integrated@scale.data, paste0(vgm_out_path, "scaled_GEX_OVA_RBD_int_harmony_overlappingclonesexcl.csv"), row.names = TRUE, col.names = TRUE)










