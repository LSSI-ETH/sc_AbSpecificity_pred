############ 
# R analysis of "Evaluating predictive patterns of antigen-specific B cells by single-cell transcriptome and antibody repertoire sequencing.
# Author: Lena Erlach 
# 17th June 2024

# edit 12.12.2022 : VGM was run additionally with harmony



setwd('/Users/lerlach/Documents/current_work/Publications/2024_single-cell_Ab_specificity_predictions/CellSystems_sc_Ab_specificity/Rcode/')
source("call_MIXCR_fct.R")

source("utils.R")
# Set working directory

# set path to MiXCR
mixcr.directory <- "/usr/local/Cellar/mixcr/4.2.0-1/mixcr"

# cellranger output files
cellranger_out_path <- paste0("/Users/lerlach/Documents/current_work/OVA_seq/aligned_data/OVA_specific/")
vgm_out_path <-  paste0(getwd(), "/Analysis_test/")

ifelse(!dir.exists(file.path(vgm_out_path)), dir.create(file.path(vgm_out_path), recursive = TRUE), FALSE)


# Setups
species <- "mmu"
set.seed(123)



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


################################################################################
# PLATYPUS: CELLRANGER OUTPUT PROCESSING 
################################################################################



VDJ.out.directory.list <- list()
VDJ.out.directory.list[[1]] <- c(paste0(cellranger_out_path, "vdj/S9/"))
VDJ.out.directory.list[[2]] <- c(paste0(cellranger_out_path, "vdj/S10/"))
# VDJ.out.directory.list[[3]] <- c(paste0(cellranger_out_path, "vdj/S11/"))     
# VDJ.out.directory.list[[4]] <- c(paste0(cellranger_out_path, "vdj/S12/"))

GEX.out.directory.list <- list()
GEX.out.directory.list[[1]] <- c(paste0(cellranger_out_path, "gex/S1/"))
GEX.out.directory.list[[2]] <- c(paste0(cellranger_out_path, "gex/S2/"))
# GEX.out.directory.list[[3]] <- c(paste0(cellranger_out_path, "gex/S3/")) 
# GEX.out.directory.list[[4]] <- c(paste0(cellranger_out_path, "gex/S4/")) 

# settings for VGM
group_id <- c(1,2)

VDJ_GEX_matrix <- VDJ_GEX_matrix(VDJ.out.directory.list = VDJ.out.directory.list,
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

# save the VDJ stats file
write.csv(VDJ_GEX_matrix$VDJ.GEX.stats, paste0(vgm_out_path, "VDJ_GEX_stats.csv"))





########################################################
# CALL MIXCR 
# ########################################################

VDJ_mat_mixcr <- VDJ_call_MIXCR(VDJ = VDJ_GEX_matrix[[1]],
                                mixcr.directory = mixcr.directory,
                                species = species, platypus.version = "v3", simplify = TRUE)

VDJ_GEX_matrix[['VDJ']] <- VDJ_mat_mixcr




########################################################
# ADDITIONAL DATA PREP
#########################################################

## Preprocess file; 
Idents(VDJ_GEX_matrix[["GEX"]]) <- "group_id"
VDJ_GEX_matrix[["GEX"]] <- RenameIdents(VDJ_GEX_matrix[["GEX"]], "2" = "nonspecific")
VDJ_GEX_matrix[["GEX"]] <- RenameIdents(VDJ_GEX_matrix[["GEX"]], "1" = "specific")
VDJ_GEX_matrix[["GEX"]]$group_id <- Idents(VDJ_GEX_matrix[["GEX"]])
Idents(VDJ_GEX_matrix[["GEX"]]) <- "seurat_clusters"
  



# set variables
sample_names <- unique(VDJ_GEX_matrix$sample_id)
num_clones <- 30 # number of clones to include

# data prep
VDJ_GEX_matrix <- data_prep(VDJ_GEX_matrix)



# save RDS file
saveRDS(VDJ_GEX_matrix, paste0(vgm_out_path, "VDJ_GEX_OVAspec_harmony_mixcraligned.rds"))
write.csv(VDJ_GEX_matrix[["VDJ"]], paste0(vgm_out_path, "VDJ_data_OVA_mixcr.csv"))

