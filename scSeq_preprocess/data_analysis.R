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
# cellranger_out_path <- paste0("/Users/lerlach/Documents/current_work/OVA_seq/aligned_data/OVA_specific/")
vgm_out_path <- paste0(getwd() , "/Analysis_test/")


# define path to save plots to 
save_path <- paste0(vgm_out_path, "Plots/")
ifelse(!dir.exists(file.path(save_path)), dir.create(file.path(save_path), recursive = TRUE), FALSE)



  
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
library(viridis)
library(ggseqlogo)
library(ggvenn)
library(stringdist)
library(wesanderson)
library(Platypus)




########################################################
# Load file - START OF ANALYSIS
########################################################

VDJ_GEX_matrix <- readRDS(paste0(vgm_out_path, "data/VDJ_GEX_object_OVA_RBD_integrated_harmony_overlapping_clones_rm.rds"))
  

# GEX_OVA <- readRDS(paste0(vgm_out_path, "VDJ_GEX_object_OVA_harmony_overlapping_clones_rm.rds"))[["GEX"]]
# GEX_RBD <- readRDS(paste0(vgm_out_path, "VDJ_GEX_object_RBD_harmony_overlapping_clones_rm.rds"))[["GEX"]]
# GEX_INT <- readRDS(paste0(vgm_out_path, "VDJ_GEX_object_RBD_harmony_overlapping_clones_rm.rds"))[["GEX"]]



# just rename GEX dataset & VDJ dataset
pbmc <- VDJ_GEX_matrix[[2]]
VDJ_mat <- VDJ_GEX_matrix[[1]]

DimPlot(pbmc, group.by = 'seurat_clusters') 




########################################################
# GEX ANALYSIS                                                    
########################################################

########################################################
# UMAP Plots - Highlight (non)specific cells
########################################################

# UMAPs with just showing sample 1&3
Idents(pbmc) <- "sample_id"
s1_high <- WhichCells(pbmc, ident = c("s1_OVA", "s3_OVA"))
s1R_high <- WhichCells(pbmc, ident = c("s1_RBD"))

# PCA Plot
plot1 <- DimPlot(pbmc, reduction = "pca", cells.highlight= list(s1_high,s1R_high), pt.size = 1) +
  scale_color_manual(labels = c("Others", "OVA_spec", "RBD_sepc"), values = c("grey", "darkblue", "darkred"))
plot1

# PCA Plot
plot1 <- DimPlot(pbmc, reduction = "harmony", cells.highlight= list(s1_high,s1R_high), pt.size = 1) +
  scale_color_manual(labels = c("Others", "OVA_spec", "RBD_sepc"), values = c("grey", "darkblue", "darkred"))
plot1

# UMAP Plot
plot1 <- DimPlot(pbmc, reduction = "umap", cells.highlight= list(s1_high,s1R_high), pt.size = 1) +
  scale_color_manual(labels = c("Others", "OVA_spec", "RBD_sepc"), values = c("grey", "darkblue", "darkred"))
plot1

ggsave(filename=paste0(save_path, "UMAP_OVA_RBD_spec.pdf"),plot=plot1)


# UMAPs with just showing sample 1&3
Idents(pbmc) <- "sample_id"
s1_high <- WhichCells(pbmc, ident = c("s1_OVA", "s3_OVA"))
s1R_high <- WhichCells(pbmc, ident = c("s1_RBD"))
# UMAP Plot
plot2 <- DimPlot(pbmc, reduction = "umap", cells.highlight= list(s1_high,s1R_high), pt.size = 1) +
  scale_color_manual(labels = c("Others", "OVA_spec", "RBD_sepc"), values = c("grey", "darkblue", "darkred"))
plot1 + plot2

FeaturePlot(pbmc, features = c("XBP1")) + FeaturePlot(pbmc, features = c("rna_CD3E"))

######################################################### 
# Start of VDJ analysis 
#########################################################

########################################################
#  Platypus: Clonal expansion donuts                      
########################################################

VDJ_mat_donut <- VDJ_clonal_donut(VDJ = VDJ_mat,
                             #counts.to.use='clonotype_frequency',
                             label.size=12,
                             # not.expanded.label.vjust,
                             #not.expanded.label.hjust = 3,
                             total.label.vjust=2.5,
                             # total.label.hjust,
                             expanded.colors=c('gray36', 'gray51', 'grey69', 'grey87'))
                             # non.expanded.color)

VDJ_mat_donut[[1]]
VDJ_mat_donut[[2]]  

for (i in 1:length(sample_names)) {
  ggsave(filename=paste0(save_path, "Expansion_donuts_s", i, ".pdf"), plot=VDJ_mat_donut[[i]], width = 10)
}






