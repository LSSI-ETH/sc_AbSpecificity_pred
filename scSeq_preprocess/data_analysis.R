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
save_path_GEX <- paste0(vgm_out_path, "GEX/Plots/")
save_path_VDJ <- paste0(vgm_out_path, "VDJ/Plots/")
save_path_mod <- paste0(vgm_out_path, "Modelling/Plots/")

ifelse(!dir.exists(file.path(save_path_GEX)), dir.create(file.path(save_path_GEX), recursive = TRUE), FALSE)
ifelse(!dir.exists(file.path(save_path_VDJ)), dir.create(file.path(save_path_VDJ), recursive = TRUE), FALSE)
ifelse(!dir.exists(file.path(save_path_mod)), dir.create(file.path(save_path_mod), recursive = TRUE), FALSE)

  
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

VDJ_GEX_matrix <- readRDS("/Users/lerlach/Documents/current_work/OVA_seq/OVA_RBD_Wuhan_integrated/Analysis/Modelling/data/VDJ_GEX_object_OVA_RBD_harmony_overlapping_clones_rm.rds")
 


# just rename GEX dataset & VDJ dataset
pbmc <- VDJ_GEX_matrix[[1]]
VDJ_mat <- VDJ_GEX_matrix[[2]]
 




# ###########################################################################################################################################################################
# # Additional clonotyping
# #########################################################
# 
# 
# ### Create clonotyping table (def. of clones: ident. CDRH3_CDRL3)
# clonotype_ls <- clonotype_tbl(VDJ_mat, num_clone = "all", sample_names=sample_names)
# VDJ_tbl_all <- clonotype_ls[[1]]
# VDJ_VJ_tbl_all <- clonotype_ls[[2]]
# # write_csv(VDJ_tbl, paste0(save_path_VDJ, "clone_table_all_clones_VDJ.csv"))
# # write_csv(VDJ_VJ_tbl, paste0(save_path_VDJ, "clone_table_all_clones_VDJVJ.csv"))
# 
# 
# ### Create clonotyping table (def. of clones: ident. CDRH3_CDRL3)
# clonotype_ls <- clonotype_tbl(VDJ_mat, num_clones, sample_names)
# VDJ_tbl <- clonotype_ls[[1]]
# VDJ_VJ_tbl <- clonotype_ls[[2]]
# # write_csv(VDJ_tbl, paste0(save_path_VDJ, "clone_table_top", num_clones, "clones_VDJ.csv"))
# # write_csv(VDJ_VJ_tbl, paste0(save_path_VDJ, "clone_table_top", num_clones, "clones_VDJVJ.csv"))
# 
# ## clonotyping according to 10x definition
# clonotype_ls <- clonotype_tbl(VDJ_mat, num_clones, sample_names, clone_def = "clonotype_id_10x")
# VDJ_tbl_10x <- clonotype_ls[[1]]
# VDJ_VJ_tbl_10x <- clonotype_ls[[2]]
# # write_csv(VDJ_tbl_10x, paste0(save_path_VDJ, "clone_table_top", num_clones, "clones_VDJ_10x.csv"))
# # write_csv(VDJ_VJ_tbl_10x, paste0(save_path_VDJ, "clone_table_top", num_clones, "clones_VDJVJ_10x.csv"))
# 
# 
# 


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



plot1 <- DimPlot(pbmc, reduction = "umap", cells.highlight= list(s1_high,s1R_high), pt.size = 1) +
  scale_color_manual(labels = c("Others", "OVA_spec", "RBD_sepc"), values = c("grey", "darkblue", "darkred"))
plot1
ggsave(filename=paste0(save_path_GEX, "UMAP_OVA_RBD_spec.pdf"),plot=plot1)





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

for (i in 1:length(sample_names)) {
  ggsave(filename=paste0(save_path_VDJ, "Expansion_donuts_s", i, ".pdf"), plot=VDJ_mat_donut[[i]], width = 10)
}






