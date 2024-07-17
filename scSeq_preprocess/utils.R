###########
# Utils containing all the functions necessary for the OVA_seq_analysis code
#
# Lena Erlach
# 9.3.2022
##########


########################################################
# Functions for the analysis 
########################################################

### Custom function to prepare dataset for further analysis and modelling

# data preparation function
data_prep <- function(VDJ_GEX_matrix, new.cluster.ids=NULL, filter_1HCLC = FALSE, rename_group_id_OVA=FALSE) {
  
  # just rename GEX dataset & VDJ dataset
  if (length(VDJ_GEX_matrix) > 1) {
    
    pbmc <- VDJ_GEX_matrix[["GEX"]]
    
  } else {
    pbmc <- VDJ_GEX_matrix
  }
  
  
  if (filter_1HCLC == TRUE) {
    VDJ_mat <- subset(VDJ_GEX_matrix[[1]], Nr_of_VDJ_chains == 1 & Nr_of_VJ_chains == 1)
  } else {VDJ_mat <- VDJ_GEX_matrix[[1]]}
  
  if (rename_group_id_OVA == TRUE) {
    Idents(pbmc) <- "group_id"
    pbmc <- RenameIdents(pbmc, "2" = "nonspecific")
    pbmc <- RenameIdents(pbmc, "1" = "specific")
    pbmc$group_id <- Idents(pbmc)
    Idents(pbmc) <- "seurat_clusters"
  }
  
  # assign preliminary clusterIDs
  if (!is.null(new.cluster.ids) & typeof(new.cluster.ids) == "character") {
    print("rename")
    # new.cluster.ids <- c("0 GC/LZ TFH help", "1 proliferative/DZ (enriched)", "2 DZ", "3 naive-like/PreBmem", "4 proliferative GC/DZ (enriched)", "5 GC/DZ", "6 GC/LZ (enriched)", "7 Bmem",
    #                      "8 naive/activated", "9 T cells", "10 PB (ASC-like)")
    names(new.cluster.ids) <- levels(pbmc)
    pbmc <- RenameIdents(pbmc, new.cluster.ids)
    pbmc[["cell_assigned"]] <- Idents(object = pbmc)
    Idents(pbmc) <- "seurat_clusters"
    pbmc$cell_assigned <- as.character(pbmc$cell_assigned)
  } 
  
  ### Preprocess VDJ add VDJ and VJ sequences from MIXCr
  VDJ_mat$cdr_comb <- paste0(VDJ_mat$VDJ_aaSeqCDR3, "_", VDJ_mat$VJ_aaSeqCDR3)
  VDJ_mat$VDJ_aaSeq <- paste0(VDJ_mat$VDJ_aaSeqFR1, VDJ_mat$VDJ_aaSeqCDR1, VDJ_mat$VDJ_aaSeqFR2, VDJ_mat$VDJ_aaSeqCDR2, VDJ_mat$VDJ_aaSeqFR3, VDJ_mat$VDJ_aaSeqCDR3, VDJ_mat$VDJ_aaSeqFR4)
  VDJ_mat$VJ_aaSeq <- paste0(VDJ_mat$VJ_aaSeqFR1, VDJ_mat$VJ_aaSeqCDR1, VDJ_mat$VJ_aaSeqFR2, VDJ_mat$VJ_aaSeqCDR2, VDJ_mat$VJ_aaSeqFR3, VDJ_mat$VJ_aaSeqCDR3, VDJ_mat$VJ_aaSeqFR4)
  VDJ_mat$VDJ_VJ_aaSeq <- paste0(VDJ_mat$VDJ_aaSeq, "_", VDJ_mat$VJ_aaSeq)
  
  VDJ_mat$VDJ_nSeq <- paste0(VDJ_mat$VDJ_nSeqFR1, VDJ_mat$VDJ_nSeqCDR1, VDJ_mat$VDJ_nSeqFR2, VDJ_mat$VDJ_nSeqCDR2, VDJ_mat$VDJ_nSeqFR3, VDJ_mat$VDJ_nSeqCDR3, VDJ_mat$VDJ_nSeqFR4)
  VDJ_mat$VJ_nSeq <- paste0(VDJ_mat$VJ_nSeqFR1, VDJ_mat$VJ_nSeqCDR1, VDJ_mat$VJ_nSeqFR2, VDJ_mat$VJ_nSeqCDR2, VDJ_mat$VJ_nSeqFR3, VDJ_mat$VJ_nSeqCDR3, VDJ_mat$VJ_nSeqFR4)
  
  VDJ_mat$seq_complete <- NA
  # add column specifying if sequence is complete
  for (i in 1:nrow(VDJ_mat)) {
    if (all(VDJ_mat[i,c("VDJ_aaSeqFR1", "VDJ_aaSeqCDR1", "VDJ_aaSeqFR2", "VDJ_aaSeqCDR2", "VDJ_aaSeqFR3", "VDJ_aaSeqCDR3", "VDJ_aaSeqFR4", "VJ_aaSeqFR1", "VJ_aaSeqCDR1", "VJ_aaSeqFR2", "VJ_aaSeqCDR2", "VJ_aaSeqFR3", "VJ_aaSeqCDR3", "VJ_aaSeqFR4")] != "")) {
      VDJ_mat$seq_complete[i] <- TRUE}
    else {
      VDJ_mat$seq_complete[i] <- FALSE
      #print(paste0("cell ",  i,": not complete"))
    }
  }
  
  # add specificity column
  pbmc@meta.data$ag_specificity <- NA
  pbmc@meta.data$ag_specificity[pbmc@meta.data$sample_id %in% c("s1", "s3")] <- "specific"
  pbmc@meta.data$ag_specificity[pbmc@meta.data$sample_id %in% c("s2", "s4")] <- "nonspecific"
  
  
  ret_ls <- list(pbmc, VDJ_mat)
  names(ret_ls) <- c("GEX", "VDJ")
  return(ret_ls)
}





### function to harmonize and rename the group_id and sample_id in the OVA and RBD GEX dataset
OVA_RBD_renaming <- function(GEX) {
  # rename columns
  GEX@meta.data$sample_id[GEX@meta.data$sample_id == "s1" & GEX@meta.data$group_id == "1"] <- "s1_OVA"
  GEX@meta.data$sample_id[GEX@meta.data$sample_id == "s2" & GEX@meta.data$group_id == "2"] <- "s2_OVA"
  GEX@meta.data$sample_id[GEX@meta.data$sample_id == "s3" & GEX@meta.data$group_id == "1"] <- "s3_OVA"
  GEX@meta.data$sample_id[GEX@meta.data$sample_id == "s4" & GEX@meta.data$group_id == "2"] <- "s4_OVA"
  GEX@meta.data$sample_id[GEX@meta.data$sample_id == "s1" & GEX@meta.data$group_id == "LN_Wuhan_pos"] <- "s1_RBD"
  GEX@meta.data$sample_id[GEX@meta.data$sample_id == "s2" & GEX@meta.data$group_id == "LN_Wuhan_neg"] <- "s2_RBD"
  # rename group_id
  GEX@meta.data$group_id[GEX@meta.data$group_id == "1"] <- "OVA_pos"
  GEX@meta.data$group_id[GEX@meta.data$group_id == "2"] <- "OVA_neg"
  # GEX@meta.data$group_id[GEX@meta.data$group_id == "LN_Wuhan_pos"] <- "RBD_pos"
  # GEX@meta.data$group_id[GEX@meta.data$group_id == "LN_Wuhan_neg"] <- "RBD_neg"
  
  # add specificity column
  GEX@meta.data$ag_specificity <- NA
  GEX@meta.data$ag_specificity[GEX@meta.data$group_id %in% c("OVA_pos", "LN_Wuhan_pos")] <- "specific"
  GEX@meta.data$ag_specificity[GEX@meta.data$group_id %in% c("OVA_neg", "LN_Wuhan_neg")] <- "nonspecific"
  return(GEX)
}



