# Evaluating predictive patterns of antigen-specific B cells by single- cell transcriptome and antibody repertoire sequencing
This repository contains code to perform the analysis to L. Erlach, et al. Evaluating predictive patterns of antigen-specific B cells by single- cell transcriptome and antibody repertoire sequencing.

## Abstract
The field of antibody discovery typically involves extensive experimental screening of B cells from immunized animals. Machine learning (ML)-guided prediction of antigen-specific B cells could accelerate this process, but requires sufficient training data with antigen-specificity labeling. Here, we introduce a dataset of single-cell transcriptome and antibody repertoire sequencing of B cells from immunized mice, which are labeled as antigen-specific or non-specific through experimental selections. We identify gene expression patterns associated with antigen-specificity by differential gene expression analysis and assess their antibody sequence diversity. Subsequently, we benchmark various ML models, both linear and non-linear, trained on different combinations of gene expression and antibody repertoire features. Additionally, we assess transfer learning using features from general and antibody-specific protein language models (PLMs). Our findings show that gene expression-based models outperform sequence-based models for antigen-specificity predictions, highlighting a promising avenue for computational-guided antibody discovery.

## Table of contents
1. [Working Environment](#working-environment)
2. [Datasets](#datasets)
3. [Feature generation](#feature-generation)
4. [Model evaluations](#model-evaluations)
5. [Visualization](#visualization)
6. [Citing This Work](#citing-this-work)

## Working environment
#### Setup with Conda

```console
conda env create -f environment_scabpred.yml
conda activate abmap
```

## Datasets
The raw sequencing data is deposited in SRA under the BioProject number: PRJNA1124428. 
1. The sequencing files were processed with cellranger (v5.0.0) and the scripts for the alignment of the files is in `scSeq_preprocess/1_Cellranger_alignment_GEX.sh` and `scSeq_preprocess/2_Cellranger_alignment_VDJ.sh`
2. Preprocessed single cell sequencing data was further processed in R, mainly utilizing the Playtpus and Seurat packages. The analysis, including the differential expression analysis is in `scSeq_preprocess/`
3. Preprocessing of the datasets for ML model evaluations is in `notebooks/ML_preprocess/`


## Feature generation
Features for the ML model evaluations were generated in the Jupyter notebooks in `notebooks/ML_preprocess/`
1. Gene expression data was processed in `003_GEX_dataprep.ipynb`
2. Antibody sequencing data was processed in `001_VDJ_OVA_seq_preprocessing.ipynb` and `001.2_VDJ_RBD_seq_preprocessing.ipynb`
3. The PLM embeddings were generated with the notebooks in `notebooks/ESM_embed/001_Generate_ESM_embeddings.ipynb`, `notebooks/ESM_embed/002_Extract_CDR3Embeddings.ipynb` and `notebooks/Antiberty_embed/Embed_seqs_antiberty.ipynb`


## Model evaluations
Scripts for training and evaluating the ML models are in `src/Spec_classification/Specificity_classification_script.py` which can be executed as shown in the example below.
```console
./Specificity_classification_script.py --config path_to_config --simsplit_thresh 0.05 --chaintype VH_VL --outpath path_to_save_results
```

## Visualization
Visualization and summarization of the results of the model evaluation are performed with the jupyter notebooks `notebooks/model_evaluations/Metrics_visualization.ipynb` and `notebooks/model_interpretation/LogReg_Koefficient_analysis.ipynb`. The latter also contains the biological analysis of the LogReg models. 