#################################
# 
# Class for loading the embeddings in to use in the classification models to predict 
# Binding Specificity for OVA values from sequence embeddings
# This class is used in the classification pipeline to train and evaluate ML models. 
# Author: Lena Erlach
# Date: 09 Nov 2020
#
################################


import numpy as np
import pandas as pd
import os, sys
from datetime import datetime, date 
import random
import configparser


# add root directory to path such that the utils_nb file can be imported
UTILS_DIR = './sc_AbSpecificity_pred/src/'
sys.path.append(UTILS_DIR)
sys.path.append(os.path.join(UTILS_DIR, 'AbMAP_analysis'))

import utils_nb as utils




class LoadEmbeddings_VH_VL:

    def __init__(self, CONFIG_PATH, seq_col='VDJ_VJ_aaSeq', seq_cols_load: list = ['seq_id', 'VDJ_aaSeq', 'VJ_aaSeq', 'VDJ_VJ_aaSeq', 'group_id', 'sample_id', 'seq_complete'], 
                 filter_192 = True, filter_VH_complete = False):
        # setup parser for the config file
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        self.ROOT_DIR = config['ROOT']['ROOT_DIR']

        # Set input path Sequences 
        self.seq_inputPath = os.path.join(self.ROOT_DIR, config['PATHS']['SEQ_DF'])

        assert seq_col in ['VDJ_VJ_aaSeq', 'VDJ_aaSeq'], 'seq_col must be either "VDJ_VJ_aaSeq" or "VDJ_aaSeq"'


        if seq_col == 'VDJ_VJ_aaSeq': 
            config_dir = 'VH_VL_EMBEDPATH'

        elif seq_col == 'VDJ_aaSeq':
            config_dir = 'VH_EMBEDPATH'



        # Set input path - ESM-2 embeddings
        self.emb_inputPath_ESM = os.path.join(self.ROOT_DIR, config[config_dir]['ESM2_var'])

        # CDR extracted embeddings
        self.emb_inputPath_ESM_cdrs = os.path.join(self.ROOT_DIR, config[config_dir]['ESM2_CDRextract'])

        # Antiberty embeddings
        self.emb_inputPath_antiberty = os.path.join(self.ROOT_DIR, config[config_dir]['ANTIBERTY'])

        # distance output path 
        dist_matrix_inputPath = os.path.join(self.ROOT_DIR, config[config_dir]['DISTANCE_MATRIX'])

        # Seq column name 'VDJ_aaSeq', 'VDJ_VJ_aaSeq'...
        self.seq_col = seq_col




        ########## LOAD DATASETS ##########
        # Load the datasets
        self.seq_df = pd.read_csv(self.seq_inputPath, usecols=seq_cols_load)
        
        # delete one extra row for VH_VL sequences
        if seq_col == 'VDJ_VJ_aaSeq' or filter_VH_complete is True:
            self.seq_df = self.seq_df[self.seq_df.seq_complete == True]
            

        if filter_192 is True:
            self.seq_df.drop(192, inplace=True)
            

        self.seq_df.reset_index(drop=True, inplace=True)
        self.names = self.seq_df.seq_id.tolist()
        self.seqs = self.seq_df[seq_col]

        # Load distance matrix
        self.dist_matrix = np.loadtxt(dist_matrix_inputPath, delimiter=',')


    
    ########## LOAD EMBEDDINGS FUNCTION ##########
    def load_embeddings(self, embedding_type = 'all', VH_emb_fname_suff = '', verbose=True):
        """
        Load the embeddings for the sequences in the dataset in the class
        """

        if self.seq_col == 'VDJ_VJ_aaSeq': 
            if verbose is True: print("Load embeddings for VH_VL sequences")
            
            # iterate through all embeddings and load them            
            if embedding_type == 'all' or embedding_type == 'esm':
                ### ESM2 - VH_VL
                self.emb_ESM_var = utils.load_pickle_embeddings_VH_VL(self.names, self.emb_inputPath_ESM, embedding_type = 'var')
                self.emb_ESM = utils.mean_over_HL(self.emb_ESM_var) # average embeddings over across seq_len & VH+VL
                if verbose is True: print("ESM - VH_VL embeddings loaded")

            if embedding_type == 'all' or embedding_type == 'esm_cdrs':
                ### ESM2 CDR - VH_VL
                embeddings = utils.load_pickle_embeddings(self.names, self.emb_inputPath_ESM_cdrs)
                self.emb_ESM_cdrs = np.array(embeddings)
                if verbose is True: print("ESM CDRextract - embeddings VH_VL loaded")            
            
            if embedding_type == 'all' or embedding_type == 'antiberty':
                ### Antiberty - VH_VL
                self.emb_antiberty_var = utils.load_pickle_embeddings_VH_VL(self.names, self.emb_inputPath_antiberty,  file_suffix = '', embedding_type = 'var')
                self.emb_antiberty = utils.mean_over_HL(self.emb_antiberty_var)
                if verbose is True: print("Antiberty - VH_VL embeddings loaded")


        elif self.seq_col == 'VDJ_aaSeq':
            if verbose is True: print("Load embeddings for VH sequences")

            # adjust names 
            self.names = [f'{n}{VH_emb_fname_suff}' for n in self.names]

            # iterate through all embeddings and load them
            if embedding_type == 'all' or embedding_type == 'esm':
                ### ESM2 - VH 
                self.emb_ESM_var = utils.load_pickle_embeddings(self.names, self.emb_inputPath_ESM)
                self.emb_ESM = np.array([emb.mean(0) for emb in self.emb_ESM_var]) #  average over the sequence length
                if verbose is True: print("ESM - VH embeddings loaded")

            if embedding_type == 'all' or embedding_type == 'esm_cdrs':
                ### ESM2 CDR - VH
                embeddings = utils.load_pickle_embeddings(self.names, self.emb_inputPath_ESM_cdrs)
                self.emb_ESM_cdrs = np.array(embeddings)
                if verbose is True: print("ESM CDRextract - VH embeddings loaded")            
            
            if embedding_type == 'all' or embedding_type == 'antiberty':
                ### Antiberty - VH
                self.emb_antiberty_var = utils.load_pickle_embeddings(self.names, self.emb_inputPath_antiberty, file_suffix = '_H')
                self.emb_antiberty = np.array([emb.mean(0) for emb in self.emb_antiberty_var])

            # reset the names
            self.names = self.seq_df.seq_id.tolist()


        





if __name__ == '__main__':
    pass


    # def test():
        
    #     # test the class
    #     CONFIG_PATH = './src/config_file.txt'
    #     HL_class = LoadEmbeddings_VH_VL(CONFIG_PATH, seq_col='VDJ_aaSeq', filter_192 = True, filter_VH_complete = True)
    #     # print(HL_class.names[120:128])
    #     # HL_class.load_embeddings(embedding_type = 'antiberty')
    #     # print(HL_class.emb_antiberty.shape)
    #     # print()
    #     # print(HL_class.emb_ESM.shape)
    #     # print(HL_class.emb_ESM_cdrs.shape)


    #     H_class = LoadEmbeddings_VH_VL(CONFIG_PATH, seq_col='VDJ_aaSeq', filter_192 = True, filter_VH_complete = True) 
    #     H_class.load_embeddings(embedding_type = 'antiberty')   
    #     # print(H_class.emb_AM.shape)
    #     # print(H_class.emb_ESM.shape)
    #     # print(H_class.emb_ESM_cdrs.shape)
    #     print(H_class.emb_antiberty.shape)