#!/usr/bin/env python3
#################################
# Script for classification models to predict Binding Specificity for RBD values from sequence embeddings
# This class is used in the classification pipeline to train and evaluate ML models. 
# Author: Lena Erlach
# Date: 16 Jan 2024
################################

import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
import random
import configparser, argparse



# add parser
parser = argparse.ArgumentParser(description='Run classification models to predict Binding Specificity values from gene expression')
parser.add_argument('--config', type=str, default='config_file.txt', help='Path to the config file')
parser.add_argument('--simsplit_thresh', type=float, default=0.05, help='Similarity split threshold; default 0.05')
parser.add_argument('--outpath', type=str, default='data/model_evaluation/Specificity_classification/', help='Output path for results')
args = parser.parse_args()

####################################

testing = False

####################################

# add root directory to path such that the utils_nb file can be imported
if testing == True:
    CONFIG_PATH = '/data/cb/scratch/lenae/sc_AbSpecificity_pred/src/Spec_classification/GEX_BCR/config_file_GEX_BCR.txt'
else: 
    CONFIG_PATH = args.config 


# import custom modules
UTILS_DIR = '../'
UTILS_DIR1 = '../../'
# add directory to path such that the utils_nb file can be imported
sys.path.append(UTILS_DIR)
sys.path.append(UTILS_DIR1)

import utils_nb as utils
import Specificity_classification_class as CLF
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA





######## MAIN ########


def run():

    # setup parser for the config file
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    ROOT_DIR = config['ROOT']['ROOT_DIR']

    
    ab_chain = 'VDJ_VJ_aaSeq'

    if ab_chain == 'VDJ_aaSeq': 
        c_type = 'VH'
    elif ab_chain == 'VDJ_VJ_aaSeq':
         c_type = 'VH_VL'

    # set random seed
    random.seed(123)
    today = str(datetime.now().date())
    log = utils.Logger(__name__, log_file=f'BCR_GEX_{today}_{c_type}_app.log').get_logger()
    log.info('Start Script!')

    # setup parser for the config file
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    ROOT_DIR = config['ROOT']['ROOT_DIR']

    # setup output path 
    if testing == True:
        out_path = os.path.join('./', 'GEX')
    else:
        out_path = os.path.join(args.outpath, 'GEX')


    if not os.path.exists(out_path):
        os.makedirs(out_path)


    ######## LOOP THROUGH DATASETS ########
    if testing == True:
        d_list = ['OVA']
    else:
        d_list = ['OVA', 'RBD', 'INTEGRATED']


    for dataset in d_list:

        log.info(f'Start evaluating models - {dataset}')

        ######## LOADING DATASET ########
        try:
            # dataset = 'INTEGRATED'

            feat_inputPath = os.path.join(ROOT_DIR, config['BCR_GEX'][dataset])

            # Load dataframe
            feature_df = pd.read_csv(feat_inputPath)
            

            # Calculate distance matrix between all sequences
            distance_matrix = utils.calc_norm_levens_dist(feature_df[ab_chain], verbose=0)


            #### Prepare datasets
            GEX = feature_df.iloc[:, -2000:].values
            kmer = feature_df.iloc[: , 6:-2000].values
            GEX_kmer = feature_df.iloc[: , 6:].values
            y = feature_df.loc[:, 'group_id'].values

            if np.unique(y)[0] == 1 and np.unique(y)[1] == 2: 
                y = np.where(y == 2, 0, 1)
                
            


        except Exception as e:
            log.exception(f'ERROR Loading files: {e}')



        ######## TRAIN TEST SPLITS ########
        try:
            # create train test splits - sequence clustering
            N_SPLITS=5
            if testing == True:
                SIM_SPLIT = 0.05
            else: 
                SIM_SPLIT = args.simsplit_thresh 
                

            # best splits are created with N_SPLITS=6 (based on manual inspection of train test splits)
            train_ls, test_ls, _ = CLF.train_test_split_idx(kmer, y, cluster_thresh=SIM_SPLIT, distance_matrix=distance_matrix, 
                                                            n_splits=N_SPLITS, cluster_method= 'levenshtein_sequence_based',
                                                            verbose=0)

            print(f'Sequence-based clustering with {SIM_SPLIT} cluster threshold')
            for i in range(len(test_ls)):
                print(len(test_ls[i]))
                print(np.unique(y[test_ls[i]], return_counts=True)[1])


            # create train test splits - Random split
            train_ls_rd, test_ls_rd, _ = CLF.train_test_split_idx(kmer, y,# cluster_thresh=SIM_SPLIT, 
                                                                n_splits=N_SPLITS, cluster_method= 'random_split', verbose=0)
            print(f'Random split')
            for i in range(len(test_ls_rd)):
                print(len(test_ls_rd[i]))
                print(np.unique(y[test_ls_rd[i]], return_counts=True)[1])


            # Summarize train test split
            train_test_splits = [[train_ls, test_ls], [train_ls_rd, test_ls_rd]]

            log.info(f'Train test splits based on sequence-based clustering with {SIM_SPLIT} threshold prepared')

        
        except Exception as e:
            log.exception(f'ERROR TRAIN-TEST Splits: {e}')





        ########### RUN CLASSIFICATION ###########
        try:
            # define embedding names
            feat_n_ls = ['GEX_2000_var', '3-mer', 'GEX_2000_var_3-mer']

            # define embedding list
            feat_list = [GEX, kmer, GEX_kmer]


            # define file path
            file_path = os.path.join(out_path,f'{today}_{dataset}_{SIM_SPLIT}_GEX_Spec_classification_CV_results.csv')

            # pipe_n = ['', '_pca']# pipes_ls = [['', '_pca'], [[('scaler', StandardScaler())], [('scaler', StandardScaler()), ('pca', PCA(n_components = 50))]]]
            pipe_n = ['']
            # pipes_ls = [[( 'scaler', StandardScaler() )], [('scaler', StandardScaler()), ('pca', PCA(n_components = 50))]]
            pipes_ls = [[( 'scaler', StandardScaler() )]]
            results_l = []
            for n, pipes in zip(pipe_n, pipes_ls):
                for emb_name, emb in zip(feat_n_ls, feat_list):
                    for rf in [None, True]:
                        e = f'{emb_name}{n}'
                        X = emb
                        log.info(f'Evaluate classifier on {e} data')
                        results = CLF.run_clf_on_splits(X, y, train_test_splits, SIM_SPLIT, emb_name=e, RF=rf,
                                                    pipe_ls = pipes, log=log)
                        results_l.append(results)

                        # save intermediate result
                        results = pd.concat(results_l)
                        results.to_csv(file_path, index=False)

            log.info(f'Evaluation done')

        except Exception as e:
            log.exception(f'ERROR Running classifier {emb_name}: {e}')

        ########### SAVE RESULTS ###########


        results = pd.concat(results_l)
        results.to_csv(file_path, index=False)

        log.info(f'Results saved in {file_path}')






def main():
    run()




if __name__ == '__main__':
    main()