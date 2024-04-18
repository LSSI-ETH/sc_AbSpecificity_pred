#!/usr/bin/env python3
#################################
# Script for classification models to predict Binding Specificity for OVA values from sequence embeddings
# This class is used in the classification pipeline to train and evaluate ML models. 
# Author: Lena Erlach
# Date: 20 Mar 2024
################################


import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
import random
import configparser, argparse


# add parser
parser = argparse.ArgumentParser(description='Run classification models to predict Binding Specificity for OVA values from sequence embeddings')
parser.add_argument('--config', type=str, default='config_file.txt', help='Path to the config file')
parser.add_argument('--simsplit_thresh', type=float, default=0.05, help='Similarity split threshold; default 0.05')
parser.add_argument('--outpath', type=str, default='data/model_evaluation/Specificity_classification/', help='Output path for results')
args = parser.parse_args()

# add root directory to path such that the utils_nb file can be imported
CONFIG_PATH = args.config 
UTILS_DIR = '../'
UTILS_DIR1 = './'
sys.path.append(UTILS_DIR)
sys.path.append(UTILS_DIR1)


# import custom modules
import utils_nb as utils
import Load_embs_class as lec
import Specificity_classification_class as CLF
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def run():
    
    # setup parser for the config file
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    ROOT_DIR = config['ROOT']['ROOT_DIR']

    # set dataset (OVA or RBD)
    dataset = config['SETUP']['DATASET']
    print(f'Dataset: {dataset}')
    # create output directory
    out_path = os.path.join(args.outpath, dataset)
    print(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # set random seed
    random.seed(123)
    today = str(datetime.now().date())
    log = utils.Logger(__name__, log_file=f'{today}_{dataset}_app.log').get_logger()
    log.info('Start Script!')


    for ab_chain in ['VDJ_VJ_aaSeq', 'VDJ_aaSeq']:

        if ab_chain == 'VDJ_aaSeq': 
            c_type = 'VH'
            filter_192 = False
        if ab_chain == 'VDJ_VJ_aaSeq':
            c_type = 'VH_VL'
            
            if dataset == 'OVA': 
                filter_192 = True
            else:
                filter_192 = False
            
        if dataset == 'OVA': 
            filter_VH_complete = False
        else:
            filter_VH_complete = True


        ######## LOADING CLASS ########
        try:
            Embeddings = lec.LoadEmbeddings_VH_VL(CONFIG_PATH=CONFIG_PATH, seq_col=ab_chain, filter_192 = filter_192,
                                                filter_VH_complete = filter_VH_complete)
            
            Embeddings.load_embeddings(embedding_type = 'all', verbose=False)
            # Seq column name 'VDJ_aaSeq', 'VDJ_aaSeqCDR3', 'cdr_comb'...            

            ### Load mAb sequences
            seq_df = Embeddings.seq_df
            seqs = Embeddings.seqs


            ### Load embeddings - ESM2 - VH_VL
            ESM_fl_embeddings = Embeddings.emb_ESM
            log.info("ESM - embeddings loaded")


            ### Load embeddings - ESM2 CDRextract - VH_VL
            ESM_cdr_fl_embeddings = Embeddings.emb_ESM_cdrs
            log.info("ESM CDRextract - embeddings loaded")

            
            ### Load embeddings - Antiberty - VH_VL
            antiberty_embeddings = Embeddings.emb_antiberty
            log.info("Antiberty - embeddings loaded")

            # Calculate the kmer vectors
            k=3
            all_kmers = utils.generate_all_kmers(seqs, k)
            vectors = [utils.freqs_to_vector(utils.kmer_frequencies(seq, k), all_kmers) for seq in seqs]
            kmer_arr_3 = np.array(vectors)

            
            # Load sequence distance matrix
            distance_matrix = Embeddings.dist_matrix
            log.info("distance matrix loaded")

        except Exception as e:
            log.exception(f'ERROR Loading files: {e}')




        ######## TRAIN TEST SPLITS ########
        try:
            # create train test splits - sequence clustering
            N_SPLITS=5
            SIM_SPLIT = args.simsplit_thresh
            X = ESM_fl_embeddings
            y = np.array(seq_df['group_id'])

            # best splits are created with N_SPLITS=6 (based on manual inspection of train test splits)
            train_ls, test_ls, clusters = CLF.train_test_split_idx(X, y, cluster_thresh=SIM_SPLIT, distance_matrix=distance_matrix, 
                                                            n_splits=N_SPLITS, cluster_method= 'levenshtein_sequence_based',
                                                            verbose=0)
            # # manually remove an item for bad split
            print(f'Sequence-based clustering with {SIM_SPLIT} cluster threshold')
            for i in range(len(test_ls)):
                print(len(test_ls[i]))
                print(np.unique(y[test_ls[i]], return_counts=True)[1])


            # create train test splits - Random split
            train_ls_rd, test_ls_rd, _ = CLF.train_test_split_idx(X, y, cluster_thresh=SIM_SPLIT, n_splits=N_SPLITS, cluster_method= 'random_split',
                                                            verbose=0)
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
            emb_n_list = ['ESM-2', 'ESM-2-CDRextract',
                          '3-mer', 'Antiberty']

            # define embedding list
            emb_list = [ESM_fl_embeddings,ESM_cdr_fl_embeddings, 
                kmer_arr_3, antiberty_embeddings]

            # define file path
            file_path = os.path.join(out_path,f'{today}_{dataset}_{c_type}_Spec_classification_CV_results.csv')


            results_l = []
            for n, pipes in zip(['', '_pca'], [[('scaler', StandardScaler())], [('scaler', StandardScaler()), ('pca', PCA(n_components = 50))]]):
                for emb_name, emb in zip(emb_n_list, emb_list):
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