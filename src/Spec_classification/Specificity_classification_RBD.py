
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
import configparser

# add parser
parser = argparse.ArgumentParser(description='Run classification models to predict Binding Specificity for OVA values from sequence embeddings')
parser.add_argument('--config', type=str, default='config_file.txt', help='Path to the config file')
parser.add_argument('--simsplit_tresh', type=float, default=0.05, help='Similarity split threshold; default 0.05')
parser.add_argument('--out_path', type=str, default='data/model_evaluation/Specificity_classification/', help='Output path for results')


# add root directory to path such that the utils_nb file can be imported
CONFIG_PATH = parser.parse_args().config 
UTILS_DIR = '/data/cb/scratch/lenae/sc_AbSpecificity_pred/src'
sys.path.append(UTILS_DIR)



# import custom modules
import sc_AbSpecificity_pred.src.utils_nb as utils
import Load_embs_class as lec
import Specificity_classification_class as CLF


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



######## MAIN ########


def run():

    # set random seed
    random.seed(123)
    today = str(datetime.now().date())
    log = utils.Logger(__name__, log_file=f'RBD_{today}_app.log').get_logger()
    log.info('Start Script!')


    for ab_chain in ['VDJ_VJ_aaSeq', 'VDJ_aaSeq']:
        # ab_chain = 'VDJ_aaSeq'
        # ab_chain = 'VDJ_VJ_aaSeq'
        if ab_chain == 'VDJ_aaSeq': 
            c_type = 'VH'
        elif ab_chain == 'VDJ_VJ_aaSeq':
            c_type = 'VH_VL'

        # setup parser for the config file
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH)
        ROOT_DIR = config['ROOT']['ROOT_DIR']



        ######## LOADING CLASS ########
        try:
            
            Embeddings =  lec.LoadEmbeddings_VH_VL(CONFIG_PATH=CONFIG_PATH, seq_col=ab_chain, filter_192 = False, 
                                            filter_VH_complete = True)

            Embeddings.load_embeddings(embedding_type = 'all', verbose=True)
            # Seq column name 'VDJ_aaSeq', 'VDJ_aaSeqCDR3', 'cdr_comb'...
            seq_col = Embeddings.seq_col
            

            ### Load mAb sequences
            seq_df = Embeddings.seq_df
            names = Embeddings.names
            seqs = Embeddings.seqs


            ### AbMAP 
            Abmap_fl_embeddings = Embeddings.emb_AM
            log.info("AbMap - embeddings loaded")


            ### Load embeddings - ESM2 
            ESM_fl_embeddings = Embeddings.emb_ESM
            log.info("ESM - embeddings loaded")


            ### Load embeddings - ESM2 augmented 
            ESM_aug_fl_embeddings = Embeddings.emb_ESM_aug
            log.info("ESM augmented - embeddings loaded")


            ### Load embeddings - ESM2 CDRextract
            ESM_cdr_fl_embeddings = Embeddings.emb_ESM_cdrs
            log.info("ESM CDRextract - embeddings loaded")


            ### Load embeddings - Antiberty
            antiberty_embeddings = Embeddings.emb_antiberty
            log.info("Antiberty - embeddings loaded")

            # Calculate the kmer vectors
            k=3
            all_kmers = utils.generate_all_kmers(seqs, k)
            vectors = [utils.freqs_to_vector(utils.kmer_frequencies(seq, k), all_kmers) for seq in seqs]
            kmer_arr = np.array(vectors)

            k=2
            all_kmers = utils.generate_all_kmers(seqs, k)
            vectors = [utils.freqs_to_vector(utils.kmer_frequencies(seq, k), all_kmers) for seq in seqs]
            kmer_arr_2 = np.array(vectors)
            log.info("kmer embeddings calculated")


            # Load sequence distance matrix
            distance_matrix = Embeddings.dist_matrix
            log.info("distance matrix loaded")


        except Exception as e:
            log.exception(f'ERROR Loading files: {e}')



        ######## TRAIN TEST SPLITS ########
        try:
            # create train test splits - sequence clustering
            N_SPLITS=5
            SIM_SPLIT = parser.parse_args().simsplit_tresh

                
            X = ESM_fl_embeddings
            y = np.array(seq_df['group_id'])

            # best splits are created with N_SPLITS=6 (based on manual inspection of train test splits)
            train_ls, test_ls, clusters = CLF.train_test_split_idx(X, y, cluster_thresh=SIM_SPLIT, distance_matrix=distance_matrix, 
                                                            n_splits=N_SPLITS, cluster_method= 'levenshtein_sequence_based',
                                                            verbose=0)
            # # manually remove an item for bad split
            # test_ls.pop(2)
            # train_ls.pop(2)
            print(f'Sequence-based clustering with {SIM_SPLIT} cluster threshold')
            for i in range(len(test_ls)):
                print(len(test_ls[i]))
                print(np.unique(y[test_ls[i]], return_counts=True)[1])


            # create train test splits - Random split
            train_ls_rd, test_ls_rd, _ = CLF.train_test_split_idx(X, y, cluster_thresh=SIM_SPLIT, n_splits=N_SPLITS, cluster_method= 'random_split',
                                                            verbose=0)
            print(f'Random split')
            # test_ls_rd.pop(2)
            # train_ls_rd.pop(2)
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
            emb_n_list = [#'AbMAP', 
                          'ESM-2', 'ESM-2-CDRextract', 
                          'ESM-2-augmented', 
                          '3-mer', '2-mer',
                            'Antiberty'  ]

            # define embedding list
            emb_list = [# Abmap_fl_embeddings, 
                ESM_fl_embeddings, ESM_cdr_fl_embeddings, ESM_aug_fl_embeddings, 
                kmer_arr, kmer_arr_2,
                antiberty_embeddings]
            
            # define file path
            file_path = os.path.join(ROOT_DIR, parser.out_path,f'{today}_{c_type}_Spec_classification_CV_results.csv')

            results_l = []
            for n, pipes in zip(['', '_pca'], [[('scaler', StandardScaler())], [('scaler', StandardScaler()), ('pca', PCA(n_components = 50))]]):
                for emb_name, emb in zip(emb_n_list, emb_list):
                    e = f'{emb_name}{n}'
                    X = emb
                    log.info(f'Evaluate classifier on {e} data')
                    results = CLF.run_clf_on_splits(X, y, train_test_splits, SIM_SPLIT, emb_name=e, RF=True,
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