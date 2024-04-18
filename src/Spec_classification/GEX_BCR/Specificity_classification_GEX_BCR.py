
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



# add root directory to path such that the utils_nb file can be imported
CONFIG_PATH = '/data/cb/scratch/lenae/p-GP-LLM-AbPred/src/models/Spec_classification/GEX_BCR/config_file_GEX_BCR.txt'

# setup parser for the config file
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
ROOT_DIR = config['ROOT']['ROOT_DIR']
UTILS_DIR = config['ROOT']['UTILS_DIR']
sys.path.append(UTILS_DIR)
sys.path.append(os.path.join(UTILS_DIR, 'AbMAP_analysis'))


# import custom modules
import sc_AbSpecificity_pred.src.utils_nb as utils
import utils_abmap_analysis as utilsa
import Load_embs_class as lec
import Specificity_classification_class as CLF


from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
# clustering
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import pairwise_distances, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, f1_score
from sklearn.decomposition import PCA


#############################  CLASS  #############################


# class Specificity_classificiation_model_evaluation:

#     """ 
#     Evaluation of Classification models with nested cross-validation based on sklearn;

#         This class can be defned with various models, which will be evaluated for:
#          - outputs/saves nested accuracy metrics; for 3-5 rounds of train test splits; the test splits 
#          can be grouped by similarity in embedding or sequence space.


#         Attributes:
#             X: numpy.ndarray of shape (n_samples, p_length * 20)
#                 protein (embedding/kmer) input data;
#             y: numpy.ndarray of shape (n_samples, 1)
#                 CamSol value to be predicted from the protein sequences;
#             model_name: str
#                 name of the model type as a prefix for saving files/plots;
#             metrics: dict
#                 metric names (str) as keys and strings as values of the corresponding sklearn metrics for fitting the models;
#                 currently up to 2 metrics are supported in this class; the first one in the dictionary will be used as the main
#                 metric for refitting the model; the second will be only reported; optimized for MSE and R2 value;
#                 for default set to: {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}
#             param_grid: dict
#                 parameters names (str) as keys and lists of parameter settings to try as values in the parameter tuning of the model
            
            
#         """

#     def __init__(self, X: np.ndarray, y: np.ndarray, metrics: dict = {'AUC': 'roc_auc', 'f1': 'f1', 'recall': 'recall'}, 
#                  pipe_ls = [('scaler', StandardScaler())]):
#         """
#         Initializes the instance based on classification model, data, model name and metrics.
#         """
#         self.X = X
#         self.y = y
#         self.clf = None
#         self.model_name = None
#         self.metrics = metrics
#         self.param_grid = None
#         self.y_pred = None
#         self.pipe_ls = pipe_ls
#         self.pipe = None
#         self.best_model = None # best model from non-nested CV
        
#         self.train_id_ls = None
#         self.test_id_ls = None


#     def Log_reg_nosp(self, X_train, X_test, y_train, y_test, verbose):

#         self.model_name = 'Log_reg'
#         # Setup classifier
#         self.clf = LogisticRegression(random_state=123)

#         # prepare pipeline
#         p_ls = self.pipe_ls.copy()
#         p_ls.append(('clf', self.clf))
#         self.pipe = Pipeline(p_ls)


#         # define parameter grid
#         self.param_grid = {'clf__penalty': ['l2', None],
#                       'clf__class_weight': ['balanced', None],
#                       'clf__C': [10, 1, 0.1, 0.01, 0.001],
#                       #'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#                       'clf__max_iter': [500, 1000, 2000]
#                      }


#         # Perfrom hyperparameter tuning
#         grid_search = GridSearchCV(estimator=self.pipe, param_grid=self.param_grid, cv=5, 
#                                    scoring=self.metrics, refit='recall',
#                                    n_jobs=-1, verbose=verbose)
#         grid_search.fit(X_train, y_train)


#         if verbose > 1:
#             print("Best Hyperparameters:", grid_search.best_params_)
#             print("Best score:", grid_search.best_score_)

#         self.best_model = grid_search.best_estimator_
#         self.y_pred = self.best_model.predict(X_test)


#         # save metrics
#         metric_dict = {}

#         accuracy = accuracy_score(y_test, self.y_pred)
#         prec = precision_score(y_test, self.y_pred)
#         rec = recall_score(y_test, self.y_pred)
#         mcc = matthews_corrcoef(y_test, self.y_pred)

#         metric_dict['accuracy'] = accuracy
#         metric_dict['MCC'] = mcc
#         metric_dict['F1'] = f1_score(y_test, self.y_pred)
#         metric_dict['precision'] = prec
#         metric_dict['recall'] = rec

#         if verbose > 0:
#             # print scores
#             print("Test Accuracy:", np.round(accuracy, 3))
#             print("Test MCC:", np.round(mcc, 3))
#             print("Test Precision:", np.round(prec, 3))
#             print("Test Recall:", np.round(rec, 3))

#         return metric_dict


#     ### SVC for the below wrapper
#     def SVC_eval_nosp(self, X_train, X_test, y_train, y_test, verbose=0):

#         self.model_name = 'kSVC'
#         # Setup classifier
#         self.clf = SVC(probability=True, random_state=123)
        
#         # prepare pipeline
#         p_ls = self.pipe_ls.copy()
#         p_ls.append(('clf', self.clf))
#         self.pipe = Pipeline(p_ls)


#         self.param_grid = {
#             'clf__C': [1, 0.1, 0.01, 0.001],
#             'clf__kernel': ['poly', 'rbf', 'sigmoid'],
#             'clf__degree': [2, 3, 4],  # Only used when kernel is 'poly'
#             'clf__gamma': ['scale', 'auto'] #+ list(np.logspace(-3, 3, 7))
#         }

#         # Perfrom hyperparameter tuning
#         grid_search = RandomizedSearchCV(estimator=self.pipe, param_distributions=self.param_grid, cv=5, 
#                                          scoring=self.metrics, refit='recall',
#                                          n_jobs=-1, verbose=verbose)
#         grid_search.fit(X_train, y_train)

#         if verbose > 1:
#             print("Best Hyperparameters:", grid_search.best_params_)
#             print("Best score:", grid_search.best_score_)


#         self.best_model = grid_search.best_estimator_
#         self.y_pred = self.best_model.predict(X_test)

#             # save metrics
#         metric_dict = {}

#         accuracy = accuracy_score(y_test, self.y_pred)
#         prec = precision_score(y_test, self.y_pred)
#         rec = recall_score(y_test, self.y_pred)
#         mcc = matthews_corrcoef(y_test, self.y_pred)

#         metric_dict['accuracy'] = accuracy
#         metric_dict['MCC'] = mcc
#         metric_dict['F1'] = f1_score(y_test, self.y_pred)
#         metric_dict['precision'] = prec
#         metric_dict['recall'] = rec

#         if verbose >0:
#             # print scores
#             print("Test Accuracy:", np.round(accuracy, 3))
#             print("Test MCC:", np.round(mcc, 3))
#             print("Test Precision:", np.round(prec, 3))
#             print("Test Recall:", np.round(rec, 3))

#         return metric_dict

    
  
    
#     # Wrapps SVC and LogReg classifier for evaluation
#     def evaluate_clf(self, X, y, group_thresh, train_id_ls, test_id_ls, log=None,
#                      verbose=1):

        
#         metric_dict_Log = {}
#         for k in ['accuracy', 'MCC', 'F1', 'precision', 'recall']:
#             metric_dict_Log[k] = []

#         metric_dict_SVC = {}
#         for k in ['accuracy', 'MCC', 'F1', 'precision', 'recall']:
#             metric_dict_SVC[k] = []
#         i=1
        
# #         # iterate throught different reshuffeled group splits
# #         for train_idx, test_idx in gss.split(X, y, groups=clusters): 
#         for train_idx, test_idx in zip(train_id_ls, test_id_ls):
            
#             #if i > 3: break
            
#             if verbose > 0: print(f'\nROUND {i}')
#             i += 1
            
#             print(len(test_idx))
            
#             X_train, X_test = X[train_idx], X[test_idx]
#             y_train, y_test = y[train_idx], y[test_idx]
#             if log is not None: 
#                 log.info(f'Train LogReg classifier')
#             else: 
#                 print('Train LogReg classifier')
#             m_dict_l = self.Log_reg_nosp(X_train, X_test, y_train, y_test, verbose=verbose)

#             if log is not None: 
#                 log.info(f'Train SVC classifier')
#             else: 
#                 print('Train SVC classifier')
#             m_dict_s = self.SVC_eval_nosp(X_train, X_test, y_train, y_test, verbose=verbose)

#             for k in m_dict_l.keys():
#                 metric_dict_Log[k].append([m_dict_l[k]])
#                 metric_dict_SVC[k].append([m_dict_s[k]])

#         return metric_dict_Log, metric_dict_SVC, test_idx

    
    

    
# # Run the classificaton on X and y with the train test splits
# def run_clf_on_splits(X, y, train_test_splits, sim_split, emb_name, pipe_ls = [('scaler', StandardScaler())], log=None):
#     # Instantiate class 
#     clf = Specificity_classificiation_model_evaluation(X, y, pipe_ls = pipe_ls)
    
#     # save results
#     results = pd.DataFrame()

#     for splits, cluster_m in zip(train_test_splits, ['levenshtein_sequence_based', 'random_split']):

#         if log is not None: 
#             log.info(f'Fit models on {cluster_m} method')
#         else: 
#             print(f'Fit models on {cluster_m} method')
#         # test sequence based splitting

#         metric_dict_Log, metric_dict_SVC, test_idx = clf.evaluate_clf(X, y, group_thresh=sim_split, train_id_ls=splits[0], test_id_ls=splits[1], 
#                                                             log=None, verbose=1)


#         if log is not None: 
#             log.info('\nLogReg - MCC: ', np.mean(metric_dict_Log['MCC']), np.std(metric_dict_Log['MCC']))
#         else: 
#             print('\nLogReg - MCC: ', np.mean(metric_dict_Log['MCC']), np.std(metric_dict_Log['MCC']))
        
#         if log is not None: 
#             log.info('kSVC - MCC: ', np.mean(metric_dict_SVC['MCC']), np.std(metric_dict_SVC['MCC']))
#         else: 
#             print('kSVC - MCC: ', np.mean(metric_dict_SVC['MCC']), np.std(metric_dict_SVC['MCC']))


#             # add dataframe 
#         for model, m_dict in zip(['LogReg', 'SVC'], [metric_dict_Log, metric_dict_SVC]):
#             row = len(results)
#             results.loc[row, 'Model'] = f'{model}_{emb_name}' 
#             results.loc[row, 'group_thresh'] = sim_split 
#             results.loc[row, 'train_test_split'] = cluster_m 
#             for m in m_dict.keys():

#                 mean = np.round(np.mean(m_dict[m]), 3)
#                 std = np.round(np.std(m_dict[m]), 3)
#                 # save results
#                 results.loc[row, m] = mean 
#                 results.loc[row, f'{m}_std'] = std 

#     return results




######## MAIN ########



def run():

    # for c_type in ['VDJ_VJ_aaSeq', 'VDJ_aaSeq']: ## only VDJ_VJ!
    ab_chain = 'VDJ_VJ_aaSeq'

    if ab_chain == 'VDJ_aaSeq': 
        c_type = 'VH'
    elif ab_chain == 'VDJ_VJ_aaSeq':
         c_type = 'VH_VL'

    # set random seed
    random.seed(123)
    today = str(datetime.now().date())
   log = utils.Logger(__name__, log_file=f'BCR_GEX_{today}_{c_type}_05splitINTonly_app.log').get_logger()
    log.info('Start Script!')

    # setup parser for the config file
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    ROOT_DIR = config['ROOT']['ROOT_DIR']




    ######## LOOP THROUGH DATASETS ########
    for dataset in ['OVA', 'RBD', 
        'INTEGRATED']:

        
        log.info(f'Start evaluating models - {dataset}')

        ######## LOADING DATASET ########
        try:
            # dataset = 'INTEGRATED'

            feat_inputPath = os.path.join(ROOT_DIR, config['BCR_GEX'][dataset])

            # Load dataframe
            feature_df = pd.read_csv(feat_inputPath)
            

            # Calculate distance matrix between all sequences
            distance_matrix = utils.calc_norm_levens_dist(feature_df[ab_chain])


            #### Prepare datasets
            GEX = feature_df.iloc[:, -2000:].values
            kmer = feature_df.iloc[: , 6:-2000].values
            GEX_kmer = feature_df.iloc[: , 6:].values
            y = feature_df.loc[:, 'group_id'].values



        except Exception as e:
            log.exception(f'ERROR Loading files: {e}')



        ######## TRAIN TEST SPLITS ########
        try:
            # create train test splits - sequence clustering
            N_SPLITS=5
            SIM_SPLIT = 0.05
                

            # best splits are created with N_SPLITS=6 (based on manual inspection of train test splits)
            train_ls, test_ls, clusters = CLF.train_test_split_idx(kmer, y, cluster_thresh=SIM_SPLIT, distance_matrix=distance_matrix, 
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
            train_ls_rd, test_ls_rd, _ = CLF.train_test_split_idx(kmer, y,# cluster_thresh=SIM_SPLIT, 
                                                                n_splits=N_SPLITS, cluster_method= 'random_split', verbose=0)
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
            feat_n_ls = ['GEX_2000_var', '3-mer', 'GEX_2000_var_3-mer']

            # define embedding list
            feat_list = [GEX, kmer, GEX_kmer]


            # define file path
            file_path = os.path.join(ROOT_DIR, parser.out_path,f'{today}_{c_type}_Spec_classification_CV_results.csv')

            results_l = []
            for n, pipes in zip(['', '_pca'], [[('scaler', StandardScaler())], [('scaler', StandardScaler()), ('pca', PCA(n_components = 50))]]):
                for emb_name, emb in zip(feat_n_ls, feat_list):
                    e = f'{emb_name}_{n}'
                    X = emb
                    log.info(f'Evaluate classifier on {e} data')
                    results = CLF.run_clf_on_splits(X, y, train_test_splits, SIM_SPLIT, emb_name=e,
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