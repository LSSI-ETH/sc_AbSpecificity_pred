#################################
# 
# Class for classification models to predict Binding Specificity for OVA values from sequence embeddings
# This class is used in the classification pipeline to train and evaluate ML models. 
# Author: Lena Erlach
# Date: 09 Nov 2020
################################

import numpy as np
import pandas as pd
import os, sys
from datetime import datetime
import random
import configparser



# add root directory to path such that the utils_nb file can be imported
CONFIG_PATH = '/data/cb/scratch/lenae/p-GP-LLM-AbPred/notebooks/config_file.txt'
UTILS_DIR = '/data/cb/scratch/lenae/p-GP-LLM-AbPred/notebooks'
sys.path.append(UTILS_DIR)
sys.path.append(os.path.join(UTILS_DIR, 'AbMAP_analysis'))


# import custom modules
import sc_AbSpecificity_pred.src.utils_nb as utils
import utils_abmap_analysis as utilsa
import Load_embs_class as lec



from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
# clustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import pairwise_distances, auc, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef , RocCurveDisplay, roc_auc_score, average_precision_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



#############################  CUSTOM FUNTIONS  #############################


# split train test set by clustering similar sequences
def return_grouped_train_test_split(X, y, group_thresh, distance_matrix = None, cluster_method = 'levenshtein_sequence_based', 
                                    n_splits=5, verbose=1):
    
    '''
    Function for returning grouped train test split. When cluster_method == 'embedding_dbscan', the embeddings are clustered
    based on DBSCAN and a train-test split based on group_thresh (distance of sequences wihtin a cluster).
    Here euclidean distances between embeddings are computed within the function.
    
    When cluster_method = 'levenshtein_sequence_based', a distance matrix has to be supplied and based on group_thresh distance
    hierarchical cluster dendogram is cut to assign sequences to clusters.
    
    group_thresh: similarity measure for splitting
    '''
    
    if cluster_method in ['levenshtein_sequence_based', 'embedding_dbscan', 'embedding_hclust']:
        if cluster_method == 'levenshtein_sequence_based':

            assert distance_matrix is not None, 'No distance matrix supplied!' 

            if verbose > 0: print('Calculate clusters from precomputed distance matrix')
            # Assuming distances is your distance matrix
            linked = linkage(squareform(distance_matrix), 'single')  # You can also use 'complete', 'average', etc.
            # Forms flat clusters so that the original observations in each flat cluster have no greater a cophenetic distance than t
            clusters = fcluster(linked, t=group_thresh, criterion='distance')



        elif cluster_method == 'embedding_dbscan':

            if verbose > 0: print('Calculate clusters from embeddings with DBSCAN')
            # Compute the pairwise distance matrix
            distance_matrix = pairwise_distances(X)

            # Use DBSCAN clustering
            dbscan = DBSCAN(eps=group_thresh, min_samples=2, metric='precomputed')
            clusters = dbscan.fit_predict(distance_matrix)
            #print(np.unique(clusters, return_counts=True))

            # replace -1 as being assigned to no cluster
            for i, c in enumerate(clusters): 
                if c == -1: 
                    clusters[i] = i + clusters[-1]
        
        elif cluster_method == 'embedding_hclust':

            if verbose > 0: print('Calculate clusters from embeddings with hierarchical clustering')
            # Compute the pairwise distance matrix
            distance_matrix = pairwise_distances(X)

            # Assuming distances is your distance matrix
            linked = linkage(squareform(distance_matrix), 'single')  # You can also use 'complete', 'average', etc.
            
            # Forms flat clusters so that the original observations in each flat cluster have no greater a cophenetic distance than t
            clusters = fcluster(linked, t=group_thresh, criterion='distance')
            

        # just return the GroupShuffleSplit object
        # gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=123)
        gss = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=12)
        gss.get_n_splits()

    elif cluster_method == 'random_split':
        if verbose > 0: print('Calculate clusters from embeddings with DBSCAN')
        gss = StratifiedShuffleSplit(test_size=1/n_splits, random_state=123)
        clusters = None
        
    return gss, clusters    



def train_test_split_idx(X, y, cluster_thresh=0, distance_matrix=None, n_splits=5, cluster_method= 'random_split',
                             verbose=0):

        '''
        Function for returning train test splits (list of train & test indices) for evaluate_clf(). 
        When cluster_method == 'embedding_dbscan', the embeddings are clustered
        based on DBSCAN and a train-test split based on group_thresh (distance of sequences wihtin a cluster).
        Here euclidean distances between embeddings are computed within the function.

        When cluster_method = 'levenshtein_sequence_based', a distance matrix has to be supplied and based on group_thresh distance
        hierarchical cluster dendogram is cut to assign sequences to clusters.

        cluster_thresh: similarity measure for splitting
        '''
        # lists for the train test splits
        train_id_ls = []
        test_id_ls = []
        
        # Check for the different methods
        if cluster_method in ['levenshtein_sequence_based', 'embedding_dbscan', 'embedding_hclust']:
            if cluster_method == 'levenshtein_sequence_based':

                assert distance_matrix is not None, 'No distance matrix supplied!' 

                if verbose > 0: print('Calculate clusters from precomputed distance matrix')
                # Assuming distances is your distance matrix
                linked = linkage(squareform(distance_matrix), 'single')  # You can also use 'complete', 'average', etc.
                # Forms flat clusters so that the original observations in each flat cluster have no greater a cophenetic distance than t
                clusters = fcluster(linked, t=cluster_thresh, criterion='distance')



            elif cluster_method == 'embedding_dbscan':

                if verbose > 0: print('Calculate clusters from embeddings with DBSCAN')
                # Compute the pairwise distance matrix
                distance_matrix = pairwise_distances(X)

                # Use DBSCAN clustering
                dbscan = DBSCAN(eps=cluster_thresh, min_samples=2, metric='precomputed')
                clusters = dbscan.fit_predict(distance_matrix)
                #print(np.unique(clusters, return_counts=True))

                # replace -1 as being assigned to no cluster
                for i, c in enumerate(clusters): 
                    if c == -1: 
                        clusters[i] = i + clusters[-1]

            elif cluster_method == 'embedding_hclust':

                if verbose > 0: print('Calculate clusters from embeddings with hierarchical clustering')
                # Compute the pairwise distance matrix
                distance_matrix = pairwise_distances(X)

                # Assuming distances is your distance matrix
                linked = linkage(squareform(distance_matrix), 'single')  # You can also use 'complete', 'average', etc.

                # Forms flat clusters so that the original observations in each flat cluster have no greater a cophenetic distance than t
                clusters = fcluster(linked, t=cluster_thresh, criterion='distance')


            # just return the GroupShuffleSplit object
            # gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=123)
            gss = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=123)
            
            # get the train-test splits for grouped splits
            gss.get_n_splits()
            for train_idx, test_idx in gss.split(X, y, groups=clusters):
                train_id_ls.append(train_idx)
                test_id_ls.append(test_idx)
            
            
        elif cluster_method == 'random_split':
            if verbose > 0: print('Return random splits')
            
            gss = StratifiedShuffleSplit(test_size=1/n_splits, n_splits= n_splits, random_state=123)
            clusters = None
            
            # get the train-test splits for grouped splits
            gss.get_n_splits()
            for train_idx, test_idx in gss.split(X, y):
                train_id_ls.append(train_idx)
                test_id_ls.append(test_idx)
           

        return train_id_ls, test_id_ls, clusters    

    

    


############################# NEW CLASS  #############################




class Specificity_classificiation_model_evaluation:

    """ 
    Evaluation of Classification models with nested cross-validation based on sklearn;

        This class can be defned with various models, which will be evaluated for:
         - outputs/saves nested accuracy metrics; for 3-5 rounds of train test splits; the test splits 
         can be grouped by similarity in embedding or sequence space.


        Attributes:
            X: numpy.ndarray of shape (n_samples, p_length * 20)
                protein (embedding/kmer) input data;
            y: numpy.ndarray of shape (n_samples, 1)
                CamSol value to be predicted from the protein sequences;
            model_name: str
                name of the model type as a prefix for saving files/plots;
            metrics: dict
                metric names (str) as keys and strings as values of the corresponding sklearn metrics for fitting the models;
                currently up to 2 metrics are supported in this class; the first one in the dictionary will be used as the main
                metric for refitting the model; the second will be only reported; optimized for MSE and R2 value;
                for default set to: {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}
            param_grid: dict
                parameters names (str) as keys and lists of parameter settings to try as values in the parameter tuning of the model
            
            
        """

    def __init__(self, X: np.ndarray, y: np.ndarray, metrics: dict = {'AUC': 'roc_auc', 'f1': 'f1', 'recall': 'recall'}, 
                 pipe_ls = [('scaler', StandardScaler())]):
        """
        Initializes the instance based on classification model, data, model name and metrics.
        """
        self.X = X
        self.y = y
        self.clf = None
        self.model_name = None
        self.metrics = metrics
        self.param_grid = None
        self.y_pred = None
        self.pipe_ls = pipe_ls
        self.pipe = None
        self.best_model = None # best model from non-nested CV
        
        self.train_id_ls = None
        self.test_id_ls = None


    def run_clf(self, model_name, param_grid, clf, X_train, X_test, y_train, y_test, verbose=1):
        '''
        Function for evaluating the regression model with nested cross-validation.

        param: 
            model_name: str; name of the model type as a prefix for saving files/plots; ' kSVC' or 'Log_reg' implemented;
            clf: sklearn classifier object; classifier to be evaluated;
            param_grid: dict; parameters names (str) as keys and lists of parameter settings to try as values in the parameter tuning of the model;
            X_train: numpy.ndarray of shape (n_samples, dim)
            X_test: numpy.ndarray of shape (n_samples, dim) 
            y_train: numpy.ndarray of shape (n_samples, 1)
            y_test: numpy.ndarray of shape (n_samples, 1)
            verbose: int; verbosity level of the function
        '''


        self.model_name = model_name
        # Setup classifier
        self.clf = clf

        # prepare pipeline
        p_ls = self.pipe_ls.copy()
        p_ls.append(('clf', self.clf))
        self.pipe = Pipeline(p_ls)

        # define parameter grid
        self.param_grid = param_grid
        

        # Perfrom hyperparameter tuning
        grid_search = RandomizedSearchCV(estimator=self.pipe, param_distributions=self.param_grid, cv=5, 
                                         scoring=self.metrics, refit='recall',
                                         n_jobs=-1, random_state=123, verbose=verbose)
        
        grid_search.fit(X_train, y_train)

        if verbose > 1:
            print("Best Hyperparameters:", grid_search.best_params_)
            print("Best score:", grid_search.best_score_)


        self.best_model = grid_search.best_estimator_
        self.y_pred = self.best_model.predict(X_test)

        # save metrics
        metric_dict = {}

        accuracy = accuracy_score(y_test, self.y_pred)
        prec = precision_score(y_test, self.y_pred)
        rec = recall_score(y_test, self.y_pred)
        mcc = matthews_corrcoef(y_test, self.y_pred)
        # add ROC metrics
        roc_auc = roc_auc_score(y_test, self.y_pred) 
        roc_pr = average_precision_score(y_test, self.y_pred)

        metric_dict['accuracy'] = accuracy
        metric_dict['MCC'] = mcc
        metric_dict['F1'] = f1_score(y_test, self.y_pred)
        metric_dict['precision'] = prec
        metric_dict['recall'] = rec
        metric_dict['roc_auc'] = roc_auc
        metric_dict['roc_pr'] = roc_pr

        if verbose >0:
            # print scores
            print("Test Accuracy:", np.round(accuracy, 3))
            print("Test MCC:", np.round(mcc, 3))
            print("Test Precision:", np.round(prec, 3))
            print("Test Recall:", np.round(rec, 3))
            print("Test ROC_AUC:", np.round(roc_auc, 3))
            print("Test ROC_PR:", np.round(roc_pr, 3))

        return metric_dict



    
    # Wrapps SVC and LogReg classifier for evaluation
    def evaluate_clf(self, X, y, train_id_ls, test_id_ls, log=None, RF=None,
                     verbose=1):

        
        metric_dict_Log = {}
        metrics_ls = ['accuracy', 'MCC', 'F1', 'precision', 'recall', 'roc_pr', 'roc_auc']
        for k in metrics_ls:
            metric_dict_Log[k] = []

        metric_dict_SVC = {}
        for k in metrics_ls:
            metric_dict_SVC[k] = []
        i=1
        
        # iterate throught different reshuffeled group splits
        for train_idx, test_idx in zip(train_id_ls, test_id_ls):
            
            # if testing is True: 
            # if i > 2: break
            
            if verbose > 0: print(f'\nROUND {i}')
            i += 1
            
            print(len(test_idx))
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # setup parameters
            if RF is None:
                param_grids = {
                    'Log_reg': {'clf__penalty': ['l2', None],
                            'clf__class_weight': ['balanced', None],
                            'clf__C': [10, 1, 0.1, 0.01, 0.001],
                            'clf__max_iter': [500, 1000, 2000]
                            }, 
                    'kSVC': {'clf__C': [1, 0.1, 0.01, 0.001],
                                'clf__kernel': ['poly', 'rbf', 'sigmoid'],
                                'clf__degree': [2, 3, 4],  # Only used when kernel is 'poly'
                                'clf__gamma': ['scale', 'auto'] #+ list(np.logspace(-3, 3, 7))
                                }}

                models = {'Log_reg': LogisticRegression(random_state=123),
                        'kSVC': SVC(probability=True, random_state=123)}
                
                m_names = ['Log_reg', 'kSVC']
            
            else:
                param_grids = {
                    'RF': {
                        'clf__n_estimators': [100, 200, 300],
                        'clf__max_depth': [None, 5, 10],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__min_samples_leaf': [1, 2, 4],
                        'clf__max_features': ['auto', 'sqrt', 'log2']
                    },
                    'GBoost': {
                        'clf__n_estimators': [100, 200, 300], # Number of boosting stages to be run
                        'clf__learning_rate': [0.05, 0.1, 0.2], # Shrinks the contribution of each tree
                        'clf__max_depth': [3, 4, 5], # Maximum depth of the individual estimators
                        'clf__min_samples_split': [2, 5, 10], # Minimum number of samples required to split a node
                        'clf__min_samples_leaf': [1, 2, 4], # Minimum number of samples required to be at a leaf node
                        'clf__max_features': ['auto', 'sqrt', 'log2', None], # Number of features to consider when looking for the best split
                        'clf__subsample': [0.8, 0.9, 1.0] # Fraction of samples used for fitting the individual base learners

                    }
                }

                m_names = ['RF', 
                    'GBoost']

                models = {'RF': RandomForestClassifier(random_state=123),
                          'GBoost': GradientBoostingClassifier(random_state=123)}
            

            metric_dicts = []

            # Train models 
            for m_name in m_names:
                if log is not None: 
                    log.info(f'Train {m_name} classifier')
                else: 
                    print(f'Train {m_name} classifier')
            
                # Train Logreg models
                m_dict = self.run_clf(model_name=m_name, 
                                                param_grid=param_grids[m_name], 
                                                clf=models[m_name], 
                                                X_train = X_train, X_test = X_test, 
                                                y_train=y_train, y_test=y_test, verbose=verbose)

                metric_dicts.append(m_dict)



            for k in metrics_ls:
                metric_dict_Log[k].append([metric_dicts[0][k]])
                metric_dict_SVC[k].append([metric_dicts[1][k]])

                # # if RF is only one model to be evaluated
                # if RF is None:


        return metric_dict_Log, metric_dict_SVC, test_idx


# # Wrapps SVC and LogReg classifier for evaluation
#     def evaluate_clf_with_ROCplot(self, X, y, group_thresh, train_id_ls, test_id_ls, 
#                                   log=None, verbose=1):

        
#         metric_dict_Log = {}
#         for k in ['accuracy', 'MCC', 'F1', 'precision', 'recall']:
#             metric_dict_Log[k] = []

#         metric_dict_SVC = {}
#         for k in ['accuracy', 'MCC', 'F1', 'precision', 'recall']:
#             metric_dict_SVC[k] = []
        

#         tprs_L = []
#         aucs_L = []
#         tprs_S = []
#         aucs_S = []
#         mean_fpr = np.linspace(0, 1, 100)
#         n_splits = len(train_id_ls)
#         fig, ax = plt.subplots(1,2, figsize=(10, 5))

#          # iterate throught different reshuffeled group splits
#         for fold, (train_idx, test_idx) in enumerate(zip(train_id_ls, test_id_ls)):
            
#             if fold > 1: break
            
#             if verbose > 0: print(f'\nROUND {fold}')

#             print(len(test_idx))
            
#             X_train, X_test = X[train_idx], X[test_idx]
#             y_train, y_test = y[train_idx], y[test_idx]

#             ##### LOGREG
#             if log is not None: 
#                 log.info(f'Train LogReg classifier')
#             else: 
#                 print('Train LogReg classifier')

#             m_dict_l = self.Log_reg_nosp(X_train, X_test, y_train, y_test, verbose=verbose)

#             # ROC CURVE
#             viz = RocCurveDisplay.from_estimator(
#                 self.best_model,
#                 X_test,
#                 y_test,
#                 name=f"ROC fold {fold}",
#                 alpha=0.3,
#                 lw=1,
#                 ax=ax[0],
#                 #plot_chance_level=(fold == n_splits - 1),
#             )
#             interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#             interp_tpr[0] = 0.0
#             tprs_L.append(interp_tpr)
#             aucs_L.append(viz.roc_auc)


#             ##### SVC
#             if log is not None: 
#                 log.info(f'Train SVC classifier')
#             else: 
#                 print('Train SVC classifier')
#             m_dict_s = self.SVC_eval_nosp(X_train, X_test, y_train, y_test, verbose=verbose)
            
#             # ROC CURVE
#             viz = RocCurveDisplay.from_estimator(
#                 self.best_model,
#                 X_test,
#                 y_test,
#                 name=f"ROC fold {fold}",
#                 alpha=0.3,
#                 lw=1,
#                 ax=ax[1],
#                 #plot_chance_level=(fold == n_splits - 1),
#             )
#             interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#             interp_tpr[0] = 0.0
#             tprs_S.append(interp_tpr)
#             aucs_S.append(viz.roc_auc)

#             for k in m_dict_l.keys():
#                 metric_dict_Log[k].append([m_dict_l[k]])
#                 metric_dict_SVC[k].append([m_dict_s[k]])




#         # Calculate mean and var for plot
#         for i, (tprs, aucs) in enumerate(zip([tprs_L, tprs_S], [aucs_L, aucs_S])):    

#             mean_tpr=np.mean(tprs, axis=0)
            
#             mean_tpr[-1] = 1.0
#             mean_auc = auc(mean_fpr, mean_tpr)
#             std_auc = np.std(aucs)
#             ax[i].plot(
#                 mean_fpr,
#                 mean_tpr,
#                 color="b",
#                 label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
#                 lw=2,
#                 alpha=0.8,
#             )

#             std_tpr = np.std(tprs, axis=0)
#             tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#             tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#             ax[i].fill_between(
#                 mean_fpr,
#                 tprs_lower,
#                 tprs_upper,
#                 color="grey",
#                 alpha=0.2,
#                 label=r"$\pm$ 1 std. dev.",
#             )

#             ax[i].set(
#                 xlabel="False Positive Rate",
#                 ylabel="True Positive Rate",
#                 title=['LogReg', 'SVC'][i],
#             )
#             ax[i].legend(loc="lower right")

#         plt.show()    

#         return metric_dict_Log, metric_dict_SVC, test_idx, plt

    



# Run the classificaton on X and y with the train test splits
def run_clf_on_splits(X, y, train_test_splits, sim_split, emb_name, RF=None, log=None, pipe_ls = [('scaler', StandardScaler())]):
    
    # Instantiate class 
    clf = Specificity_classificiation_model_evaluation(X, y, pipe_ls = pipe_ls)
    
    # save results
    results = pd.DataFrame()

    for splits, cluster_m in zip(train_test_splits, ['levenshtein_sequence_based', 'random_split']):

        if log is not None: 
            log.info(f'Fit models on {cluster_m} method')
        else: 
            print(f'Fit models on {cluster_m} method')

        # test sequence based splitting
        metric_dict_Log, metric_dict_SVC, test_idx = clf.evaluate_clf(X, y, train_id_ls=splits[0], test_id_ls=splits[1], 
                                                            RF=RF, log=None, verbose=1)

        
        if RF is None: 
            model1 = 'LogReg'
            model2 = 'kSVC'
        else:
            model1 = 'RF'
            model2 = 'GBoost'

        if log is not None: 
            log.info(f'\n{model1} - MCC: ', np.mean(metric_dict_Log['MCC']), np.std(metric_dict_Log['MCC']))
        else: 
            print(f'\n{model1} - MCC: ', np.mean(metric_dict_Log['MCC']), np.std(metric_dict_Log['MCC']))
        
        if log is not None: 
            log.info(f'\n{model2} - MCC: ', np.mean(metric_dict_SVC['MCC']), np.std(metric_dict_SVC['MCC']))
        else: 
            print(f'\n{model2} - MCC: ', np.mean(metric_dict_SVC['MCC']), np.std(metric_dict_SVC['MCC']))

        

            # add dataframe 
        for model, m_dict in zip([model1, model2], [metric_dict_Log, metric_dict_SVC]):
            row = len(results)
            results.loc[row, 'Model'] = f'{model}_{emb_name}' 
            results.loc[row, 'group_thresh'] = sim_split 
            results.loc[row, 'train_test_split'] = cluster_m 
            for m in m_dict.keys():

                mean = np.round(np.mean(m_dict[m]), 3)
                std = np.round(np.std(m_dict[m]), 3)
                # save results
                results.loc[row, m] = mean 
                results.loc[row, f'{m}_std'] = std 

    return results






# test()




# # define embedding names
# def test():
#     emb_n_list = ['AbMAP']#, 'ESM-2', 'ESM-2-CDRextract', 'ESM-2-augmented', '3-mer']

#     # define embedding list
#     emb_list = [Abmap_fl_embeddings]#, ESM_fl_embeddings, ESM_cdr_fl_embeddings, ESM_aug_fl_embeddings, kmer_arr]

#     results_l = []
#     for emb_name, emb in zip(emb_n_list, emb_list):
#         X = emb
#         log.info(f'Evaluate classifier on {emb_name} data')
#         results = run_clf_on_splits(X, y, train_test_splits, SIM_SPLIT, emb_name=emb_name,
#                                     pipe_ls = [('scaler', StandardScaler()), ('pca', PCA(n_components = 50))], log=log)
#         results_l.append(results)


#     log.info(f'Evaluation done')










######## MAIN ########


# def run():
#     # pass

#     # set random seed
#     random.seed(123)
#     today = str(datetime.now().date())
#     log = utils.Logger(__name__, log_file=f'test{today}_app.log').get_logger()
#     log.info('Start Script!')

#     # for ab_chain in ['VDJ_VJ_aaSeq']:#, 'VDJ_aaSeq']:
#     ab_chain = 'VDJ_VJ_aaSeq'

#     if ab_chain == 'VDJ_aaSeq': 
#         c_type = 'VH'
#         filter_192 = False
#     elif ab_chain == 'VDJ_VJ_aaSeq':
#         c_type = 'VH_VL'
#         filter_192 = True


#     # setup parser for the config file
#     config = configparser.ConfigParser()
#     config.read(CONFIG_PATH)
#     ROOT_DIR = config['ROOT']['ROOT_DIR']


#     ######## LOADING CLASS ########
#     try:
#         Embeddings = lec.LoadEmbeddings_VH_VL(CONFIG_PATH=CONFIG_PATH, seq_col=ab_chain, filter_192 = filter_192,
#                                             filter_VH_complete = False)
        
#         Embeddings.load_embeddings(embedding_type = 'antiberty', verbose=False)
#         # Seq column name 'VDJ_aaSeq', 'VDJ_aaSeqCDR3', 'cdr_comb'...
#         seq_col = Embeddings.seq_col
        

#         ### Load mAb sequences
#         seq_df = Embeddings.seq_df
#         names = Embeddings.names
#         seqs = Embeddings.seqs


# #             ### AbMAP  - VH_VL
# #             Abmap_fl_embeddings = Embeddings.emb_AM
# #             log.info("AbMap - embeddings loaded")


# #             ### Load embeddings - ESM2 - VH_VL
# #             ESM_fl_embeddings = Embeddings.emb_ESM
# #             log.info("ESM - embeddings loaded")


# #             ### Load embeddings - ESM2 augmented - VH_VL
# #             ESM_aug_fl_embeddings = Embeddings.emb_ESM_aug
# #             log.info("ESM augmented - embeddings loaded")


# #             ### Load embeddings - ESM2 CDRextract - VH_VL
# #             ESM_cdr_fl_embeddings = Embeddings.emb_ESM_cdrs
# #             log.info("ESM CDRextract - embeddings loaded")

        
#         ### Load embeddings - Antiberty - VH_VL
#         antiberty_embeddings = Embeddings.emb_antiberty
#         log.info("Antiberty - embeddings loaded")

#         # Calculate the kmer vectors
#         k=3
#         all_kmers = utils.generate_all_kmers(seqs, k)
#         vectors = [utils.freqs_to_vector(utils.kmer_frequencies(seq, k), all_kmers) for seq in seqs]
#         kmer_arr_3 = np.array(vectors)

#         # k=2
#         # all_kmers = utils.generate_all_kmers(seqs, k)
#         # vectors = [utils.freqs_to_vector(utils.kmer_frequencies(seq, k), all_kmers) for seq in seqs]
#         # kmer_arr_5 = np.array(vectors)           
#         # log.info("kmer embeddings calculated")


#         # Load sequence distance matrix
#         distance_matrix = Embeddings.dist_matrix
#         log.info("distance matrix loaded")

#     except Exception as e:
#         log.exception(f'ERROR Loading files: {e}')




#     ######## TRAIN TEST SPLITS ########
#     try:
#         # create train test splits - sequence clustering
#         N_SPLITS=5
#         SIM_SPLIT = 0.05
#         X = antiberty_embeddings
#         y = np.array(seq_df['group_id'])

#         # best splits are created with N_SPLITS=6 (based on manual inspection of train test splits)
#         train_ls, test_ls, clusters = train_test_split_idx(X, y, cluster_thresh=SIM_SPLIT, distance_matrix=distance_matrix, 
#                                                         n_splits=N_SPLITS, cluster_method= 'levenshtein_sequence_based',
#                                                         verbose=0)
#         # # manually remove an item for bad split
#         # test_ls.pop(4)
#         # train_ls.pop(4)
#         print(f'Sequence-based clustering with {SIM_SPLIT} cluster threshold')
#         for i in range(len(test_ls)):
#             print(len(test_ls[i]))
#             print(np.unique(y[test_ls[i]], return_counts=True)[1])


#         # create train test splits - Random split
#         train_ls_rd, test_ls_rd, _ = train_test_split_idx(X, y, cluster_thresh=SIM_SPLIT, n_splits=N_SPLITS, cluster_method= 'random_split',
#                                                         verbose=0)
#         print(f'Random split')
#         # test_ls_rd.pop(4)
#         # train_ls_rd.pop(4)
#         for i in range(len(test_ls_rd)):
#             print(len(test_ls_rd[i]))
#             print(np.unique(y[test_ls_rd[i]], return_counts=True)[1])


#         # Summarize train test split
#         train_test_splits = [[train_ls, test_ls], [train_ls_rd, test_ls_rd]]

#         log.info(f'Train test splits based on sequence-based clustering with {SIM_SPLIT} threshold prepared')

#     except Exception as e:
#         log.exception(f'ERROR TRAIN-TEST Splits: {e}')
    

#     ########### RUN CLASSIFICATION ###########
#     try:
#         # define embedding names
#         emb_n_list = [#'AbMAP', 
#                     #   'ESM-2', 'ESM-2-CDRextract', 
#                     #   'ESM-2-augmented', 
#                         '3-mer', #'5-mer',
#                         'Antiberty']

#         # define embedding list
#         emb_list = [# Abmap_fl_embeddings, 
#             # ESM_fl_embeddings,ESM_cdr_fl_embeddings, ESM_aug_fl_embeddings, 
#             kmer_arr_3, #kmer_arr_5, 
#             antiberty_embeddings]

#         results_l = []
#         for n, pipes in zip(['', '_pca'], [[('scaler', StandardScaler())], [('scaler', StandardScaler()), ('pca', PCA(n_components = 50))]]):
#             for emb_name, emb in zip(emb_n_list, emb_list):
#                 e = f'{emb_name}{n}'
#                 X = emb
#                 log.info(f'Evaluate classifier on {e} data')
#                 results = run_clf_on_splits(X, y, train_test_splits, SIM_SPLIT, emb_name=e, RF=True,
#                                             pipe_ls = pipes, log=log)
#                 results_l.append(results)

#                 # save intermediate result
#                 results = pd.concat(results_l)
#                 results.to_csv(os.path.join(ROOT_DIR, f'data/model_evaluation/Specificity_classification/{today}_{c_type}_Spec_classification_CV_results.csv'), index=False)

#         log.info(f'Evaluation done')

#     except Exception as e:
#         log.exception(f'ERROR Running classifier {emb_name}: {e}')

#     ########### SAVE RESULTS ###########


#     # results = pd.concat(results_l)
#     # results.to_csv(os.path.join(ROOT_DIR, f'data/model_evaluation/Specificity_classification/{today}_{c_type}_Spec_classification_CV_results.csv'), index=False)

#     # log.info(f'Results saved in {os.path.join(ROOT_DIR, f"data/model_evaluation/Specificity_Classification/{today}_{c_type}_Spec_classification_CV_results.csv")}')






def main():
    pass




if __name__ == '__main__':
    main()