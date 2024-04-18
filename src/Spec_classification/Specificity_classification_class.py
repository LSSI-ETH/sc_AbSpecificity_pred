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
# CONFIG_PATH = parser.parse_args().config 
UTILS_DIR = '../'
UTILS_DIR1 = './'
sys.path.append(UTILS_DIR)
sys.path.append(UTILS_DIR1)

# import custom modules
import utils_nb as utils
import Load_embs_class as lec



from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
# clustering
from scipy.cluster.hierarchy import linkage, fcluster
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







def main():
    pass




if __name__ == '__main__':
    main()