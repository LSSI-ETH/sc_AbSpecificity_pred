### UTILITIES FOR GP-LLM project

import numpy as np
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pickle
import seaborn as sns
import abmap


from sklearn import decomposition
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, train_test_split, GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, pairwise_distances, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef 
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC


import warnings, random, logging
from scipy.stats import pearsonr, spearmanr
import umap
from sklearn.cluster import DBSCAN
from Bio import SeqIO

########################################################################
### Basics
########################################################################

def read_fasta_to_list(filename):
    sequences = []
    seq_ids = []
    for record in SeqIO.parse(filename, "fasta"):
        
        sequence_id = record.id
        seq_ids.append(sequence_id)
        sequences.append(str(record.seq))
    return seq_ids, sequences







def load_pickle_embeddings(names, inputPath, file_suffix = '', verbose=False):
    loaded_data = []

    for file_name in names:
        try:
            with open(os.path.join(inputPath, f'{file_name}{file_suffix}.p'), 'rb') as file:
                data = pickle.load(file)
                loaded_data.append(data)
                if verbose is True: print(f"Loaded {file_name}")
        except FileNotFoundError:
            print(f"File {file_name} not found.")
        except Exception as e:
            print(f"An error occurred while loading {file_name}: {str(e)}")
    
#     embeddings = [x.numpy() for x in loaded_data]
    return loaded_data

def load_pickle_embeddings_wo_p(names, inputPath, verbose=False):
    loaded_data = []

    for file_name in names:
        try:
            with open(os.path.join(inputPath, f'{file_name}'), 'rb') as file:
                data = pickle.load(file)
                loaded_data.append(data)
                if verbose is True: print(f"Loaded {file_name}")
        except FileNotFoundError:
            print(f"File {file_name} not found.")
        except Exception as e:
            print(f"An error occurred while loading {file_name}: {str(e)}")
    
#     embeddings = [x.numpy() for x in loaded_data]
    return loaded_data


### load pickle embeddings of ESM model for heavy and light chains: 
# - take mean over tokens --> fixed-length embeddings
# - and append fixed-length embedding of VH and VL antibody sequence

def load_pickle_embeddings_VH_VL(names, inputPath, embedding_type = 'fixed', file_suffix = '', verbose=False):
    loaded_data = []
    names_VL = [str(n) + f'_L{file_suffix}.p' for n in names]
    names_VH = [str(n) + f'_H{file_suffix}.p' for n in names]

    for file_name in range(len(names)):
        try:
            # open VH
            with open(os.path.join(inputPath, f'{names_VH[file_name]}'), 'rb') as file:
                data_VH = pickle.load(file)

                if verbose is True: print(f"Loaded {names_VH[file_name]}")

            # open VL embedding
            with open(os.path.join(inputPath, f'{names_VL[file_name]}'), 'rb') as file:
                data_VL = pickle.load(file)

                if verbose is True: print(f"Loaded {names_VL[file_name]}")


            # concatenate ESM VH and VL embeddings for fixed length
            if embedding_type == 'fixed':
                data_VH = data_VH.mean(0)
                data_VL = data_VL.mean(0)
                data = np.concatenate((data_VH, data_VL))
                loaded_data.append(data)

            if embedding_type == 'var':
                data = [data_VH, data_VL]
                loaded_data.append(data)

        except FileNotFoundError:
            print(f"File {names_VH[file_name]} not found.")
        except Exception as e:
            print(f"An error occurred while loading {names_VH[file_name]}: {str(e)}")

    
    return loaded_data



### TAKES THE MEAN OVER HEAVY AND LIGHT CHAIN EMBEDDING & RETURNS THE FIXED-LENGTH EMBEDDING
###

def mean_over_HL(embeddings):

    # take mean over HL and var embeddings to get fixed length embeddings
    fl_embeddings = []

    for i, emb in enumerate(embeddings):
        # vstack H and L embeddings and take mean over embeddings for a fixed lenght
        fl_emb_chains = np.vstack((emb[0], emb[1]))

        fl_embeddings.append(fl_emb_chains.mean(0))

    return np.array(fl_embeddings)



# combination of functions for calculating kmer frequencies
def kmer_frequencies(seq, k):
    freqs = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        freqs[kmer] = freqs.get(kmer, 0) + 1 # add count for that kmer
    # Normalize by the total number of k-mers
    total_kmers = sum(freqs.values())
    for kmer in freqs:
        freqs[kmer] /= total_kmers
    return freqs

def freqs_to_vector(freqs, all_kmers):
    return [freqs.get(kmer, 0) for kmer in all_kmers]

# Generate a list of all possible k-mers from sequences
def generate_all_kmers(sequences, k):
    all_kmers = set()
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            all_kmers.add(seq[i:i+k])
    return sorted(list(all_kmers))

def compute_distance_matrix(sequences, k):
    all_kmers = generate_all_kmers(sequences, k)
    vectors = [freqs_to_vector(kmer_frequencies(seq, k), all_kmers) for seq in sequences]
    distance_matrix = euclidean_distances(vectors)
    return distance_matrix






class Logger:
    def __init__(self, name, log_file='app.log'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger


########################################################################
### GENERATE EMBEDDINGS WITH ESM, ESM-AUG AND AbMAP - CUSTOM FUNCTIONS
########################################################################


# prep data for ESM Embedding
def generate_ESM_embedding(df, seq_column_HC='VDJ_aaSeq', seq_column_LC='VJ_aaSeq', 
                                k = 50, model_typ = 'esm2', 
                                out_folder = 'embeddings/', VH_only = False, save_plm=True, save_PLM_aug=True,
                                cuda_dev_num = 0): 
    '''
    Function for generating ESM embeddings and augmenting them with contrastive augmentation. 
    Also saves the variable length embeddings in out_folder;
    
    params:
            df: pd.DataFrame() with sequences of the heavy and light chains in seq_column_HC & seq_column_LC
            seq_column_HC: str, column name in df of the heavy chain sequences
            seq_column_LC: str, column name in df of the light chain sequences
            k: int, number of in-silico mutants for contrastive CDR augmentation
            model_typ: str, foundational PLM to use (supported by AbMAP)
    
    '''
    
    esm_embeddings_VH_VL = {}
    aug_esm_embeddings_VH_VL = {}
    emb_ids = []
    ids_to_drop = []


    if VH_only == True: 
        print("VH only")

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):

            seq = row['VDJ_aaSeq']
            chain_type = 'H'
            
            prot_embed = abmap.ProteinEmbedding(seq, chain_type, dev=cuda_dev_num)
            p_id = row['seq_id']+'_k{}.p'.format(k)

            # create folder for esm embeddings
            for model_typ in ['esm2']:
                
                # create paths if not existent yet
                out_path_PLM = os.path.join(out_folder, model_typ + '_VH/')
                if not os.path.isdir(out_path_PLM):
                    os.makedirs(out_path_PLM)

                out_path_aug = os.path.join(out_folder, model_typ + '_aug_'+str(k)+'_VH/')
                if not os.path.isdir(out_path_aug):
                    os.makedirs(out_path_aug)

                #try:
                prot_embed.embed_seq(embed_type = model_typ)

                # add embedding to dict
                file_name = '{}.p'.format(row['seq_id'])

                # add embedding to dict
                esm_embeddings_VH_VL[file_name] = prot_embed.embedding.cpu().numpy()



                if save_plm is True: 
                    with open(os.path.join(out_path_PLM, file_name), 'wb') as fh:
                        print("Saving", row['seq_id'])
                        pickle.dump(prot_embed.embedding.cpu().numpy(), fh)



                prot_embed.create_cdr_mask()
                kmut_matr = prot_embed.create_kmut_matrix(num_muts=k, embed_type=model_typ)
                cdr_embed = prot_embed.create_cdr_embedding(kmut_matr, sep = False, mask = True)

                # save ids and embeddings as list for AbMap embeddings
                emb_ids.append(p_id)
                aug_esm_embeddings_VH_VL[file_name] = cdr_embed


                with open(os.path.join(out_path_aug, p_id), 'wb') as f:
                    pickle.dump(cdr_embed.cpu().numpy(), f)

                # except:
                #     ids_to_drop.append(index)


    # generate VH_VL embeddings
    else:
        print("VH_VL")
        
        
        # create folder for esm embeddings
        if save_plm == True:
            out_path_PLM = os.path.join(out_folder, model_typ + '_VH_VL')
            if not os.path.isdir(out_path_PLM):
                os.makedirs(out_path_PLM)

        if save_PLM_aug == True:
            out_path_aug = os.path.join(out_folder, model_typ + '_aug_'+str(k)+'_VH_VL')
            if not os.path.isdir(out_path_aug):
                os.makedirs(out_path_aug)
        
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            seq_h, seq_l = row[seq_column_HC], row[seq_column_LC]

            for seq, chain_type in [(seq_h, 'H'), (seq_l, 'L')]:
                
                # create instance and file name
                prot_embed = abmap.ProteinEmbedding(seq, chain_type, dev=cuda_dev_num)
                file_name = '{}_{}.p'.format(row['seq_id'], prot_embed.chain_type)

                        
                # Generate the embeddings
                try:
                    prot_embed.embed_seq(embed_type = model_typ)
                    
                    # add embedding to dict
                    esm_embeddings_VH_VL[file_name] = prot_embed.embedding.cpu().numpy()

                    if save_plm is True: 
                        with open(os.path.join(out_path_PLM, file_name), 'wb') as fh:
                            abmap.utils.log(f'Saving {row["seq_id"]} {chain_type}')
                            pickle.dump(prot_embed.embedding.cpu().numpy(), fh)
                    
                    
                    # augment the PLM embeddings with AbMAP
                    prot_embed.create_cdr_mask()
                    kmut_matr = prot_embed.create_kmut_matrix(num_muts=k, embed_type=model_typ)
                    cdr_embed = prot_embed.create_cdr_embedding(kmut_matr, sep = False, mask = True)

                    # save ids and embeddings as dict for AbMap embeddings
                    emb_ids.append(file_name)
                    aug_esm_embeddings_VH_VL[file_name] = cdr_embed

                    
                    if save_PLM_aug is True: 
                        with open(os.path.join(out_path_aug, file_name), 'wb') as f:
                            pickle.dump(cdr_embed.cpu().numpy(), f)

                except:
                    ids_to_drop.append(index)


    abmap.utils.log("# ESM & ESM augmented embedding done!")
    return emb_ids, [esm_embeddings_VH_VL, aug_esm_embeddings_VH_VL], ids_to_drop



def generate_AbMAP_embedding(df, aug_esm_embeddings_VH_VL, ids_to_drop, seq_column_HC='VDJ_aaSeq', seq_column_LC='VJ_aaSeq', 
                                model_typ='AbMap', pretrained_path_H_ls = [], embed_type='fixed', VH_only = False, k = 50,
                                out_folder = 'embeddings/', save_embedding=True,
                                dev = 'cuda:0', cuda_dev_num=0): 
    '''
    Function for generating ESM embeddings and augmenting them with contrastive augmentation. 
    Also saves the variable length embeddings in out_folder;
    
    params:
            df: pd.DataFrame() with sequences of the heavy and light chains in seq_column_HC & seq_column_LC
            seq_column_HC: str, column name in df of the heavy chain sequences
            seq_column_LC: str, column name in df of the light chain sequences
            aug_esm_embeddings_VH_VL: dict, dictionary of the augmented PLM embeddings; output of generate_ESM_embedding
            model_typ: str, just a name for creating a folder
    
    '''
    
    if VH_only == True:
        out_path = os.path.join(out_folder, f'{model_typ}_k_{str(k)}_VH_{embed_type}')

    if VH_only == False:
        out_path = os.path.join(out_folder, f'{model_typ}_k_{str(k)}_VH_VL_{embed_type}')
        
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    abmap_emb_H_L_fl = {}
    
    # Load models
    abmap_H = abmap.load_abmap(pretrained_path=pretrained_path_H_ls[0], device=cuda_dev_num, plm_name='esm2')
    
    if VH_only == False:
        abmap_L = abmap.load_abmap(pretrained_path=pretrained_path_H_ls[0], device=cuda_dev_num, plm_name='esm2')
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        # pass, if in invalid sequences
        if index in ids_to_drop: continue

        seq_h, seq_l = row[seq_column_HC], row[seq_column_LC]
        
        # create file names
        if VH_only == True:
            p_id_H = '{}.p'.format(row['seq_id'])
        elif VH_only == False:
            p_id_H = '{}_{}.p'.format(row['seq_id'], 'H')
            p_id_L = '{}_{}.p'.format(row['seq_id'], 'L')
            
            
            
        # generate the HC embedding
        z_H = torch.unsqueeze(aug_esm_embeddings_VH_VL[p_id_H], dim=0).to(dev)
        with torch.no_grad():
            emb_H = abmap_H.embed(z_H, embed_type=embed_type)
            emb_H = torch.squeeze(emb_H, dim=0)
            
            
        # if VH and VL to be processed embed VL
        if VH_only == False:
            
            # generate the LC embedding
            z_L = torch.unsqueeze(aug_esm_embeddings_VH_VL[p_id_L], dim=0).to(dev)
            with torch.no_grad():
                emb_L = abmap_L.embed(z_L, embed_type=embed_type)
                emb_L = torch.squeeze(emb_L, dim=0)

            # concatentate the embeddings and save if 'fixed'
            if embed_type == 'fixed': 
                emb_H_L = torch.cat((emb_H[:256], emb_L[:256]), dim=-1)
                # add VH_VL embedding to dict
                abmap_emb_H_L_fl[f'{row["seq_id"]}'] = emb_H_L.cpu().numpy()
            
            elif embed_type == 'variable':
                abmap_emb_H_L_fl[f'{row["seq_id"]}_H'] = emb_H.cpu().numpy()
                abmap_emb_H_L_fl[f'{row["seq_id"]}_L'] = emb_L.cpu().numpy()

                
        # if only VH add embedding to dict
        if VH_only == True: 
            emb_H_L = emb_H[:256]
            abmap_emb_H_L_fl[row['seq_id']] = emb_H_L.cpu().numpy()
        
        
        if save_embedding == True:
            # if fixed save one embedding per sequence
            if embed_type == 'fixed': 
                fname = os.path.join(out_path, f'{row["seq_id"]}.p')  
                # save the abmap embedding
                with open(fname, 'wb') as f:
                    pickle.dump(emb_H_L.cpu().numpy(), f)
            
            # if fixed save one embedding per sequence
            elif embed_type == 'variable':
                if VH_only == False:
                    for ct, e in zip(['H', 'L'], [emb_H, emb_L]):
                        fname = os.path.join(out_path, f'{row["seq_id"]}_{ct}.p')  
                        # save the abmap embedding
                        with open(fname, 'wb') as f:
                            pickle.dump(e.cpu().numpy(), f)
                
                
                else:
                    fname = os.path.join(out_path, f'{row["seq_id"]}.p')  
                    # save the abmap embedding
                    with open(fname, 'wb') as f:
                        pickle.dump(emb_H.cpu().numpy(), f)
                    
                    
                    

    abmap.utils.log("# AbMap embedding done!")  
    
    return abmap_emb_H_L_fl



########################################################################
### GENERATE EMBEDDINGS WITH ANTIBERTY - CUSTOM FUNCTIONS
########################################################################


def generate_Antiberty_Seq_embedding(seq_HL, name, antiberty,
                                 out_folder = 'embeddings/', save_plm=True): 

    '''
    Function for generating Anitberty embeddings and saving them to a folder. Generates variable length embeddings of heavy and light chains in out_folder;

    params:
    seq_HL: list of 2 str, sequences of the heavy and light chains 
    name: str, seq_id of the sequence
    antiberty: loaded AntiBERTyRunner() object
    out_folder: str, path to the folder where the embeddings should be saved
    '''

    ids_to_drop = []

    try:
        # embed the sequences
        embeddings_r = antiberty.embed(seq_HL)
        embeddings = [embeddings_r[0][1:-1,:].cpu().numpy(), embeddings_r[1][1:-1,:].cpu().numpy()]

        # create folder for esm embeddings
        if save_plm == True:
            out_path_PLM = os.path.join(out_folder)
            if not os.path.isdir(out_path_PLM):
                os.mkdir(out_path_PLM)

            # save the embeddings
            for embedding, chain_type in zip([embeddings[0], embeddings[1]], ['H', 'L']):
                file_name = '{}_{}.p'.format(name, chain_type)
                #print(os.path.join(out_path_PLM, file_name))

                with open(os.path.join(out_path_PLM, file_name), 'wb') as fh:
                    pickle.dump(embedding, fh)

    except:
        ids_to_drop.append(name)
    
    return embeddings, ids_to_drop


########################################################################
### PCA and PLOT function
########################################################################
# functions used in 003.1_CamSol_Regression_Embeddings.ipynb

def run_pca(fl_embeddings: np.array, n_comps: int = 2): 
    # Run PCA 
    pca = decomposition.PCA(n_components=n_comps)
    pca.fit(fl_embeddings)
    PCs = pca.transform(fl_embeddings)
    
    return(PCs)



def pca_plot(fl_embeddings: np.array, seq_df: pd.DataFrame, camsol_raw: pd.DataFrame = None, sample_ids: list = None, group_ids: list = None,
             n_comps: int = 2, n_comps_plot: tuple = (0,1), 
             model_name: str = "", plot_subtype = 'all'): 
    '''
    Custom PCA plots for Analysis of the OVA datasets
        param:
            fl_embeddings: np.array() of shape (n_samples, n_dim_embeddings) sequence based fixed-length embeddings
            n_comps: int, number of components to reduce the embeddings to with PCA.
            n_comps_plot: tuple, idxs of the PCs to plot on 2d scatter plot
            camsol_raw pd.dataframe: raw CamSol data loaded as pandas dataframe 
    
    
    '''
    
    # Run PCA 
    PCs = run_pca(fl_embeddings, n_comps = 2)
    
    # Setup of the Plots
    #if plot_subtype == 'all': dim = 2 else: dim = 1
        
    fig, axs = plt.subplots(2,2)
    fig.set_figwidth(14)
    fig.set_figheight(10)

    
    # Plot the with camsol
    if plot_subtype == 'camsol' or plot_subtype == 'all':
    
        # plot with CamSol
        points = axs[0,0].scatter(PCs[:,n_comps_plot[0]], PCs[:,n_comps_plot[1]], c=camsol_raw['protein variant score'], 
                                s=8, cmap="plasma")
        axs[0,0].set_title("PCA - {} Embeddings\nCamSol-intrinsic".format(model_name))
        fig.colorbar(points)


        axs[0,1].scatter(PCs[camsol_raw['protein variant score'] > 0][:,0], PCs[camsol_raw['protein variant score'] > 0][:,1], 
                       s=8, label='> 0')
        axs[0,1].scatter(PCs[camsol_raw['protein variant score'] <= 0][:,0], PCs[camsol_raw['protein variant score'] <= 0][:,1], 
                       s=8, label='<= 0')
        axs[0,1].legend(loc="upper right")

        plt.title("PCA - {} embeddings\nCamSol-intrinsic".format(model_name))
        
        
    # Plot the by group 
    if plot_subtype == 'groups' or plot_subtype == 'all':
            
        axs[1, 0].scatter(PCs[seq_df["group_id"] == group_ids[0]][:,n_comps_plot[0]], PCs[seq_df["group_id"] == group_ids[0]][:,n_comps_plot[1]], 
                          c='orange', s=15, alpha = 0.2, label='non-spec')
        axs[1, 0].scatter(PCs[seq_df["group_id"] == group_ids[1]][:,n_comps_plot[0]], PCs[seq_df["group_id"] == group_ids[1]][:,n_comps_plot[1]], 
                          c='purple', alpha = 0.5, s=15, label='spec')
        axs[1, 0].legend(loc="upper right")
        axs[1, 0].set_title("PCA - {} embeddings\ngroup_id".format(model_name))

        # plot samples_ids 
        samples_ids = np.unique(seq_df.sample_id)

        axs[1, 1].scatter(PCs[seq_df["sample_id"] == samples_ids[0]][:,n_comps_plot[0]], PCs[seq_df["sample_id"] == samples_ids[0]][:,n_comps_plot[1]], s=8, label='s2')
        axs[1, 1].scatter(PCs[seq_df["sample_id"] == samples_ids[1]][:,n_comps_plot[0]], PCs[seq_df["sample_id"] == samples_ids[1]][:,n_comps_plot[1]], s=8, label='s4')
        axs[1, 1].scatter(PCs[seq_df["sample_id"] == samples_ids[2]][:,n_comps_plot[0]], PCs[seq_df["sample_id"] == samples_ids[2]][:,n_comps_plot[1]], s=8, label='s1')
        axs[1, 1].scatter(PCs[seq_df["sample_id"] == samples_ids[3]][:,n_comps_plot[0]], PCs[seq_df["sample_id"] == samples_ids[3]][:,n_comps_plot[1]], s=8, label='s3')
        axs[1, 1].legend(loc="upper right")

        axs[1, 1].set_title("PCA - {} embeddings\nsample_id".format(model_name))
        
    fig.show()

    


def run_UMAP(X: np.array, y=None, scaler = None, n_neighbors: int = 15, min_dist: float = 0.1, verbose: int = 0):

    if scaler is not None: X = scaler.fit_transform(X)
        
    manifold = umap.UMAP(random_state=42, n_neighbors=n_neighbors, min_dist=min_dist).fit(X, y)
    X_reduced = manifold.transform(X)

    return X_reduced
    
    
    
def UMAP_emb_plots(X_reduced: np.array, camsol_raw: pd.DataFrame, seq_df: pd.DataFrame, 
                   sample_ids: list, group_ids: list,
                   model_name: str = ""):


    fig, axs = plt.subplots(2,2)
    fig.set_figwidth(14)
    fig.set_figheight(10)


    axs[0,0].scatter(X_reduced[:, 0][seq_df["sample_id"] == sample_ids[0]], X_reduced[:, 1][seq_df["sample_id"] == sample_ids[0]], c='lightgrey', s=20, label='s4', alpha=0.2)
    axs[0,0].scatter(X_reduced[:, 0][seq_df["sample_id"] == sample_ids[1]], X_reduced[:, 1][seq_df["sample_id"] == sample_ids[1]], c='lightgrey', s=20, label='s2', alpha=0.2)
    axs[0,0].scatter(X_reduced[:, 0][seq_df["sample_id"] == sample_ids[2]], X_reduced[:, 1][seq_df["sample_id"] == sample_ids[2]], c='y', s=20, label='s1', alpha=0.5)
    axs[0,0].scatter(X_reduced[:, 0][seq_df["sample_id"] == sample_ids[3]], X_reduced[:, 1][seq_df["sample_id"] == sample_ids[3]], c='r', s=20, label='s3', alpha=0.5)
    axs[0,0].legend()
    axs[0,0].set_title("UMAP - {} embeddings\nsample_id".format(model_name))

    axs[0,1].scatter(X_reduced[:, 0][seq_df["group_id"] == group_ids[0]], X_reduced[:, 1][seq_df["group_id"] == group_ids[0]], c='orange', s=20, label='nonspec', alpha=0.2)
    axs[0,1].scatter(X_reduced[:, 0][seq_df["group_id"] == group_ids[1]], X_reduced[:, 1][seq_df["group_id"] == group_ids[1]], c='purple', s=20, label='spec', alpha=0.5)
    axs[0,1].legend()
    axs[0,1].set_title("UMAP - {} embeddings\ngroup_id".format(model_name))

    points = axs[1,0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=camsol_raw['protein variant score'], s=10,
                         cmap="plasma")
    fig.colorbar(points)
    axs[1,0].set_title("UMAP - {} embeddings\nCamSol".format(model_name))

    axs[1,1].scatter(X_reduced[camsol_raw['protein variant score'] > 0][:,0], X_reduced[camsol_raw['protein variant score'] > 0][:,1], 
                   s=10, label='> 0')
    axs[1,1].scatter(X_reduced[camsol_raw['protein variant score'] <= 0][:,0], X_reduced[camsol_raw['protein variant score'] <= 0][:,1], 
                   s=10, label='<= 0')
    axs[1,1].set_title("UMAP - {} embeddings\nCamSol binned".format(model_name))
    axs[1,1].legend()


    plt.show()





    
########################################################################
### REGRESSION TESTING
########################################################################
### for 003.1_CamSol_Regression_Embeddings

### Parameter tuning setup
def reg_param_tuning(X_train, y_train, pipeline, param_grid, metrics,k =5, n_jobs=-1, timed = True, verbose=0):
    # setup k fold
    kf = KFold(n_splits=k, shuffle=True, random_state=1)  # Define the n_split = number of folds

    if timed is True: tic = time.perf_counter()    
    
    # Define the grid search object - non-nested
    grid_search = GridSearchCV(estimator= pipeline,
                               param_grid=param_grid,
                               scoring=metrics,
                               cv=kf,
                               refit=list(metrics.keys())[1], 
                               n_jobs=n_jobs, verbose=1)


    # Non_nested parameter search and scoring
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    
    if verbose > 0: 
        print('Best model: ', best_model)
        print('Best score: ', np.round(best_score, 4))
    
    if timed is True: 
        t = time.perf_counter() - tic
        if t > 60: 
            print(f"Param tuning done in {t/60:0.2f} minutes")
        else: 
            print(f"Param tuning done in {t:0.2f} seconds")
        
    return(grid_search)


### Correlation plot
def regression_correlation_plot(y_test: np.array, y_pred: np.array, title: str = "Correlation plot",
                               print_metrics: bool = True):
    # Plot Correlation
    
    if y_test.shape != (-1, ):# reshape
        y_test = y_test.reshape(-1, )
    if y_pred.shape != (-1, ):
        y_pred = y_pred.reshape(-1, )

    # Plot the correlation of predicted values 
    fig, axs = plt.subplots()
    # fig.set_figwidth(15)
    points = axs.scatter(y_test, y_pred, s=8)
    axs.set_title(title)
    axs.set_xlabel("True values")
    axs.set_ylabel("Predictions")
    
    # correlation 
    spear = spearmanr(y_test, y_pred)[0]
    pears = pearsonr(y_test, y_pred)[0]
    
    # define legend
    l0 = "PearsonCorr = {:.3f}".format(pears)
        
    leg = plt.legend(labels = [l0], handlelength=0, handletextpad=0,
                 loc = 4)
    for item in leg.legendHandles:
        item.set_visible(False)
    
    plt.show()
    
    if print_metrics is True:
        print('R2: ', np.round(r2_score(y_test, y_pred), 3))
        print('MSE ', np.round(mean_squared_error(y_test, y_pred), 3))

        print('Spearman Correlation coef: {}'.format( np.round(spear, 3 )))
        print('Pearson Correlation coef: {}'.format( np.round(pears, 3 )))


        
# Summarizes reg_param_tuning function with KRR
def tune_predict_KRR(X_train, X_test, y_train, y_test, k): 

    # setup parameters
    scaler = StandardScaler()
    reg = KernelRidge()
    pipeline = Pipeline([
        ('scaler', scaler),
        ('regressor', reg)
    ])
    metrics = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}
    # define parameter grid
    param_grid = {'regressor__alpha': [0.1, 1.0, 10.0, 50, 100],
                  'regressor__kernel': ['linear', 'rbf', 'polynomial'],
                  'regressor__degree': [2, 3, 4],
                  'regressor__gamma': [0.1, 1.0, 10.0]}


    # Perfrom hyperparameter tuning
    parma_tuned_GS = reg_param_tuning(X_train, y_train, pipeline, param_grid, metrics, k = k, n_jobs=-1, timed = True)
    # Obtain best model and predict values from test set
    best_model = parma_tuned_GS.best_estimator_
    y_pred = best_model.predict(X_test)
    
    return y_pred

# Summarizes reg_param_tuning function with GP
def tune_predict_GP(X_train, X_test, y_train, y_test, k):

    # setup parameters
    scaler = StandardScaler()
    reg = GaussianProcessRegressor()
    pipeline = Pipeline([
        ('scaler', scaler),
        ('regressor', reg)
    ])
    metrics = {'neg_MSE': 'neg_mean_squared_error', 'r2': 'r2'}

    # define parameter grid
    kernel_list = [RBF(l) for l in np.logspace(-1, 1, 3)]+[Matern(l) for l in np.logspace(-1, 1, 3)]
    param_grid = {'regressor__kernel': kernel_list,
                 'regressor__alpha': [1e-10, 1e-3, 0.1]
                 }

    # Perfrom hyperparameter tuning
    parma_tuned_GS = reg_param_tuning(X_train, y_train, pipeline, param_grid, metrics, k = k, n_jobs=-1, timed = True)
    # Obtain best model and predict values from test set
    best_model = parma_tuned_GS.best_estimator_
    y_pred = best_model.predict(X_test)

    return y_pred
   
    
# split train test set by clustering similar sequences
def grouped_train_test_split(X, y, test_size, group_thresh, verbose=1):
    sim = 1 - group_thresh 

    # Compute the pairwise distance matrix
    distance_matrix = pairwise_distances(X)


    # Use DBSCAN clustering
    dbscan = DBSCAN(eps=sim, min_samples=2, metric='precomputed')
    clusters = dbscan.fit_predict(distance_matrix)
    #print(np.unique(clusters, return_counts=True))
    
    # replace -1 as being assigned to no cluster
    for i, c in enumerate(clusters): 
        if c == -1: 
            clusters[i] = i + clusters[-1]


    # Split the data based on groups
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=123).split(X, groups=clusters))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if verbose > 0:
        print('y_test: ', len(y_test))
        
    return X_train, X_test, y_train, y_test        
                
        
### Correlation plot for 2 methods with legends
def regression_correlation_plots(y_test: np.array, y_preds: list, title: str = "Correlation plot", methods = ['KRR', 'GP'],
                               print_metrics: bool = True):
    # Plot Correlation
    
    if y_test.shape != (-1, ):# reshape
        y_test = y_test.reshape(-1, )


    # Plot the correlation of predicted values 
    fig, axs = plt.subplots(1,2)
    fig.set_figwidth(15)
    
    for i, (method, y_pred) in enumerate(zip(methods, y_preds)):
        if y_pred.shape != (-1, ):
            y_pred = y_pred.reshape(-1, )
        
        points = axs[i].scatter(y_test, y_pred, s=8)
        axs[i].set_title(title + " " + method)
        axs[i].set_xlabel("True values")
        axs[i].set_ylabel("Predictions")


        # correlation 
        spear = spearmanr(y_test, y_pred)[0]
        pears = pearsonr(y_test, y_pred)[0]

        # define legend
        l0 = "PearsonCorr = {:.3f}".format(pears)

        leg = axs[i].legend(labels = [l0], handlelength=0, handletextpad=0,
                     loc = 4, markerscale=0)

        
        
        # print 
        print(method)
        if print_metrics is True:
            print('R2: ', np.round(r2_score(y_test, y_pred), 3))
            print('MSE ', np.round(mean_squared_error(y_test, y_pred), 3))

            print('Spearman Correlation coef: {}'.format( np.round(spear, 3 )))
            print('Pearson Correlation coef: {}'.format( np.round(pears, 3 )))   
    
    plt.show()
    
 


# split train test set by clustering similar sequences
def return_grouped_train_test_split(X, y, test_size, group_thresh, n_splits=5, verbose=1):
    sim = 1 - group_thresh 

    # Compute the pairwise distance matrix
    distance_matrix = pairwise_distances(X)


    # Use DBSCAN clustering
    dbscan = DBSCAN(eps=sim, min_samples=2, metric='precomputed')
    clusters = dbscan.fit_predict(distance_matrix)
    #print(np.unique(clusters, return_counts=True))
    
    # replace -1 as being assigned to no cluster
    for i, c in enumerate(clusters): 
        if c == -1: 
            clusters[i] = i + clusters[-1]

    # just return the GroupShuffleSplit object
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=123)
    gss.get_n_splits()

        
    return gss, clusters    








####### updated grouped split classifier evaluation in k=5 fold CV
# logisitc regression 
def Log_reg_nosp(X_train, X_test, y_train, y_test, verbose=1):
    # Setup classifier
    clf = LogisticRegression(random_state=123)

    scaler = StandardScaler()

    pipe = Pipeline([
        ('scaler', scaler),
        ('clf', clf)
    ])


    metrics = {'AUC': 'roc_auc', 'f1': 'f1', 'recall': 'recall'}


    # define parameter grid
    param_grid = {'clf__penalty': ['l1', 'l2', None],
                  'clf__class_weight': ['balanced', None],
                  'clf__C': [1, 0.1, 0.01, 0.001],
                  #'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'clf__max_iter': [500, 1000, 2000]
                 }


    # Perfrom hyperparameter tuning
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring=metrics, refit='recall',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    
    if verbose >1:
        print("Best Hyperparameters:", best_params)
        print("Best score:", grid_search.best_score_)

    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test)

    # save metrics
    metric_dict = {}
    
    accuracy = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    metric_dict['accuracy'] = accuracy
    metric_dict['MCC'] = mcc
    metric_dict['precision'] = prec
    metric_dict['recall'] = rec

    if verbose >0:
        # print scores
        print("Test Accuracy:", np.round(accuracy, 3))
        print("Test MCC:", np.round(mcc, 3))
        print("Test Precision:", np.round(prec, 3))
        print("Test Recall:", np.round(rec, 3))
    
    return metric_dict


### SVC for the below wrapper
def SVC_eval_nosp(X_train, X_test, y_train, y_test, verbose=1):
    
    clf = SVC(probability=True, random_state=123)

    scaler = StandardScaler()

    pipe = Pipeline([
        ('scaler', scaler),
        ('clf', clf)
    ])


    metrics = {'AUC': 'roc_auc', 'f1': 'f1', 'recall': 'recall'}

    param_grid = {
        'clf__C': [1, 0.1, 0.01, 0.001],
        'clf__kernel': ['poly', 'rbf', 'sigmoid'],
        'clf__degree': [2, 3, 4],  # Only used when kernel is 'poly'
        'clf__gamma': ['scale', 'auto'] #+ list(np.logspace(-3, 3, 7))
    }

    # Perfrom hyperparameter tuning
    grid_search = RandomizedSearchCV(estimator=pipe, param_distributions=param_grid, cv=5, scoring=metrics, refit='recall',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    if verbose > 1:
        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)


    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    
        # save metrics
    metric_dict = {}
    
    accuracy = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    metric_dict['accuracy'] = accuracy
    metric_dict['MCC'] = mcc
    metric_dict['precision'] = prec
    metric_dict['recall'] = rec

    if verbose >0:
        # print scores
        print("Test Accuracy:", np.round(accuracy, 3))
        print("Test MCC:", np.round(mcc, 3))
        print("Test Precision:", np.round(prec, 3))
        print("Test Recall:", np.round(rec, 3))
    
    return metric_dict



#### Wrapper for evaluating Log_reg_nosp() & SVC_reg_nosp() classifier with grouped_split
# uses return_grouped_train_test_split clusters for iterating through n_splits shuffled train-test splits
def evaluate_clf(X, y, test_size, group_thresh, n_splits = 5):
    gss, clusters = return_grouped_train_test_split(X, y, test_size, group_thresh, n_splits=5, verbose=1)

    metric_dict_Log = {}
    for k in ['accuracy', 'MCC', 'precision', 'recall']:
        metric_dict_Log[k] = []

    metric_dict_SVC = {}
    for k in ['accuracy', 'MCC', 'precision', 'recall']:
        metric_dict_SVC[k] = []

    # iterate throught different reshuffeled group splits
    for train_idx, test_idx in gss.split(X, groups=clusters): 
    #     print(len(train_idx))
    #     print(len(test_idx))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        m_dict_l = Log_reg_nosp(X_train, X_test, y_train, y_test, verbose=0)
        m_dict_s = SVC_eval_nosp(X_train, X_test, y_train, y_test, verbose=0)

        for k in m_dict_l.keys():
            metric_dict_Log[k].append([m_dict_l[k]])
            metric_dict_SVC[k].append([m_dict_s[k]])
    
    return metric_dict_Log, metric_dict_SVC
        
    
    
########################################################################
### FUNCTIONS FOR SEQUENCE SIMILARITY BASED TRAIN TEST SPLITTING
########################################################################  
from Levenshtein import ratio as norm_dist

# Calculate levenshtein distance (normalized) matrix
def calc_norm_levens_dist(seqs: list, verbose=1):
    sim_matrix = np.ndarray((len(seqs), len(seqs)))
    for j in range(len(seqs)):
        
        if verbose > 0:
            if (j % 100 == 0):  print(j)
        
        LD_arr = []
        for i in range(len(seqs)):    
            LD_arr.append( norm_dist(seqs[j], seqs[i]) )
        
        # store distances in matrix
        sim_matrix[j,:] = LD_arr
        
    # return distance matrix
    dist_matrix = 1-sim_matrix
    return dist_matrix
    
    
    

    

        
 ########################################################################
### EMBEDDING SIMILARITY ANALYSIS - CUSTOM FUNCTIONS
########################################################################
# USED IN 002.2_Distance_analysis_emb_seq_space



###############


def cluster_map(data: np.array, linkage: np.array, figsize: tuple=(8,6),
                        row_cluster=True, col_cluster=True, 
                        row_colors =None, title: str = 'Pairwise distances'):
            
            
    cg = sns.clustermap(data=data, row_linkage=linkage, col_linkage=linkage, 
                        figsize=figsize,
                        row_cluster=row_cluster, col_cluster=col_cluster, 
                        row_colors=row_colors, cbar_pos=(0, .2, .03, .4))
    cg.ax_heatmap.set_title(title)
    cg.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
    cg.ax_col_dendrogram.set_visible(False) #suppress column dendrogram
    plt.title("Pairwise Euclidean distance")

    plt.show()


    
    
    def emb_scatter_plots(distance_matrix1, distance_matrix2, seq_df, x_lab='embedding dist1', y_lab='embedding dist2'): 
    
        # prepare the data
        df1 = np.tril(distance_matrix1, k=0)
        df1 = df1[df1 != 0.].ravel()
        df2 = np.tril(distance_matrix2, k=0)
        df2 = df2[df2 != 0.].ravel()

        # Plot the correlation of predicted values 
        fig, axs = plt.subplots(1,2)
        fig.set_figwidth(15)
        # Calculate the point density (for the plot)
        xy = np.vstack([df1,df2])
        z = gaussian_kde(xy)(xy)

        # plot with density 
        points = axs[0].scatter(df1, df2, c=z,
                             s=.5, alpha = 1)
        axs[0].set_title("Correlation plot of pairwise distances")
        axs[0].set_xlabel(x_lab)
        axs[0].set_ylabel(y_lab)
        #fig.colorbar(points)

        for i, j in zip([2, 1], ['nonspec', 'spec']):
            df1 = np.tril(distance_matrix1[seq_df['group_id'] == i])
            df1 = df1[df1 != 0.].ravel()

            df2 = np.tril(distance_matrix2[seq_df['group_id'] == i])
            df2 = df2[df2 != 0.].ravel()

            axs[1].scatter(df1, df2, s=.5, alpha = 1, label = j)

        axs[1].set_title("Correlation of pairwise distances - Specificity")
        axs[1].legend()
        plt.show()

        
        
        
        
        
        
        
        

