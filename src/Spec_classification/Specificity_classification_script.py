#!/usr/bin/env python3
#################################
# Script for classification models to predict Binding Specificity for OVA values from sequence embeddings
# This class is used in the classification pipeline to train and evaluate ML models.
# Author: Lena Erlach
# Date: 20 Mar 2024
################################


import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime
import random
import configparser
import argparse


# add parser
parser = argparse.ArgumentParser(
    description="Run classification models to predict Binding Specificity for OVA values from sequence embeddings"
)
parser.add_argument(
    "--config", type=str, default="config_file.txt", help="Path to the config file"
)
parser.add_argument(
    "--simsplit_thresh",
    type=float,
    default=0.05,
    help="Similarity split threshold; default 0.05",
)
parser.add_argument(
    "--chaintype",
    type=str,
    default="both",
    choices=["VH", "VH_VL", "both"],
    help="Output path for results",
)
parser.add_argument(
    "--outpath",
    type=str,
    default="data/model_evaluation/Specificity_classification/",
    help="Output path for results",
)
args = parser.parse_args()


####################################

testing = False

####################################

# add root directory to path such that the utils_nb file can be imported
if testing:
    CONFIG_PATH = "/data/cb/scratch/lenae/sc_AbSpecificity_pred/config_file_RBD.txt"
else:
    CONFIG_PATH = args.config

UTILS_DIR = "../"
UTILS_DIR1 = "./"
sys.path.append(UTILS_DIR)
sys.path.append(UTILS_DIR1)


# import custom modules
import utils_nb as utils
import Load_embs_class as lec
import Specificity_classification_class as CLF
from sklearn.preprocessing import StandardScaler


def run():
    # setup parser for the config file
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    # set dataset (OVA or RBD)
    dataset = config["SETUP"]["DATASET"]
    print(f"Dataset: {dataset}")

    # create output directory
    if testing:
        out_path = os.path.join("./", dataset)
    else:
        out_path = os.path.join(args.outpath, dataset)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # set random seed
    random.seed(123)
    today = str(datetime.now().date())
    log = utils.Logger(__name__, log_file=f"{today}_{dataset}_app.log").get_logger()
    log.info("Start Script!")

    if testing:
        VH_list = ["VDJ_aaSeq"]
    else:
        if args.chaintype == "both":
            VH_list = ["VDJ_VJ_aaSeq", "VDJ_VJ_aaSeq"]
        elif args.chaintype == "VH":
            VH_list = ["VDJ_aaSeq"]
        else:
            VH_list = ["VDJ_VJ_aaSeq"]

    for ab_chain in VH_list:
        if ab_chain == "VDJ_aaSeq":
            c_type = "VH"
            filter_192 = False
        if ab_chain == "VDJ_VJ_aaSeq":
            c_type = "VH_VL"

            if dataset == "OVA":
                filter_192 = True
            else:
                filter_192 = False

        if dataset == "OVA":
            filter_VH_complete = False
        else:
            filter_VH_complete = True

        ######## LOADING CLASS ########
        try:
            if testing:
                embedding_type = "esm_cdrs"
            else:
                embedding_type = "all"

            Embeddings = lec.LoadEmbeddings_VH_VL(
                CONFIG_PATH=CONFIG_PATH,
                seq_col=ab_chain,
                filter_192=filter_192,
                filter_VH_complete=filter_VH_complete,
            )

            Embeddings.load_embeddings(embedding_type=embedding_type, verbose=False)
            # Seq column name 'VDJ_aaSeq', 'VDJ_aaSeqCDR3', 'cdr_comb'...

            ### Load mAb sequences
            seq_df = Embeddings.seq_df
            seqs = Embeddings.seqs


            # Calculate the kmer vectors
            k = 3
            all_kmers = utils.generate_all_kmers(seqs, k)
            vectors = [
                utils.freqs_to_vector(utils.kmer_frequencies(seq, k), all_kmers)
                for seq in seqs
            ]
            kmer_arr_3 = np.array(vectors)

            # Load sequence distance matrix
            distance_matrix = Embeddings.dist_matrix
            log.info("distance matrix and embeddings loaded")

        except Exception as e:
            log.exception(f"ERROR Loading files: {e}")

        ######## TRAIN TEST SPLITS ########
        try:
            # create train test splits - sequence clustering
            N_SPLITS = 5

            if testing:
                SIM_SPLIT = 0.05
            else:
                SIM_SPLIT = args.simsplit_thresh

            y = np.array(seq_df["group_id"])
            if np.unique(y)[0] == 1 and np.unique(y)[1] == 2:
                y = np.where(y == 2, 0, 1)

            # best splits are created with N_SPLITS=6 (based on manual inspection of train test splits)
            train_ls, test_ls, _ = CLF.train_test_split_idx(
                X=np.zeros((len(seqs), 1)),
                y=y,
                cluster_thresh=SIM_SPLIT,
                distance_matrix=distance_matrix,
                n_splits=N_SPLITS,
                cluster_method="levenshtein_sequence_based",
                verbose=0,
            )

            print(f"Sequence-based clustering with {SIM_SPLIT} cluster threshold")
            for i in range(len(test_ls)):
                print(len(test_ls[i]))
                print(np.unique(y[test_ls[i]], return_counts=True)[1])

            # create train test splits - Random split
            train_ls_rd, test_ls_rd, _ = CLF.train_test_split_idx(
                X=np.zeros((len(seqs), 1)),
                y=y,
                cluster_thresh=SIM_SPLIT,
                n_splits=N_SPLITS,
                cluster_method="random_split",
                verbose=0,
            )
            print("Random split")
            for i in range(len(test_ls_rd)):
                print(len(test_ls_rd[i]))
                print(np.unique(y[test_ls_rd[i]], return_counts=True)[1])

            # Summarize train test split
            train_test_splits = [[train_ls, test_ls], [train_ls_rd, test_ls_rd]]

            log.info(
                f"Train test splits based on sequence-based clustering with {SIM_SPLIT} threshold prepared"
            )

        except Exception as e:
            log.exception(f"ERROR TRAIN-TEST Splits: {e}")

        ########### RUN CLASSIFICATION ###########
        try:
            if testing:
                emb_n_list = ["ESM-2-CDRextract"]
                emb_list = [Embeddings.emb_ESM_cdrs]
            else:
                # define embedding names
                emb_n_list = ["ESM-2", "ESM-2-CDRextract", "3-mer", "Antiberty"]

                # define embedding list
                emb_list = [
                    Embeddings.emb_ESM,
                    Embeddings.emb_ESM_cdrs,
                    kmer_arr_3,
                    Embeddings.emb_antiberty,
                ]

            # define file path
            file_path = os.path.join(
                out_path,
                f"{today}_{dataset}_{c_type}_{SIM_SPLIT}_Spec_classification_CV_results.csv",
            )

            # pipe_n = ['', '_pca']# pipes_ls = [['', '_pca'], [[('scaler', StandardScaler())], [('scaler', StandardScaler()), ('pca', PCA(n_components = 50))]]]
            pipe_n = [""]
            # pipes_ls = [[( 'scaler', StandardScaler() )], [('scaler', StandardScaler()), ('pca', PCA(n_components = 50))]]
            pipes_ls = [[("scaler", StandardScaler())]]
            results_l = []
            for n, pipes in zip(pipe_n, pipes_ls):
                for emb_name, emb in zip(emb_n_list, emb_list):
                    for rf in [None, True]:
                        e = f"{emb_name}{n}"
                        X = emb
                        log.info(f"Evaluate classifier on {e} data")
                        results = CLF.run_clf_on_splits(
                            X,
                            y,
                            train_test_splits,
                            SIM_SPLIT,
                            emb_name=e,
                            RF=rf,
                            pipe_ls=pipes,
                            log=log,
                        )
                        results_l.append(results)

                        # save intermediate result
                        results = pd.concat(results_l)
                        results.to_csv(file_path, index=False)

            log.info("Evaluation done")

        except Exception as e:
            log.exception(f"ERROR Running classifier {emb_name}: {e}")

        ########### SAVE RESULTS ###########

        results = pd.concat(results_l)
        results.to_csv(file_path, index=False)

        log.info(f"Results saved in {file_path}")


def main():
    run()


if __name__ == "__main__":
    main()
