import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse
import torch
import time
import preprocessing as preprocessing
import training as training
import models as models
from torch_geometric.nn import GAE

debug = False

def parse_arguments():
    parser = argparse.ArgumentParser(description='GAE arguments')
    parser.add_argument("-m", "--mode", type=str, default = "train",
        help="GAE mode: preprocess,train")
    parser.add_argument("-i", "--inputdirpath", type=str,default = "../dataset/seqFISH/seqfish_dataframe.csv",
                    help="Input directory path where ST data is stored")
    parser.add_argument("-o", "--outputdirpath", type=str,default = "../output/seqfish",
                    help="Output directory path where results will be stored ")
    parser.add_argument("-s", "--studyname", type=str,default='TENET',
                     help="study name")
    parser.add_argument("-t", "--split", type=float,default = 0.7,
        help="# of test edges [0,1)")
    parser.add_argument("-n", "--numgenespercell", type=int, default = 60,
                help="Number of genes in each gene regulatory network")
    parser.add_argument("-k", "--nearestneighbors", type=int, default = 2,
                help="Number of nearest neighbors for each cell")
    parser.add_argument("--fp", type=float, default=0,
        help="(experimentation only) # of false positive test edges [0,1)")
    parser.add_argument("--fn", type=float, default=0,
        help="(experimentation only) # of false negative test edges [0,1)")
    parser.add_argument("-a", "--ownadjacencypath", type=str,default = None,
        help="Using your own cell level adjacency (give path)")
    parser.add_argument("-lr",type=float,default = 0.8,
        help="Using your lr (learning rate)")
    args = parser.parse_args()
    return args



def preprocess(st_data, num_nearestneighbors, lrgene_ids, cespgrn_hyperparameters, ownadjacencypath = None):
    if debug:
        print("1. Skipping CeSpGRN inference (for debug mode)")
        grns = np.load("../output/scmultisim/1_preprocessing_output/initial_grns.npy")
    else:
        grns = preprocessing.infer_initial_grns(st_data, cespgrn_hyperparameters)
    if ownadjacencypath is not None:
        celllevel_adj = np.load(ownadjacencypath)
    else:
        celllevel_adj, _ = preprocessing.construct_celllevel_graph(st_data, num_nearestneighbors, get_edges=False)
    gene_level_graph, num2gene, gene2num, grn_components = preprocessing.construct_genelevel_graph(grns, celllevel_adj, node_type="int", lrgenes = lrgene_ids)

    gene_features, genefeaturemodel = preprocessing.get_gene_features(grn_components, type="node2vec")

    return celllevel_adj, gene_level_graph, num2gene, gene2num, grns, gene_features, genefeaturemodel


def model_building(data, hyperparams = None):
    num_cells, num_cellfeatures = data[0][0].x.shape[0], data[0][0].x.shape[1]
    num_genes, num_genefeatures = data[1].x.shape[0], data[1].x.shape[1]
    hidden_dim = hyperparams["concat_hidden_dim"] // 2
    num_genespercell = hyperparams["num_genespercell"]
    cellEncoder = models.CellEncoder(num_cellfeatures, hidden_dim)
    geneEncoder = models.GeneEncoder(num_features=num_genefeatures, hidden_dim=hidden_dim, num_vertices = num_cells, num_subvertices = num_genespercell)
    
    multiviewEncoder = models.MultiviewEncoder(GeneEncoder = geneEncoder, CellEncoder = cellEncoder)
    
    gae = GAE(multiviewEncoder)
    return gae

def main():

    args = parse_arguments()
    print("=======================================================")
    print(args)
    print("=======================================================")
        
    mode = args.mode
    input_dir_path = args.inputdirpath
    output_dir_path = args.outputdirpath
    num_nearestneighbors = args.nearestneighbors
    num_genespercell = args.numgenespercell
    LR_database = args.lrdatabase
    studyname = args.studyname
    ownadjacencypath = args.ownadjacencypath
        
    preprocess_output_path = os.path.join(output_dir_path, "preprocessing_output")
    training_output_path = os.path.join(output_dir_path, "model_pth")
    evaluation_output_path = os.path.join(output_dir_path, "training_output")

        
    if "preprocess" in mode:
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)
        st_data = pd.read_csv(input_dir_path, index_col=None)
        assert {"Cell_ID", "X", "Y", "Cell_Type"}.issubset(set(st_data.columns.to_list()))
        
        numcells, totalnumgenes = st_data.shape[0], st_data.shape[1] - 4

        print(f"{numcells} Cells & {totalnumgenes} Total Genes\n")
        
        cespgrn_hyperparameters = {
            "bandwidth" : 0.1,
            "n_neigh" : 30,
            "lamb" : 0.1,
            "max_iters" : 1000
        }
        
        print(f"Hyperparameters:\n # of Nearest Neighbors: {num_nearestneighbors}\n # of Genes per Cell: {num_genespercell}\n")
        
        selected_st_data, lrgene2id = preprocessing.select_LRgenes(st_data, num_genespercell, LR_database)
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)

        celllevel_features = st_data.drop(["Cell_ID", "Cell_Type", "X", "Y"], axis = 1).values

        celllevel_adj, genelevel_graph, num2gene, gene2num, grns, genelevel_features, genelevel_feature_model = preprocess(selected_st_data, num_nearestneighbors,lrgene2id.values(), cespgrn_hyperparameters, ownadjacencypath)
            
        celllevel_edgelist = preprocessing.convert_adjacencylist2edgelist(celllevel_adj)
        genelevel_edgelist = nx.to_pandas_edgelist(genelevel_graph).drop(["weight"], axis=1).to_numpy().T
        genelevel_adjmatrix = nx.adjacency_matrix(genelevel_graph, weight=None)
        
        assert celllevel_edgelist.shape == (2, celllevel_adj.shape[0] * celllevel_adj.shape[1])
        
        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencylist.npy"),celllevel_adj)
        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencymatrix.npy"),preprocessing.convert_adjacencylist2adjacencymatrix(celllevel_adj))
        np.save(os.path.join(preprocess_output_path, "celllevel_edgelist.npy"),celllevel_edgelist)
        np.save(os.path.join(preprocess_output_path, "celllevel_features.npy"),celllevel_features)
        np.save(os.path.join(preprocess_output_path, "genelevel_edgelist.npy"),genelevel_edgelist)
        np.save(os.path.join(preprocess_output_path, "genelevel_adjmatrix.npy"),genelevel_adjmatrix)
        np.save(file = os.path.join(preprocess_output_path, "initial_grns.npy"), arr = grns) 
        np.save(os.path.join(preprocess_output_path, "genelevel_features.npy"), genelevel_features) 
        genelevel_feature_model.save(os.path.join(preprocess_output_path, "genelevel_feature_model")) 
    

    if "train" in mode:
        hyperparameters = {
            "num_genespercell": num_genespercell,
            "concat_hidden_dim": 64,
            "optimizer" : "adam",
            "criterion" : torch.nn.BCELoss(),
            "num_epochs": 120,
            "split": args.split,
        }

        false_edges = None if args.fp == 0 and args.fn == 0 else {"fp": args.fp, "fn": args.fn}

        celllevel_data, genelevel_data = training.create_pyg_data(preprocess_output_path, hyperparameters["split"],false_edges)
        print(genelevel_data["edge_index"])
        
        celllevel_str = str(celllevel_data)
        
        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)


        print("Training_process")

        data = (celllevel_data, genelevel_data)

        model = model_building(data, hyperparameters)
        
        if hyperparameters["optimizer"] == "adam":
            hyperparameters["optimizer"] = torch.optim.Adam(model.parameters(), lr=0.01), 
            
        split = hyperparameters["split"]
        trained_model, metrics_df = training.train_gae(model=model, data=data, hyperparameters = hyperparameters , lr = args.lr)
        
        torch.save(trained_model.state_dict(), os.path.join(training_output_path,f'{studyname}_trained_gae_model.pth'))
        
        if not os.path.exists(evaluation_output_path):
            os.mkdir(evaluation_output_path)
        metrics_df.to_csv(os.path.join(evaluation_output_path, f"{studyname}_metrics_{split}.csv"))

    return


if __name__ == "__main__":
    main()
