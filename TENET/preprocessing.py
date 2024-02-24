import numpy as np
import pandas as pd
import networkx as nx
import time
import random
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from node2vec import Node2Vec
sys.path.insert(0, os.path.abspath('./submodules/'))
sys.path.append('./submodules/CeSpGRN/src/')
sys.path.append('../')


def convert_adjacencylist2edgelist(adj_list):
    edge_list = []
    
    for node, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            edge_list.append([node, neighbor])
            
    return np.array(edge_list).T


def convert_adjacencylist2adjacencymatrix(adj_list):
    
    num_vertices = len(adj_list)
    adj_matrix = np.zeros(shape=(num_vertices,num_vertices))
     
    for i in range(num_vertices):
        for j in adj_list[i]:
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1
     
    return adj_matrix




def select_LRgenes(data_df, num_genespercell, lr_database = 0):

    if lr_database == 0:
        sample_counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis = 1)
        lr_df = pd.read_csv("../data/celltalk_mouse_lr_pair.txt", sep="\t")
        
        receptors = set(lr_df["receptor_gene_symbol"].str.upper().to_list())
        ligands = set(lr_df["ligand_gene_symbol"].str.upper().to_list())

        real2uppercase = {x:x.upper() for x in sample_counts.columns}
        uppercase2real = {upper:real for real,upper in real2uppercase.items()}
        candidate_genes = set(np.vectorize(real2uppercase.get)(sample_counts.columns.to_numpy()))
        
        selected_ligands = candidate_genes.intersection(ligands)
        selected_receptors = candidate_genes.intersection(receptors)
        selected_lrs = selected_ligands | selected_receptors
        
        if len(selected_lrs) > num_genespercell // 2 + 1:
            selected_lrs = set(random.sample(tuple(selected_lrs), num_genespercell // 2 + 1))
            selected_ligands = selected_lrs.intersection(selected_ligands)
            selected_receptors = selected_lrs.intersection(selected_receptors)
        
        num_genesleft = num_genespercell - len(selected_ligands) - len(selected_receptors)
        candidate_genesleft = candidate_genes - selected_ligands - selected_receptors
        selected_randomgenes = set(random.sample(tuple(candidate_genesleft), num_genesleft))
        
        selected_genes = list(selected_randomgenes | selected_ligands | selected_receptors)
                
        selected_columns = ["Cell_ID", "X", "Y", "Cell_Type"] + np.vectorize(uppercase2real.get)(selected_genes).tolist()
        selected_df = data_df[selected_columns]

                
        lr2id = {gene:list(selected_df.columns).index(gene)-4 for gene in np.vectorize(uppercase2real.get)(list(selected_ligands|selected_receptors))}
        
        return selected_df, lr2id
    
    elif lr_database==2:
        sample_counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis = 1)
        candidate_genes = sample_counts.columns.to_numpy()
        scmultisim_lrs = pd.read_csv("../data/scMultiSim/simulated/cci_gt.csv")[["ligand", "receptor"]]
        scmultisim_lrs["ligand"] = scmultisim_lrs["ligand"]
        scmultisim_lrs["receptor"] = scmultisim_lrs["receptor"]
        selected_ligands = np.unique(scmultisim_lrs["ligand"])
        selected_receptors = np.unique(scmultisim_lrs["receptor"])
        selected_lrs = np.concatenate((selected_ligands,selected_receptors),axis=0)
        num_genesleft = num_genespercell - len(selected_ligands) - len(selected_receptors)
        indices  = np.argwhere(candidate_genes == selected_lrs)
        candidate_genesleft = np.delete(candidate_genes, indices)
        selected_randomgenes = random.sample(set(candidate_genesleft), num_genesleft)
        selected_genes = np.concatenate((selected_lrs, selected_randomgenes),axis=0)
        new_columns = ["Cell_ID", "X", "Y", "Cell_Type"] + list(selected_genes)
        selected_df = data_df[new_columns]
        lr2id = {gene:list(selected_df.columns).index(gene)-4 for gene in selected_genes}

        return selected_df, lr2id
        
    else:
        raise Exception("Invalid lr_database type")
    

def infer_initial_grns(data_df, cespgrn_hyperparams):
    
    console = Console()
    
    from submodules.CeSpGRN.src import kernel
    from submodules.CeSpGRN.src import g_admm as CeSpGRN
    
    with console.status("[cyan] Preparing CeSpGRN ...") as status:
        status.update(spinner="aesthetic", spinner_style="cyan")
        counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis = 1).values
        pca_op = PCA(n_components = 20)
        X_pca = pca_op.fit_transform(counts)
        bandwidth = cespgrn_hyperparams["bandwidth"]
        n_neigh = cespgrn_hyperparams["n_neigh"]
        lamb = cespgrn_hyperparams["lamb"]
        max_iters = cespgrn_hyperparams["max_iters"]
        K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = n_neigh)
        empir_cov = CeSpGRN.est_cov(X = counts, K_trun = K_trun, weighted_kt = True)
        cespgrn = CeSpGRN.G_admm_minibatch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize = 120)
        
        
    grns = cespgrn.train(max_iters=max_iters, n_intervals=100, lamb=lamb)
    
    return grns



def construct_celllevel_graph(data_df, k, get_edges=False):   # Top k closest neighbors for each cell

    adjacency = np.zeros(shape=(len(data_df), k),dtype=int) # shape = (numcells, numneighbors of cell)
    coords = np.vstack([data_df["X"].values,data_df["Y"].values]).T

    edges = None
    edge_x = []
    edge_y = []

    for i in track(range(len(data_df)), description=f"[cyan]2. Constructing Cell-Level Graph from ST Data"):
        cell_id = data_df["Cell_ID"][i]
        x0, y0 = data_df["X"].values[i],data_df["Y"].values[i]
        candidate_cell = coords[i]
        candidate_neighbors = coords
        euclidean_distances = np.linalg.norm(candidate_neighbors - candidate_cell,axis=1)
        neighbors = np.argsort(euclidean_distances)[1:k+1]
        adjacency[i] = neighbors
        assert i not in adjacency[i]
        if get_edges:
            for ncell in adjacency[i]:
                x1, y1 = data_df["X"].values[ncell],data_df["Y"].values[ncell]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
        
    edges=[edge_x,edge_y]
                
    return adjacency,edges

def construct_genelevel_graph(disjoint_grns, celllevel_adj_list, node_type = "int", lrgenes = None):

    numgenes = disjoint_grns[0].shape[0]
    numcells = disjoint_grns.shape[0]
    num2gene = {}   #dictionary that maps the supergraph integer nodes to the gene name (Cell#_gene#)
    gene2num = {}   #dictionary that maps the gene name (Cell#_gene#) to the supergraph integer node

    assert max(lrgenes) <= numgenes

    grn_graph_list = []
    for cellnum, grn in enumerate(track(disjoint_grns, description=f"[cyan]3a. Combining individual GRNs")):
        G =  nx.from_numpy_matrix(grn)
        grn_graph_list.append(G)
        for i in range(numgenes):
            num2gene[cellnum*numgenes+i] = f"Cell{cellnum}_Gene{i}"
            gene2num[f"Cell{cellnum}_Gene{i}"] = cellnum * numgenes + i

    union_of_grns = nx.disjoint_union_all(grn_graph_list)
        
    gene_level_graph = nx.relabel_nodes(union_of_grns, num2gene)  # relabel nodes to actual gene names
    
    # print(len(lrgenes), lrgenes)

    # for cell, neighborhood in enumerate(track(celllevel_adj_list,\
    #     description=f"[cyan]3b. Constructing Gene-Level Graph")): # for each cell in the ST data
        
    #     if lrgenes == None:     # randomize selection of genes to connect spatial edges
    #         for genenum1 in range(numgenes):                            # for each gene in the cell                         45
    #             node1 = f"Cell{cell}_Gene{genenum1}"
    #             for ncell in neighborhood:                                  # for each neighborhood cell adjacent to cell   5
    #                 candidate_neighbors=np.arange(numgenes,dtype=int)
    #                 np.random.shuffle(candidate_neighbors)
    #                 for genenum2 in candidate_neighbors[:numgenes//3]:          # for each gene in the neighborhood cell    45
    #                     node2 = f"Cell{ncell}_Gene{genenum2}"
    #                     gene_level_graph.add_edge(node1, node2)
    #     else:
    #         for genenum1 in lrgenes:                            # for each lr gene in the cell                         45
    #             node1 = f"Cell{cell}_Gene{genenum1}"
    #             for ncell in neighborhood:                                  # for each neighborhood cell adjacent to cell   5
    #                 for genenum2 in lrgenes:                                    # for each gene in the neighborhood cell    45
    #                     node2 = f"Cell{ncell}_Gene{genenum2}"
    #                     gene_level_graph.add_edge(node1, node2)
    
        
    for cell, neighborhood in enumerate(track(celllevel_adj_list,\
        description=f"[cyan]3b. Constructing Gene-Level Graph")): # for each cell in the ST data
        for neighbor_cell in neighborhood:
            if neighbor_cell != -1:
                for lrgene1 in lrgenes:
                    for lrgene2 in lrgenes:
                        node1 = f"Cell{cell}_Gene{lrgene1}"
                        node2 = f"Cell{neighbor_cell}_Gene{lrgene2}"
                        if not gene_level_graph.has_node(node1) or not gene_level_graph.has_node(node2): 
                            raise Exception(f"Nodes {node1} or {node2} not found. Debug the Gene-Level Graph creation.")
                        
                        gene_level_graph.add_edge(node1, node2)

    if node_type == "str":
        gene_level_graph = gene_level_graph
    elif node_type == "int":
        gene_level_graph = nx.convert_node_labels_to_integers(gene_level_graph)

    assert len(gene_level_graph.nodes()) == numcells * numgenes

    return gene_level_graph, num2gene, gene2num, union_of_grns
        



def get_gene_features(graph, type="node2vec"):
    
    if type == "node2vec":
        node2vec = Node2Vec(graph, dimensions=64, walk_length=15, num_walks=100, workers=4)
        model = node2vec.fit()

        gene_feature_vectors = model.wv.vectors
        
    return gene_feature_vectors, model

