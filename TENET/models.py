import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv
class MultiviewEncoder(torch.nn.Module):
    def __init__(self, GeneEncoder, CellEncoder, is_training=False):
        super(MultiviewEncoder, self).__init__()
        self.encoder_g = GeneEncoder                            #  encoder for gene level graph
        self.encoder_c = CellEncoder                               #  encoder for cell level graph
        self.is_training = is_training


    def forward(self, x_c, x_g, edge_index_c, edge_index_g):
        Z_g, gene_embeddings = self.encoder_g(x_g, edge_index_g)
        Z_c = self.encoder_c(x_c, edge_index_c)
        Z = torch.cat((Z_c,Z_g),dim=1)
        # print(torch.max(Z), torch.min(Z))
        # print(Z)

        assert Z.shape[1] == 2 * Z_g.shape[1]

        return Z, Z_c, Z_g, gene_embeddings

class CellEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.2, is_training=False):
        super(CellEncoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.is_training = is_training
        self.dropout = 0.2

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x



class GeneEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_vertices, num_subvertices, dropout=0.2, is_training=False):
        super(GeneEncoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(num_subvertices * hidden_dim, hidden_dim)
        self.is_training = is_training
        self.dropout = 0.2
        self.num_vertices = num_vertices
        self.num_subvertices = num_subvertices

    def embed(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        embeddings = self.embed(x,edge_index)
        x=embeddings
        x = x.view(x.shape[0]//self.num_subvertices, x.shape[1] * self.num_subvertices)
        x = self.linear(x)
        return x, embeddings
    
class InnerProductDecoder(torch.nn.Module):

    def forward(self, z, edge_index,sigmoid = True):
        print("---------------------------------------------------")
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value




    
