import sys
import os
import numpy as np
import torch
import torch_geometric.transforms as T
import pandas as pd
import torch.nn as nn
import random
import argparse
from torch_geometric.data import HeteroData
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag
from tqdm import trange
from EEB import EEB_stage2, EEB_stage1
from SEB import Local_stage2, Local_stage1, Global
from CECB import CECB
from torch_geometric.data import Data
from supervised_loss import ContrastiveLoss

def add_fake_edges(num_vertices, num_old_edges, fp):
    num_add_edges = int((fp) * num_old_edges)
    
    add_edges = torch.from_numpy(np.random.randint(0,high=num_vertices, size=(2,num_add_edges)))
        
    print(f"Added {fp*100}% false edges")
    return add_edges
  

def remove_real_edges(old_edge_indices, fn):
    new_edge_indices = None
    num_remove_edges = int(fn * old_edge_indices.shape[0])
    new_edge_indices = torch.from_numpy(np.random.choice(old_edge_indices, size = num_remove_edges, replace=False)).int()

    print(f"Removed {fn*100}% real edges")
    return new_edge_indices
    

def create_pyg_data(preprocessing_output_folderpath, split=0.1, false_edges = None):

    if not os.path.exists(preprocessing_output_folderpath) or \
        not {"celllevel_adjacencylist.npy","celllevel_adjacencymatrix.npy",  "celllevel_edgelist.npy", "genelevel_edgelist.npy", "celllevel_features.npy", "genelevel_features.npy"}.issubset(set(os.listdir(preprocessing_output_folderpath))):
        
        raise Exception("Proper preprocessing files not found. Please run the 'preprocessing' step.")

    celllevel_adjacencymatrix = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_adjacencymatrix.npy"))).type(torch.LongTensor)
    celllevel_features = torch.from_numpy(normalize(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_features.npy")))).type(torch.float32)
    celllevel_edgelist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_edgelist.npy"))).type(torch.LongTensor)
    genelevel_edgelist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "genelevel_edgelist.npy"))).type(torch.LongTensor)
    genelevel_features = torch.from_numpy(normalize(np.load(os.path.join(preprocessing_output_folderpath, "genelevel_features.npy")))).type(torch.float32)
    genelevel_grns_flat = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "initial_grns.npy"))).type(torch.float32).flatten()


    cell_level_data = Data(x=celllevel_features, edge_index = celllevel_edgelist, y = celllevel_adjacencymatrix)
    gene_level_data = Data(x= genelevel_features, edge_index = genelevel_edgelist, y= genelevel_grns_flat)
    
    if split is not None:
      print(f"{1-split} training edges | {split} testing edges")

      transform = T.RandomLinkSplit(
          num_test=split,
          num_val=0,
          is_undirected=True, 
          add_negative_train_samples=True,
          neg_sampling_ratio=1.0,
          key = "edge_label",
          disjoint_train_ratio=0,
      )
      train_cell_level_data, _, test_cell_level_data = transform(cell_level_data)
      cell_level_data = (train_cell_level_data, test_cell_level_data)


    if false_edges is not None:
      fp = false_edges["fp"]
      fn = false_edges["fn"]
      
      if fn != 0:
        new_indices = train_cell_level_data.edge_label.clone()
        old_edge_indices = np.argwhere(train_cell_level_data.edge_label == 1).squeeze()
        new_neg_edge_indices = remove_real_edges(old_edge_indices, fn).long()
        new_indices[new_neg_edge_indices] = 0
        train_cell_level_data.edge_label = new_indices
        
      if fp !=0:
        posmask = train_cell_level_data.edge_label == 1
        newedges = add_fake_edges(train_cell_level_data.x.size()[0], train_cell_level_data.edge_label_index[:, posmask].shape[1], fp)
        train_cell_level_data.edge_label  = torch.cat([train_cell_level_data.edge_label , torch.ones(newedges.shape[1])])
        train_cell_level_data.edge_label_index =  torch.cat([train_cell_level_data.edge_label_index , newedges],dim=1)
    return cell_level_data, gene_level_data
  


def train(data, model, hyperparameters):
  num_epochs = hyperparameters["num_epochs"]
  optimizer = hyperparameters["optimizer"][0]
  criterion = hyperparameters["criterion"]

  with trange(num_epochs,desc="") as pbar:
    for epoch in pbar:
      pbar.set_description(f"Epoch {epoch}")
      model.train()
      optimizer.zero_grad()
      recon_Ac, recon_Ag = model(data[0].x,data[1].x, data[0].edge_index, data[1].edge_index)


      loss = criterion(recon_Ac, data[0].y.float())


      loss.backward() 
      optimizer.step()  

      pbar.set_postfix(loss=loss.item())
      
      
def create_intracellular_gene_mask(num_cells, num_genespercell):
  I = np.ones(shape=(num_genespercell,num_genespercell))
  block_list = [I for _ in range(num_cells)]
  return block_diag(*block_list).astype(bool)
 
    
def autoChooselr(epoch,lr):
    # learning rate warm-up -> learning rate decay
    # default  =0.8
    # if epoch  < 30:
    #     lr = 0.01
        # seqfish lr
#     if epoch < 30:
#         lr =0.1
#     elif epoch >= 30 and epoch < 70:
#         lr = 0.06
#     elif epoch >= 70 and epoch < 100:
#         lr = 0.01
#     else:
#         lr = 0.005
    if epoch < 50:
        lr = 0.01  #.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
    # after the epoch reaches 25 use 10 epoch to adaptively decay
    # min -> 0.8*0.1*0.1
    # elif epoch >= 30 and epoch < 75:
    #     lr = 0.005
#     elif epoch >= 37 and epoch < 60:
        
#         lr = 0.01
#     elif epoch>=60 and epoch<83:
#         lr =0.01
    else:
        lr = 0.001
    return lr

def train_gae(data, model, hyperparameters,lr):
  num_epochs = hyperparameters["num_epochs"]
  optimizer = hyperparameters["optimizer"][0]
  criterion = hyperparameters["criterion"]
  split = hyperparameters["split"]
  num_genespercell = hyperparameters["num_genespercell"]
  
  if split is not None:
    cell_train_data = data[0][0]
    cell_test_data = data[0][1]
  else:
    cell_train_data = data[0]
  gene_train_data = data[1]

  num_cells = cell_train_data.x.shape[0]
  intracellular_gene_mask = create_intracellular_gene_mask(num_cells, num_genespercell)
  mse = torch.nn.MSELoss()

  test_roc_scores = []
  test_ap_scores = []
  test_auprc_scores = []

  with trange(num_epochs,desc="") as pbar:
    for epoch in pbar:
      pbar.set_description(f"Epoch {epoch}")
      model.train()
      optimizer.zero_grad()

      posmask = cell_train_data.edge_label == 1
      z, _, _, z_g = model.encode(cell_train_data.x, gene_train_data.x, cell_train_data.edge_label_index[:, posmask], gene_train_data.edge_index)
      hyperparameters["optimizer"] = torch.optim.Adam(model.parameters(), lr=autoChooselr(epoch,lr))
      print("========================================================================")
      # seqFISH
      # torch.Size([1597, 64])
      # torch.Size([71865, 32])
    
      # MERFISH
      # torch.Size([2000, 64])
      # torch.Size([90000, 32])
        
      # print(z.shape)
      # print(z_g.shape)
#######CECB###############
      temp_z = z
      temp = z_g
      CECB_cell = CECB(z.shape[1])
      CECB_gene = CECB(z_g.shape[1])
      z = CECB_cell(z)
      z_g = CECB_gene(z_g)
      temp1 = z_g
#######CECB###############


#######SEB###############
        
      in_channel = z_g.shape[1]
      out_channel =z_g.shape[1]
      SEB_gene = Global(in_channel,out_channel)
      data = Data(x = z_g,edge_index=gene_train_data.edge_index)
      z_g,temp2 = SEB_gene(data)
        
#######SEB###############        

#######EEB###############    
      EEB_cell = EEB_stage2(z.shape[1],z.shape[1])
      z = EEB_cell(z)
#######EEB###############     

######Triple_enhancement_Fusionist####### 
      Linear1 = nn.Linear(z_g.shape[1],16)
      ReLu = nn.ReLU()
      Linear2 = nn.Linear(16,z_g.shape[1])
      Linear3 = nn.Linear(temp2.shape[1],temp1.shape[1])
      temp2 = Linear3(temp2)
      z_g = temp + temp1 + temp2 + z_g
      z_g = Linear1(z_g)
      z_g = ReLu(z_g)
      z_g = Linear2(z_g)



      # Extra fusionist for cell can be added, but it won't have a great improve

######Triple_enhancement_Fusionist#######         
      recon_Ag = model.decoder.forward_all(z_g)
      intracellular_penalty_loss =  mse(recon_Ag[intracellular_gene_mask], gene_train_data.y)
      # print(temp_z.dtype)
      # print(z.dtype)
        
      Sloss = ContrastiveLoss(799)
      supervisedloss = Sloss(temp_z,z)
      recon_loss = model.recon_loss(z, cell_train_data.edge_label_index[:, posmask])
      
      loss = recon_loss + intracellular_penalty_loss + supervisedloss
      # loss = recon_loss + intracellular_penalty_loss
      auroc,ap = model.test(z,  cell_train_data.edge_label_index[:, posmask], cell_train_data.edge_label_index[:,~posmask])

      
      loss.backward()  
      optimizer.step()

      model.eval()
      posmask = cell_test_data.edge_label == 1
      test_recon_loss = model.recon_loss(z, cell_test_data.edge_label_index[:, posmask])
      test_rocauc, test_ap = model.test(z, cell_test_data.edge_label_index[:, posmask], cell_test_data.edge_label_index[:,~posmask])
      test_precision, test_recall, _ = precision_recall(model,z, cell_test_data.edge_label_index[:, posmask], cell_test_data.edge_label_index[:,~posmask])
      test_auprc = auc(test_recall, test_precision)
      test_roc_scores.append(test_rocauc)
      test_auprc_scores.append(test_auprc)
      test_ap_scores.append(test_ap)

        
      if (epoch+1) % 5 == 0:
            print("Test_roauc: {}, Test_ap : {}".format(test_rocauc,test_ap))
      pbar.set_postfix(train_loss=loss.item(), train_recon_loss = recon_loss.item(),test_recon_loss =test_recon_loss.item())

  metrics_df = pd.DataFrame({"Epoch":range(num_epochs), f"TENET Test AP": test_ap_scores, f"TENET Test ROC": test_roc_scores})

  return model, metrics_df


def precision_recall(model, z, pos_edge_index, neg_edge_index):
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = model.decode(z, pos_edge_index, sigmoid=True)
    neg_pred = model.decode(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return precision_recall_curve(y, pred)