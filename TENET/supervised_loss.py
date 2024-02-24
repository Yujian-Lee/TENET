import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        # self.register_buffer("negatives_mask", (~torch.eye((batch_size * 4 - 2), dtype=torch.bool)).float().detach())
        # # self.register_buffer("negatives_mask", (~torch.eye((batch_size * 4 - 2), dtype=torch.bool)).float())
        # # self.register_buffer("negatives_mask", (~torch.eye((batch_size * 4 - 2), dtype=torch.bool)).float())
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 4-2, batch_size * 4-2, dtype=torch.bool)).float())	

    
    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)     
        z_j = F.normalize(emb_j, dim=1) 
        
        representations = torch.cat([z_i, z_j], dim=0)          
        #print(representations.unsqueeze(1).shape) # 48 1 2048
        #print(representations.unsqueeze(0).shape) # 1 48 2048
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      
        
        #print(similarity_matrix)
        #print()
        sim_ij_bs1 = torch.diag(similarity_matrix, self.batch_size)         
        #print('sim_ij', sim_ij)
        sim_ji_bs1 = torch.diag(similarity_matrix, -self.batch_size)       
        
        sim_ij_bs2 = torch.diag(similarity_matrix,  2*self.batch_size)
        sim_ji_bs2 = torch.diag(similarity_matrix, -2*self.batch_size)
        
        sim_ij_bs3 = torch.diag(similarity_matrix,  3*self.batch_size)
        sim_ji_bs3 = torch.diag(similarity_matrix, -3*self.batch_size)
        #print()
        #print('sim_ji', sim_ji)
        positives1 = torch.cat([sim_ij_bs1, sim_ji_bs1], dim=0)                 
        positives2 = torch.cat([sim_ij_bs2, sim_ji_bs2], dim=0) 
        positives3 = torch.cat([sim_ij_bs3, sim_ji_bs3], dim=0) 
        
        nominator1 = torch.exp(positives1 / self.temperature)           
        nominator2 = torch.exp(positives2 / self.temperature)
        nominator3 = torch.exp(positives3 / self.temperature)
        # print('nominator1',nominator1.shape)
        # print('nominator2',nominator2.shape)
        # print('nominator3',nominator3.shape)
        # print()
        
        # print('negatives_mask',self.negatives_mask.shape)
        # print('similarity', similarity_matrix.shape)
        
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)          
        
        #print('denominator', denominator.shape)
        # loss_partial1 = -torch.log(nominator1 / torch.sum(denominator))
        # loss_partial2 = -torch.log(nominator2 / torch.sum(denominator, dim=1))
        # loss_partial3 = -torch.log(nominator3 / torch.sum(denominator, dim=1))
        nominator = torch.sum(nominator1) + torch.sum(nominator2) + torch.sum(nominator3)
        denominator = torch.sum(denominator) - nominator
        loss_partial = -torch.log(nominator/denominator)
        loss = torch.sum(loss_partial) / (4 * self.batch_size)
        return loss
    
    
if __name__ == '__main__':
    dummy_x = torch.randn([1597,64]).float().cpu()
    dummy_y = torch.randn([1597,64]).float().cpu()
    model = ContrastiveLoss(799).cpu()
    output = model(dummy_x,dummy_y)
    print(output)
    
