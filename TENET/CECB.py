import torch
import torch.nn as nn
import random

class CECB(torch.nn.Module):
    def __init__(self, in_ch):
        super(CECB, self).__init__()
        self.GAP = nn.AdaptiveAvgPool1d(512)
        self.FC1 = nn.Linear(in_ch, 32)
        self.ReLu = nn.ReLU()
        self.FC2 = nn.Linear(32, in_ch)
        self.Sigmoid = nn.Sigmoid()
        self.BN = nn.BatchNorm1d(in_ch)
        self.GAP2 = nn.AdaptiveAvgPool1d(512)

    def forward(self, Finput):
        Fv = self.BN(Finput)
        Mv = Finput - Fv
        Mv_1 = self.GAP(Mv)
        Mv_1 = Mv_1.view([Finput.shape[0], Finput.shape[1]])
        Mv_2 = self.FC1(Mv_1)
        Mv_3 = self.ReLu(Mv_2)
        Mv_4 = self.FC2(Mv_3)
        Mv_5 = self.Sigmoid(Mv_4)
        Mv_x = torch.mul(Mv, Mv_5)

        cat = Fv + Mv_x

        catBN = self.BN(cat)
        catBN = self.GAP2(catBN)
        return catBN

if __name__ == "__main__":
    dummy_x_1 = torch.rand([1597,64])
    dummy_x_2 = torch.rand([71865, 32])
    model1 = CECB(64)
    out_1 = model1(dummy_x_1)
    # out_2 = model1(dummy_x_2)

    print(out_1.shape)
    # print(out_2.shape)

