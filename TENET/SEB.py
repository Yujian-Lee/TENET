import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
class Local_stage1(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)

        U = sum(conv_outs)

        S = U.mean(-1).mean(-1)
        Z = self.fc(S)

        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))
        attention_weughts = torch.stack(weights, 0)
        attention_weughts = self.softmax(attention_weughts)

        V = (attention_weughts * feats).sum(0)
        return V
class Local_stage2(nn.Module):
    def __init__(self, in_ch):
        super(Local_stage2, self).__init__()
        self.se1 = Local_stage1(channel=in_ch, reduction=in_ch / 4)

    def forward(self, input1):
        input1 = input1.view([input1.shape[0], input1.shape[1]])
        s1, c1 = input1.shape
        input1 = input1.view(s1, c1, 1, 1)
        input1_en = self.se1(input1)
        input1_en = input1_en.view(input1_en.shape[0], input1_en.shape[1])
        return input1_en

class Global(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.4, bias=True):
        super(Global, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True, dropout=dropout, bias=bias)
        self.se = Local_stage2(512)
        self.conv2 = GATConv(heads * out_channels, out_channels, heads=heads, concat=False, dropout=dropout, bias=bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # print(edge_index.shape)
        # print(x.shape)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = self.se(x)
        temp2 = x
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1),temp2

# block test
if __name__ == '__main__':
    dummy_input =  torch.rand([100, 32])
    dummy_edge = torch.randint(0,100,[2,10000])
    data = Data(x=dummy_input, edge_index=dummy_edge)
    in_channels = 32
    out_channels = 64
    SEB_model = Global(in_channels, out_channels)
    output = SEB_model(data)
    print("Dummy_test_output_shape: ", output.shape)

    # input = torch.randn(50, 512, 7, 7)
    # se = Local(channel=512, reduction=8)
    # output = se(input)
    # print(output.shape)

