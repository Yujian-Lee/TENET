import torch
import torch.nn as nn
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class EEB_stage1(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(EEB_stage1, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class EEB_stage2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EEB_stage2, self).__init__()
        self.ES1 = EEB_stage1(in_ch, out_ch, reduction=in_ch / 4)

    def forward(self, input1):
        s1, c1 = input1.shape
        input1 = input1.view(s1, c1, 1, 1)
        input1_en = self.ES1(input1)
        input1_en = input1_en.view([input1_en.shape[0], input1_en.shape[1]])
        return input1_en


if __name__ == '__main__':
    dummy_input =  torch.rand([100, 128])
    in_channels = 128
    out_channels =128
    EEB_model = EEB_stage2(in_ch=in_channels,out_ch=out_channels)
    output = EEB_model(dummy_input)
    print("Dummy_test_output_shape: ", output.shape)