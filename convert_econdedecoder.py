import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
#########################################################################
#########                          GNN部分              #################
########################################################################
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
def conv_block_decode(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2)
    )
class Convnet_code(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.decoder = nn.Sequential()
        self.decoder.add_module('conv5', nn.Conv2d(in_channels=z_dim, out_channels=hid_dim,kernel_size=3,padding=1))
        self.decoder.add_module('rule5', nn.ReLU(True))
        self.decoder.add_module('pool5',nn.Upsample(scale_factor=2,mode='bilinear'))

        self.decoder.add_module('conv6', nn.Conv2d(in_channels=hid_dim, out_channels=hid_dim,kernel_size=3,padding=1))
        self.decoder.add_module('rule6', nn.ReLU(True))
        self.decoder.add_module('pool6',nn.Upsample(scale_factor=2,mode='bilinear'))

        self.decoder.add_module('conv7', nn.Conv2d(in_channels=hid_dim, out_channels=hid_dim,kernel_size=3,padding=1))
        self.decoder.add_module('rule7', nn.ReLU(True))
        self.decoder.add_module('pool7',nn.Upsample(scale_factor=2,mode='bilinear'))

        self.decoder.add_module('conv8', nn.Conv2d(in_channels=hid_dim, out_channels=x_dim,kernel_size=3,padding=2))
        self.decoder.add_module('rule8', nn.ReLU(True))
        self.decoder.add_module('pool8',nn.Upsample(scale_factor=2,mode='bilinear'))

        self.decoder.add_module('conv9',nn.Conv2d(in_channels=x_dim, out_channels=x_dim, kernel_size=3, padding=1))

        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        feature = x.view(x.size(0), -1)
        img_rec = self.decoder(x)

        return feature, img_rec

class GConvnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, N_way = 5, N_shot = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

        self.out_channels = 1600
        self.gcn = GcnNet(input_dim=self.out_channels, output_dim=256,test_N_shot = 5, test_N_way = 5)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.gcn(x)

