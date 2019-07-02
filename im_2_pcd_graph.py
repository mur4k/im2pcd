import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from data_utils import sample_minibatch_from_sphere

# ------------------------------------------------------------------------------ #

BILINEAR = 'bilinear'
NEAREST = 'nearest'

ZEROS = 'zeros'
BORDER = 'border'
REFLECTION = 'reflection'

MODE = BILINEAR
PADDING_MODE = BORDER


def weight_init(m):
    if isinstance(m, nn.Conv2d) or \
       isinstance(m, nn.Linear) or \
       isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif  isinstance(m, nn.BatchNorm2d):
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or \
       isinstance(m, nn.Linear) or \
       isinstance(m, nn.BatchNorm2d) or \
       isinstance(m, nn.Conv1d):
        m.reset_parameters()

# ---------------------- GRAPH CONV WITH R.V.  ---------------------- #

class SharedMLP(nn.Module):
    
    def __init__(self, channels):
        super(SharedMLP, self).__init__()
        shared_mlp = []
        for i in range(len(channels)-2):
            shared_mlp.extend([nn.Conv1d(in_channels=channels[i],
                                         out_channels=channels[i+1],
                                         kernel_size=1, 
                                         stride=1, 
                                         padding=0),
                               nn.ReLU(True)])
        shared_mlp.append(nn.Conv1d(in_channels=channels[-2],
                                    out_channels=channels[-1],
                                    kernel_size=1,
                                    stride=1,
                                    padding=0))
        self.shared_mlp = nn.Sequential(*shared_mlp)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

    def forward(self, x):
        x = self.shared_mlp(x)
        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda


class SPFP(nn.Module):
    
    def __init__(self, channels):
        super(SPFP, self).__init__()
        self.loc = nn.Sequential(SharedMLP(channels),
                                 nn.ReLU(True))
        self.resid_loc = nn.Conv1d(in_channels=channels[-1]+3,
                                   out_channels=2,
                                   kernel_size=1, 
                                   stride=1, 
                                   padding=0)
        # initialize resid_loc as a projction onto XoY
        self.resid_loc.weight.data.zero_()
        self.resid_loc.weight.data[0, 0] = 1.0
        self.resid_loc.weight.data[1, 1] = 1.0
        self.resid_loc.bias.data.zero_()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

    def forward(self, x_loc, x_feat, x_to_pool_from):
        x_pool_feat = torch.cat([x_loc, x_feat], dim=1)
        x_pool_feat = self.loc(x_pool_feat)
        x_pool_feat = torch.cat([x_loc, x_pool_feat], dim=1)
        grid = self.resid_loc(x_pool_feat)
        # create grid of size N x 1 x NUM_PTS x 2
        grid = grid.unsqueeze(1).transpose(-2, -1)
        # sample from grid -> N x C x 1 X NUM_PTS
        x_pool_feat = [F.grid_sample(x_to_pool, grid, mode=MODE, padding_mode=PADDING_MODE) for x_to_pool in x_to_pool_from]
        x_pool_feat = torch.cat(x_pool_feat, dim=1)
        return x_pool_feat.squeeze(2)


    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda


class PCDConv(nn.Module):

    def __init__(self, in_channels, out_channels, k):
        super(PCDConv, self).__init__()
        self.graph_conv = gnn.GraphConv(in_channels+3, out_channels)
        self.relu = nn.ReLU(True)
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

    def forward(self, x):
        x_loc, x_feat = x
        x_new_feat = torch.cat([x_loc, x_feat], dim=1)
        x_new_feat = x_new_feat.transpose(-2, -1)
        x_loc = x_loc.transpose(-2, -1)

        batch_size = x_new_feat.size(0)
        x_batch = torch.ones(x_new_feat.size()[:2], 
                             dtype=torch.int64,
                             device=self.device)
        x_batch *= torch.arange(start=0, 
                                end=batch_size, 
                                dtype=torch.int64, 
                                device=self.device).view(-1, 1)
        x_batch = x_batch.flatten()

        x_new_feat = x_new_feat.contiguous().view(-1, x_new_feat.size(-1))
        x_loc = x_loc.contiguous().view(-1, 3)
        edge_index = gnn.knn_graph(x=x_loc, k=self.k, batch=x_batch)
        x_new_feat = self.graph_conv(x_new_feat, edge_index)
        x_new_feat = self.relu(x_new_feat)
        x_new_feat = x_new_feat.view(batch_size, -1, x_new_feat.size(1)).transpose(-2, -1)
        x_loc = x_loc.view(batch_size, -1, 3).transpose(-2, -1)
        return (x_loc, x_new_feat)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda


class PCDRefinement(nn.Module):

    def __init__(self, in_channels, out_channels, num_conv_layers, k):
        super(PCDRefinement, self).__init__()
        middle_channels = (in_channels + out_channels) // 2
        pcd_conv = [PCDConv(in_channels, middle_channels, k)]
        pcd_conv.extend([PCDConv(middle_channels, middle_channels, k)]*(num_conv_layers-2))
        pcd_conv.append(PCDConv(middle_channels, out_channels, k))
        self.pcd_conv = nn.Sequential(*pcd_conv)
        self.loc = nn.Conv1d(in_channels=out_channels+3,
                             out_channels=3,
                             kernel_size=1, 
                             stride=1, 
                             padding=0)
        self.tanh = nn.Tanh()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

    def forward(self, x_loc, x_feat):
        x_new_loc, x_new_feat = self.pcd_conv((x_loc, x_feat))
        x_loc_logits = torch.cat([x_loc, x_new_feat], dim=1)
        x_loc_logits = self.loc(x_loc_logits)
        x_new_loc = self.tanh(x_loc_logits)
        x_new_loc = x_loc + x_new_loc
        return x_new_loc, x_new_feat

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda


class GraphConvDecoder(nn.Module):

    def __init__(self, in_channels, pool_channels, feat_channels, out_channels):
        super(GraphConvDecoder, self).__init__()
        middle_channels = in_channels // 2
        self.spfp = SPFP([in_channels+3, 
                          middle_channels, 
                          middle_channels])
        self.linear = SharedMLP([pool_channels, 
                                 feat_channels,
                                 feat_channels])
        self.refine = PCDRefinement(feat_channels+in_channels, out_channels, num_conv_layers=3, k=3)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

    def forward(self, x_loc, x_feat, x_to_pool_from):
        x_pool_feat = self.spfp(x_loc, x_feat, x_to_pool_from)
        x_pool_feat = self.linear(x_pool_feat)
        x_new_feat = torch.cat([x_feat, x_pool_feat], dim=1)
        x_new_loc, x_new_feat = self.refine(x_loc, x_new_feat)
        return x_new_loc, x_new_feat

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda



class Im2PcdGraph(nn.Module):

    def __init__(self, num_points=1024):
        super(Im2PcdGraph, self).__init__()
        self.num_points = num_points

        resnet_encoder = models.resnet50(pretrained=True) 

        # encoder operations
        features = list(resnet_encoder.children())
        self.encoder_block1 = nn.Sequential(*features[:4])  # -> N x 64 x 56 x 56
        self.encoder_block2 = features[4]  # -> N x 256 x 56 x 56
        self.encoder_block3 = features[5]  # -> N x 512 x 28 x 28
        self.encoder_block4 = features[6]  # -> N x 1024 x 14 x 14
        self.encoder_block5 = features[7]  # -> N x 2048 x 7 x 7
        self.avgpool = resnet_encoder.avgpool
        self.fcn_encoder = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(True))
        
        # decoder operations
        self.decoder_block5 = GraphConvDecoder(in_channels=2048, 
                                               pool_channels=2048+1024+512+256+64,
                                               feat_channels=128, 
                                               out_channels=128)
        self.decoder_block4 = GraphConvDecoder(in_channels=128, 
                                               pool_channels=2048+1024+512+256+64,
                                               feat_channels=128, 
                                               out_channels=128)
        self.decoder_block3 = GraphConvDecoder(in_channels=128, 
                                               pool_channels=2048+1024+512+256+64,
                                               feat_channels=128, 
                                               out_channels=128)
        self.decoder_block2 = GraphConvDecoder(in_channels=128, 
                                               pool_channels=2048+1024+512+256+64,
                                               feat_channels=128,                                               out_channels=128)
        self.decoder_block1 = GraphConvDecoder(in_channels=128, 
                                               pool_channels=2048+1024+512+256+64,
                                               feat_channels=128, 
                                               out_channels=128)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
                                  
    def forward(self, x):
        batch_size = x.size(0)
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        # # print(x.size())
        x_enc1 = self.encoder_block1(x)
        # # print(x_enc1.size())
        x_enc2 = self.encoder_block2(x_enc1)
        # # print(x_enc2.size())
        x_enc3 = self.encoder_block3(x_enc2)
        # # print(x_enc3.size())
        x_enc4 = self.encoder_block4(x_enc3)
        # # print(x_enc4.size())
        x_enc5 = self.encoder_block5(x_enc4)
        # # print(x_enc5.size())
        x_enc6 = self.avgpool(x_enc5)
        # # print(x_enc6.size())

        x_enc6 = x_enc6.view(x_enc6.size(0), -1)
        # # print(x_enc6.size())
        x_enc6 = self.fcn_encoder(x_enc6)
        # # print(x_enc6.size())

        x_loc = sample_minibatch_from_sphere(batch_size, self.num_points)
        x_loc = x_loc.to(self.device)
        x_feat = x_enc6.unsqueeze(-1).repeat(1, 1, self.num_points)
        # # print(x_loc.size(), x_feat.size())

        x_loc, x_feat = self.decoder_block5(x_loc, 
                                            x_feat, 
                                            [x_enc5, x_enc4, x_enc3, x_enc2, x_enc1])
        # # print(x_loc.size(), x_feat.size())
        x_loc, x_feat = self.decoder_block4(x_loc, 
                                            x_feat, 
                                            [x_enc5, x_enc4, x_enc3, x_enc2, x_enc1])
        # # print(x_loc.size(), x_feat.size())
        x_loc, x_feat = self.decoder_block3(x_loc, 
                                            x_feat, 
                                            [x_enc5, x_enc4, x_enc3, x_enc2, x_enc1])
        # # print(x_loc.size(), x_feat.size())
        x_loc, x_feat = self.decoder_block2(x_loc, 
                                            x_feat, 
                                            [x_enc5, x_enc4, x_enc3, x_enc2, x_enc1])
        # # print(x_loc.size(), x_feat.size())
        x_loc, x_feat = self.decoder_block1(x_loc, 
                                            x_feat, 
                                            [x_enc5, x_enc4, x_enc3, x_enc2, x_enc1])
        # # print(x_loc.size(), x_feat.size())

        return x_loc.transpose(-2, -1)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)