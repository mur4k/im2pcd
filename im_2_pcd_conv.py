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
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif  isinstance(m, nn.BatchNorm2d):
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()

# ---------------------- MULTIBRANCH-UPCONV WITHOUT R.V.  ---------------------- #

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_conv_layers, num_groups):
        super(BasicBlock, self).__init__()
        middle_channels = (in_channels + out_channels) // 2
        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, groups=num_groups, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, groups=num_groups, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
        ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, groups=num_groups,  padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_conv_layers, num_groups):
        super(EncoderBlock, self).__init__()
        self.enc_block = nn.Sequential(BasicBlock(in_channels, out_channels, num_conv_layers, num_groups), 
                                       nn.MaxPool2d(2)
                                      )

    def forward(self, x):
        return self.enc_block(x)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_conv_layers, kernel_size, stride, num_groups):
        super(DecoderBlock, self).__init__()
        self.dec_block = nn.Sequential(nn.ConvTranspose2d(in_channels, 
                                                          in_channels, 
                                                          kernel_size=kernel_size, 
                                                          groups=num_groups, 
                                                          stride=stride), 
                                       BasicBlock(in_channels, out_channels, num_conv_layers, num_groups)
                                      )

    def forward(self, x):
        return self.dec_block(x)


class Localization(nn.Module):

    def __init__(self, in_channels, num_groups, input_size):
        super(Localization, self).__init__()
        h, w = input_size
        self.input_size = input_size
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.in_channels_per_group = self.in_channels // self.num_groups
        self.out_channels = self.in_channels // 4
        self.out_channels_per_group = self.out_channels // self.num_groups
        self.l_in_per_group = (h // 4) * (w // 4) * self.out_channels_per_group // 4
        self.l_out_per_group = self.l_in_per_group // 4
        self.l_in = self.l_in_per_group * self.num_groups
        self.l_out = self.l_in // 4

        # define the structure of localization predictor
        self.loc = nn.Sequential(EncoderBlock(self.in_channels, self.out_channels, 2, self.num_groups),
                                 EncoderBlock(self.out_channels, self.out_channels // 4, 2, self.num_groups))
        self.fc_loc = nn.Sequential(nn.Conv1d(in_channels=self.l_in,
                                              out_channels=self.l_out,
                                              kernel_size=1,
                                              stride=1,
                                              groups=self.num_groups),
                                    nn.ReLU(True),
                                    nn.Conv1d(in_channels=self.l_out,
                                              out_channels=self.l_out,
                                              kernel_size=1,
                                              stride=1,
                                              groups=self.num_groups),
                                    nn.ReLU(True),
                                    nn.Conv1d(in_channels=self.l_out,
                                              out_channels=3*2*self.num_groups,
                                              kernel_size=1,
                                              stride=1,
                                              groups=self.num_groups)
                                   )

        # initialize parameters with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]*self.num_groups, dtype=torch.float))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.loc(x)
        x = x.view(batch_size, -1).unsqueeze(-1)
        theta = self.fc_loc(x).squeeze().view(-1, 2, 3)
        return theta


class SpatialFeaturePoolingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_conv_layers, num_groups, input_size):
        super(SpatialFeaturePoolingBlock, self).__init__()
        self.num_groups = num_groups
        self.block = BasicBlock(in_channels+2*self.num_groups,
                                out_channels,
                                num_conv_layers,
                                num_groups)
        self.localization = Localization(in_channels, num_groups, input_size)

    def forward(self, x, x_to_pool_from):
        batch_size = x.size(0)
        c_, h_, w_ = x_to_pool_from.size(-3), x_to_pool_from.size(-2), x_to_pool_from.size(-1)
        c, h, w = x.size(-3) // self.num_groups, x.size(-2), x.size(-1)
        
        assert c == c_, "Channel dimensions of augmented and pooled tensors should be equal, got [{}, {}]".format(c, c_)

        theta = self.localization(x)
        grid = F.affine_grid(theta, torch.Size((batch_size*self.num_groups, c, h, w)))
        
        x_to_pool_from = x_to_pool_from.repeat(1, self.num_groups, 1, 1)
        x_to_pool_from = x_to_pool_from.view(-1, c_, h_, w_)
        
        x_to_pool_from = F.grid_sample(x_to_pool_from, grid, mode=MODE, padding_mode=PADDING_MODE)
        x_to_pool_from = x_to_pool_from.view(batch_size, self.num_groups*c, h, w)

        x_out = x + x_to_pool_from
        x_out = x_out.view(batch_size, self.num_groups, c, h, w)

        grid = grid.permute(0, 3, 1, 2)
        grid = grid.view(batch_size, self.num_groups, 2, h, w)

        x_out = torch.cat([x_out, grid], dim=2).view(batch_size, self.num_groups*(c+2), h, w)

        x_out = self.block(x_out)

        return x_out


class Im2PcdConv(nn.Module):

    def __init__(self, num_points=14*14*6):
        super(Im2PcdConv, self).__init__()
        self.num_points = num_points

        vgg_encoder = models.vgg19(pretrained=False) 
        
        # reset conv layers
        for feat in vgg_encoder.features:
            weight_reset(feat)
            weight_init(feat)
        
        # reset fc layers
        for fc in vgg_encoder.classifier:
            weight_reset(fc)
            weight_init(fc)

        # encoder operations
        features = list(vgg_encoder.features.children())
        self.maxpool_ind = [4, 9, 18, 27, 36]
        self.encoder_block1 = nn.Sequential(*features[:5])  # -> N x 64 x 112 x 112
        self.encoder_block2 = nn.Sequential(*features[5:10])  # -> N x 128 x 56 x 56
        self.encoder_block3 = nn.Sequential(*features[10:19])  # -> N x 256 x 28 x 28
        self.encoder_block4 = nn.Sequential(*features[19:28])  # -> N x 512 x 14 x 14
        self.encoder_block5 = nn.Sequential(*features[28:])  # -> N x 512 x 7 x 7

        # modify fc part
        vgg_encoder.classifier[-1] = nn.Linear(4096, 1024) 
        vgg_encoder.classifier.add_module('7', nn.ReLU(True)) # -> N x 4096
        self.avgpool = vgg_encoder.avgpool
        self.fcn_encoder = nn.Sequential(*vgg_encoder.classifier)
        
        # decoder operations
        self.decoder_block5 = DecoderBlock(in_channels=1024, 
                                           out_channels=1536, 
                                           num_conv_layers=3, 
                                           kernel_size=7, 
                                           stride=1, 
                                           num_groups=1)  # -> N x (1*1280) x 7 x 7
        self.decoder_block4 = DecoderBlock(in_channels=1536, 
                                           out_channels=1536, 
                                           num_conv_layers=3, 
                                           kernel_size=2, 
                                           stride=2, 
                                           num_groups=6)  # -> N x (6*256) x 14 x 14
        
        # spatial feature pooling operations
        # self.spfp_block4 = SpatialFeaturePoolingBlock(in_channels=2560, 
        #                                               out_channels=1280,
        #                                               num_conv_layers=3,
        #                                               num_groups=5,
        #                                               input_size=(14, 14))  # -> N x (6*256) x 14 x 14
        self.spfp_block3 = SpatialFeaturePoolingBlock(in_channels=1536, 
                                                      out_channels=768,
                                                      num_conv_layers=3,
                                                      num_groups=6,
                                                      input_size=(14, 14))  # -> N x (6*128) x 14 x 14
        self.spfp_block2 = SpatialFeaturePoolingBlock(in_channels=768, 
                                                      out_channels=384,
                                                      num_conv_layers=3,
                                                      num_groups=6,
                                                      input_size=(14, 14))  # -> N x (6*64) x 14 x 14
        self.spfp_block1 = SpatialFeaturePoolingBlock(in_channels=384, 
                                                      out_channels=192,
                                                      num_conv_layers=3,
                                                      num_groups=6,
                                                      input_size=(14, 14))  # -> N x (6*32) x 14 x 14

        self.fcn_decoder = nn.Sequential(nn.Conv2d(192, 96, kernel_size=3, groups=6, padding=1),
                                         nn.ReLU(True),
                                         nn.Conv2d(96, 18, kernel_size=1, groups=6, padding=0)
                                        )  # -> N x (5*3) x 14 x 14

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
                                  
    def forward(self, x):
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
        x_enc5 = self.avgpool(x_enc5)
        # # print(x_enc5.size())

        x_enc5 = x_enc5.view(x_enc5.size(0), -1)
        # # print(x_enc5.size())
        x_enc5 = self.fcn_encoder(x_enc5)
        # # print(x_enc5.size())

        x_enc5 = x_enc5.unsqueeze(-1).unsqueeze(-1)
        # # print(x_enc5.size())

        x_out = self.decoder_block5(x_enc5)
        # # print(x_out.size())
        x_out = self.decoder_block4(x_out)
        # # print(x_out.size())

        # x_out = self.spfp_block4(x_out, x_enc4)
        # # print(x_out.size())
        x_out = self.spfp_block3(x_out, x_enc3)
        # # print(x_out.size())
        x_out = self.spfp_block2(x_out, x_enc2)
        # # print(x_out.size())
        x_out = self.spfp_block1(x_out, x_enc1)
        # # print(x_out.size())
        
        x_out = self.fcn_decoder(x_out)
        # # print(x_out.size())

        yv, xv = torch.meshgrid([torch.linspace(-1, 1, 14), torch.linspace(-1, 1, 14)])
        x_out[:, 0::3, :, :] += xv
        x_out[:, 1::3, :, :] += yv

        x_out = x_out.permute(0, 2, 3, 1)
        # # print(x_out.size())
        x_out = x_out.contiguous().view(x_out.size(0), -1, 3)
        return x_out

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