import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from
#  https://github.com/fanhqme/PointSetGeneration/blob/master/depthestimate/train_nn.py


def make_conv_layer(repeat, input, output, filter=3, stride=1):
    layers = []
    for i in range(repeat):
        layers.append(
            nn.Conv2d(input, output, filter, stride=stride, padding=filter//2))
        layers.append(nn.BatchNorm2d(output))
        layers.append(nn.ReLU(True))
        input = output
        stride = 1
    return nn.Sequential(*layers)


class DeconvBlock(nn.Module):

    def __init__(self, input, input_, output):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            input, output, 4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(output)
        self.conv1 = nn.Conv2d(input_, output, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(output)
        self.conv2 = nn.Conv2d(output, output, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(output)

    def forward(self, x, x_enc):
        # import pdb; pdb.set_trace()
        x = self.norm(self.deconv(x))
        x_enc = self.norm1(self.conv1(x_enc))
        # import pdb; pdb.set_trace()
        x = F.relu(x + x_enc)
        x = F.relu(self.norm2(self.conv2(x)))
        return x, x_enc


class PointSetGen(nn.Module):

    def __init__(self):
        super().__init__()
        self.net0 = make_conv_layer(2, 3, 16)
        self.net1 = make_conv_layer(3, 16, 32, stride=2)  # 112, 112
        self.net2 = make_conv_layer(3, 32, 64, stride=2)  # 56, 56
        self.net3 = make_conv_layer(3, 64, 128, stride=2)  # 28, 28
        self.net4 = make_conv_layer(3, 128, 256, stride=2)  # 14, 14
        self.net5 = make_conv_layer(3, 256, 512, stride=2)  # 7, 7
        # FC branch
        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 14*14*6*3 - 28*28*3),
        )
        # Deconv branch 1
        self.net5d = DeconvBlock(512, 256, 128)  # 14 14
        self.net4d = DeconvBlock(128, 128, 64)  # 28 28
        self.net3d = DeconvBlock(64, 64, 32)  # 56 56
        self.net2d = DeconvBlock(32, 32, 16)  # 112 112
        # Deconv branch 2
        self.net1_2 = make_conv_layer(3, 16, 32, stride=2)  # 56, 56
        self.net2_2 = make_conv_layer(3, 32, 64, stride=2)  # 28, 28
        self.net3_2 = make_conv_layer(3, 64, 128, stride=2)  # 14, 14
        self.net4_2 = make_conv_layer(3, 128, 256, stride=2)  # 7, 7
        self.net1_2d = DeconvBlock(256, 128, 128)  # 14 14
        self.net2_2d = DeconvBlock(128, 64, 64)  # 28, 28
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.net0(x)
        x = x1 = self.net1(x)
        x = x2 = self.net2(x)
        x = x3 = self.net3(x)
        x = x4 = self.net4(x)
        x = x5 = self.net5(x)
        assert x.shape[-2:] == (7, 7)
        # FC out
        yfc = self.fc(x.view(x.size(0), -1)).view(x.size(0), -1, 3)
        # Deconv Branch 1
        x, x5 = self.net5d(x, x4)
        x, x4 = self.net4d(x, x3)
        x, x3 = self.net3d(x, x2)
        x, x2 = self.net2d(x, x1)
        # Deconv Branch 2
        x = self.net1_2(x)
        x = x2 = self.net2_2(x)
        x = x3 = self.net3_2(x)
        x = x4 = self.net4_2(x)
        x, _ = self.net1_2d(x, x3)
        x, _ = self.net2_2d(x, x2)
        # Deconv out
        yconv = self.out(x)
        yconv = yconv.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 3)
        return torch.cat([yconv, yfc], -2)

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



if __name__ == "__main__":
    x = torch.rand(4, 3, 224, 224)
    net = PointSetGen()
    print(net(x).shape)
    print(net)
