import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18_3d(**kwargs):
    r"""ResNet-18 (3D)model
    """
    return ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)


class UNetEnc(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # load pre-trained UNet
        unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                              in_channels=3, out_channels=1, init_features=32, pretrained=True)

        unet_named_childs = unet.named_children()

        unet_enc = {}

        while 1:
            name, child = next(unet_named_childs)
            if 'up' in name:
                break
            unet_enc[name] = child

        self.model = nn.ModuleDict(unet_enc)

        # replace first conv layer with newone
        self.model.encoder1.enc1conv1 = nn.Conv2d(
            in_channels, out_channels=32, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.model.encoder1(x)
        x = self.model.encoder2(self.model.pool1(x))
        x = self.model.encoder3(self.model.pool2(x))
        x = self.model.encoder4(self.model.pool3(x))
        x = self.model.bottleneck(self.model.pool4(x))

        return x


class HalfUNet(nn.Module):
    def __init__(self, in_channels, n_class=2):
        super().__init__()
        self.enc = UNetEnc(in_channels=in_channels)   # pretrained UNet encoder
        self.fc = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.enc(x)   # (N,512,16,16)

        # GAP
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class SparseAutoencoderKL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7*7*7),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Pretrained_C2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # resnet
        net = models.resnet18(pretrained=True)

        layers = list(net.children())

        for _ in range(3):
            layers.pop(0)

        layers.insert(0, ResidualBlock(in_channels, 64))

        self.feature_dim = layers.pop(-1).in_features

        # feature extractor
        self.feature = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 2, bias=False)
        )

    def forward(self, x):
        x = self.feature(x).view(-1, self.feature_dim)

        x = self.fc(x)

        return x


class Simple_C3D(nn.Module):
    def __init__(self, ae_w):
        super().__init__()

        self.pretrained_conv = nn.Conv3d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.feature = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),

            nn.Conv3d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.fc = nn.Linear(256, 2)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            if name == 'pretrained_conv':
                # pre-trained sparseAE weight
                m.weight.data = ae_w
                m.weight.requires_grad_(False)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # convolve with pre-trained sparse-AE weights & bias
        x = self.pretrained_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.feature(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class ConvRNN(nn.Module):
    def __init__(self, ):
        super().__init__()
        # embedding layer : pretrained CNN
        layers = list(models.resnet152(pretrained=True).children())
        self.embedding = nn.Sequential(*layers[:-1])
        self.lstm = nn.LSTM(layers[-1].in_features, 128, 2)
        self.fc = nn.Linear(128, 2)

    def feature_extraction(self, x):

        xs = x.permute(0, 2, 1, 3, 4)  # seq of x

        feats = []
        for t in range(xs.size(1)):
            emb = self.embedding(x)   # (b,c,1,1)
            emb = torch.flatten(emb, 1)  # (b,c)
            feats.append(emb)
        feats = torch.stack(feats)

        return feats

    def forward(self, x):
        feats = self.feature_extraction(x)
        lstm_outs, _ = self.lstm(feats)
        final_out = lstm_outs[-1]

        pred = self.fc(final_out)

        return pred


if __name__ == '__main__':

    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # init net
    autoencoder = SparseAutoencoderKL()
    autoencoder.load_state_dict(torch.load(
        './history/sparse_autoencoder_KL.pt'))

    fc_layers = [m for m in list(
        autoencoder.encoder.children()) if type(m) == nn.Linear]
    cascaded_weight = fc_layers[0].weight.t()
    for layer in fc_layers[1:]:
        cascaded_weight = cascaded_weight.matmul(
            layer.weight.t()
        )

    ae_w = cascaded_weight.t().view(-1, 1, 7, 7, 7)

    net = Simple_C3D(ae_w=ae_w)
    net.cuda()

    # test code
    N = 4
    C = 1
    D = 178
    H = 112
    W = 112

    args = (N, C, D, H, W)

    test_input = torch.Tensor(*args)           # (N,C,D,H,W)
    out = net(test_input.cuda(0))

    print(out.shape)

    # net = HalfUNet(
    #     in_channels=5,
    #     n_class=2)
    # test_input = torch.Tensor(1,5,256,256)
    # out = net(test_input)
    # print(out.shape)
