import torch
import torch.nn as nn
from networks import (resnet18_3d, HalfUNet, Simple_C3D,
                      SparseAutoencoderKL, Pretrained_C2D, ConvRNN)


def freeze_layers(layers):
    for child in layers:
        for p in child.parameters():
            p.requires_grad = False


def init_net(opt, in_channels):
    if opt == '1':
        # Replace the first layer with pre-trained sparse-AE weights
        # load pre-trained sparse-AE
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
    elif opt == '2':
        # Transfer learning with 3D ConvNets pre-trained with Kinetics-400
        net = Pretrained_C2D(in_channels=in_channels)

        locator = 0
        drop_rate = 0.3

        # append dropout at each conv group
        for i in range(2, len(net.feature)-1):
            net.feature[i] = nn.Sequential(
                *(list(net.feature[i].children()) + [nn.Dropout(drop_rate)]))
        # for param in net.feature[i].parameters():
        #     param.requires_grad_(False)

        locator += 1
        if locator % 2 == 0:
            drop_rate += 0.1

    elif opt == '3':
        # Transfer learning with pre-trained UNets
        net = HalfUNet(
            in_channels=in_channels,
            n_class=2)

    elif opt == '4':
        net = resnet18_3d(num_classes=2)

    elif opt == '5':
        net = ConvRNN()

    net.cuda()

    return net
