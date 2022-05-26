import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNNModel(nn.Module):
    def __init__(self, num_x_chan, D):
        """
        num_x_chan - int: the number of channels of the input images
        D - int: the total number of layers the denoiser should have. The paper
                 used 17 and 20
        """
        super(DnCNNModel, self).__init__()

        # Conv+ReLU: 64 chan
        self.init_layer = nn.Conv2d(in_channels=num_x_chan, out_channels=64, kernel_size=3, padding='same')

        # Conv+BN+ReLU
        hidden_layers = []
        for i in range(D - 2):
            hidden_layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'))

        self.conv_bn_layers = nn.ModuleList(hidden_layers)

        # Conv layer for output
        self.final_layer = nn.Conv2d(in_channels=64, out_channels=num_x_chan, kernel_size=3, padding='same')


    def forward(self, x):
        x = F.ReLU(self.init_layer(x))

        for i in range(len(self.conv_bn_layers)):
            x = self.conv_bn_layers[i](x)
            bn = nn.BatchNorm1d(64)
            x = F.ReLU(bn(x))
        # not sure if i should use a ReLU on the final layer
        v = self.final_layer(x)
        return v
