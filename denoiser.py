import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNNModel(nn.Module):
    """
    This class is from the work by Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and
    Lei Zhang. The original code is here:
    https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_pytorch/main_train.py
    """
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        """
        Tensor values coming in should already be in the range [0, 1]
        """
        super(DnCNNModel, self).__init__()
        padding = 1

        layers = []
        # Initial layer
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                       kernel_size=3, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # Hidden layers
        for i in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))

        # Conv layer for output
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                                       kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out
