import torch
import torch.nn as nn

from denoiser_training import DnCNN
import classifier_models as cm


class CombinedModel(nn.Module):
    def __init__(self, device):
        super(CombinedModel, self).__init__()

        if device == 'cuda':
            map_loc = torch.device('cuda')
        else:
            map_loc = torch.device('cpu')

        self.denoiser = torch.load('denoiser_models/DnCNN_sigma25/model_005.pth',
                                       map_location=map_loc)
        self.classifier = cm.binaryResNet()
        self.classifier.load_state_dict(torch.load('cats_dogs_resNet18.pt',
                                                   map_location=map_loc))

    def forward(self, x):
        denoised_img = self.denoiser(x)
        y = self.classifier(denoised_img)
        return y
        
