import os
import re
import glob
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from denoiser_training import DnCNN
import classifier_models as cm

# to make a DataLoader, do the following:

# from torch.utils.data import DataLoader
# dataset = DogsCatsDataset()
# dataloader = DataLoader(dataset=dataset, batch_size=some_num, shuffle=True)
# data_iter = iter(dataloader) ... or ... enumerate(dataloader)

class DogsCatsDataset(Dataset):
    """
    Creates the dogs_cats training dataset. Must have the 'dc_train_filenames.txt' file
    in the 'data/dogs_cats/' directory.
    Assumes that the images are in the directory: 'data/dogs_cats/train/'
    Resizes all images to a standard size given as height, width parameters.
    Defaults to 200x200.
    The tensors of the image data are in the range [0, 1].
    A cat image has a label of 0.
    A dog image has a label of 1.
    """

    def __init__(self, height=200, width=200):
        self.directory = "data/dogs_cats/"
        self.h = height
        self.w = width
        self.data_keys_list = []
        self.labels = []
        self.map_fn_2_idx = {}
        self.map_idx_2_fn = {}

        with open(self.directory + "dc_train_filenames.txt", "r") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                filename = line.strip()
                self.map_fn_2_idx[filename] = i
                self.map_idx_2_fn[i] = filename
                self.data_keys_list.append(filename)
                if filename[:3] == "cat":
                    self.labels.append(0)
                else:
                    self.labels.append(1)

        self.num_samples = len(self.data_keys_list)
        x0, y0 = self.__getitem__(0)
        self.num_x_chan = x0.size()[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """index - int"""
        path = self.directory + "train/" + self.data_keys_list[index]
        img = read_image(path)
        img_resize = resize(img, [self.h, self.w])
        img_normed = img_resize / 255.0
        return img_normed, self.labels[index]


class CombinedModel(nn.Module):
    """returns soft predictions"""
    def __init__(self, device, sigma, denoiser_dir='./denoiser_models/'):
        super(CombinedModel, self).__init__()

        # if device == 'cuda':
        #     map_loc = torch.device('cuda')
        # else:
        #     map_loc = torch.device('cpu')

        directory = denoiser_dir + f'DnCNN_sigma{int(sigma)}/'
        denoiser_fn = self.get_latest_model_fn(directory)
        self.denoiser_path = directory + denoiser_fn

        print(f"denoiser filename: {denoiser_fn}")

        self.denoiser = torch.load(self.denoiser_path, map_location=device)
        self.classifier = cm.binaryResNet()
        self.classifier.load_state_dict(torch.load('cats_dogs_resNet18.pt',
                                                   map_location=device))
        self.classifier.requires_grad_(requires_grad=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        denoised_img = self.denoiser(x)
        y = self.classifier(denoised_img)
        y = self.sm(y)
        return y

    def get_latest_model_fn(self, directory):
        file_list = os.listdir(directory)
        latest = 0
        latest_str = '000'
        for filename in file_list:
            if filename[:5] == 'model':
                num_str = filename[6:9]
                num_str_strip = num_str.lstrip('0')
                if int(num_str_strip) > latest:
                    latest = int(num_str_strip)
                    latest_str = num_str
        return f'model_{latest_str}.pth'


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def train():
    # torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device('cpu')
    print(f"device is {device}")


    #### Hypers
    sigma = 9.0
    lr = 1e-3
    n_epoch = 90
    # data_dir = None #TODO
    save_dir = f"./combined_models/DnCNN_sigma{int(sigma)}/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    batch_size = 8


    #### build model
    print('===> Building model')
    model = CombinedModel(device, sigma)
    model.to(device=device)
    sole_classifier = cm.binaryResNet()
    sole_classifier.load_state_dict(torch.load('cats_dogs_resNet18.pt', map_location=device))
    sole_classifier.requires_grad_(requires_grad=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # make data
    dataset = DogsCatsDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    # data_iter = iter(dataloader) ... or ... enumerate(dataloader)

    epoch_loss_list = []
    #### training loop
    for epoch in range(n_epoch):
        epoch_loss = 0
        start_time = time.time()

        for n_count, (x, y) in enumerate(dataloader):
            x = x.to(device=device)
            y = y.to(device=device)
            
            no_noise_pred = sole_classifier(x)
            hard_pred = torch.argmax(no_noise_pred, dim=1)

            noise = (torch.randn(x.size()) * sigma / 255.0).to(device=device)
            x_plus_noise = x + noise
            soft_pred = model(x_plus_noise)
            #print(soft_pred)
            #print(f"size of pred: {soft_pred.size()}")
            #print(no_noise_pred)

            loss = criterion(soft_pred, hard_pred)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss_list.append(epoch_loss)
        elapsed_time = time.time() - start_time
        log('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        np.savetxt('stab_train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))


if __name__ == "__main__":
    train()
