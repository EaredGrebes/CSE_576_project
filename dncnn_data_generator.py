# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# no need to run this code separately


import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, sigma, data_dir='data/denoising_train_data/Train400'):
        super(DenoisingDataset, self).__init__()
        self.data_dir = data_dir
        self.sigma = sigma
        with open(self.data_dir + f"/patched/data_len.txt", "r") as f:
            line = f.readline()
        self.num_images = int(line.strip())


    def __getitem__(self, index):
        # loads the image (which has already been put in [0, 1] range
        # adds gaussian noise normalized for the image range

        batch_x = np.load(self.data_dir + f"/patched/patched{index}.npy")
        batch_x = torch.from_numpy(batch_x.transpose((2, 0, 1)))
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.num_images


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    # get multiscale patches from a single image
    # img = cv2.imread(file_name, 0)  # gray scale
    img = cv2.imread(file_name)  # color
    # img is a np.ndarray at this point of shape: (h, w, 3)
    h, w, c = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        ### WARNING: this cv2.resize function flips the dimension order
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[j:j+patch_size, i:i+patch_size, :]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir='data/denoising_train_data/Train400', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.jpg')  # using jpg
    # initialize
    data = []
    # generate patches
    num_orig_imgs = len(file_list)
    counter = 0
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        num_patches = len(patches)
        for j, patch in enumerate(patches):
            #data.append(patch)
            counter += 1
            idx = i * num_patches + j
            np.save(data_dir + f"/patched/patched{idx}.npy", patch.astype('float32') / 255.0)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')

    # no need to add a dimension, because we're keeping the color dims
        
    with open(data_dir + f"/patched/data_len.txt", "w") as f:
        f.write(f"{counter}\n")

    print('^_^-training data finished-^_^')



if __name__ == '__main__': 

    data = datagenerator(verbose=True)
