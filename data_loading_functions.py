import numpy as np
import matplotlib.pyplot as plt
import tqdm
from functools import partial
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2 
from random import shuffle 
import seaborn as sns


#------------------------------------------------------------------------------
# MNIST functions
def get_next_data_batch(loader, device):
    
    batches = enumerate(loader)
    batch_idx, (data, targets) = next(batches)
    
    data = data.to(device)
    targets = targets.to(device)
    
    return data, targets


def convert_batches_to_device(loader, device):
    
    batch_data = []
    batch_targets = []
    for batch_idx, (data, targets) in enumerate(loader):
        batch_data.append(data.to(device))
        batch_targets.append(targets.to(device))
        
    return batch_data, batch_targets


#------------------------------------------------------------------------------
# Cats and Dogs functions

# main data loading function
def load_cats_dogs_data(train_batch_size, img_size, trainFolder):
    
    train_size = 24000
    val_size = 500
    test_size = 500
    
    # 25000 total images, don't exceed that (GPU issues)
    total_size = train_size + val_size + test_size
    
    # only use train data, test data is un-labeled
    train_image_list = os.listdir(trainFolder)[0:total_size]
     
    # spot-check some data 
    #plot_image_list_count(train_image_list)
    
    raw_data = process_raw_image_data(train_image_list, trainFolder)
    
    # split data into train, validation, and test
    raw_data_train = raw_data[0:train_size]
    raw_data_val   = raw_data[train_size:train_size + val_size]
    raw_data_test  = raw_data[train_size + val_size:train_size + val_size + test_size]
    
    # convert raw data into torch batches
    # NOTE: this data-set is so big, it cannot be fit on the GPU all at once
    # instead, keep it all on the cpu, and swap out batches on the gpu
    train_data, train_targets = images_to_torch_batch(raw_data_train, train_batch_size, 'cpu')
    val_data,   val_targets   = images_to_torch_batch(raw_data_val, val_size, 'cpu')  # singleton batch
    test_data,  test_targets   = images_to_torch_batch(raw_data_test, val_size, 'cpu')  # singleton batch
    
    # val and test data are single batches
    val_data    = val_data[0]
    val_targets = val_targets[0]
    test_data    = test_data[0]
    test_targets = test_targets[0]
    
    return train_data, train_targets, val_data, val_targets, 

def binary_label(img):
    pet = img.split('.')[-3]
    if pet == 'cat': return 1
    elif pet == 'dog': return 0
    
    
def process_raw_image_data(data_image_list, data_folder, img_size):
    data_list = []
    for img in tqdm.tqdm(data_image_list):
        
        path = os.path.join(data_folder,img)
        label = binary_label(img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_size,img_size))
        data_list.append([np.array(img),np.array(label)])
    shuffle(data_list)
    return data_list


def images_to_torch_batch(raw_data, batch_size, device):
    N = len(raw_data)
    if batch_size < N:
        steps = np.floor(N/batch_size).astype(int)
        intervals = np.arange(0, (steps+1)*batch_size+1, batch_size)
        intervals[-1] = N
    else:
        intervals = np.array([0, N])
    
    print(intervals)
    
    im_h, im_w, im_c = raw_data[0][0].shape
    
    torch_data = []
    torch_targets = []
    for batch in range(1, len(intervals)):
        
        batch_indexes = np.arange(intervals[batch-1], intervals[batch])
        X_tensor = torch.zeros(len(batch_indexes), im_c, im_h, im_w)
        y_vector = torch.zeros((len(batch_indexes)))
        
        for ii, batch_ind in enumerate(batch_indexes):
            
            X_tensor[ii,0,:,:] = torch.from_numpy(raw_data[batch_ind][0][:,:,2])
            X_tensor[ii,1,:,:] = torch.from_numpy(raw_data[batch_ind][0][:,:,1])
            X_tensor[ii,2,:,:] = torch.from_numpy(raw_data[batch_ind][0][:,:,0])
            
            y_vector[ii] = torch.from_numpy(raw_data[batch_ind][1])
         
        y_vector = y_vector.long()
        torch_data.append(X_tensor.to(device))
        torch_targets.append(y_vector.to(device))
    
    return torch_data, torch_targets
    

def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split('.')[-3])
    sns.countplot(labels)
    plt.title('Cats and Dogs')
    
    
def plot_torch_image(tensor, title):
    
    np_tensor = tensor.cpu().detach().numpy()
    im_c, im_h, im_w = np_tensor.shape
    np_tensor2 = np.stack((np_tensor[0,:,:], np_tensor[1,:,:], np_tensor[2,:,:]), axis = 2)
    
    plt.figure(title)
    plt.title(title)
    plt.imshow(np_tensor2/255)
    
def plot_torch_feature_map(tensor, title): 
    mat = tensor.cpu().detach().numpy()
    map_normed = (mat - mat.min()) / (mat.max() - mat.min())
    
    plt.figure(title)
    plt.title(title)
    plt.imshow(map_normed)
    