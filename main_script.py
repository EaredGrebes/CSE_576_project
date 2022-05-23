import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2 
from random import shuffle 
import seaborn as sns
import gc

# custom functions
import data_loading_functions as datFun
import classifier_models as classFun

np.random.seed(1)
torch.manual_seed(1)

plt.close('all')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(device)

script_folder_path = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------------
# configuration
load_data   = True  # if you run the script in the same worspace, you don't need to load the data each time
force_train = False

# evaluate ideas against the Cats and Dogs dataset
CD_config = {
'name':          'cats_dogs_resNet18',
'epochs':        40,
'learning_rate': 0.0005,
'optimizer':     'Adam',
'img_size':      224,
'batch_size':    128,
'data_folder':   script_folder_path + '/data/dogs_cats/train',
'load_function': datFun.load_cats_dogs_data,
'model':         classFun.binaryResNet}

# for quicker code development, use MNIST
MNIST_config = {
'name':          'MNIST_LeNet',
'epochs':        10,
'learning_rate': 0.001,
'optimizer':     'Adam',
'img_size':      28,
'batch_size':    256,
'data_folder':   script_folder_path + '/data/',
'load_function': datFun.load_MNIST_data,
'model':         classFun.my_LeNet_Model}

# select which configuration to run
#config = CD_config
config = MNIST_config

#------------------------------------------------------------------------------
# load data
if load_data:
    print('loading data')
    # train data is batched, val and test are not
    train_data,    \
    train_targets, \
    val_data,      \
    val_targets,   \
    test_data,     \
    test_targets = config['load_function'](config['batch_size'],\
                                           config['img_size'],  \
                                           config['data_folder'])

#------------------------------------------------------------------------------
# load model (or train from scratch)
 
nnModel, \
loss_val_list, \
loss_train_list, \
accuracy_val_list = classFun.load_model(train_data, train_targets, val_data, val_targets, config, device, force_train)    

# plot loss and accuracy curves from training
plt.figure()
plt.plot(loss_train_list)
plt.plot(loss_val_list)
plt.title('Loss')
plt.xlabel('training epoch')
plt.legend(['training loss (single batch)', 'validation loss'])

plt.figure()
plt.plot(accuracy_val_list)
plt.title('validation accuracy')
plt.xlabel('training epoch')

#------------------------------------------------------------------------------
# evaluate test data performance
y_targets = test_targets.detach().numpy()

test_data = test_data.to(device)
y_test_pred = nnModel(test_data) 
y_test_pred = torch.softmax(y_test_pred, dim=1)
y_test_pred = y_test_pred.cpu().detach().numpy()
pred = np.argmax(y_test_pred, axis=1)
prob = np.max(y_test_pred, axis = 1)

correct = (pred == y_targets)
test_accuracy = correct.mean()  
print('Test dataset accuracy: {}'.format(test_accuracy))

# plot some test predictions
for sample_index in range(10):
    title = 'label: {} predict: {}, probability: {:.2f}'.format(y_targets[sample_index], pred[sample_index], prob[sample_index])
    datFun.plot_torch_image(test_data[sample_index,:,:,:], title)
        

