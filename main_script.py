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

# custom functions
import data_loading_functions as datFun
import models as nnModels

np.random.seed(1)
torch.manual_seed(1)

plt.close('all')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

script_folder_path = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------------
# configuration
config_dict = {
'name':          'baseline',
'epochs':        10,
'learning_rate': 0.00001,
'optimizer':     'Adam',
'img_size':      224,
'batch_size':    256,
'data_folder':   script_folder_path + '/data/train'}


#------------------------------------------------------------------------------
# load data




#------------------------------------------------------------------------------
# train and evaluate network

# the neural network object
nnModel = CNNModel()
nnModel = nnModel.to(device)
  
opt_dict = {'SGD': torch.optim.SGD(nnModel.parameters(), lr=cfg['learning_rate']),
            'Adam': torch.optim.Adam(nnModel.parameters(), lr = cfg['learning_rate'])}
   
optimizer = opt_dict[cfg['optimizer']]
loss_fn   = nn.CrossEntropyLoss()

# train model
loss_train_list   = np.zeros((cfg['epochs'],))
loss_val_list     = np.zeros((cfg['epochs'],))
accuracy_val_list = np.zeros((cfg['epochs'],))

# epoch - one loop through all the data points
for epoch in tqdm.trange(cfg['epochs']):
#for epoch in range(cfg['epochs']):
    
    # some layers (like dropout) have different behavior when training and 
    # evaluating the model, switch to train mode
    nnModel.train()
    
    # batch update the weights
    for X_train, y_train in zip(train_data, train_targets):
        
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        
        optimizer.zero_grad()
        y_pred = nnModel(X_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
    # evaluate loss and accuracy on validation data
    with torch.no_grad():
        
        nnModel.eval()
        y_val_pred = nnModel(val_data) 
        loss_val = loss_fn(y_val_pred, val_targets)
        
        # accuracy
        correct = (torch.argmax(y_val_pred, dim=1) == val_targets).type(torch.FloatTensor)
        acc = correct.cpu().mean()
        accuracy_val_list[epoch] = acc.item()
    
    loss_val_list[epoch] = loss_val.detach().item()
    loss_train_list[epoch] = loss.detach().item()
    
# fee up GPU mem
del train_data
del train_targets
del X_train
del y_train

# evaluate test data performance

test_data = test_data[0]
test_targets = test_targets[0]
y_targets = test_targets.cpu().detach().numpy()

nnModel.plot_filters = True
y_test_pred = nnModel(test_data) 
y_test_pred = torch.softmax(y_test_pred, dim=1)
y_test_pred = y_test_pred.cpu().detach().numpy()
pred = np.argmax(y_test_pred, axis=1)
prob = np.max(y_test_pred, axis = 1)

correct = (pred == y_targets)
test_accuracy = correct.mean()  
  
print('final training batch loss: {}'.format(loss.item()))
print('final validation accuracy: {}'.format(acc.item()))
print('Test dataset accuracy: {}'.format(test_accuracy))

# plot some test predictions
for sample_index in range(10):
    title = 'label: {} predict: {}, probability: {:.2f}'.format(y_targets[sample_index], pred[sample_index], prob[sample_index])
    plot_torch_image(test_data[sample_index,:,:,:], title)
        
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


torch.cuda.empty_cache() 

