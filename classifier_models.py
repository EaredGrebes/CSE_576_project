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
from os.path import exists
from torchvision import models

#------------------------------------------------------------------------------
# MNIST models

# example model given in lab notes
class MNIST_CNNModel(nn.Module):
    def __init__(self):
        super(MNIST_CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, 
                              kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, 
                              kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 5 * 5, 10) 
    
    def forward(self, x):
        # Input x dimensions:   #nx1x28x28
        # Set 1
        out = self.cnn1(x)      #nx16x26x26
        out = self.relu1(out) 
        out = self.maxpool1(out)#nx16x13x13
        
        # Set 2
        out = self.cnn2(out)    #nx32x11x11
        out = self.relu2(out)   
        out = self.maxpool2(out)#nx32x5x5
        
        #Flatten
        out = out.view(out.size(0), -1) #nx800

        #Dense
        out = self.fc1(out)     #nx10
        
        return out


# an implementation of the LeNet model
class my_LeNet_Model(nn.Module):
    def __init__(self):
        super(my_LeNet_Model, self).__init__()    
        
        # layer 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=6, 
                              kernel_size=5, stride=1, padding=2)
        self.tanh1 = torch.tanh
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
     
        # layer 1
        self.cnn2 = nn.Conv2d(in_channels=6, out_channels=16, 
                              kernel_size=5, stride=1, padding=0)
        self.tanh2 = torch.tanh
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.tanh3 = torch.tanh
        
        # Fully connected 1
        self.fc2 = nn.Linear(120, 84) 
        self.tanh4 = torch.tanh
    
        # Fully connected 1
        self.fc3 = nn.Linear(84, 10) 
        self.tanh5 = torch.tanh
        
    def forward(self, x):
        # Input x dimensions:   #nx1x28x28
        # Set 1
        out = self.cnn1(x)      
        out = self.tanh1(out) 
        out = self.pool1(out)
        
        # Set 2
        out = self.cnn2(out)  
        out = self.tanh2(out)   
        out = self.pool2(out)
        
        #Flatten
        out = out.view(out.size(0), -1) #nx800

        #Dense
        out = self.fc1(out)    
        out = self.tanh3(out)
        
        out = self.fc2(out)    
        out = self.tanh4(out)
        
        out = self.fc3(out)    
        out = self.tanh5(out)
        
        #out = F.softmax(out, dim=1)
        
        return out

#------------------------------------------------------------------------------
# Dogs and Cats models

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class DG_CNNModel(nn.Module):
    def __init__(self):
        super(DG_CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5)
        self.relu1 = nn.ReLU()
        #self.relu1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        #self.relu2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # convolution 3
        self.cnn3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        #self.relu3 = nn.Tanh()
        
        # convolution 4
        self.cnn4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        #self.relu4 = nn.Tanh()
        
        # convolution 5
        self.cnn5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        #self.relu5 = nn.Tanh()
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # image to fully connected 1
        self.fc1 = nn.Linear(9216, 4096) 
        self.relu6 = nn.ReLU()
        #self.relu6 = nn.Tanh()
        self.dropout1 = nn.Dropout(p=0.3)
        
        # fc 1 to fc2
        self.fc2 = nn.Linear(4096, 4096) 
        self.relu7 = nn.ReLU()
        #self.relu7 = nn.Tanh()
        self.dropout2 = nn.Dropout(p=0.3)
        
        # fc3 to output
        self.fc3 = nn.Linear(4096, 2) 
        
        self.init_bias()
        
        
    def init_bias(self):
        nn.init.normal_(self.cnn1.weight, mean=0, std=0.01)
        nn.init.normal_(self.cnn2.weight, mean=0, std=0.01)
        nn.init.normal_(self.cnn3.weight, mean=0, std=0.01)
        nn.init.normal_(self.cnn4.weight, mean=0, std=0.01)
        nn.init.normal_(self.cnn5.weight, mean=0, std=0.01)
        
        nn.init.constant_(self.cnn1.bias, 0)
        nn.init.constant_(self.cnn2.bias, 1)
        nn.init.constant_(self.cnn3.bias, 0)
        nn.init.constant_(self.cnn4.bias, 1)
        nn.init.constant_(self.cnn5.bias, 1)
    
    
    def forward(self, out):

        out = self.cnn1(out)   
        out = self.relu1(out) 
        out = self.maxpool1(out) # [b x 96 x 27 x 27]
        
        out = self.cnn2(out)     
        out = self.relu2(out) 
        out = self.maxpool2(out)
        
        out = self.cnn3(out)    
        out = self.relu3(out)  
        
        out = self.cnn4(out)    
        out = self.relu4(out)  
        
        out = self.cnn5(out)     
        out = self.relu5(out) 
        out = self.maxpool5(out)

        out = out.view(out.size(0), -1) 
        
        # fully connected
        out = self.fc1(out)    
        out = self.relu6(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu7(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out
    
#------------------------------------------------------------------------------
# torchvision resnet
def binaryResNet():
    
    model = models.resnet18()
    num_classes = 2
    model.fc = nn.Linear(512, num_classes)
    return model

def binaryVGGNet():
    
    model = models.vgg11_bn()
    num_classes = 2
    model.classifier[6] = nn.Linear(4096,num_classes)
    return model


#------------------------------------------------------------------------------
# helper functions related to models

def load_model(train_data, train_targets, val_data, val_targets, config, device, force_train):
    nnModel = config['model']()
    model_filename = config['name'] + '.pt'
    
    # train from scratch    
    if force_train or not exists(model_filename):
        print('training model')
        loss_val_list, \
        loss_train_list, \
        accuracy_val_list = train_model(nnModel, \
                                        train_data, \
                                        train_targets, \
                                        val_data, \
                                        val_targets, \
                                        config, \
                                        device)
            
        torch.save(nnModel.state_dict(), model_filename)
        
        np.savez(config['name'], 
                 loss_val_list     = np.array(loss_val_list), 
                 loss_train_list   = np.array(loss_train_list),
                 accuracy_val_list = np.array(accuracy_val_list) )
    
    # check to see if model has already been saved
    else:
        nnModel.load_state_dict(torch.load(model_filename))
        nnModel = nnModel.to(device)
        print('Model loaded.')
        
        filez = np.load(config['name'] + '.npz', allow_pickle=True)
        loss_val_list = filez['loss_val_list']
        loss_train_list = filez['loss_train_list']
        accuracy_val_list = filez['accuracy_val_list']
        
    return nnModel, loss_val_list, loss_train_list, accuracy_val_list     
    

def train_model(nnModel, train_data, train_targets, val_data, val_targets, config, device):

    nnModel = nnModel.to(device)
    val_data = val_data.to(device)
    val_targets = val_targets.to(device)
      
    opt_dict = {'SGD': torch.optim.SGD(nnModel.parameters(),   lr = config['learning_rate']),
                'Adam': torch.optim.Adam(nnModel.parameters(), lr = config['learning_rate'])}
       
    optimizer = opt_dict[config['optimizer']]
    loss_fn   = nn.CrossEntropyLoss()
    
    # train model
    loss_train_list   = np.zeros((config['epochs'],))
    loss_val_list     = np.zeros((config['epochs'],))
    accuracy_val_list = np.zeros((config['epochs'],))
    
    # epoch - one loop through all the data points
    for epoch in tqdm.trange(config['epochs']):
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
            print(f'validation accuracy: {acc.item()}')
        
        loss_val_list[epoch] = loss_val.detach().item()
        loss_train_list[epoch] = loss.detach().item()
        
    # fee up GPU mem
    del train_data
    del train_targets
    del X_train
    del y_train
    
    return loss_val_list, loss_train_list, accuracy_val_list