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


#------------------------------------------------------------------------------
# MNIST models

# example model given in lab notes
class MNIST_CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
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
    def __init__(self,  cfg):
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
        
        out = F.softmax(out, dim=1)
        
        return out

#------------------------------------------------------------------------------
# Dogs and Cats models

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # convolution 3
        self.cnn3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        # convolution 4
        self.cnn4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        
        # convolution 5
        self.cnn5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # image to fully connected 1
        self.fc1 = nn.Linear(9216, 4096) 
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        
        # fc 1 to fc2
        self.fc2 = nn.Linear(4096, 4096) 
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        
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
# helper functions related to models

def initialize_weights(cfg, m):
    
  if isinstance(m, nn.Linear):
      if cfg['initialization'] == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
      elif cfg['initialization'] == 'xavier':
          nn.init.xavier_uniform_(m.weight.data)
      elif cfg['initialization'] == 'normal':
          nn.init.normal_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)