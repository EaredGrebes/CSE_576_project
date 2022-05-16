import numpy as np
import matplotlib.pyplot as plt
import tqdm
from functools import partial
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

np.random.seed(1)
torch.manual_seed(1)

plt.close('all')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device "{device}"')

script_folder_path = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------------
# functions
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
# example model given in lab notes
class CNNModel(nn.Module):
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
    
    
def initialize_weights(cfg, m):
    
  if isinstance(m, nn.Linear):
      if cfg['initialization'] == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
      elif cfg['initialization'] == 'xavier':
          nn.init.xavier_uniform_(m.weight.data)
      elif cfg['initialization'] == 'normal':
          nn.init.normal_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
  #------------------------------------------------------------------------------
  # load data
  train_batch_size = 256  # Define train batch size
  train_val_split = 0.8

  dataFolder = os.path.dirname(script_folder_path) + '/data/'

  # Use the following code to load and normalize the dataset
  torch_train_dataset = torchvision.datasets.MNIST(dataFolder, train=True, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

  # split the training data into training and validation
  train_size = len(torch_train_dataset)
  new_train_size = int(train_val_split * train_size)
  val_size = train_size - new_train_size

  train_dataset, val_dataset = torch.utils.data.random_split(torch_train_dataset, [new_train_size, val_size])

  # use batches for training
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_batch_size, shuffle=True)
  train_data, train_targets = convert_batches_to_device(train_loader, device)

  # use full data for validation set (no batches)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = len(val_dataset), shuffle=True)
  val_data, val_targets = get_next_data_batch(val_loader, device)

  # use full data for test set (no batches)
  torch_test_dataset = torchvision.datasets.MNIST(dataFolder, train=False, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

  test_loader = torch.utils.data.DataLoader(torch_test_dataset, batch_size = len(torch_test_dataset), shuffle=True)
  test_data, test_targets = get_next_data_batch(test_loader, device)
    

  #------------------------------------------------------------------------------
  # train and evaluate network
  def create_baseline_config():
      cfg = {'name': 'baseline',
             'network_config': 'config_1',
             'epochs': 10,
             'learning_rate': 0.0015,
             'optimizer': 'Adam',
             'initialization': 'None'}
      return cfg

  cfg = create_baseline_config()

  cfg['epochs'] = 6 # Shorter compute time!
      
  # the neural network object
  nnModel = my_LeNet_Model(cfg)
  #nnModel = CNNModel()
  #init_weights_fun = partial(initialize_weights, cfg)
  #nnModel.apply(init_weights_fun)
  nnModel = nnModel.to(device)
    
  opt_dict = {'SGD': torch.optim.SGD(nnModel.parameters(), lr=cfg['learning_rate']),
              'Adam': torch.optim.Adam(nnModel.parameters(), lr = cfg['learning_rate'])}
     
  optimizer = opt_dict[cfg['optimizer']]
  loss_fn   = nn.CrossEntropyLoss()

  # train model
  loss_train_list = np.zeros((cfg['epochs'],))
  loss_val_list = np.zeros((cfg['epochs'],))
  accuracy_val_list = np.zeros((cfg['epochs'],))

  # epoch - one loop through all the data points
  for epoch in tqdm.trange(cfg['epochs']):
  #for epoch in range(cfg['epochs']):
      
      # some layers (like dropout) have different behavior when training and 
      # evaluating the model, switch to train mode
      
      nnModel.train()
      # batch update the weights
      for X_train, y_train in zip(train_data, train_targets):
          
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
         
      #print('training batch loss: {}'.format(loss.item()))
      #print('validation accuracy: {}'.format(acc.item()))
      
      loss_val_list[epoch] = loss_val.detach().item()
      loss_train_list[epoch] = loss.detach().item()
      
  y_test_pred = nnModel(test_data) 
  correct = (torch.argmax(y_test_pred, dim=1) == test_targets).type(torch.FloatTensor)
  test_accuracy = correct.cpu().mean()    
  print('final training batch loss: {}'.format(round(loss.item(), 4)))
  print('final validation accuracy: {}%'.format(round(acc.item()*100, 1)))
  print('Test dataset accuracy: {}%'.format(round(test_accuracy.item()*100, 1)))

  # SAVE!
  torch.save(nnModel.state_dict(), './scatch/MNIST_CNN_MODEL.pt')
  # torch.save(nnModel, './scatch/MNIST_CNN_MODEL.pt')
          
  # some metrics
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


  # look at some examples
  if False:
      sample_data, sample_targets = get_next_data_batch(train_loader, 'cpu')
      
      y_samp_pred = nnModel(sample_data.to(device)) 
      pred_max, idx_max = torch.max(y_samp_pred, axis = 1)
      y_samp2 = y_samp_pred.cpu().detach().numpy()
      
      img_dim_x = sample_data.shape[2]
      img_dim_y = sample_data.shape[3]
      
      # check that the data wasn't messed up
      fig = plt.figure()
      offset = 65
      for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(sample_data[i+offset,0,:,:])
        plt.title("prediction: {}, probability: {:.2f}".format(idx_max[i+offset], pred_max[i+offset]))
        plt.xticks([])
        plt.yticks([])
        
      fig = plt.figure()
      offset = 15
      for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(sample_data[i+offset,0,:,:])
        plt.title("prediction: {}, probability: {:.2f}".format(idx_max[i+offset], pred_max[i+offset]))
        plt.xticks([])
        plt.yticks([])      

      plt.show()

