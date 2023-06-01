import os
import sys
import copy
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler, TensorDataset
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, BatchNorm1d, Dropout, Flatten, BCELoss
from torch.optim import Adam, SGD
from torch import nn
# from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter

from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from Distillation_loss import *

gpu_id=3
if gpu_id>=0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cuda_id = "cuda:" + str(0)  # cuda:2

device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")
print("Device:", device)
if (torch.cuda.is_available()):
    torch.cuda.set_device(cuda_id)
    print("Current GPU ID:", torch.cuda.current_device())

def prepare_data_PAMAP2(root_path='../../PAMAP2_Dataset/Protocol/subject10'):
    X=[]
    user_labels=[]
    act_labels=[]

    window_len = 512
    stride_len = 20
    # columns for IMU data
    imu_locs = [11,12,13,28,29,30,45,46,47
            ] 
    
    act_list = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

    scaler = MinMaxScaler()
    # scaler = StandardScaler()

    for uid in np.arange(1,10):
        path = root_path + str(uid) + '.dat'
        df = pd.read_table(path, sep=' ', header=None)
        act_imu_filter = df.iloc[:, imu_locs] 

        for act_id in range(len(act_list)):
            act_filter =  act_imu_filter[df.iloc[:, 1] == act_list[act_id]]
            act_data = act_filter.to_numpy()
                
            act_data = np.transpose(act_data)
            # sliding window segmentation
            start_idx = 0
            while start_idx + window_len < act_data.shape[1]:
                window_data = act_data[:, start_idx:start_idx + window_len]
                downsamp_data = window_data[:, ::3] # downsample from 100hz to 33.3hz
                downsamp_data = np.nan_to_num(downsamp_data) # remove nan

                X.append(downsamp_data)
                user_labels.append(uid)
                act_labels.append(act_id)
                start_idx = start_idx + stride_len

    X_n = np.array(X).astype('float32')

    normalized_X = np.zeros_like(X_n) # allocate numpy array for normalized data
    for ch_id in range(X_n.shape[1]): # loop the 27 sensor channels
        ch_data = X_n[:, ch_id, :] # the data of channel id
        scaler = MinMaxScaler() # maybe different scalers?
        ch_data = scaler.fit_transform(ch_data) # scale the data in this channel to [0,1]
        normalized_X[:, ch_id, :] = ch_data # assign normalized data to normalized_X
    normalized_X = np.transpose(normalized_X, (0, 2, 1)) # overwrote X here, changed dimensions into: num_samples, sequence_length, feature_length
        
    # convert list to numpy array
    # normalized_X= normalized_X.reshape(normalized_X.shape[0], 1, normalized_X.shape[1], normalized_X.shape[2]) 
    act_labels = np.array(act_labels).astype('float32')
    act_labels = act_labels.reshape(act_labels.shape[0],1)
    act_labels = to_categorical(act_labels, num_classes=len(act_list))

    return normalized_X, act_labels
normalized_X, act_labels = prepare_data_PAMAP2()

print(normalized_X.shape)

X_train, X_test, y_train, y_test = train_test_split(normalized_X, act_labels, test_size=0.2, random_state=42)
 

class HARModel(nn.Module):
    
    def __init__(self, n_sensor_channels=113, len_seq=24, n_hidden=128, n_layers=1, n_filters=64, 
                 n_classes=5, filter_size=(1,5), drop_prob=0.5):
        super(HARModel, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.n_sensor_channels = n_sensor_channels
        self.len_seq = len_seq

             
        self.conv1 = nn.Conv2d(1, n_filters, filter_size)
        self.conv2 = nn.Conv2d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv2d(n_filters, n_filters, filter_size)
        #self.conv4 = nn.Conv2d(n_filters, n_filters, filter_size)
        
        # self.lstm1  = nn.LSTM(64, n_hidden, n_layers)
        # self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=n_sensor_channels*n_filters, num_heads=1) # 7232=113*64
        # self.fc0 = nn.Linear(57856, 128)
        self.fc = nn.Linear(n_sensor_channels*n_filters*(len_seq-4*(filter_size[1]-1)), n_classes) #57856 = 8*113*64

        self.dropout = nn.Dropout(drop_prob)
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
     
        x = torch.permute(x, (0,2,1))
        x = torch.unsqueeze(x, dim=1)
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
    
        x = torch.permute(x, (3,0,1,2))
        x = x.view(x.shape[0], x.shape[1],-1)
        
        # print(x.shape)
        # x = x.view(8, x.shape[0], -1) # bak
        
    
        x, attn_output_weights = self.multihead_attn(x,x,x)
        x = self.dropout(x)
        x = F.relu(x)
#         x, attn_output_weights = self.multihead_attn1(x,x,x)
#         # x = self.dropout(x)
#         x = F.relu(x)    
        
        x = torch.permute(x, (1,0,2))
    

        x = torch.reshape(x, (x.shape[0],-1))
        # x = F.relu(self.fc0(x))
        # x = self.dropout(x)
        x = self.fc(x)
        
        # out = x.view(batch_size, -1, self.n_classes)[:,-1,:]
        return x

    
net_st3 = HARModel(n_sensor_channels=X_train.shape[2], len_seq=X_train.shape[1], n_classes=12)

def train(net, epochs=10, batch_size=64, lr=0.01):
    # opt = torch.optim.Adam(net.parameters(), lr=lr)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # opt = torch.optim.RMSprop(net.parameters(), lr=lr, momentum=0.1)
    # opt = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, drop_last = True)  

    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=batch_size, shuffle=False, drop_last = True) 
    
    if(train_on_gpu):
        net.cuda()
     
    for e in range(epochs):
        
        # initialize hidden state
        # h = net.init_hidden(batch_size)         
        train_losses = []    
        net.train()
        # for batch in iterate_minibatches(X_train, y_train, batch_size):
        for batch in train_loader:
            x, y = batch

            # inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            inputs, targets = x.to(device), y.to(device)  

            # zero accumulated gradients
            opt.zero_grad()   
            
            # get the output from the model
            output = net(inputs)
            # loss = criterion(output, torch.from_numpy(to_categorical(y, num_classes=NUM_CLASSES)).to(device))
            loss = DistillationLoss(output, torch.argmax(targets,dim=1),net3(inputs))
            # print(output.shape)
            # print(targets.shape)
            # loss = criterion(output, targets)
            train_losses.append(loss.item())
            loss.backward()
            opt.step()
            
        # val_h = net.init_hidden(batch_size)
        val_losses = []
        accuracy=0
        f1score=0
        
        correct = 0
        total = 0
        total_true = []
        total_pred = []
        
        net.eval()
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                inputs, targets = x.to(device), y.to(device)  
 
                # print(images.shape)            
            # for batch in iterate_minibatches(X_test, y_test, batch_size):
            #     x, y = batch     

                # inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                # val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    
                output = net(inputs)

                # val_loss = criterion(output, torch.from_numpy(to_categorical(y, num_classes=NUM_CLASSES)).to(device))
                val_loss = DistillationLoss(output, torch.argmax(targets,dim=1),net3(inputs))
                # val_loss = criterion(output, targets)
                val_losses.append(val_loss.item())
                
                predicted = torch.argmax(output.data, dim=1)
                total += targets.size(0)
                correct += (predicted == torch.argmax(targets, dim=1)).sum().item()

                total_pred = total_pred + predicted.cpu().numpy().tolist()
                total_true = total_true + (torch.argmax(targets, dim=1).cpu().numpy().tolist())



        net.train() # reset to train mode after iterationg through validation data
    
        # print(f'Test Accuracy: {100.0 * correct / total} %')
        # print(" | ".join(act_labels_txt))
        # conf_mat = confusion_matrix(y_true = total_true, y_pred = total_pred)
        # conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        # print(np.array(conf_mat).round(3) * 100)  
        f1_score = metrics.f1_score(y_true = total_true, y_pred = total_pred, average='weighted')
        # print('F1 score:', f1_score)
        # print('')      

        print("Epoch: {}/{}...".format(e+1, epochs),
        "Train Loss: {:.4f}...".format(np.mean(train_losses)),
        "Val Loss: {:.4f}...".format(np.mean(val_losses)),
        "Val Acc: {:.4f}...".format(correct / total),
        "F1-Score: {:.4f}...".format(f1_score))
        
        PATH = 'pamap2_ConvAttn_ep'+str(e)+'.pt'
        torch.save(net.state_dict(), PATH)
        
## check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')

train(net_st3)
