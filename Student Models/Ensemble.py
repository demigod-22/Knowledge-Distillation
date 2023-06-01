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

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split




import Helper
from accs import net_st1
from gyrs import net_st2
from mags import net_st3


def prepare_data_PAMAP2(root_path='../../PAMAP2_Dataset/Protocol/subject10'):
    X=[]
    user_labels=[]
    act_labels=[]

    window_len = 512
    stride_len = 20
    # columns for IMU data
    imu_locs = [4,5,6, 10,11,12, 13,14,15, 
                21,22,23, 27,28,29, 30,31,32, 
                38,39,40, 44,45,46, 47,48,49
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

X_train, X_test, y_train, y_test = train_test_split(normalized_X, act_labels, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, drop_last = True)  

test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=batch_size, shuffle=False, drop_last = True) 




class Ensemble(nn.Module):

    def __init__(self, modelA, modelB, modelC, input):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

        self.fc1 = nn.Linear(input, 16)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)

        out = out1 + out2 + out3

        x = self.fc1(out)
        return torch.softmax(x, dim=1)
    

model = Ensemble(net_st1, net_st2 , net_st3, 16)


model.to(device)


optimizer = optim.Adam(model.parameters(),lr=0.003)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

model, train_loss, test_loss = Helper.train(model, train_loader, test_loader, epoch, optimizer, criterion)
