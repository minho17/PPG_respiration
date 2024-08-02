from torch.utils.data import Dataset
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import util 

class dataset1(Dataset):
    def __init__(self, ind, data0, win_anal, win_move):
        
        data1 = data0[ind,:,:].copy()
        data1 = data1.reshape(-1,data1.shape[-2],data1.shape[-1])

        n_sub = data1.shape[0] 
        n_sig = data1.shape[1]

        self.n_win = int((n_sig - win_anal)/win_move) + 1
        self.n_data = n_sub * self.n_win
        self.data = np.zeros((self.n_data, win_anal,2))
        self.raw_sig = data1.copy()

        count = 0
        for i in range(n_sub):
            for i1 in range(self.n_win):
                ind_start = i1*win_move
                temp_sig = data1[ i, ind_start : ind_start + win_anal, 0 ]
                self.data[count,:,0] = (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)

                temp_sig = data1[ i, ind_start : ind_start + win_anal, 1 ]
                self.data[count,:,1] = (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)
                count = count + 1

    def __getitem__(self, index): 
        x = np.squeeze(self.data[index,:,0])
        y =  np.squeeze(self.data[index,:,1])
        return x[np.newaxis,:], y[np.newaxis,:]
    
    def __len__(self):
        return self.n_data



class dataset0(Dataset):
    def __init__(self, x,y):
        
        self.x = x
        self.y = y

        self.n_data = x.shape[0]
        
    def __getitem__(self, index): 
        return self.x[index,np.newaxis,:], self.y[index,np.newaxis,:]
    
    def __len__(self):
        return self.n_data
