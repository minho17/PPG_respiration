from torch.utils.data import Dataset
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import util 


class dataset1(Dataset):
    def __init__(self, ind, data0,data1,data2,data3,real_rr, win_anal, win_move,is_tr=0):
        
        data0 = data0[ind,:,:].copy()
        data0 = data0.reshape(-1,data0.shape[-2],data0.shape[-1])

        real_rr = real_rr[ind,:].copy()

        n_sub = data0.shape[0] 
        n_sig = data0.shape[1]

        self.n_win = int((n_sig - win_anal)/win_move) + 1
        self.n_data = n_sub * self.n_win
        self.data = np.zeros((self.n_data, win_anal,2))

        count = 0
        for i in range(n_sub):
            for i1 in range(self.n_win):
                ind_start = i1*win_move
                temp_sig = data0[ i, ind_start : ind_start + win_anal, 0 ]
                self.data[count,:,0] = (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)

                temp_sig = data0[ i, ind_start : ind_start + win_anal, 1 ]
                self.data[count,:,1] = (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)
                count = count + 1

        if is_tr == 1:
            count = 0
            data_aug = np.zeros((self.n_data*3, win_anal,2))
            for i_aug in range(3):
                if i_aug == 0:
                    data0 = data1[ind,:,:].copy()
                    win_move2 = win_move*2
                elif i_aug==1:
                    data0 = data2[ind,:,:].copy()
                    win_move2 = int(win_move/1.5)
                elif i_aug==2:
                    data0 = data3[ind,:,:].copy()
                    win_move2 = int(win_move/2)

                n_sub = data0.shape[0] 
                n_sig = data0.shape[1]
                n_win2 = int((n_sig - win_anal)/win_move2) + 1

                count = 0
                for i in range(n_sub):
                    if (real_rr[i,i_aug+1] >= 6) and (real_rr[i,i_aug+1] <= 50):
                        for i1 in range(n_win2):
                            ind_start = i1*win_move2
                            temp_sig = data0[ i, ind_start : ind_start + win_anal, 0 ]
                            data_aug[count,:,0] = (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)

                            temp_sig = data0[ i, ind_start : ind_start + win_anal, 1 ]
                            data_aug[count,:,1] = (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)
                            count = count + 1

            data_aug = data_aug[:count,:,:]
            self.n_data = self.n_data + count

            self.data = np.append(self.data,data_aug,axis=0)
            # a=1

    def __getitem__(self, index): 
        x = np.squeeze(self.data[index,:,0])
        y =  np.squeeze(self.data[index,:,1])
        
        return x[np.newaxis,np.newaxis,:], y[np.newaxis,np.newaxis,:]
    
    def __len__(self):
        return self.n_data


class dataset3(Dataset):
    def __init__(self, ind, data0, win_anal, win_move,fs,metric,is_tr = 0):
        
        data1 = data0[ind,:,:].copy()
        data1 = data1.reshape(-1,data1.shape[-2],data1.shape[-1])

        n_sub = data1.shape[0] 
        n_sig = data1.shape[1]

        self.n_win = int((n_sig - win_anal)/win_move) + 1
        self.n_data = n_sub * self.n_win
        self.data = np.zeros((self.n_data, win_anal,2))
        self.true_f = np.zeros((self.n_data))
        self.is_tr = is_tr
        # self.raw_sig = np.zeros((self.n_data, win_anal,2))

        # if self.is_tr == 1:
        #     self.w = np.ones((self.n_data))
        #     metric = metric[ind,:].copy()
        #     self.w0 = np.ones((n_sub,2))

        #     for i_w in range(2):
        #         temp = np.squeeze(metric[:,i_w].copy())
        #         ind = np.where( np.abs(temp - np.mean(temp)) < 2*np.std(temp) )
        #         temp = temp[ind]

        #         for i_w2 in range(n_sub):
        #             if (np.abs(metric[i_w2,i_w] - np.mean(temp)) > 0.5*np.std(temp)) and i_w == 0 :
        #                 self.w0[i_w2,i_w] = np.minimum(np.abs(metric[i_w2,i_w] - np.mean(temp)) / (0.5*np.std(temp)),4)
        #             elif ( (metric[i_w2,i_w] - np.mean(temp)) > 1*np.std(temp)) and i_w == 1 :
        #                 self.w0[i_w2,i_w] = np.minimum(np.abs(metric[i_w2,i_w] - np.mean(temp)) / np.std(temp),2)

        count = 0
        for i in range(n_sub):
            for i1 in range(self.n_win):
                ind_start = i1*win_move
                temp_sig = data1[ i, ind_start : ind_start + win_anal, 0 ]
                self.data[count,:,0] = (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)

                temp_sig = data1[ i, ind_start : ind_start + win_anal, 1 ]
                self.data[count,:,1] = (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)
                
                # sig = self.data[count,:,1]
                # n = len(sig) 
                # k = np.arange(n)
                # T = n/fs
                # freq = k/T 
                # freq = freq[range(int(n/2))]
                # ind_f = np.argwhere( (freq >= 0.1) & (freq <= 0.83) )[:,0]
                # Y = np.abs(np.fft.fft(sig)/n) 
                # Y = Y[range(int(n/2))]
                # freq = freq[ind_f]
                # Y = Y[ind_f]
                # ind = np.argmax(Y) + 1
                # n = Y.shape[0]
                # self.true_f[count] = (2*ind - (n+1)  )/n

                # if self.is_tr == 1:
                #     self.w[count] = self.w0[i,0]
                            
                count = count + 1

    def __getitem__(self, index): 
        x = np.squeeze(self.data[index,:,0])
        y =  np.squeeze(self.data[index,:,1])

        # if self.is_tr == 1:
        #     return x[np.newaxis,np.newaxis,:], y[np.newaxis,np.newaxis,:] #, self.true_f[index], self.w[index]
        # else: 
        return x[np.newaxis,np.newaxis,:], y[np.newaxis,np.newaxis,:] #, self.true_f[index]
    
    def __len__(self):
        return self.n_data