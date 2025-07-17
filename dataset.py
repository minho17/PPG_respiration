from torch.utils.data import Dataset
import numpy as np

class dataset1(Dataset):
    def __init__(self, ind, data0, win_anal, win_move,metric,flag=0):
        
        data1 = data0[ind,:,:].copy()
        data1 = data1.reshape(-1,data1.shape[-2],data1.shape[-1])

        n_sub = data1.shape[0] 
        n_sig = data1.shape[1]

        self.n_win = int((n_sig - win_anal)/win_move) + 1
        self.n_data = n_sub * self.n_win
        self.data = np.zeros((self.n_data*2, win_anal,2))
        self.flag = flag

        if flag == 1:
            self.w = np.zeros((self.n_data,2))
            w_max = np.max(metric)
            w_min = np.min(metric)
            w_med = np.median(metric)
            w_low_bound = 0.1

        count = 0
        for i in range(n_sub):
            if flag == 1:
                w_sub0 = (metric[i] -  w_med)/(w_max-w_med)
                w_sub0 = np.maximum(w_sub0,w_low_bound)

                w_sub1 = (w_med - metric[i])/(w_med-w_min)
                w_sub1 = np.maximum(w_sub1,w_low_bound)

            for i1 in range(self.n_win):
                ind_start = i1*win_move
                temp_sig = data1[ i, ind_start : ind_start + win_anal, 0 ]
                self.data[count,:,0] = (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)

                temp_sig = data1[ i, ind_start : ind_start + win_anal, 1 ]
                self.data[count,:,1] = (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)

                if flag == 1:
                    self.w[count,0] = w_sub0
                    self.w[count,1] = w_sub1
                            
                count = count + 1

        self.data = self.data[0:count,:,:]

        if flag ==1 :
            self.w = self.w[0:count,:]

    def __getitem__(self, index): 
        x = np.squeeze(self.data[index,:,0])
        y =  np.squeeze(self.data[index,:,1])
        x = x[np.newaxis,np.newaxis,:]

        if self.flag == 1:
            return x, y[np.newaxis,np.newaxis,:],self.w[index,:]
        else:
            return x, y[np.newaxis,np.newaxis,:], 0
    
    def __len__(self):
        return self.n_data
    