import numpy as np
from sklearn.metrics import mean_squared_error 
import os
import torch
from util_PPG import peak_AT2 

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim

class log():
    def __init__(self, path, init=0):
        self.path = path
        self.init = init

    def w(self, text):
        if self.init == 0:
            self.init = 1
            with open(self.path, "w") as file:
                file.write(text)
        else:
            with open(self.path, "a") as file:
                file.write(text)

def cal_performance1(data,fs):
    n_win = data.shape[0]
    metric = np.zeros((n_win,2))

    for i in range(n_win):
        temp_sig = data[i,:,0].copy()
        temp_sig = temp_sig - np.mean(temp_sig)
        rr_sig,freq_sig,Y_sig = cal_resp1(temp_sig,fs)

        temp_resp = data[i,:,1].copy()
        temp_resp = temp_resp - np.mean(temp_resp)
        rr_resp,freq_resp,Y_resp  = cal_resp1(temp_resp,fs)
        metric[i,0] = np.abs( rr_sig - rr_resp )
        metric[i,1] = rr_resp

    rr_mae = np.mean(metric[:,0])
    true_rr = np.mean(metric[:,1])
    mse = mean_squared_error(data[:,:,0],data[:,:,1])
    
    return mse , rr_mae, true_rr


def cal_performance1_ensem(data,fs, n_model):
    n_win = data.shape[0]
    metric = np.zeros((n_win,2))

    for i in range(n_win):
        temp_sig = data[i,:,0].copy()
        temp_sig = temp_sig - np.mean(temp_sig)

        rr_sig,freq_sig,Y_sig = cal_resp1_ensem(data[i,:,0:n_model],fs,n_model)

        temp_resp = data[i,:,n_model].copy()
        temp_resp = temp_resp - np.mean(temp_resp)
        rr_resp,freq_resp,Y_resp  = cal_resp1(temp_resp,fs)
        metric[i,0] = np.abs( rr_sig - rr_resp )
        metric[i,1] = rr_resp

    rr_mae = np.mean(metric[:,0])
    true_rr = np.mean(metric[:,1])
    mse = mean_squared_error(data[:,:,0],data[:,:,1])
    
    return mse , rr_mae, true_rr


def cal_performance_ensem_w(data,fs, n_model, raw_ppg,lb,ub):
    n_win = data.shape[0]
    metric = np.zeros((n_win,2))

    n_win_check = 5
    metric_rr = np.zeros((4,n_win_check))

    for i in range(n_win):
        temp_sig = data[i,:,0].copy()
        temp_sig = temp_sig - np.mean(temp_sig)

        rr_sig,freq_sig,Y_sig = cal_resp1_ensem(data[i,:,0:n_model-2],fs,n_model-2)

        temp_ppg = raw_ppg[i,:].copy()
        temp_ppg = temp_ppg - np.mean(temp_ppg)
        rr_ori,_ = cal_resp1_ppg(temp_ppg,fs)

        temp_sig_w = data[i,:,n_model-2].copy()
        temp_sig_w = temp_sig_w - np.mean(temp_sig_w)
        rr1,_,y1 = cal_resp1(temp_sig_w,fs)

        temp_sig_w = data[i,:,n_model-1].copy()
        temp_sig_w = temp_sig_w - np.mean(temp_sig_w)
        rr2,_,y2 = cal_resp1(temp_sig_w,fs)

        metric_rr[0,:] = est_p(rr_ori,metric_rr[0,:])
        metric_rr[1,:] = est_p(rr1,metric_rr[1,:])
        metric_rr[2,:] = est_p(rr2,metric_rr[2,:])
        metric_rr[3,:] = est_p(rr_sig,metric_rr[3,:])

        if (np.mean(metric_rr[0,:] ) > ub) and (i >= 5) and (np.std(metric_rr[1,:] ) < np.std(metric_rr[3,:] )):
            Y_sig = Y_sig + y1/np.max(y1)*3
            ind = np.argmax(Y_sig)
            rr_sig = freq_sig[ind] * 60
        elif (np.mean(metric_rr[0,:] ) < lb) and (i >= 5)  and (np.std(metric_rr[2,:] ) < np.std(metric_rr[3,:] )):
            Y_sig = Y_sig + y2/np.max(y2)*3
            ind = np.argmax(Y_sig)
            rr_sig = freq_sig[ind] * 60

        temp_resp = data[i,:,n_model].copy()
        temp_resp = temp_resp - np.mean(temp_resp)
        rr_resp,freq_resp,Y_resp  = cal_resp1(temp_resp,fs)
        metric[i,0] = np.abs( rr_sig - rr_resp )
        metric[i,1] = rr_resp

    rr_mae = np.mean(metric[:,0])
    true_rr = np.mean(metric[:,1])
    mse = mean_squared_error(data[:,:,0],data[:,:,1])
    
    return mse , rr_mae, true_rr


def est_p(res,xt):
    xt[0:-1] = xt[1:].copy()
    xt[-1] = res

    return xt


def cal_resp1_ensem(data,fs,n_model):

    for i in range(n_model):
        sig = data[:,i].copy()
        sig = sig - np.mean(sig)
   
        n = len(sig) 
        k = np.arange(n)
        T = n/fs
        freq = k/T 
        freq = freq[range(int(n/2))]
        ind_f = np.argwhere( (freq >= 0.1) & (freq <= 0.83) )[:,0]

        Y = np.abs(np.fft.fft(sig * np.hamming(n) )/n) 
        Y = Y[range(int(n/2))]

        freq = freq[ind_f]
        Y = Y[ind_f]

        if i == 0:
            y_total = Y/np.max(Y)
        else:
            y_total = y_total + Y/np.max(Y)

    ind = np.argmax(y_total)
    rr = freq[ind] * 60

    return rr,freq,y_total


def cal_resp1(sig,fs):

    n = len(sig) 
    k = np.arange(n)
    T = n/fs
    freq = k/T 
    freq = freq[range(int(n/2))]

    ind_f = np.argwhere( (freq >= 0.1) & (freq <= 0.83) )[:,0]

    Y = np.abs(np.fft.fft(sig * np.hamming(n) )/n) 
    Y = Y[range(int(n/2))]

    freq = freq[ind_f]
    Y = Y[ind_f]

    ind = np.argmax(Y)
    rr = freq[ind] * 60

    return rr,freq,Y

def cal_resp1_ppg(sig,fs):

    [peak_max, _, _] = peak_AT2(sig,fs,0)
    peak_inter = np.diff(peak_max)
    ind_peak = np.argwhere( np.abs(peak_inter - np.mean(peak_inter)) < (3*np.std(peak_inter)))[:,0]
    peak_inter = np.mean(peak_inter[ind_peak])
    peak_f = fs/peak_inter

    n = len(sig) 
    k = np.arange(n)
    T = n/fs
    freq = k/T 
    freq = freq[range(int(n/2))]

    ind_f = np.argwhere( (freq >= 0.1) & (freq <= 0.83) )[:,0]

    Y = np.abs(np.fft.fft(sig * np.hamming(n) )/n) 
    Y = Y[range(int(n/2))]

    freq = freq[ind_f]
    Y = Y[ind_f]
    if (peak_f >= 0.1) and (peak_f <= 0.83):
        f_compare = np.abs(freq - peak_f)
        ind_min = np.argmin(f_compare)
        Y[ind_min] = 0

    ind = np.argmax(Y)
    rr = freq[ind] * 60

    return rr,freq

