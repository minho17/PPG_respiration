import numpy as np
import matplotlib.pyplot as plt
import biosppy
import scipy
from sklearn.metrics import mean_squared_error 
from datetime import datetime
import os
import torch
import torch.nn as nn

def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if isinstance(net, nn.DataParallel):
        torch.save({'net': net.module.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))
    else:
        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

def mk_folders(flag_data):
    now = datetime.now()
    path_folder = os.getcwd() + '/result/D' + str(flag_data) + '_' + now.strftime('%Y-%m-%d_%H_%M_%S')
    os.makedirs(path_folder)
    
    return path_folder

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

def cal_performance2(data,fs, win_anal, win_move, n_win, true_resp,flag_debug = 0):
    n_data = data.shape[0]
    n_sub = int(n_data/n_win)

    mse = mean_squared_error(data[:,:,0],data[:,:,1])
    rr_mae = np.zeros((n_sub,2))
    for i in range(n_sub):
        rr_mae[i,:] = cal_resp_w2(data,fs,win_anal, win_move, 0)

        # if flag_debug == 1: #(n_sub > 1 and i == 5) or 
        #     plt.subplot(2,1,1)
        #     plt.plot(re_data)
        #     plt.subplot(2,1,2)
        #     plt.plot(true_resp[i,:])
        #     plt.show()

        #     a=1
    rr_mae = np.mean(rr_mae,axis = 0)

    return mse , rr_mae



def cal_performance3(data,fs, win_anal, win_move, n_win, raw_sig, flag_pic = 0):
    n_data = data.shape[0]
    n_sub = int(n_data/n_win)

    mse = mean_squared_error(data[:,:,0],data[:,:,1])
    rr_mae = np.zeros((n_sub,2))
    true_rr = np.zeros((n_sub,2))
    for i in range(n_sub):
        re_data = np.zeros((win_anal + win_move * (n_win-1),2))
        for i1 in range(n_win):
            w_start = win_move*i1
            temp = data[ i*n_win + i1,:,0].copy() 
            re_data[w_start:w_start+win_anal,0] = re_data[w_start:w_start+win_anal,0] + (temp - np.mean(temp) )
            re_data[w_start:w_start+win_anal,1] = re_data[w_start:w_start+win_anal,1] + np.ones((win_anal))

        re_data = np.divide(re_data[:,0],re_data[:,1])
        rr_mae[i,:], true_rr[i,:] = cal_resp_w1(re_data,raw_sig[i,:,:],fs,0,flag_pic)

        # if flag_debug == 1: #(n_sub > 1 and i == 5) or 
        #     plt.subplot(2,1,1)
        #     plt.plot(re_data)
        #     plt.subplot(2,1,2)
        #     plt.plot(true_resp[i,:])
        #     plt.show()

        #     a=1
    rr_mae = np.mean(rr_mae,axis = 0)
    true_rr = np.mean(true_rr,axis = 0)

    return mse , rr_mae, true_rr




def cal_performance1(data,fs, win_anal, win_move, n_win, raw_sig, flag_pic = 0):
    n_sub = data.shape[0]

    mse = 0
    rr_mae = np.zeros((n_sub,2))
    true_rr = np.zeros((n_sub,2))
    for i in range(n_sub):
        mse = mse + mean_squared_error(data[i,:,:,0],data[i,:,:,1])

        re_data = np.zeros((win_anal + win_move * (n_win-1),2))
        for i1 in range(n_win):
            w_start = win_move*i1
            re_data[w_start:w_start+win_anal,0] = re_data[w_start:w_start+win_anal,0] + (data[ i, i1,:,0].copy() )
            re_data[w_start:w_start+win_anal,1] = re_data[w_start:w_start+win_anal,1] + np.ones((win_anal))

        re_data = np.divide(re_data[:,0],re_data[:,1])
        rr_mae[i,:], true_rr[i,:] = cal_resp_w1(re_data,raw_sig[i,:,:],fs,0,flag_pic)

        # if flag_debug == 1: #(n_sub > 1 and i == 5) or 
        #     plt.subplot(2,1,1)
        #     plt.plot(re_data)
        #     plt.subplot(2,1,2)
        #     plt.plot(true_resp[i,:])
        #     plt.show()

        #     a=1
    rr_mae = np.mean(rr_mae,axis = 0)
    true_rr = np.mean(true_rr,axis = 0)
    mse = mse/n_sub

    return mse , rr_mae, true_rr

    # return mse, rr_mae

def cal_performance0(data,fs):
    n_data = data.shape[0]
    mse = mean_squared_error(data[:,:,0],data[:,:,1])

    rr_mae = np.zeros((n_data))
    for i in range(n_data):
        # rr_mae[0] = rr_mae[0] + np.abs(cal_resp1(data[i,:,0],fs,0) - cal_resp1(data[i,:,1],fs,0))
        # rr_mae[1] = rr_mae[1] + np.abs(cal_resp1(data[i,:,0],fs,1) - cal_resp1(data[i,:,1],fs,1))
        rr_mae[i] = np.abs(cal_resp1(data[i,:,0],fs,2) - cal_resp1(data[i,:,1],fs,2))

    rr_mae_temp = rr_mae.copy()
    rr_mae = np.mean(rr_mae)

    # if rr_mae == 0:
    #     aa0 = np.squeeze(data[0,:,0])
    #     aa1 = np.squeeze(data[0,:,1])
    #     plt.subplot(4,1,1)
    #     plt.plot(aa0)
    #     plt.subplot(4,1,2)
    #     plt.plot(aa1)
        

    #     [freq,Y] = scipy.signal.periodogram(aa0, fs)
    #     ind = np.argmax(Y)
    #     rr = freq[ind] * 60

    #     plt.subplot(4,1,3)
    #     plt.plot(freq,Y)
    #     [freq2,Y2] = scipy.signal.periodogram(aa1, fs)
    #     plt.subplot(4,1,4)
    #     plt.plot(freq2,Y2)
    #     plt.show()

    #     ind2 = np.argmax(Y2)
    #     rr2 = freq2[ind2] * 60

    #     a=1

    return mse, rr_mae


def cal_resp_w2(data,fs,win_anal0, win_move0, flag_method):
    win_anal = [32*fs,64*fs]
    win_move = [3*fs,6*fs]
    result = np.zeros((2))

    n_seg = data.shape[0]
    for i in range(2):
        n_anal_seg = int((win_anal[i] - win_anal0) / win_move0) + 1
        n_win = n_seg - n_anal_seg + 1
        metric = np.zeros((n_win))

        for i1 in range(n_win):
            metric2 = np.zeros((n_anal_seg))
            for i2 in range(n_anal_seg):
                temp_sig = data[ i1 + i2 , : , 0 ].copy()
                temp_sig = temp_sig - np.mean(temp_sig)
                rr_sig  = cal_resp1(temp_sig,fs,flag_method)

                temp_resp = data[ i1 + i2 , : , 1].copy()
                temp_resp = temp_resp - np.mean(temp_resp)
                rr_resp  = cal_resp1(temp_resp,fs,flag_method)
                metric2[i2] = np.abs( rr_sig - rr_resp )

            metric[i1] = np.mean(metric2)
        result[i] = np.mean(metric)

    return result


def cal_resp_w1(sig,raw_sig,fs,flag_method,flag_pic=0):
    win_anal = [32*fs,64*fs]
    win_move = [3*fs,6*fs]

    n_sig = sig.shape[0]
    result = np.zeros((2))
    true_rr = np.zeros((2))

    # plt.subplot(2,1,1)
    # plt.plot(sig)
    # plt.subplot(2,1,2)
    # plt.plot(sig_resp)
    # plt.show()

    for i in range(2):
        n_win = int((n_sig-win_anal[i])/win_move[i]) + 1
        metric = np.zeros((n_win))
        metric2 = np.zeros((n_win))

        if flag_pic != 0:
            path_fic = flag_pic + '/pic/' + str(i)
            if os.path.isdir(path_fic) == 0:
                os.makedirs(path_fic)

        for i1 in range(n_win):
            temp_sig = sig[ i1*win_move[i] : i1*win_move[i] + win_anal[i] ].copy()
            temp_sig = temp_sig - np.mean(temp_sig)
            rr_sig,freq_sig,Y_sig = cal_resp1(temp_sig,fs,flag_method)

            temp_resp = raw_sig[ i1*win_move[i] : i1*win_move[i] + win_anal[i],1 ].copy()
            temp_resp = temp_resp - np.mean(temp_resp)
            rr_resp,freq_resp,Y_resp  = cal_resp1(temp_resp,fs,flag_method)
            metric[i1] = np.abs( rr_sig - rr_resp )
            metric2[i1] = rr_resp

            if flag_pic != 0:
                temp_ppg = raw_sig[ i1*win_move[i] : i1*win_move[i] + win_anal[i],0 ].copy()
                temp_ppg = temp_ppg - np.mean(temp_ppg)

                file_save = path_fic + '/' +  str(i1) + '_' + str(metric[i1]) + '.jpg'
                plt.figure(1)
                plt.subplot(5,1,1)
                plt.plot(temp_ppg)
                plt.subplot(5,1,2)
                plt.plot(temp_sig)
                plt.subplot(5,1,3)
                plt.plot(temp_resp)
                plt.subplot(5,1,4)
                plt.plot(freq_sig,Y_sig)
                ind = np.argmax(Y_sig)
                plt.plot(freq_sig[ind],Y_sig[ind],'*')
                plt.subplot(5,1,5)
                plt.plot(freq_resp,Y_resp)
                ind = np.argmax(Y_resp)
                plt.plot(freq_resp[ind],Y_resp[ind],'*')
                plt.savefig(file_save)
                plt.close()

        result[i] = np.mean(metric)
        true_rr[i] = np.mean(metric2)

    return result, true_rr

def cal_resp1(sig,fs,flag_method):

    if flag_method == 0:
        n = len(sig) 
        k = np.arange(n)
        T = n/fs
        freq = k/T 
        freq = freq[range(int(n/2))]

        ind_f = np.argwhere( (freq >= 0.1) & (freq <= 0.83) )[:,0]

        Y = np.abs(np.fft.fft(sig)/n) 
        Y = Y[range(int(n/2))]

        freq = freq[ind_f]
        Y = Y[ind_f]

        # plt.plot(freq,Y)
        # plt.show()

        ind = np.argmax(Y)
        rr = freq[ind] * 60

    elif flag_method == 1:
        [_,_,_,_,rr]=biosppy.signals.resp.resp(sig,fs,show=False)
        rr = rr[0] * 60

    elif flag_method == 2:
        [freq,Y] = scipy.signal.periodogram(sig, fs)

        ind_f = np.argwhere( (freq >= 0.1) & (freq <= 0.83) )[:,0]
        freq = freq[ind_f]
        Y = Y[ind_f]

        ind = np.argmax(Y)
        rr = freq[ind] * 60

    return rr,freq,Y



class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, best_fitness, patience=30):
        self.best_fitness = best_fitness  # i.e. MSE Loss
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness <= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        return stop
