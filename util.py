import numpy as np
import matplotlib.pyplot as plt
# import biosppy
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


def cal_performance2(data,pred2,fs, win_anal, win_move, raw_ppg, flag_pic = 0):
    n_win = data.shape[0]
    metric = np.zeros((n_win,2))

    if flag_pic != 0:
        path_fic = flag_pic + '/pic/' + str(int(win_anal/fs))
        if os.path.isdir(path_fic) == 0:
            os.makedirs(path_fic)

    for i in range(n_win):
        temp_sig = data[i,:,0].copy()
        temp_sig = temp_sig - np.mean(temp_sig)
        rr_sig,freq_sig,Y_sig = cal_resp2(temp_sig,pred2[i],fs)

        temp_resp = data[i,:,1].copy()
        temp_resp = temp_resp - np.mean(temp_resp)
        rr_resp,freq_resp,Y_resp  = cal_resp1(temp_resp,fs,0)

        metric[i,0] = np.abs( rr_sig - rr_resp )
        metric[i,1] = rr_resp

        if flag_pic != 0:
            temp_ppg = raw_ppg[i,:].copy()
            temp_ppg = temp_ppg - np.mean(temp_ppg)

            file_save = path_fic + '/' +  str(i) + '_' + str(metric[i,0]) + '.jpg'
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

    rr_mae = np.mean(metric[:,0])
    true_rr = np.mean(metric[:,1])
    mse = mean_squared_error(data[:,:,0],data[:,:,1])
    
    return mse , rr_mae, true_rr

    # return mse, rr_mae

def cal_performance1(data,fs, win_anal, win_move, raw_ppg, flag_pic = 0):
    n_win = data.shape[0]
    metric = np.zeros((n_win,2))

    if flag_pic != 0:
        path_fic = flag_pic + '/pic/' + str(int(win_anal/fs))
        if os.path.isdir(path_fic) == 0:
            os.makedirs(path_fic)

    for i in range(n_win):
        temp_sig = data[i,:,0].copy()
        temp_sig = temp_sig - np.mean(temp_sig)
        rr_sig,freq_sig,Y_sig = cal_resp1(temp_sig,fs,0)

        temp_resp = data[i,:,1].copy()
        temp_resp = temp_resp - np.mean(temp_resp)
        rr_resp,freq_resp,Y_resp  = cal_resp1(temp_resp,fs,0)
        metric[i,0] = np.abs( rr_sig - rr_resp )
        metric[i,1] = rr_resp

        if flag_pic != 0:
            temp_ppg = raw_ppg[i,:].copy()
            temp_ppg = temp_ppg - np.mean(temp_ppg)

            file_save = path_fic + '/' +  str(i) + '_' + str(metric[i,0]) + '.jpg'
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

    rr_mae = np.mean(metric[:,0])
    true_rr = np.mean(metric[:,1])
    mse = mean_squared_error(data[:,:,0],data[:,:,1])
    
    return mse , rr_mae, true_rr

    # return mse, rr_mae

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

def cal_resp1(sig,fs,flag_method=0):

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

    # elif flag_method == 1:
    #     [_,_,_,_,rr]=biosppy.signals.resp.resp(sig,fs,show=False)
    #     rr = rr[0] * 60

    elif flag_method == 2:
        [freq,Y] = scipy.signal.periodogram(sig, fs)

        ind_f = np.argwhere( (freq >= 0.1) & (freq <= 0.83) )[:,0]
        freq = freq[ind_f]
        Y = Y[ind_f]

        ind = np.argmax(Y)
        rr = freq[ind] * 60

    return rr,freq,Y

def cal_resp2(sig,pred,fs):

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

    # ind = np.argmax(Y)

    n = Y.shape[0]
    ind = int( np.round(((pred*n) + (n+1))/2 - 1) )
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
