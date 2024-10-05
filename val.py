import torch
import numpy as np
import util
import matplotlib.pyplot as plt

def val1(loader,model,device,data,win_anal,win_move,fs,flag_pic=0):
    with torch.no_grad(): 
        pred = np.zeros((data.n_data,win_anal,2))
        raw_ppg = np.zeros((data.n_data,win_anal))
        count = 0
        for j,(sig,label)in enumerate(loader):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)

            out_temp = model.forward(x)
            out_temp = np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1)
            y_temp = np.squeeze(y.to('cpu').detach().numpy(),axis=1)
            ppg = np.squeeze(x.to('cpu').detach().numpy(),axis=1)
            
            raw_ppg[count:count+out_temp.shape[0], :]  = ppg
            pred[count:count+out_temp.shape[0], :, 0] =  out_temp
            pred[count:count+y_temp.shape[0],   :, 1] =  y_temp
            count = count + out_temp.shape[0]

        [mse, rr_mae, true_rr] = util.cal_performance1(pred,fs, win_anal, win_move, raw_ppg, flag_pic)

    return mse, rr_mae, pred, true_rr


def val2(loader,model,device,data,win_anal,win_move,fs,flag_pic=0):
    with torch.no_grad(): 
        pred = np.zeros((data.n_data,win_anal,2))
        pred2 = np.zeros((data.n_data))
        raw_ppg = np.zeros((data.n_data,win_anal))
        count = 0
        for j,(sig,label,_)in enumerate(loader):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)
            # y2 = label_f.to(torch.float32)

            out_temp,_,y2 = model.forward(x)
            # out_temp,_,_ = model.forward(x)
            out_temp = np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1)
            y_temp = np.squeeze(y.to('cpu').detach().numpy(),axis=1)
            ppg = np.squeeze(x.to('cpu').detach().numpy(),axis=1)
            y_temp2 = np.squeeze(y2.to('cpu').detach().numpy())
            
            pred2[count:count+out_temp.shape[0]] = y_temp2
            raw_ppg[count:count+out_temp.shape[0], :]  = ppg
            pred[count:count+out_temp.shape[0], :, 0] =  out_temp
            pred[count:count+y_temp.shape[0],   :, 1] =  y_temp
            count = count + out_temp.shape[0]

        [mse, rr_mae, true_rr] = util.cal_performance2(pred,pred2,fs, win_anal, win_move, raw_ppg, flag_pic)

    return mse, rr_mae, pred, true_rr

def val3(loader,model,device,data,win_anal,win_move,fs,flag_pic=0):
    with torch.no_grad(): 
        pred = np.zeros((data.n_data,win_anal,2))
        raw_ppg = np.zeros((data.n_data,win_anal))
        count = 0

        for j,(sig,label)in enumerate(loader):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)

            out_temp = model.forward(x)
            out_temp = np.squeeze(np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1),axis=1)
            y_temp = np.squeeze(np.squeeze(y.to('cpu').detach().numpy(),axis=1),axis=1)
            ppg = np.squeeze(np.squeeze(x.to('cpu').detach().numpy(),axis=1),axis=1)
            
            raw_ppg[count:count+out_temp.shape[0], :]  = ppg
            pred[count:count+out_temp.shape[0], :, 0] =  out_temp
            pred[count:count+y_temp.shape[0],   :, 1] =  y_temp
            count = count + out_temp.shape[0]

        [mse, rr_mae, true_rr] = util.cal_performance1(pred,fs, win_anal, win_move, raw_ppg, flag_pic)

    return mse, rr_mae, pred, true_rr



def sel_al(sig0,sig1,sig2,fs):

    sigs = np.append(sig0,sig1,axis=0)
    sigs = np.append(sigs,sig2,axis=0)
    result = np.zeros((3))
    for i in range(3):
        sig = sigs[i,:]
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

        p = plt.hist(Y,bins=10)
        p = p[0]
        p = p / np.sum(p)
        # result[i] = -np.sum(p * np.log2(p)) 

        for i2 in range(len(p)):
            if p[i2] != 0:
                result[i] = result[i] - (p[i2] * np.log2(p[i2]))

    ind = np.argmin(result)

    return sigs[ind,:]


def val3_ensemble(loader,model0,model1,model2,device,data,win_anal,win_move,fs,flag_pic=0):
    with torch.no_grad(): 
        pred = np.zeros((data.n_data,win_anal,2))
        raw_ppg = np.zeros((data.n_data,win_anal))
        count = 0

        for j,(sig,label,_)in enumerate(loader):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)

            out_temp0 = model0.forward(x)
            out_temp0 = np.squeeze(np.squeeze(out_temp0.to('cpu').detach().numpy(),axis=1),axis=1)

            out_temp1 = model1.forward(x)
            out_temp1 = np.squeeze(np.squeeze(out_temp1.to('cpu').detach().numpy(),axis=1),axis=1)

            out_temp2 = model2.forward(x)
            out_temp2 = np.squeeze(np.squeeze(out_temp2.to('cpu').detach().numpy(),axis=1),axis=1)

            out_temp = sel_al(out_temp0,out_temp1,out_temp2,fs)
            out_temp = out_temp[np.newaxis,:]

            y_temp = np.squeeze(np.squeeze(y.to('cpu').detach().numpy(),axis=1),axis=1)
            ppg = np.squeeze(np.squeeze(x.to('cpu').detach().numpy(),axis=1),axis=1)
            
            raw_ppg[count:count+out_temp.shape[0], :]  = ppg
            pred[count:count+out_temp.shape[0], :, 0] =  out_temp
            pred[count:count+y_temp.shape[0],   :, 1] =  y_temp
            count = count + out_temp.shape[0]

        [mse, rr_mae, true_rr] = util.cal_performance1(pred,fs, win_anal, win_move, raw_ppg, flag_pic)

    return mse, rr_mae, pred, true_rr



def val3_tr(loader,model,device,data,win_anal,win_move,fs,flag_pic=0):
    with torch.no_grad(): 
        pred = np.zeros((data.n_data,win_anal,2))
        raw_ppg = np.zeros((data.n_data,win_anal))
        count = 0

        for j,(sig,label,_,_)in enumerate(loader):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)

            out_temp = model.forward(x)
            out_temp = np.squeeze(np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1),axis=1)
            y_temp = np.squeeze(np.squeeze(y.to('cpu').detach().numpy(),axis=1),axis=1)
            ppg = np.squeeze(np.squeeze(x.to('cpu').detach().numpy(),axis=1),axis=1)
            
            raw_ppg[count:count+out_temp.shape[0], :]  = ppg
            pred[count:count+out_temp.shape[0], :, 0] =  out_temp
            pred[count:count+y_temp.shape[0],   :, 1] =  y_temp
            count = count + out_temp.shape[0]

        [mse, rr_mae, true_rr] = util.cal_performance1(pred,fs, win_anal, win_move, raw_ppg, flag_pic)

    return mse, rr_mae, pred, true_rr