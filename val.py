import torch
import numpy as np
import util

def val(loader,model,device,data,win_anal,fs):
    with torch.no_grad(): 
        pred = np.zeros((data.n_data,win_anal,2))
        raw_ppg = np.zeros((data.n_data,win_anal))
        count = 0

        for j,(sig,label,_)in enumerate(loader):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)

            out_temp = model.forward(x)
            out_temp = np.squeeze(np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1),axis=1)
            y_temp = np.squeeze(np.squeeze(y.to('cpu').detach().numpy(),axis=1),axis=1)
            ppg = np.squeeze(x[:,0,:,:].to('cpu').detach().numpy(),axis=1)
            
            raw_ppg[count:count+out_temp.shape[0], :]  = ppg
            pred[count:count+out_temp.shape[0], :, 0] =  out_temp
            pred[count:count+y_temp.shape[0],   :, 1] =  y_temp
            count = count + out_temp.shape[0]

        [mse, rr_mae, true_rr] = util.cal_performance1(pred,fs)

    return mse, rr_mae, pred, true_rr


def val_ensem(loader,models,n_model,device,data,win_anal,fs):
    with torch.no_grad(): 
        pred = np.zeros((data.n_data,win_anal,n_model+1))
        raw_ppg = np.zeros((data.n_data,win_anal))
        count = 0

        for j,(sig,label,_)in enumerate(loader):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)

            for i in range(n_model):
                out_temp = models[i].forward(x)
                out_temp = np.squeeze(np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1),axis=1)
                pred[count:count+out_temp.shape[0], :, i] =  out_temp

            y_temp = np.squeeze(np.squeeze(y.to('cpu').detach().numpy(),axis=1),axis=1)
            ppg = np.squeeze(x[:,0,:,:].to('cpu').detach().numpy(),axis=1)
            
            raw_ppg[count:count+out_temp.shape[0], :]  = ppg
            pred[count:count+y_temp.shape[0],   :, n_model] =  y_temp
            count = count + out_temp.shape[0]

        [mse, rr_mae, true_rr] = util.cal_performance1_ensem(pred,fs, n_model)

    return mse, rr_mae, pred, true_rr


def val_w(loader,models,n_model,device,data,win_anal,fs,Q1,Q3):
    with torch.no_grad(): 
        pred = np.zeros((data.n_data,win_anal,n_model+1))
        raw_ppg = np.zeros((data.n_data,win_anal))
        count = 0

        IQR = Q3 - Q1
        lb = Q1 - 1.5 * IQR  
        ub = Q3 + 1.5 * IQR 

        for j,(sig,label,_)in enumerate(loader):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)

            for i in range(n_model):
                out_temp = models[i].forward(x)
                out_temp = np.squeeze(np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1),axis=1)
                pred[count:count+out_temp.shape[0], :, i] =  out_temp

            y_temp = np.squeeze(np.squeeze(y.to('cpu').detach().numpy(),axis=1),axis=1)
            ppg = np.squeeze(x[:,0,:,:].to('cpu').detach().numpy(),axis=1)
            
            raw_ppg[count:count+out_temp.shape[0], :]  = ppg
            pred[count:count+y_temp.shape[0],   :, n_model] =  y_temp
            count = count + out_temp.shape[0]

        [mse, rr_mae, true_rr] = util.cal_performance_ensem_w(pred,fs, n_model, raw_ppg,lb,ub)

    return mse, rr_mae, pred, true_rr

