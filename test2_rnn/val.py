import torch
import numpy as np
import util

def val1(loader,model,device,data,win_anal,win_move,fs,flag_pic=0):
    with torch.no_grad(): 
        pred = np.zeros((data.n_sub,data.n_win,win_anal,2))
        count = 0
        for j,(sig,label)in enumerate(loader):
            if len(sig.shape) == 2:
                sig = sig.unsqueeze(0)
            x = torch.swapaxes(sig.to(torch.float32).to(device),0,1)
            y = torch.swapaxes(label.to(torch.float32),0,1)

            out_temp = model.forward(x)
            out_temp = np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1)
            y_temp = np.squeeze(y.to('cpu').detach().numpy(),axis=1)

            pred[:, j, :, 0] =  out_temp
            pred[:, j, :, 1] =  y_temp

        [mse, rr_mae, true_rr] = util.cal_performance1(pred,fs, win_anal, win_move, data.n_win, data.raw_sig,flag_pic)

    return mse, rr_mae, pred, true_rr


def val2(loader,model,device,data,win_anal,win_move,fs):
    with torch.no_grad(): 
        pred = np.zeros((data.n_data,win_anal,2))
        count = 0
        for j,(sig,label)in enumerate(loader):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)

            out_temp = model.forward(x)
            out_temp = np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1)
            y_temp = np.squeeze(y.to('cpu').detach().numpy(),axis=1)

            pred[count:count+out_temp.shape[0], :, 0] =  out_temp
            pred[count:count+y_temp.shape[0],   :, 1] =  y_temp
            count = count + out_temp.shape[0]

        [mse, rr_mae] = util.cal_performance2(pred,fs, win_anal, win_move, data.n_win, data.true_resp)

    return mse, rr_mae, pred


def val3(loader,model,device,data,win_anal,win_move,fs,flag_pic=0):
    with torch.no_grad(): 
        pred = np.zeros((data.n_data,win_anal,2))
        count = 0
        for j,(sig,label)in enumerate(loader):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)

            out_temp = model.forward(x)
            out_temp = np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1)
            y_temp = np.squeeze(y.to('cpu').detach().numpy(),axis=1)

            pred[count:count+out_temp.shape[0], :, 0] =  out_temp
            pred[count:count+y_temp.shape[0],   :, 1] =  y_temp
            count = count + out_temp.shape[0]

        [mse, rr_mae, true_rr] = util.cal_performance3(pred,fs, win_anal, win_move, data.n_win, data.raw_sig,flag_pic)

    return mse, rr_mae, pred, true_rr