import torch
import numpy as np
import util

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

