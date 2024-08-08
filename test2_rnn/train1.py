
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
import torch
import gc
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.optim.lr_scheduler import ReduceLROnPlateau

import build_model
from pytorch_model_summary import summary
# import matplotlib.pyplot as plt
import util
import val
from dataset import dataset1
from random import shuffle
from sklearn.metrics import mean_squared_error 

def main():
    flag_data = 0 # 0=Capno / 1=BIDMC
    flag_re_train = 0

    n_epoc = 100
    batch_size = 5
    lr = 0.0001

    win_anal = 10
    win_move = 1

    if flag_re_train == 1:
        start_sub = 12
        path_result = os.getcwd() + '/result/' + 'D1_2024-07-26_14_02_02'
        log = util.log(path_result + '/log_tr.txt',1)
        log.w('===== retrain ===='+"\n")
    else:
        start_sub = 0
        path_result = util.mk_folders(flag_data)
        log = util.log(path_result + '/log_tr.txt')
        log.w('Data: ' + str(flag_data) + ' / lr: ' + str(lr) + ' / batch_size: ' + str(batch_size) + ' / n_epoc: ' + str(n_epoc) + ' / win_anal: ' + str(win_anal) + ' / win_move: ' + str(win_move))

    if flag_data == 0:
        path_data = "C:/Users/USER/Desktop/minho/PPG_resp/algorithm/matlab/data1_Capno.mat"
    elif flag_data == 1:
        path_data = "C:/Users/USER/Desktop/minho/PPG_resp/algorithm/matlab/data1_BIDMC.mat"

    mat_file = loadmat(path_data)
    data = mat_file['data']
    fs = mat_file['fs'][0,0]
    n_sub = data.shape[0]

    win_anal = int(win_anal*fs)
    win_move = int(win_move*fs)

    ind = np.array([i for i in range(n_sub)])
    
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    result = np.zeros((n_sub,3))
    for ind_te in range(start_sub,n_sub):
        ind_tr = np.argwhere(ind != ind_te)[:,0]
        [ind_tr,ind_val] = train_test_split(ind_tr,test_size=0.2)

        path_result2 = path_result + '/' + str(ind_te)
        os.makedirs(path_result2)
        path_result3 = path_result2 + '/model'
        os.makedirs(path_result3)

        model = build_model.Correncoder_model( [8,8,8], [150,75,50], [20,20,10], 0.5).to(device)
        # print(summary(model, torch.zeros(batch_size,1,win_anal), show_input=False))

        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience = 5,min_lr=lr*0.0001)
        stopper = util.EarlyStopping(best_fitness = 9999, patience = 10)

        if ind_te == 0:
            log.w(" / model: " + model.name + '\n')

        n_sub_tr = len(ind_tr)
        n_batch = int(n_sub_tr/batch_size)
        n_last_batch = n_sub_tr - n_batch*batch_size

        tr_data = dataset1(ind_tr, data, win_anal, win_move) 
        val_data = dataset1(ind_val, data, win_anal, win_move)
        te_data = dataset1(ind_te, data, win_anal, win_move)

        loader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2)
        loader_te = DataLoader(te_data, batch_size=1, shuffle=False, num_workers=2)

        # loss_arr = []
        best_val = 9999
        ind_tr2 = np.array([i for i in range(n_sub_tr)])
        for i_epoc in range(n_epoc):
            tr_loss = []
            model.train()
            shuffle(ind_tr2)

            for i_batch in range(n_batch + 1):
                if i_batch == n_batch:
                    ind_tr3 = ind_tr2[ i_batch*batch_size  : ]
                else:
                    ind_tr3 = ind_tr2[ i_batch*batch_size  : (i_batch+1)*batch_size ]

                tr_data.change_batch_ind(ind_tr3)
                loader_tr = DataLoader(tr_data, batch_size=1, shuffle=False, num_workers=2)

                pbar = tqdm(enumerate(loader_tr), total=len(loader_tr), leave = False ,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

                for batch, (sig,label) in pbar:
                    x = torch.swapaxes(sig.to(torch.float32).to(device),0,1)
                    y = torch.swapaxes(label.to(torch.float32).to(device),0,1)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=True):
                        out = model.forward(x)
                        loss = loss_func(out, y)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    pbar.set_description(f"Ep {i_epoc} _ Bat {i_batch}")
                    pbar.set_postfix(loss = int(loss.cpu().detach().numpy()*1000) )
                    
                    tr_loss.append(loss.cpu().detach().numpy())

            model.eval()
            [val_mse, val_rr_mae, _, _] = val.val1(loader_val,model,device,val_data,win_anal,win_move,fs)

            if best_val > val_mse and i_epoc > 10:
                best_val = val_mse
                util.save(ckpt_dir= path_result3 , net=model, optim=optimizer, epoch=i_epoc)

            scheduler.step(val_mse)
            if stopper(epoch = i_epoc, fitness = val_mse):
                if i_epoc <= 10:
                    util.save(ckpt_dir= path_result3 , net=model, optim=optimizer, epoch=i_epoc)
                break

            if (i_epoc > 0 and i_epoc % 2 == 0): # or i_epoc == n_epoc-1 :
                    model.eval()
                    tr_data.change_batch_ind(ind_tr2)
                    loader_tr = DataLoader(tr_data, batch_size=1, shuffle=False, num_workers=2)

                    [tr_mse, tr_rr_mae, _, _] = val.val1(loader_tr,model,device,tr_data,win_anal,win_move,fs)
                    [te_mse, te_rr_mae, _, _] = val.val1(loader_te,model,device,te_data,win_anal,win_move,fs)

                    log.w(str(i_epoc) + ": " + str('{:.3f}'.format(tr_mse)) + " - " + str('{:.3f}'.format(val_mse)) + " - " + str('{:.3f}'.format(te_mse)) )
                    log.w( " / tr: " + str('{:.3f}'.format(tr_rr_mae[0])) + " - " + str('{:.3f}'.format(tr_rr_mae[1])))
                    log.w( " / val: " + str('{:.3f}'.format(val_rr_mae[0])) + " - " + str('{:.3f}'.format(val_rr_mae[1])))
                    log.w( " / te: " + str('{:.3f}'.format(te_rr_mae[0])) + " - " + str('{:.3f}'.format(te_rr_mae[1])) +"\n")

                    print(tr_mse, tr_rr_mae)
                    print(val_mse, val_rr_mae)
                    print(te_mse, te_rr_mae)

        [model,optim,_] = util.load(ckpt_dir = path_result3, net=model, optim=optimizer)
        model.eval()
        [result[ind_te,0], result[ind_te,1:], pred, _] = val.val1(loader_te,model,device,te_data,win_anal,win_move,fs)
        log.w("Sub_" + str(ind_te) + ": " + str('{:.3f}'.format(result[ind_te,0])) + " / " + str('{:.3f}'.format(result[ind_te,1])) + " / " + str('{:.3f}'.format(result[ind_te,2])) +"\n")
        np.savez(path_result2 + '/results', ind_tr=ind_tr, ind_val=ind_val, result=result[ind_te,:], pred=pred, flag_data=flag_data, win_anal=win_anal, win_move=win_move)

    log.w("\n")
    log.w( "Final result: " + str('{:.3f}'.format(np.mean(result[:,0]))) + " / " + str('{:.3f}'.format(np.mean(result[:,1]))) + " / " + str('{:.3f}'.format(np.mean(result[:,2]))) )

if __name__ == '__main__':
    main()
    