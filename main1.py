
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
import torch
import gc
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

import build_model
# from pytorch_model_summary import summary
# import matplotlib.pyplot as plt
import util
from dataset import dataset1
from datetime import datetime
from sklearn.metrics import mean_squared_error 

def main():
    flag_data = 0 # 0=Capno / 1=BIDMC

    n_epoc = 100
    batch_size = 30
    lr = 0.00001

    win_anal = 9.6
    win_move = 1

    if flag_data == 0:
        path_data = "C:/Users/USER/Desktop/minho/PPG_resp/algorithm/matlab/data1_Capno.mat"
    elif flag_data == 1:
        path_data = "C:/Users/USER/Desktop/minho/PPG_resp/algorithm/matlab/data1_BIDMC.mat"

    now = datetime.now()
    file_name_txt = './log_' + now.strftime('%Y-%m-%d %H_%M_%S') + '.txt'
    with open(file_name_txt, "w") as file:
        file.write('Data: ' + str(flag_data) + ' / lr: ' + str(lr) + ' / batch_size: ' + str(batch_size) + ' / n_epoc: ' + str(n_epoc) + ' / win_anal: ' + str(win_anal) + ' / win_move: ' + str(win_move))
    
    mat_file = loadmat(path_data)
    data = mat_file['data']
    fs = mat_file['fs'][0,0]
    n_sub = data.shape[0]

    win_anal = int(win_anal*fs)
    win_move = int(win_move*fs)

    kf = KFold(n_sub)
    result = np.zeros((n_sub,3))
    for ind_tr, ind_te in kf.split(data):
        ind_tr = shuffle(ind_tr)

        torch.cuda.empty_cache()
        gc.collect()
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        model = build_model.Correncoder_model( [8,8,8], [150,75,50], [20,20,10], 0.5).to(device)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        # stopper = util.EarlyStopping(best_fitness = train_cfg['best_fitness'], patience = train_cfg['patience'])

        if ind_te[0] == 0:     
            with open(file_name_txt, "a") as file:
                file.write(" / model: " + model.name + '\n')

        tr_data = dataset1(ind_tr, data, win_anal, win_move)
        te_data = dataset1(ind_te, data, win_anal, win_move)

        loader_tr = DataLoader(tr_data, batch_size=batch_size, shuffle=True, num_workers=2)
        loader_te = DataLoader(te_data, batch_size=1, shuffle=False, num_workers=2)

        loss_arr = []
        for i_epoc in range(n_epoc):       
            tr_loss = []
            model.train()
            pbar = tqdm(enumerate(loader_tr), total=len(loader_tr), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

            for batch, (sig,label) in pbar:
                x = sig.to(torch.float32).to(device)
                y = label.to(torch.float32).to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=True):
                    out = model.forward(x)
                    loss = loss_func(out, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                pbar.set_description(f"Ep {i_epoc}")
                pbar.set_postfix(loss = int(loss.cpu().detach().numpy()*1000) )
                
                tr_loss.append(loss.cpu().detach().numpy())

            tr_loss = np.mean(tr_loss) *1000
            loss_arr.append(tr_loss)

            # if stopper(epoch=epoch, fitness=np.mean(val_losses)):
            #     patience = train_cfg['patience']
            #     logger.info(f'Early stopping as validaiton loss did not improve for {patience} epochs')
            #     break

            if (i_epoc > 0 and i_epoc % 2 == 0) or i_epoc == n_epoc-1 :
                    model.eval()
                    with torch.no_grad(): 
                        pred = np.zeros((tr_data.n_data,win_anal,2))
                        count = 0
                        for j,(sig,label)in enumerate(loader_tr):
                            x = sig.to(torch.float32).to(device)
                            y = label.to(torch.float32)

                            out_temp = model.forward(x)
                            out_temp = np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1)
                            y_temp = np.squeeze(y.to('cpu').detach().numpy(),axis=1)

                            pred[count:count+out_temp.shape[0], :, 0] =  out_temp
                            pred[count:count+y_temp.shape[0],   :, 1] =  y_temp
                            count = count + out_temp.shape[0]
                    
                        tr_mse = mean_squared_error(pred[:,:,0],pred[:,:,1])
                        # [tr_mse, tr_rr_mae] = util.cal_performance1(pred,fs, win_anal, win_move, tr_data.n_win, tr_data.true_resp)

                        pred = np.zeros((te_data.n_data,win_anal,2))
                        count = 0
                        for j,(sig,label)in enumerate(loader_te):
                            x = sig.to(torch.float32).to(device)
                            y = label.to(torch.float32)

                            out_temp = model.forward(x)
                            out_temp = np.squeeze(out_temp.to('cpu').detach().numpy(),axis=1)
                            y_temp = np.squeeze(y.to('cpu').detach().numpy(),axis=1)

                            pred[count:count+out_temp.shape[0], : ,0] =  out_temp
                            pred[count:count+y_temp.shape[0],   : ,1] =  y_temp
                            count = count + out_temp.shape[0]

                        te_mse = mean_squared_error(pred[:,:,0],pred[:,:,1])
                        # [te_mse, te_rr_mae] = util.cal_performance1(pred,fs, win_anal, win_move, te_data.n_win, te_data.true_resp)
                        # print(tr_mse, te_mse)

                        with open(file_name_txt, "a") as file:
                            file.write( str(i_epoc) + ": " + str('{:.3f}'.format(tr_mse)) + " / " + str('{:.3f}'.format(te_mse)) +"\n")

                        # print(tr_mse, tr_rr_mae)
                        # print(te_mse, te_rr_mae)

        model.eval()
        pred = np.zeros((te_data.n_data,win_anal,2))
        count = 0
        for j,(sig,label)in enumerate(loader_te):
            x = sig.to(torch.float32).to(device)
            y = label.to(torch.float32)

            out_temp = model.forward(x)
            out_temp = np.squeeze(out_temp.to('cpu').detach().numpy())
            y_temp = np.squeeze(y.to('cpu').detach().numpy())

            pred[count:count+y_temp.shape[0], : ,0] =  out_temp
            pred[count:count+y_temp.shape[0], : ,1] =  y_temp
            count = count + y_temp.shape[0]

        [result[ind_te,0], result[ind_te,1:]] = util.cal_performance1(pred,fs, win_anal, win_move, te_data.n_win, te_data.true_resp)
        with open(file_name_txt, "a") as file:
            file.write( str(ind_te[0]) + ": " + str('{:.3f}'.format(result[ind_te[0],0])) + " / " + str('{:.3f}'.format(result[ind_te[0],1])) + " / " + str('{:.3f}'.format(result[ind_te[0],2])) +"\n")

    with open(file_name_txt, "a") as file:
            file.write("\n")
            file.write( "Final result: " + str('{:.3f}'.format(np.mean(result[:,0]))) + " / " + str('{:.3f}'.format(np.mean(result[:,1]))) + " / " + str('{:.3f}'.format(np.mean(result[:,2]))) )

if __name__ == '__main__':
    main()
    