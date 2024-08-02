
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

import build_model
# from pytorch_model_summary import summary
# import matplotlib.pyplot as plt
import util
import val
from dataset import dataset1

from sklearn.metrics import mean_squared_error 

def main():

    path_target = 'D1_2024-07-26_14_02_02'
    flag_pic = 0

    path_result = os.getcwd() + '/result/' + path_target
    n_folder = 0
    for item in os.listdir(path_result): 
        sub_folder = os.path.join(path_result, item)
        if os.path.isdir(sub_folder):
            n_folder = n_folder + 1

    flag_data=int(path_target[1])
    if flag_data == 0:
        path_data = "C:/Users/USER/Desktop/minho/PPG_resp/algorithm/matlab/data1_Capno.mat"
    elif flag_data == 1:
        path_data = "C:/Users/USER/Desktop/minho/PPG_resp/algorithm/matlab/data1_BIDMC.mat"

    lr = 0.0001
    win_anal = 9.6
    win_move = 1
    
    mat_file = loadmat(path_data)
    data = mat_file['data']
    fs = mat_file['fs'][0,0]
    win_anal = int(win_anal*fs)
    win_move = int(win_move*fs)

    log = util.log(path_result + '/log_te.txt')
    log.w('Data: ' + str(flag_data) + ' / win_anal: ' + str(win_anal) + ' / win_move: ' + str(win_move) + "\n")

    results = np.zeros((n_folder,3))
    for i in range(n_folder):
        sub_folder = os.path.join(path_result, str(i))
        try:
            load_data = np.load(sub_folder + '/results.npz')
        except:
            results = results[0:-1,:]
            break

        ind_tr = load_data['ind_tr']
        ind_val = load_data['ind_val']
        result = load_data['result']
        pred = load_data['result']
        flag_data = load_data['flag_data']

        ind_te = i
        torch.cuda.empty_cache()
        gc.collect()
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        model = build_model.Correncoder_model( [8,8,8], [150,75,50], [20,20,10], 0.5).to(device)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        stopper = util.EarlyStopping(best_fitness = 9999, patience = 5)
        [model,optim,_] = util.load(ckpt_dir = sub_folder + '/model' , net=model, optim=optimizer)

        te_data = dataset1(ind_te, data, win_anal, win_move)
        loader_te = DataLoader(te_data, batch_size=1, shuffle=False, num_workers=2)

        model.eval()
        if flag_pic == 0:
            [results[i,0], results[i,1:], pred, true_rr] = val.val1(loader_te,model,device,te_data,win_anal,win_move,fs)
        else:
            [results[i,0], results[i,1:], pred, true_rr] = val.val1(loader_te,model,device,te_data,win_anal,win_move,fs,sub_folder)
 
        log.w("Sub_" + str(ind_te) + ": " + str('{:.3f}'.format(results[i,0])) + " / " + str('{:.3f}'.format(results[i,1])) + " / " + str('{:.3f}'.format(results[i,2])) + "  // True RR: " + str('{:.3f}'.format(true_rr[0])) + " / " + str('{:.3f}'.format(true_rr[1])) +"\n")

        # util.save(ckpt_dir= path_result2 , net=model, optim=optimizer, epoch=i_epoc)
        # np.savez(path_result2 + '/results', ind_tr=ind_tr, ind_val=ind_val, result=result[ind_te,:], pred=pred, flag_data=flag_data)

    print(np.median(results,axis=0))
    log.w("\n" + "Final result" + "\n")
    log.w( "Median: " + str('{:.3f}'.format(np.median(results[:,0]))) + " / " + str('{:.3f}'.format(np.median(results[:,1]))) + " / " + str('{:.3f}'.format(np.median(results[:,2]))) + "\n")
    log.w( "Mean: " + str('{:.3f}'.format(np.mean(results[:,0]))) + " / " + str('{:.3f}'.format(np.mean(results[:,1]))) + " / " + str('{:.3f}'.format(np.mean(results[:,2]))) )

if __name__ == '__main__':
    main()
    