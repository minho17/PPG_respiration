
import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader
import torch
import gc
import os

import build_model
import util
import val
from dataset import dataset1

def main():

    flag_data = 1 # 0, 1
    win_anal = 32 # 32, 64
    win_move = 3 # 3, 6

    path_model = os.getcwd() + '/model/'
    path_result = os.getcwd()

    if flag_data == 0:
        path_data = "C:/Users/USER/Desktop/minho/PPG_resp/algorithm/matlab/data1_Capno.mat"
    else:
        path_data = "C:/Users/USER/Desktop/minho/PPG_resp/algorithm/matlab/data1_BIDMC.mat"

    mat_file = loadmat(path_data)
    data = mat_file['data']
    fs = mat_file['fs'][0,0]

    file_name = 'metric_' + str(flag_data) + '_' + str(win_anal) + '.mat'
    mat_file = loadmat(file_name)
    metric = mat_file['metric']

    win_anal = int(win_anal*fs)
    win_move = int(win_move*fs)
    n_sub = data.shape[0]

    log = util.log(path_result + '/result.txt')
    log.w('Data: ' + str(flag_data) + ' / win_anal: ' + str(win_anal) + ' / win_move: ' + str(win_move) + "\n")

    n_report = 7
    results = np.zeros((n_sub,n_report))
    ind = np.array([i for i in range(n_sub)])

    count = 0
    # for i in range(0,n_sub):
    for i in range(7,8):
        ind_te = i

        ind_tr0 = np.argwhere(ind != ind_te)[:,0]
        metric2 = metric[ind_tr0,0].copy()

        Q1 = np.percentile(metric2, 25)  
        Q3 = np.percentile(metric2, 75) 

        torch.cuda.empty_cache()
        gc.collect()
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        te_data = dataset1(ind_te, data, win_anal, win_move,0)
        loader_te = DataLoader(te_data, batch_size=1, shuffle=False, num_workers=2)

        models = []
        for i_path in range(5):
            sub_folder_i = os.path.join(path_model, str(i_path))

            model_i = build_model.model( [4], [128,64,32], [32,64,128], 0.1).to(device)
            optimizer_i = torch.optim.Adam(model_i.parameters(), lr=0.1)
            [model_i,_] = util.load(ckpt_dir = sub_folder_i , net=model_i, optim=optimizer_i)
            model_i.eval()
            models.append(model_i)
        
            [_, results[count,i_path], _, _] = val.val(loader_te,model_i,device,te_data,win_anal,fs)

        [_, results[count,5], _, _] = val.val_ensem(loader_te,models,3,device,te_data,win_anal,fs)
        [_, results[count,6], _, _] = val.val_w(loader_te,models,5,device,te_data,win_anal,fs,Q1,Q3)    

        log.w("Sub_" + str(ind_te) + ": ")
        for i_report in range(n_report):
            if i_report != (n_report-1):
                log.w(str('{:.3f}'.format(results[count,i_report])) + " / " )
            else:
                log.w(str('{:.3f}'.format(results[count,i_report])) +"\n")

        count = count + 1

    results = results[0:count,:]


if __name__ == '__main__':
    main()
    