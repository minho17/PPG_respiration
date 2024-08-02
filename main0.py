
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
# # import lstm_encoder_decoder
# import util1
from dataset import dataset0
# from datetime import datetime

def main():
    flag_data = 0 # 0=Capno / 1=BIDMC

    n_epoc = 80
    batch_size = 30
    lr = 0.001

    if flag_data == 0:
        path_data = "C:/Users/USER/Desktop/minho/PPG_resp/algorithm/matlab/data1_Capno.mat"
    elif flag_data == 1:
        path_data = "C:/Users/USER/Desktop/minho/PPG_resp/algorithm/matlab/data1_BIDMC.mat"

    mat_file = loadmat(path_data)
    data = mat_file['data']
    fs = mat_file['fs'][0,0]
    n_sub = data.shape[0]

    kf = KFold(n_sub)
    for ind_tr, ind_te in kf.split(data):
        ind_tr = shuffle(ind_tr)

        x_tr = data[ind_tr,:,1]
        x_te = data[ind_te,:,1]
        y_tr = data[ind_tr,:,0]
        y_te = data[ind_te,:,0]

        torch.cuda.empty_cache()
        gc.collect()
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        model = build_model.Correncoder_model( [8,8,8], [150,75,50], [20,20,10], 0.5).to(device)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        tr_data = dataset0(x_tr, y_tr)
        te_data = dataset0(x_te, y_te)

        loader_tr = DataLoader(tr_data, batch_size=batch_size, shuffle=True, num_workers=2)
        loader_te = DataLoader(te_data, batch_size=1, shuffle=False, num_workers=2)

        loss_list = []
        acc_list = []
        acc_list_test_epoch = []
        test_error = []

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

            if (i_epoc > 0 and i_epoc % 2 == 0) or i_epoc == n_epoc-1 :
                    model.eval()
                    with torch.no_grad(): 
                        for j,(sig,label)in enumerate(loader_te):
                            x = sig.to(torch.float32).to(device)

                            out_temp = model.forward(x)
                            out_temp = out_temp.to('cpu').detach().numpy()
                            y_temp = label.to('cpu').detach().numpy()

                            mean_error_bpm = breaths_per_min_zc(out_temp, y_temp)

                            print("Epoch:", i_epoc)
                            print(mean_error_bpm[0])
                            print(mean_error_bpm[1])



def breaths_per_min_zc(output_array_zc, input_array_zc):
    peak_count_output = []
    peak_count_cap = []
    for ind_output in range(output_array_zc.shape[0]):
        output_array_zc_temp = output_array_zc[ind_output, 0, :]
        input_array_zc_temp = input_array_zc[ind_output, :]
        output_array_zc_temp = output_array_zc_temp - 0.5
        input_array_zc_temp = input_array_zc_temp - 0.5
        zero_crossings_output = ((output_array_zc_temp[:-1] * output_array_zc_temp[1:]) < 0).sum()
        zero_crossings_input = ((input_array_zc_temp[:-1] * input_array_zc_temp[1:]) < 0).sum()
        peak_count_output.append(zero_crossings_output)
        peak_count_cap.append(zero_crossings_input)
        # breaths_per_min_output = (zero_crossings_output / 2)*6.25
    peak_count_output = np.array(peak_count_output)
    peak_count_cap = np.array(peak_count_cap)
    #6.5 is used ot scale up to 1 minute, as each segment here is 60/6.5 seconds long.
    mean_error = ((np.mean(peak_count_output - peak_count_cap)) / 2) * 6.5 
    mean_abs_error = ((np.mean(np.abs(peak_count_output - peak_count_cap))) / 2) * 6.5
    return mean_abs_error, mean_error




if __name__ == '__main__':
    main()
    