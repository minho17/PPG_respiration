
import numpy as np
from scipy.io import loadmat, savemat
import util

def main():
    flag_data = 0 # 0=Capno / 1=BIDMC

    win_anal = 64 # 32 / 64
    win_move = 6 # 3 / 6

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
    metric = np.zeros((n_sub,1))

    for i in range(0,n_sub):

        data1 = data[i,:,:].copy()
        data1 = data1.reshape(-1,data1.shape[-2],data1.shape[-1])

        n_sig = data1.shape[1]
        n_win = int((n_sig - win_anal)/win_move) + 1

        result = np.zeros((n_win,1))
        for i1 in range(n_win):
            ind_start = i1*win_move
            temp_sig = data1[ 0, ind_start : ind_start + win_anal, 1 ]
            temp_sig= (temp_sig - np.mean(temp_sig)) / np.std(temp_sig)

            result[i1,0],_,_ = util.cal_resp1(temp_sig,fs,0)

        metric[i,0] = np.mean(result)

    mdic = {"metric": metric}
    file_name = 'metric_' + str(flag_data) + '_' + str(int(win_anal/fs)) + '.mat'
    savemat(file_name, mdic)

if __name__ == '__main__':
    main()
    