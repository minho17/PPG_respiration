import os
import numpy as np

print("start cord")
path = "/root"
file_list = os.listdir(path)

# print ("file_list: {}".format(file_list))
path_data = "C:/Users/minho/Desktop/work/kiom/PPG_resp/algorithm/matlab/data1_Capno.mat"
mat_file = loadmat(path_data)
data = mat_file['data']

a =3
np.savez('/result/results', a=a)
