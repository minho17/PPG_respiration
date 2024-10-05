import os
import numpy as np
from scipy.io import loadmat

print("start cord")
path = "/root"
file_list = os.listdir(path)

# print ("file_list: {}".format(file_list))
path_data = "/root/data1_Capno.mat"
mat_file = loadmat(path_data)
data = mat_file['data']

a =3
b=2
np.savez('/result/results', a=a)
np.savez('/result/results2', b=b)
