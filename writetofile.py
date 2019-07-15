import pandas as pd
import numpy as np
import pickle
from numpy import linalg as LA
import os

vlad = pd.read_csv("Dataset_processing/netvlad/68netvlad.csv", sep = ' ', header=None)

length = vlad.shape[0]
for i in range(length):
    np.savetxt('nv_txt/'+vlad.iloc[i,0], vlad.iloc[i,1:])
    print("{} / {}".format(i, length))
# jpg = pd.read_csv("Dataset_processing/netvlad/68_data.csv", skipinitialspace = True, usecols = ['imagesource'], header = None).to_numpy()

# x = np.loadtxt("nv_txt/0000068_0104480_0000001_0016027.jpg")
#
# count = 0
# for i in np.unique(jpg):
#     if os.path.exists('nv_txt/'+i):
#         count += 1
