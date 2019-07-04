"""
Author: Hamza
Dated: 20.04.2019
Project: texttomap

"""
import torch
import pickle
import numpy as np
from util.utilities import readcsv2jpgdict, wordskip

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open('training_data_pytorch05.pickle', "rb") as a:
    [klass, words_sparse, words, jpgs, enc, modes] = pickle.load(a)

def createjpgassociations(klas):
    dict = {}
    seq = 0
    for iter, q in enumerate(set(klas)):
        if not wordskip(modes[iter]):
            dict[seq] = np.asarray(jpgs)[np.argwhere(np.array(klas) == q).squeeze()]
            seq+=1
        if (iter+1) % 500 == 0:
            print("{} /{} ". format(iter+1, len(set(klas))))
    return dict

jpgklass = createjpgassociations(klass)

jpg2worddict = readcsv2jpgdict('Dataset_processing/netvlad/68_data.csv')

with open('training_data_pytorch06.pickle', "wb") as a:
    pickle.dump([klass, words_sparse, words, jpgs, enc, modes, jpgklass, jpg2worddict], a)
