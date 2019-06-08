"""
Author: Hamza
Dated: 13.05.2019
Project: texttomap

"""
import pickle
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from util.models import *
from util.updatelibrary import jpg_dict_lib as Reader
from util.loaders import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

jpg_dict_train = Reader(path='Dataset_processing/train03.txt')
with open('training_data_pytorch04.pickle', "rb") as a:
    [klass, words_sparse, words_strings, jpgs, enc, modes] = pickle.load(a)

def limitklass(klas, word, word_sparse, jpg):
    klas = np.array(klas)
    klass2 = []
    word2 = []
    word_sparse2 = []
    jpgs2 = []
    for itera, i in enumerate(klas):
        x = np.sum(np.asarray(klass2) == i).item()
        if x<=5:
            klass2.append(i)
            word2.append(word[itera])
            word_sparse2.append(word_sparse[itera])
            jpgs2.append(jpg[itera])
    return klass2, word2, word_sparse2, jpgs2

klass, words_strings, words_sparse, jpgs = limitklass(klass, words_strings, words_sparse, jpgs)



dataset = image_word_training_loader(jpgs, words_strings, words_sparse, klass,"Dataset_processing/jpeg_patch/")
train_loader = DataLoader(dataset, batch_size=512, shuffle=False)

# embed = p_embed_net(128, 0.2)
# model = TripletNet(embed)
model = torch.load("/home/presage3/2zip/2zip/models/25timodelcom.pt", map_location = 'cpu')
lib = []
model.eval()
model.embedding_net.eval()
for batch_idx, data in enumerate(train_loader):
    im, word_sp, identifier, _ = data
    output = model.get_embedding(im.to(device), word_sp.to(device))

    if len(lib) == 0:
        lib = output.detach().numpy()
        filenames = np.array(identifier)
    else:
        lib = np.concatenate((lib, output.detach().numpy()), axis=0)
        filenames = np.append(filenames, np.array(identifier))
    print("{}/{}".format(batch_idx, train_loader.__len__()))

with open('lib/final_lib_triplet_mined' + str(27).zfill(2) + '.pickle', 'wb') as q:
    pickle.dump([lib, filenames], q)

# filenames2 =[]
# for i in filenames:
#     filenames2.append(i[30:61]+'.jpg')
# filenames2 = np.array(filenames2)

def get_dicts(lib, filenames, klass):
    lib_dict = {}
    jpg_dict = {}
    for itera, i in enumerate(klass):
        try:
            lib_dict[i].append(lib[itera])
            jpg_dict[i].append(filenames[itera])
        except:
            lib_dict[i] = [lib[itera]]
            jpg_dict[i] = [filenames[itera]]
    return lib_dict, jpg_dict


libdict, jpgdict = get_dicts(lib, filenames, klass)

with open('lib/final_lib_triplet_mined_processed_libjpgdict' + str(27).zfill(2) + '.pickle', 'wb') as q:
    pickle.dump([libdict, jpgdict], q)

indice_dict = {}
for i in jpg_dict_train.keys():
    indice_dict[i] = np.argwhere(filenames == i)

with open('lib/final_lib_triplet_mined_processed' + str(27).zfill(2) + '.pickle', 'wb') as q:
    pickle.dump([lib, filenames, indice_dict], q)




def check(klass, lib):
    pmean = []
    nmean = []
    pmin = []
    nmin = []
    for itera, i  in enumerate(klass):
        indi = np.argwhere(np.asarray(klass) == i)
        samenorm = np.sort(np.linalg.norm(lib[itera] - lib[indi].squeeze(), axis=1))
        indi = np.argwhere(np.asarray(klass) != i)
        othernorm = np.linalg.norm(lib[itera] - lib[indi].squeeze(), axis=1)
        pmean.append(np.mean(samenorm[1:]))
        nmean.append(np.mean(othernorm))
        pmin.append(np.min(samenorm[1:]))
        nmin.append(np.min(othernorm))
        if itera%100 == 0:
            print("{}/{}".format(itera, len(klass)))
    return pmean, nmean, pmin, nmin

q,w,e,r = check(klass, lib)
print(np.sum(np.asarray(q)<np.asarray(w)))
print(np.sum(np.asarray(e)<np.asarray(r)))