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


dataset = image_word_triplet_loader(jpgs, words_strings, words_sparse, klass,"Dataset_processing/jpeg_patch/" , ld= 30)
train_loader = DataLoader(dataset, batch_size=512, shuffle=False)

# embed = p_embed_net(128, 0.2)
# model = TripletNet(embed)
model = torch.load("gpu_models/02timodelcom.pt", map_location = 'cpu')
lib = []
model.eval()
model.embedding_net.eval()
for batch_idx, data in enumerate(train_loader):
    im, word_sp, identifier, _, _, _, _, _, _ = data
    output = model.get_embedding(im.to(device), word_sp.to(device))

    if len(lib) == 0:
        lib = output.detach().numpy()
        filenames = np.array(identifier)
    else:
        lib = np.concatenate((lib, output.detach().numpy()), axis=0)
        filenames = np.append(filenames, np.array(identifier))
    print("{}/{}".format(batch_idx, train_loader.__len__()))

with open('lib/final_lib_triplet' + str(2).zfill(2) + '.pickle', 'wb') as q:
    pickle.dump([lib, filenames], q)

# filenames2 =[]
# for i in filenames:
#     filenames2.append(i[30:61]+'.jpg')
# filenames2 = np.array(filenames2)

indice_dict = {}
for i in jpg_dict_train.keys():
    indice_dict[i] = np.argwhere(filenames2 == i)

with open('lib/final_lib_triplet_processed' + str(1).zfill(2) + '.pickle', 'wb') as q:
    pickle.dump([lib, filenames2, filenames, indice_dict], q)