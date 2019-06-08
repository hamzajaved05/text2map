"""
Author: Hamza
Dated: 13.05.2019
Project: texttomap

"""

import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
from torch.utils import data
from torch.utils.data import DataLoader

from util.updatelibrary import jpg_dict_lib as Reader
from util.utilities import word2encodedword
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# with open('lib/final_lib_triplet_mined_processed' + str(11).zfill(2) + '.pickle', 'rb') as q:
#     [lib, jpgs1, indice_dict] = pickle.load(q)

with open('lib/final_lib_triplet_mined_processed_libjpgdict' + str(25).zfill(2) + '.pickle', 'rb') as q:
    [libdict, jpgdict] = pickle.load(q)


# im = pd.read_csv("sparse/68_data_sparse.csv")["imagesouce"].to_numpy()

# def dict_filter(ind_dic, im):
#     dict2 = {}
#     for i in indice_dict.keys():
#         if i in im:
#             dict2[i] = ind_dic[i]
#     return dict2
# def libjpg_filter(lib, jpg, im):
#     lib2 = []
#     jpg2 = []
#     for itera, i in enumerate(jpg):
#         if i in im:
#             lib2.append(lib[itera])
#             jpg2.append(i)
#     return lib2, jpg2
# indice_dict = dict_filter(indice_dict, im)
# lib, jpgs = libjpg_filter(lib, jpgs, im)

# jpgs_point = []
# jpgs_pointer = []
# for i in indice_dict.keys():
# 	for j in indice_dict[i]:
# 		jpgs_point.append(i)
# 		jpgs_pointer.append(j)
# del indice_dict
#
# def reverse_dict(dict):
#     new_dic = {}
#     for k, v in dict.items():
#         for x in v.reshape(-1).tolist():
#             new_dic.setdefault(x, []).append(k)
#     return new_dic
#
#
# rev_dict_indice = reverse_dict(indice_dict)
# del indice_dict

with open('training_data_pytorch04.pickle', "rb") as a:
    [_, _, _, _, enc, _] = pickle.load(a)


class image_word_dataset(data.Dataset):
    def __init__(self, enc, length, jpgs, jpg_dict_test, path):
        self.enc = enc
        self.length = length
        self.jpgs = jpgs
        self.jpg_dict = jpg_dict_test
        self.im_path = path

    def __len__(self):
        return len(self.jpgs)

    def __getitem__(self, index):
        # print(index)
        x = self.jpg_dict[self.jpgs[index]]
        ind = 0
        for itera, word in enumerate(x):
            if len(list(word)) < 12:
                word_tensor = torch.from_numpy(word2encodedword(self.enc, word, 12).todense())
                im = torch.tensor(cv2.imread(self.im_path + self.jpgs[index][:-4] + "_" + word + ".jpg")).permute(2, 0,
                                                                                                                  1)

                im = torch.div(im.float(), 255)
                # print(im.size())
                if ind == 0:
                    word_batch = word_tensor.unsqueeze(0)
                    image_batch = im.unsqueeze(0)
                    ind += 1
                else:
                    word_batch = torch.cat((word_batch, word_tensor.unsqueeze(0)), dim=0)
                    image_batch = torch.cat((image_batch, im.unsqueeze(0)), dim=0)
        if ind:
            return image_batch, word_batch.float(), self.jpgs[index], bool(ind)
        else:
            return [], [], [], bool(ind)


torch.nn.Module.dump_patches = True
model = torch.load("models/11timodelcom.pt", map_location = 'cpu')
model.eval()
model.embedding_net.eval()

jpg_dict_test = Reader(path='Dataset_processing/test03.txt')

dataset = image_word_dataset(enc, 12, list(jpg_dict_test.keys()), jpg_dict_test, path='Dataset_processing/jpeg_patch/')

train_loader = DataLoader(dataset, batch_size=1, shuffle=False)


def prob_un(x, y):
    return x + y - x * y


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def matchold(query, lib, rev_dict_indice):
    dic = {}
    # count = 0
    # no_patches = query.shape[0]
    for iter in range(query.shape[0]):
        normal_score = LA.norm(query[iter] - lib, axis=1)
        # normal_score = torch.tensor(normal_score)
        # normal_score = F.softmax(normal_score, dim = 0)

        ind = np.argpartition(normal_score, 10)[:10]

        # ind = np.argwhere(normal_score<0.05)

        selected_normal_scores = normal_score[ind]
        # print(selected_normal_scores.shape)
        soft_norms = softmax(selected_normal_scores)
        for it, norms in enumerate(soft_norms):
            try:
                dic[rev_dict_indice[ind[it]][0]] = prob_un(dic[rev_dict_indice[ind[it]][0]], norms)
            except:
                dic[rev_dict_indice[ind[it]][0]] = norms

        # normalized_score_indices = np.argwhere(ind == normal_score)
        # file_ids = np.array([np.argwhere(indice == np.array(jpgs_pointer).reshape(-1)) for indice in ind]).reshape(-1)
        # for itera, index in enumerate(file_ids):
        #     try:
        #         dic[jpgs_point[index]].append(normal_score[ind[itera]])
        #
        #     except:
        #         dic[jpgs_point[index]] = normal_score[ind[itera]]
        #         count+=1
        # # for key, value in indice_dict.items():
        #     if value in ind:
    # assert count == len(dic)

    n = 10
    return list({key: dic[key] for key in sorted(dic, key=dic.get, reverse=True)[:n]}.items())

def match(query, lib_dict, jpgdict):
    dic = {}
    jpg = {}
    # count = 0
    # no_patches = query.shape[0]
    for iter in range(query.shape[0]):
        for i in lib_dict.keys():
            # print(len(lib_dict[i]))
            normal_score = LA.norm(query[iter] - lib_dict[i], axis=1)
            dic[i] = np.mean(normal_score)

        dic[i][i][i]
        minkey = min(dic, key=dic.get)
        minvalue = dic[minkey]
        print(minvalue)
        confidence = np.max([(minvalue-0.101)/10, 0])
        for i in jpgdict[minkey]:
            try:
                jpg[i] = prob_un(jpg[i], confidence)
            except:
                jpg[i] = confidence
    n = 5
    return list({key: jpg[key] for key in sorted(jpg, key=jpg.get, reverse=True)[:n]}.items())


results = {}
for batch_idx, data in enumerate(train_loader):
    im_batch, word_batch, query_jpg, check = data
    if check == False:
        continue
    # print(word_batch.size(), im_batch.size())

    # index = 7
    # image = plt.imshow(im_batch.squeeze()[index].permute(1,2,0).numpy())
    # tx = word_batch.squeeze()[index].numpy()
    # enc.inverse_transform(csc_matrix(tx[:,:int(np.sum(tx))]).transpose())

    res = model.get_embedding(im_batch.squeeze(0).to(device), word_batch.squeeze(0).to(device))
    query = res.detach().numpy()
    best_match = match(query, libdict, jpgdict)
    results[query_jpg[0]] = best_match
    print(batch_idx, batch_idx / train_loader.__len__())
    if batch_idx == 3000:
        break

with open('util/dl_logs/03_test_result_confidenceTriplet_mined_sparse_jpglibdict.pickle', 'wb') as a:
    pickle.dump(results, a)