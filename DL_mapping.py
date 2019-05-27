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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('lib/39lib_processed' + str(1).zfill(3) + '.pickle', 'rb') as q:
    [lib, jpgs, indice_dict] = pickle.load(q)


# jpgs_point = []
# jpgs_pointer = []
# for i in indice_dict.keys():
# 	for j in indice_dict[i]:
# 		jpgs_point.append(i)
# 		jpgs_pointer.append(j)
# del indice_dict

def reverse_dict(dict):
    new_dic = {}
    for k, v in dict.items():
        for x in v.reshape(-1).tolist():
            new_dic.setdefault(x, []).append(k)
    return new_dic


rev_dict_indice = reverse_dict(indice_dict)
del indice_dict

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


class Model(nn.Module):
    def __init__(self, classes):
        super(Model, self).__init__()
        self.i_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3)
        self.i_pool1 = nn.MaxPool2d(2)
        self.i_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.i_pool2 = nn.MaxPool2d(4)
        self.i_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.i_pool3 = nn.MaxPool2d(2)
        self.i_linear = nn.Linear(64 * 16 * 8, 512)

        self.t_conv1 = nn.Conv1d(in_channels=62, out_channels=32, kernel_size=5, padding=2)
        self.t_pool1 = nn.MaxPool1d(kernel_size=2)
        self.t_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.t_linear = nn.Linear(64 * 6, 16)

        self.c_linear1 = nn.Linear(512 + 16, 512)
        self.c_dropout1 = nn.Dropout(p=0.4)
        self.c_linear2 = nn.Linear(512, 1024)
        self.c_dropout2 = nn.Dropout(p=0.4)
        self.c_linear3 = nn.Linear(1024, 128)
        self.c_dropout3 = nn.Dropout(p=0.1)
        self.c_linear4 = nn.Linear(128, classes)

    def forward(self, im, tx):
        im = self.i_conv1(im)
        im = self.i_pool1(im)
        im = F.relu(im)
        im = self.i_conv2(im)
        im = self.i_pool2(im)
        im = F.relu(im)
        im = self.i_conv3(im)
        im = self.i_pool3(im)
        im = F.relu(im)
        im = im.view(-1, 64 * 16 * 8)
        im = self.i_linear(im)

        tx = self.t_conv1(tx)
        tx = self.t_pool1(tx)
        tx = F.relu(tx)
        tx = self.t_conv2(tx)
        tx = tx.view(-1, 64 * 6)
        tx = self.t_linear(tx)

        c = torch.cat((im, tx), 1)
        c = self.c_linear1(c)
        c = F.relu(c)
        c = self.c_dropout1(c)

        c = self.c_linear2(c)
        c = F.relu(c)
        c = self.c_dropout2(c)

        c = self.c_linear3(c)
        c = F.relu(c)
        c = self.c_dropout3(c)
        c = F.normalize(c, p=2, dim=1)

        c = self.c_linear4(c)
        return c


model = Model(19408)
model.load_state_dict(torch.load("logs/009checkdict.pt", map_location='cpu'))

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


model.c_dropout3.register_forward_hook(get_activation('embedding'))

jpg_dict_test = Reader(path='Dataset_processing/test03.txt')

dataset = image_word_dataset(enc, 12, list(jpg_dict_test.keys()), jpg_dict_test, path='Dataset_processing/jpeg_patch/')

train_loader = DataLoader(dataset, batch_size=1, shuffle=False)


def prob_un(x, y):
    return x + y - x * y


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def match(query, lib, rev_dict_indice):
    dic = {}
    # count = 0
    # no_patches = query.shape[0]
    for iter in range(query.shape[0]):
        normal_score = LA.norm(query[iter] - lib, axis=1)
        # normal_score = torch.tensor(normal_score)
        # normal_score = F.softmax(normal_score, dim = 0)

        ind = np.argpartition(normal_score, 100)[:100]
        selected_normal_scores = normal_score[ind]
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

    model(im_batch.squeeze(0).to(device), word_batch.squeeze(0).to(device))
    query = F.normalize(F.relu(activation['embedding']), p=2, dim=1).numpy()
    best_match = match(query, lib, rev_dict_indice)
    results[query_jpg[0]] = best_match
    print(batch_idx, batch_idx / train_loader.__len__())
    if batch_idx == 3000:
        break

with open('util/dl_logs/03_test_result_confidenceT.pickle', 'wb') as a:
    pickle.dump(results, a)
