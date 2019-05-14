"""
Author: Hamza
Dated: 13.05.2019
Project: texttomap

"""

from util.updatelibrary import jpg_dict_lib as Reader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import torch
from torch.utils import data
import cv2
import argparse
from math import ceil
from tensorboardX import SummaryWriter
from util.utilities import word2encdict
from util.utilities import flatten,word2encodedword
from numpy import linalg as LA
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('lib/36lib_processed'+str(1).zfill(2) +'.pickle','rb') as q:
    [lib,jpgs,indice_dict] = pickle.load(q)
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
        x = self.jpg_dict[self.jpgs[index]]
        ind = 0
        for itera, word in enumerate(x):
            if len(list(word))<12:
                word_tensor = torch.from_numpy(word2encodedword(self.enc, word, 12).todense())
                im = torch.tensor(cv2.imread(self.im_path + self.jpgs[index][:-4] + "_" + word + ".jpg")).permute(2,0,1)
                plt.imshow(im.permute(1, 2, 0))

                im = torch.div(im.float(),255)
                # print(im.size())
                if ind == 0:
                    word_batch = word_tensor.unsqueeze(0)
                    image_batch = im.unsqueeze(0)
                    ind +=1
                else:
                    word_batch = torch.cat((word_batch,word_tensor.unsqueeze(0)),dim=0)
                    image_batch = torch.cat((image_batch, im.unsqueeze(0)), dim=0)
        return image_batch, word_batch.float(), self.jpgs[index]


class Model(nn.Module):
        def __init__(self, classes):
            super(Model, self).__init__()
            self.i_conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, padding=3)
            self.i_pool1 = nn.MaxPool2d(2)
            self.i_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding=3)
            self.i_pool2 = nn.MaxPool2d(2)
            self.i_conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
            self.i_pool3 = nn.MaxPool2d(2)
            self.i_linear = nn.Linear(32 * 32 * 16, 512)

            self.t_conv1 = nn.Conv1d(in_channels=62, out_channels=32, kernel_size=5, padding=2)
            self.t_pool1 = nn.MaxPool1d(kernel_size=2)
            self.t_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
            self.t_linear = nn.Linear(64 * 6, 16)

            self.c_linear1 = nn.Linear(512 + 16, 512)
            self.c_dropout1 = nn.Dropout(p=0.4)
            self.c_linear2 = nn.Linear(512, 1024)
            self.c_dropout2 = nn.Dropout(p=0.4)
            self.c_linear3 = nn.Linear(1024, 128)
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
            im = im.view(-1, 32 * 16 * 32)
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
            c = self.c_linear4(c)
            return c

model = Model(19408)
model.load_state_dict(torch.load("Dataset_processing/logs/31_checkpoint_dict.pt", map_location = 'cpu'))


activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook
model.c_linear3.register_forward_hook(get_activation('c_linear3'))


jpg_dict_test = Reader(path = 'Dataset_processing/test03.txt')

# words = jpg_dict_test['0000068_0027733_0000002_0004729.jpg']

dataset = image_word_dataset(enc, 12, list(jpg_dict_test.keys()), jpg_dict_test, 'Dataset_processing/jpeg_patch/')

train_loader = DataLoader(dataset, batch_size=1, shuffle = False)

def match(query, lib):
    dic = {}
    count = 0
    queries = query.shape[0]
    for iter in range(query.shape[0]):
        q = LA.norm(query[iter]-lib, axis = 1)
        ind = np.argpartition(q, 100)[:100]
        ind[np.argsort(q[ind])]
        ims  = [key for key, value in indice_dict.items() if value in ind]
        for itera,images in enumerate(ims):
            try:
                dic[images] += q[ind[itera]]*queries

            except:
                dic[images] = q[ind[itera]]*queries
                count+=1
    assert count == len(dic)
    return min(dic, key=dic.get)

# def match_check(query, lib)

results = []
for batch_idx, data in enumerate(train_loader):
    im_batch, word_batch, query_jpg = data
    # print(word_batch.size(), im_batch.size())
    model(im_batch.squeeze(0).to(device), word_batch.squeeze(0).to(device))
    query = F.relu(activation['c_linear3']).numpy()
    best_match =  match(query, lib)
    results.append([query_jpg[0],best_match])
    print(batch_idx/train_loader.__len__())
