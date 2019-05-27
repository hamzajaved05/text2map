"""
Author: hamza
dated: 08/05/2019
"""
"""
Author: Hamza
Dated: 20.04.2019
Project: texttomap

"""

import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import randint


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
        c = self.c_linear4(c)
        return c


model = Model(19408)
model.load_state_dict(torch.load("Dataset_processing/logs/39_checkpoint_dict.pt", map_location='cpu'))
model.eval()

with open("training_data_pytorch04.pickle", "rb") as a:
    [klass, words_sparse, words, jpgs, enc, modes] = pickle.load(a)


def process(word_sparse, jpg, words, path):
    word_indexed = torch.from_numpy(word_sparse.todense())
    im = torch.tensor(cv2.imread(path + jpg[:-4] + "_" + words + ".jpg")).permute(2, 0, 1)
    return word_indexed.float(), torch.div(im.float(), 255).float()


path = "Dataset_processing/jpeg_patch/"
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


model.c_linear3.register_forward_hook(get_activation('c_linear3'))


def loss(tens1, tens2):
    return np.sum((tens1.numpy() - tens2.numpy()) ** 2) ** (1 / 2)


while True:
    klass1 = np.random.choice(klass)

    indices1 = np.argwhere(np.array(klass) == klass1)
    indices2 = np.argwhere(np.array(klass) != klass1)
    dummy = np.random.choice(indices2.reshape(-1), size=(100))

    actvs1 = []
    for index in indices1.squeeze():
        q, w = process(words_sparse[index], jpgs[index], words[index], path)
        _ = model(w.unsqueeze(0), q.unsqueeze(0))
        actvs1.append(F.relu(activation['c_linear3']))
    # actvs1.append(F.normalize(F.relu(activation['c_linear3']), p=2, dim=1))

    actvs2 = []
    for index in dummy.squeeze():
        q, w = process(words_sparse[index], jpgs[index], words[index], path)
        _ = model(w.unsqueeze(0), q.unsqueeze(0))
        actvs2.append(F.relu(activation['c_linear3']))
    # actvs2.append(F.normalize(F.relu(activation['c_linear3']), p=2, dim=1))

    similar = []
    different = []

    for i in range(100):
        similar.append(loss(actvs1[randint(len(actvs1))], actvs1[randint(len(actvs1))]))
        different.append(loss(actvs1[randint(len(actvs1))], actvs2[randint(len(actvs2))]))

    x = plt.figure(1)
    plt.clf()
    plt.scatter(range(100), similar, c="c")
    plt.scatter(range(100), different, c="r")

    plt.draw()
    plt.pause(1e-6)
    input()
