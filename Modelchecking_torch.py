"""
Author: Hamza
Dated: 20.04.2019
Project: texttomap

"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from util.word_encoding import getklass

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

reload = False
if reload:
    enc_dict, klasses, wordarray = getklass()
    with open("util/dl_logs/pytorch_data_00", 'wb') as f:
        pickle.dump([enc_dict, klasses, wordarray], f)
else:
    with open("Dataset_processing/split/pytorch_data_00", 'rb') as f:
        [enc_dict, klasses, wordarray] = pickle.load(f)
klasses2 = [0]
index = 0
for i in range(1, len(klasses)):
    if not klasses[i - 1] == klasses[i]:
        index += 1
    klasses2.append(index)

print("\n\nData Loaded\n\n")


class Dataset(data.Dataset):
    def __init__(self, x, labels):
        self.labels = labels
        self.x = x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feat = self.x[index]

        X = torch.from_numpy(feat.todense())
        y = torch.tensor(self.labels[index])

        return X, y


class ParkhiNet(nn.Module):
    def __init__(self, classes):
        super(ParkhiNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        #         self.pool2 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(64 * 31 * 6, 150)
        self.bnl1 = nn.BatchNorm1d(150)
        self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(50, classes)

    def forward(self, x):
        x = self.bn1(self.pool1(F.relu(self.conv1(x))))
        x = (F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 31 * 6)
        x = self.bnl1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


x = [enc_dict[word] for word in enc_dict.keys()]
training_set = Dataset(x, klasses2)
no_classes = max(klasses2) + 1

# Network.load_state_dict(torch.load("util/dl_logs/03testingdict.pt",map_location=device))
Network = torch.load("util/dl_logs/04testingcomplete.pt", map_location=device)
with open("util/dl_logs/04log.pickle", "rb") as F:
    [train_loss, train_accuracy] = pickle.load(F)

plt.figure(1)
plt.plot(train_loss)
plt.figure(2)
plt.plot(np.divide(train_accuracy, len(klasses2)))
# activation = {}

# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook


# Network.fc2.register_forward_hook(get_activation('fc2'))
#         print(activation['fc2'].size())
