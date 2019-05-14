"""
Author: Hamza
Dated: 13.05.2019
Project: texttomap

"""
from util.updatelibrary import jpg_dict_lib as Reader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import torch
from torch.utils import data
import cv2
from util.utilities import word2encdict
from util.utilities import flatten
import pandas as pd
from sklearn.decomposition import PCA

pca = PCA(n_components=32)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

jpg_dict_train = Reader(path = 'Dataset_processing/train03.txt')
# jpg_dict_test = Reader(path = 'Dataset_processing/test03.txt')
with open('training_data_pytorch04.pickle', "rb") as a:
    [klass, words_sparse, words, jpgs, enc, modes] = pickle.load(a)


class image_word_dataset(data.Dataset):
  def __init__(self, jpeg, words, words_sparse, path):
        self. words = words
        self.im_path = path
        self.jpeg = jpeg
        self.words_sparse = words_sparse

  def __len__(self):
        return len(self.labels)

  def __getitem__(self, index):
        word_indexed = torch.from_numpy(self.words_sparse[index].todense())
        im_path = self.jpeg[index]
        im = torch.tensor(cv2.imread(self.im_path+self.jpeg[index][:-4]+"_"+self.words[index] + ".jpg")).view(3,128,256)
        return torch.div(im.float(),255), word_indexed.float(), im_path

class Model(nn.Module):
    def __init__(self, classes):
        super(Model, self).__init__()
        self.i_conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, padding=3)
        self.i_pool1 = nn.MaxPool2d(2)
        self.i_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, padding=3)
        self.i_pool2 = nn.MaxPool2d(2)
        self.i_conv3 = nn.Conv2d(in_channels=16, out_channels= 32, kernel_size=7, padding=3)
        self.i_pool3 = nn.MaxPool2d(2)
        self.i_linear = nn.Linear(32*32*16, 512)

        self.t_conv1 = nn.Conv1d(in_channels=62, out_channels=32, kernel_size=5, padding=2)
        self.t_pool1 = nn.MaxPool1d(kernel_size=2)
        self.t_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.t_linear = nn.Linear(64*6,16)

        self.c_linear1 = nn.Linear(512+16, 512)
        self.c_dropout1= nn.Dropout(p = 0.4)
        self.c_linear2 = nn.Linear(512, 1024)
        self.c_dropout2= nn .Dropout(p = 0.4)
        self.c_linear3 = nn.Linear(1024, 128)
        self.c_linear4 = nn.Linear(128,classes)


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
        im = im.view(-1, 32*16*32)
        im = self.i_linear(im)

        tx = self.t_conv1(tx)
        tx = self.t_pool1(tx)
        tx = F.relu(tx)
        tx = self.t_conv2(tx)
        tx = tx.view(-1, 64*6)
        tx = self.t_linear(tx)

        c = torch.cat((im,tx), 1)
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


dummy = list(set(flatten(jpg_dict_train.values())))

words_set = [x for x in dummy if len(list(x))<12]

enc_dict = word2encdict(enc= enc, wordsarray=words_set, length=12, lowercase=False)
del words_set
del dummy

dataset = image_word_dataset(jpgs, words, words_sparse, klass, 'Dataset_processing/jpeg_patch/')
train_loader = DataLoader(dataset, batch_size=512, shuffle = False)

lib = []

model = Model(19408)
model.load_state_dict(torch.load("Dataset_processing/logs/36_checkpoint_dict.pt", map_location = 'cpu'))
activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook
model.c_linear3.register_forward_hook(get_activation('c_linear3'))
# id = 0

for batch_idx, data in enumerate(train_loader):
    im, word, jpg = data
    model(im.to(device), word.to(device))
    if len(lib)==0:
        lib = F.relu(activation['c_linear3']).numpy()
        JPGS = np.array(jpg)
    lib = np.concatenate((lib, F.relu(activation['c_linear3']).numpy()), axis = 0)
    JPGS = np.append(JPGS, np.array(jpg))
    print(batch_idx)

with open('lib/36LIB'+str(1).zfill(2) +'.pickle','wb') as q:
    pickle.dump([lib, JPGS], q)

indice_dict = {}
# for i in set(jpgs):
#     indice_dict[i] = []

for i in jpg_dict_train.keys():
    indice_dict[i] = np.argwhere(JPGS == i)

with open('lib/36lib_processed'+str(1).zfill(2) +'.pickle','wb') as q:
    pickle.dump([lib, JPGS, indice_dict], q)