"""
Author: Hamza
Dated: 20.04.2019
Project: texttomap

"""

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from util.word_encoding import getklass
from util.utilities import encoding, word2encdict
import pickle
import torch
from torch.utils import data
from util.utilities import getwords
from PIL import Image
import torchvision.transforms.functional as TF
import time
import cv2

with open("Dataset_processing/split/training_data_pytorch03.pickle", "rb") as a:
    [klass, words_sparse, words, jpgs, enc, modes] = pickle.load(a)
jpegsdir = "Dataset_processing/jpeg_patch/"
imageshape = []
# ///////////////////////////////////////

print("\n\nData Loaded\n\n")

class Dataset(data.Dataset):
  def __init__(self, jpeg, words, words_sparse, labels, path):
        self.labels = labels
        self. words = words
        self.im_path = path
        self.jpeg = jpeg
        self.words_sparse = words_sparse

  def __len__(self):
        return len(self.labels)

  def __getitem__(self, index):
        word_indexed = torch.from_numpy(self.words_sparse[index].todense())
        y = torch.tensor(self.labels[index])
        im = torch.tensor(cv2.imread(self.im_path+jpgs[index][:-4]+"_"+words[index] + ".jpg"))
        return word_indexed, y, im

class ParkhiNet(nn.Module):
    def __init__(self,text_shape, image_shape, classes):
        super(ParkhiNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=62, out_channels= 128,kernel_size= 5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size = 2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels= 64,kernel_size= 5, padding=2)
        self.fc1 = nn.Linear(64*6, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8,classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64*6)
        x = (F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

training_set = Dataset(jpgs, words, words_sparse, klass, jpegsdir)
no_classes = len(modes)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(training_set,batch_size=2048,shuffle = True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
Network = ParkhiNet(words_sparse[1].shape,  ,no_classes).double().to(device)
optimizer = optim.Adam(Network.parameters(), lr=0.0005, weight_decay=1e-5)
epochs = 5000

train_accuracy = []
train_loss = []
# activation = {}

# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook


# Network.fc2.register_forward_hook(get_activation('fc2'))
start = time.time()
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    batch_acc = []
    batch_loss = []
    for batch_idx, data in enumerate(train_loader):

        # get the inputs
        inputs, labels, im = data
        print(batch_idx)
        # zero the parameter gradients
        # print(inputs[1])
        # print(labels.size())
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = Network(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        pred = outputs.argmax(dim=1, keepdim=True)

        correct = pred.eq(labels.to(device).long().view_as(pred)).sum().item()
        batch_acc.append(correct)
        batch_loss.append(loss.item())
#         print(activation['fc2'].size())
        # print statistics
    print("Dataset accuracy >> " +str(sum(batch_acc)) + ", Epoch "+ str(epoch)+ ", loss > " +str(sum(batch_loss)))
    train_accuracy.append(sum(batch_acc))
    train_loss.append(sum(batch_loss))
print('Finished Training')

with open("30log.pickle","wb") as F:
  pickle.dump([train_loss,train_accuracy],F)
torch.save(Network.state_dict(), "30testingdict.pt")
torch.save(Network, "30testingcomplete.pt")
end = time.time()


l = klass[0]
s = words[0]
x= []
for i, j in enumerate(klass):
    if l == j:
        x.append(lev.get_raw_score(s,words[i]))
        if lev.get_raw_score(s,words[i])>1:
            print(s,words[i])
    else:
        l = j
        s = words[i]
