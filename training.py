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
import pickle
import torch
from torch.utils import data
import time
import cv2
import argparse


parser = argparse.ArgumentParser(description='Text to map - Training with image patches and text')
parser.add_argument("patch_path",type = str, help = "Path for Image patches")
parser.add_argument("output_path",type = str, help = "path of outout dicts")
parser.add_argument("pickle_path", type = str, help = "Path of pickle file")
parser.add_argument("epochs", type = int, help = "no of epochs")
parser.add_argument("batch_size", type = int, help = "batch_size")
args= parser.parse_args()

with open(args.pickle_path, "rb") as a:
    [klass, words_sparse, words, jpgs, enc, modes] = pickle.load(a)
imageshape = [128,256]

print("\n\nData Loaded\n\n")

class image_word_dataset(data.Dataset):
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
        im = torch.tensor(cv2.imread(self.im_path+jpgs[index][:-4]+"_"+words[index] + ".jpg")).view(3,128,256)
        return im.float(), word_indexed.float(), y.float()

class Model(nn.Module):
    def __init__(self, classes):
        super(Model, self).__init__()
        self.i_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3)
        self.i_pool1 = nn.MaxPool2d(4)
        self.i_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.i_pool2 = nn.MaxPool2d(4)
        # self.i_conv3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=7, padding=3)
        # self.i_pool3 = nn.MaxPool2d(4)
        self.i_linear = nn.Linear(32*16*8, 128)

        self.t_conv1 = nn.Conv1d(in_channels=62, out_channels=16, kernel_size=7, padding=3)
        self.t_pool1 = nn.MaxPool1d(kernel_size=2)
        self.t_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.t_linear = nn.Linear(32*6,16)

        self.c_linear1 = nn.Linear(128+16, classes)
        # self.c_linear2 = nn.Linear(256,classes)


    def forward(self, im, tx):
        im = self.i_conv1(im)
        im = self.i_pool1(im)
        im = F.relu(im)
        im = self.i_conv2(im)
        im = self.i_pool2(im)
        im = F.relu(im)
        # im = self.i_conv3(im)
        # im = self.i_pool3(im)
        # im = F.relu(im)
        im = im.view(-1, 32*16*8)
        im = self.i_linear(im)

        tx = self.t_conv1(tx)
        tx = self.t_pool1(tx)
        tx = F.relu(tx)
        tx = self.t_conv2(tx)
        tx = tx.view(-1, 32*6)
        tx = self.t_linear(tx)

        c = torch.cat((im,tx), 1)
        c = self.c_linear1(c)
        # c = F.relu(c)
        # c = self.c_linear2(c)

        return F.log_softmax(c, dim=1)

training_set = image_word_dataset(jpgs, words, words_sparse, klass, args.patch_path)
no_classes = len(modes)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(training_set,batch_size=args.batch_size,shuffle = True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Network = Model(no_classes)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  Network = nn.DataParallel(Network)

Network.to(device)
optimizer = optim.Adam(Network.parameters(), lr=0.01, weight_decay=1e-5)
epochs = args.epochs

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
        im, inputs, labels = data
        # print(batch_idx)
        optimizer.zero_grad()
        outputs = Network(im.to(device), inputs.to(device))
        loss = criterion(outputs, labels.long().to(device))
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.to(device).long().view_as(pred)).sum().item()
        batch_acc.append(correct)
        batch_loss.append(loss.item())

    print("Dataset accuracy >> " +str(sum(batch_acc)) + ", Epoch "+ str(epoch)+ ", loss > " +str(sum(batch_loss)))
    train_accuracy.append(sum(batch_acc))
    train_loss.append(sum(batch_loss))
print('Finished Training')

with open(args.output_path + "00trains","wb") as F:
  pickle.dump([train_loss,train_accuracy],F)
torch.save(Network.state_dict(), "00testingdict.pt")
torch.save(Network, "00testingcomplete.pt")
end = time.time()

