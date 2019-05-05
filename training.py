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
import cv2
import argparse
from math import ceil
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Text to map - Training with image patches and text')
parser.add_argument("--impath",type = str, help = "Path for Image patches")
parser.add_argument("--inpickle", type = str, help = "Path of pickle file")
parser.add_argument("--epoch", type = int, help = "no of epochs")
parser.add_argument("--batch", type = int, help = "batch_size")
parser.add_argument("--lr", default = 0.001, type = float, help = "learning rate")
parser.add_argument("--logid",type = str, help = "logid")
parser.add_argument("--write", default=True,type = bool, help = "Write on tensorboard")
parser.add_argument("--limit", default=-1, type = int, help = "Limit dataset")
parser.add_argument("--ratio", default = 0.8, type = float, help= "Ratio of train to complete dataset")
args= parser.parse_args()

with open(args.inpickle, "rb") as a:
    [klass, words_sparse, words, jpgs, enc, modes] = pickle.load(a)
klass = klass[:args.limit]
words_sparse = words_sparse[:args.limit]
words = words[:args.limit]
jpgs = jpgs[:args.limit]
modes = modes[:args.limit]

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
        self.c_dropout = nn.Dropout(p = 0.2)
        self.c_linear2 = nn.Linear(512, 128)
        self.c_linear3 = nn.Linear(128,classes)


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
        c = self.c_dropout(c)
        c = self.c_linear2(c)
        c = F.relu(c)
        c = self.c_dropout(c)
        c = self.c_linear3(c)
        return c



com_dataset = image_word_dataset(jpgs, words, words_sparse, klass, args.impath)
train_size = args.ratio
no_classes = klass[-1]+1
data_size = len(klass)
train_dataset, val_dataset = data.random_split(com_dataset, [int(data_size*(train_size)), data_size-int(data_size*(train_size))])

if args.write:
    Writer = SummaryWriter("TBX/"+args.logid)
    Writer.add_scalars("Metadata", {"Batch_size": args.batch,
                                    "learning_rate": args.lr,
                                    "logid": int(args.logid),
                                    "training_size":train_dataset.__len__,
                                    "Validation_size": val_dataset.__len__,
                                    "No_of_classes": no_classes})

criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Network = Model(no_classes)

Network.to(device)
optimizer = optim.Adam(Network.parameters(), lr=args.lr)
epochs = args.epoch

train_accuracy = []
train_loss = []
validation_accuracy = []
validation_loss = []

# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook
# Network.fc2.register_forward_hook(get_activation('fc2'))

batches = ceil(len(klass)/args.batch)

for epoch in range(1, epochs + 1):

    training_batch_acc = []
    training_batch_loss = []
    validation_batch_acc = []
    validation_batch_loss = []

    Network.train()
    for batch_idx, data in enumerate(train_loader):
        im, inputs, labels = data
        optimizer.zero_grad()
        outputs = Network(im.to(device), inputs.to(device))
        loss = criterion(outputs, labels.long().to(device))
        loss.backward()
        optimizer.step()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.to(device).long().view_as(pred)).sum().item()
        training_batch_acc.append(correct)
        training_batch_loss.append(loss.item())
        Writer.add_scalars("Batch_Training_log", {"Epoch_acc":(correct)/labels.size()[0],
                                                  "Epoch_loss":loss.item()/labels.size()[0]},
                           (epoch-1)*batches+batch_idx)

    Network.eval()
    for batch_idx, data in enumerate(val_loader):
        im, inputs, labels = data
        outputs = Network(im.to(device), inputs.to(device))
        loss = criterion(outputs, labels.long().to(device))
        pred = outputs.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.to(device).long().view_as(pred)).sum().item()
        training_batch_acc.append(correct)
        validation_batch_loss.append(loss.item())
        Writer.add_scalars("Batch_Val_log", {"Epoch_acc":correct/labels.size()[0],
                                             "Epoch_loss":loss.item()/labels.size()[0]},
                           (epoch-1)*batches+batch_idx)

    # print("Dataset accuracy >> " + str(sum(training_batch_acc)) + ", Epoch " + str(epoch) + ", loss > " + str(sum(training_batch_loss) / len(klass)))
    train_accuracy.append(sum(training_batch_acc))
    train_loss.append(sum(training_batch_loss))
    validation_accuracy.append(sum(validation_batch_acc))
    validation_loss.append(sum(validation_batch_loss))

    if args.write:
        for name, param in Network.named_parameters():
            Writer.add_histogram(name, param.clone().cpu().data.numpy(), epochs * batches + batch_idx)
        Writer.add_scalars("Training_log", {"Epoch_acc": sum(training_batch_acc) / train_dataset.__len__,
                                            "Epoch_loss": sum(training_batch_loss) / train_dataset.__len__,
                                            "lr": optimizer.param_groups[0]["lr"],
                                            "Epoch_val_acc" : sum(validation_batch_acc)/val_dataset.__len__,
                                            "Epoch_val_loss": sum(validation_batch_loss)/val_dataset.__len__},
                           epoch)

Writer.close()
with open("logs/"+args.logid+"_data.pickle","wb") as F:
  pickle.dump([train_loss, train_accuracy, train_loss, validation_loss, validation_loss, args.batch, args.lr, args.epoch], F)
torch.save(Network.state_dict(), "logs/"+args.logid+"_dict.pt")
torch.save(Network, "logs/"+args.logid+"_dict_c.pt")
