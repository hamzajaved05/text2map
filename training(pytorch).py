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
# from word_encoding import getklass
import pickle
import torch
from torch.utils import data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

reload = False
if reload:
    enc_dict, klasses, wordarray = getklass()
    with open("plogs/pytorch_data_00", 'wb') as f:
        pickle.dump([enc_dict, klasses, wordarray], f)
else:
    with open("pytorch_data_00", 'rb') as f:
        [enc_dict, klasses, wordarray] = pickle.load(f)
klasses2= [0]
index = 0
for i in range(1,len(klasses)):
    if not klasses[i-1] == klasses[i]:
        index+=1
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
    def __init__(self,classes):
        super(ParkhiNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels= 32,kernel_size= 3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size = 2)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels= 64,kernel_size= 5, padding=2)
#         self.pool2 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(32*30*5, 100)
        self.fc2 = nn.Linear(100, 15)
        self.fc3 = nn.Linear(15,classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32*30*5)
        # print(x.size())
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

x = [enc_dict[word] for word in enc_dict.keys()]
training_set = Dataset(x,klasses2)
no_classes = max(klasses2)+1
Network = ParkhiNet(no_classes).double().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Network.parameters(), lr=0.001)
train_loader = DataLoader(training_set,batch_size=10000,shuffle = False)
epochs = 2500
train_loss = []
train_accuracy = []

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    batch_acc = []
    batch_loss = []
    for batch_idx, data in enumerate(train_loader):

        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = Network(inputs.unsqueeze(1).to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        pred = outputs.argmax(dim=1, keepdim=True)

        correct = pred.eq(labels.to(device).long().view_as(pred)).sum().item()
        batch_acc.append(correct)
        train_loss.append(loss.item())
        batch_loss.append(loss.item())
        # print statistics
    print("Dataset accuracy >> " +str(sum(batch_acc)) + ", Epoch "+ str(epoch)+ ", loss > " +str(sum(batch_loss)))
    train_accuracy.append(sum(batch_acc))
print('Finished Training')

with open("log01.pickle","wb") as F:
  pickle.dump([train_loss,train_accuracy],F)
torch.save(Network.state_dict(), "testing2_dict01.pt")
torch.save(Network, "testing2_complete01.pt")