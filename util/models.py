import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelC(nn.Module):
    def __init__(self, classes, embedding, do, em_do):
        super(ModelC, self).__init__()
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
        self.c_dropout1 = nn.Dropout(p=do)
        self.c_linear2 = nn.Linear(512, 1024)
        self.c_dropout2 = nn.Dropout(p=do)
        self.c_linear3 = nn.Linear(1024, embedding)
        self.c_dropout3 = nn.Dropout(p=em_do)
        self.c_linear4 = nn.Linear(embedding, classes)

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


class ModelT(nn.Module):
    def __init__(self, classes, embedding, do, em_do):
        super(ModelT, self).__init__()
        self.t_conv1 = nn.Conv1d(in_channels=62, out_channels=32, kernel_size=5, padding=2)
        self.t_pool1 = nn.MaxPool1d(kernel_size=2)
        self.t_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.t_linear = nn.Linear(64 * 6, 16)

        self.c_linear1 = nn.Linear(16, 512)
        self.c_dropout1 = nn.Dropout(p=do)
        self.c_linear2 = nn.Linear(512, 1024)
        self.c_dropout2 = nn.Dropout(p=do)
        self.c_linear3 = nn.Linear(1024, embedding)
        self.c_dropout3 = nn.Dropout(p=em_do)
        self.c_linear4 = nn.Linear(embedding, classes)

    def forward(self, im, tx):
        tx = self.t_conv1(tx)
        tx = self.t_pool1(tx)
        tx = F.relu(tx)
        tx = self.t_conv2(tx)
        tx = tx.view(-1, 64 * 6)
        tx = self.t_linear(tx)

        # c = torch.cat((im,tx), 1)
        c = self.c_linear1(tx)
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


class ModelI(nn.Module):
    def __init__(self, classes, embedding, do, em_do):
        super(ModelI, self).__init__()
        self.i_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, padding=3)
        self.i_pool1 = nn.MaxPool2d(2)
        self.i_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.i_pool2 = nn.MaxPool2d(4)
        self.i_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.i_pool3 = nn.MaxPool2d(2)
        self.i_linear = nn.Linear(64 * 16 * 8, 512)

        self.c_linear1 = nn.Linear(512, 512)
        self.c_dropout1 = nn.Dropout(p=do)
        self.c_linear2 = nn.Linear(512, 1024)
        self.c_dropout2 = nn.Dropout(p=do)
        self.c_linear3 = nn.Linear(1024, embedding)
        self.c_dropout3 = nn.Dropout(p=em_do)
        self.c_linear4 = nn.Linear(embedding, classes)

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

        c = self.c_linear1(im)
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
