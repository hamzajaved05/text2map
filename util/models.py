import torch
import torch.nn as nn
import torch.nn.functional as F


class p_embed_net(nn.Module):
    def __init__(self, embedding, do):
        super(p_embed_net, self).__init__()
        self.i_seq = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
                                   nn.MaxPool2d(2),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
                                   nn.MaxPool2d(2),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                   nn.MaxPool2d(2),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                   nn.MaxPool2d(2),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                   nn.MaxPool2d(2),
                                   nn.ReLU()
        )
        self.i_linear = nn.Linear(128 * 4 * 8, 512)

        self.t_seq = nn.Sequential(nn.Conv1d(in_channels=62, out_channels=32, kernel_size=5, padding=2),
                                   nn.MaxPool1d(2),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
        self.t_linear = nn.Linear(64 * 6, 16)


        self.c_seq = nn.Sequential(nn.Linear(512+16, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(p = do),
                                   nn.Linear(1024, 512),
                                   nn.ReLU(),
                                   nn.Dropout(p = do),
                                   nn.Linear(512, embedding))

    def forward(self, im, tx):
        im = self.i_seq(im)
        im = im.view(-1, 128 * 8  * 4)
        im = self.i_linear(im)

        tx = self.t_seq(tx)
        tx = tx.view(-1, 64 * 6)
        tx = self.t_linear(tx)

        c = torch.cat((im, tx), 1)
        c = self.c_seq(c)

        return F.normalize(c, p=2, dim=1)


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


class Embedding_net(nn.Module):
    def __init__(self, p_embed = 512, v_embed = 4096, c_embed = 5120):
        super(Embedding_net, self).__init__()
        self.p1 = nn.Sequential(nn.Linear(1280, p_embed))
        self.v1 = nn.Sequential(nn.Linear(4096, v_embed))
        self.c1 = nn.Sequential(nn.Linear(p_embed+v_embed, c_embed))

    def forward(self, pa, vl):
        patch = pa
        patch = self.p1(patch)
        vlad = self.v1(vl)
        com = torch.cat([vlad, patch], dim = 1)
        com = self.c1(com)
        return F.normalize(com, p=2, dim =1)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net.float()

    def forward(self, array):
        output1 = self.embedding_net(array[0], array[1])
        output2 = self.embedding_net(array[2], array[3])
        output3 = self.embedding_net(array[4], array[5])
        return output1, output2, output3

    def get_embedding(self, x, y):
        return self.embedding_net(x.float(), y.float())

class TripletLoss(nn.Module):
    def __init__(self, margin = 0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean(), distance_positive.mean(), distance_negative.mean() if size_average else losses.sum()