"""
Author: Hamza
Dated: 20.04.2019
Project: texttomap

"""

import argparse
import pickle
import torch.optim as optim
from math import ceil
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from util.early_stopping import EarlyStopping
from util.loaders import *
from util.models import *

parser = argparse.ArgumentParser(description='Text to map - Training with image patches and text')
parser.add_argument("--impath", type=str, help="Path for Image patches")
parser.add_argument("--inpickle", type=str, help="Path of pickle file")
parser.add_argument("--epoch", type=int, help="no of epochs")
parser.add_argument("--batch", type=int, help="batch_size")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--logid", type=str, help="logid")
parser.add_argument("--write", default=True, type=bool, help="Write on tensorboard")
parser.add_argument("--limit", default=-1, type=int, help="Limit dataset")
parser.add_argument("--ratio", default=0.8, type=float, help="Ratio of train to complete dataset")
parser.add_argument("--earlystopping", default=True, type=bool, help="Enable or disable early stopping")
parser.add_argument("--decay_freq", default=None, type=int, help="Decay by half after number of epochs")
parser.add_argument("--embed_size", default=256, type=int, help="Size of embedding")
parser.add_argument("--model", default="ti", type=str, help="Model type")
parser.add_argument("--embed_dropout", default=0.1, type=float, help="Dropout on embedding")
parser.add_argument("--dropout", default=0.4, type=float, help="Dropout before")

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open(args.inpickle, "rb") as a:
    [klass, words_sparse, words, jpgs, enc, modes] = pickle.load(a)

def preprocess_klass(klass):
    x = -1
    arr = []
    klass2 = []
    for i in klass:
        if i not in arr:
            x+=1
            arr.append(i)
        klass2.append(x)
    return klass2
klass = preprocess_klass(klass)


if not args.limit == -1:
    klass = klass[:args.limit]
    words_sparse = words_sparse[:args.limit]
    words = words[:args.limit]
    jpgs = jpgs[:args.limit]
    modes = modes[:args.limit]

train_size = args.ratio
no_classes = klass[-1] + 1
data_size = len(klass)


activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

criterion = nn.CrossEntropyLoss()


if args.model == 'ti':
    Model = ModelC
elif args.model == 't':
    Model = ModelT
elif args.model == 'i':
    Model = ModelI
else:
    raise ("UnIdentified Model specified")
Network = Model(no_classes, embedding=args.embed_size, do=args.dropout, em_do=args.embed_dropout)
complete_dataset = image_word_training_loader(jpgs, words, words_sparse, klass, args.impath)
train_dataset, val_dataset = data.random_split(complete_dataset, [int(data_size * (train_size)),
                                                                  data_size - int(data_size * (train_size))])
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)


if args.write:
    Writer = SummaryWriter("tbx/" + args.logid + args.model)
    Writer.add_scalars("Metadata_" + args.model, {"Batch_size": args.batch,
                                                  "learning_rate": args.lr,
                                                  "logid": int(args.logid),
                                                  "training_size": train_dataset.__len__(),
                                                  "Validation_size": val_dataset.__len__(),
                                                  "No_of_classes": no_classes,
                                                  "embed_dropout": args.embed_dropout,
                                                  "dropout": args.dropout,
                                                  "embed_size": args.embed_size})

Network.to(device)
optimizer = optim.Adam(Network.parameters(), lr=args.lr)
epochs = args.epoch
Network.c_dropout3.register_forward_hook(get_activation('embedding'))
train_accuracy = []
train_loss = []
validation_accuracy = []
validation_loss = []

early_stop = EarlyStopping(patience=100, verbose=False, name=args.logid, path='logs/')

batches = ceil(len(klass) / args.batch)

epoch_metric = {"Training_loss": [], "Training_acc": [], "Validation_loss": [], "Validation_acc": []}


def log_metric(dict, ta, tl, va, vl):
    dict["Training_loss"].append(tl)
    dict["Training_acc"].append(ta)
    dict["Validation_loss"].append(vl)
    dict["Validation_acc"].append(va)
    return dict


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

    Network.eval()
    for batch_idx, data in enumerate(val_loader):
        im, inputs, labels = data
        outputs = Network(im.to(device), inputs.to(device))
        loss = criterion(outputs, labels.long().to(device))
        pred = outputs.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.to(device).long().view_as(pred)).sum().item()
        validation_batch_acc.append(correct)
        validation_batch_loss.append(loss.item())

    Writer.add_scalars("Training_log", {"Epoch_acc": sum(training_batch_acc)*100 / train_dataset.__len__(),
                                        "Epoch_loss": sum(training_batch_loss) / train_dataset.__len__(),
                                        "Epoch_val_acc": sum(validation_batch_acc)*100 / val_dataset.__len__(),
                                        "Epoch_val_loss": sum(validation_batch_loss) / val_dataset.__len__(),
                                        "lr": optimizer.param_groups[0]["lr"]},
                       epoch)
    epoch_metric = log_metric(epoch_metric,
                              sum(training_batch_acc)/ train_dataset.__len__(),
                              sum(training_batch_loss)/ train_dataset.__len__(),
                              sum(validation_batch_acc)/ val_dataset.__len__(),
                              sum(validation_batch_loss)/val_dataset.__len__())

    with open('logs/' + args.logid + 'logfile.pickle', 'wb') as q:
        pickle.dump([epoch_metric, args], q)

    if args.earlystopping:
        early_stop(sum(validation_batch_loss), Network)

    if args.decay_freq is not None:
        if epoch % args.decay_freq == 0:
            for g in optimizer.param_groups:
                g['lr'] = args.lr / 2 ** (epoch // args.decay_freq)


Writer.close()
torch.save(Network.state_dict(), "logs/" + args.logid + "_dict.pt")
torch.save(Network, "logs/" + args.logid + "_dict_c.pt")
