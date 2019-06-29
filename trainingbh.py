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
import pandas as pd
from torch.utils.data import DataLoader
from util.loaders import *
from util.models import *
import os
from sklearn.model_selection import train_test_split
import logging
from util.valid_training import validation
import random

parser = argparse.ArgumentParser(description='Text to map - Training with image patches and text')
parser.add_argument("--impath", type=str, help="Path for Image patches")
parser.add_argument("--inpickle", type=str, help="Path of pickle file")
parser.add_argument("--epoch", type=int, help="no of epochs")
parser.add_argument("--batch", type=int, help="batch_size")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--logid", type=str, help="logid")
parser.add_argument("--write", default=True, type=bool, help="Write on tensorboard")
parser.add_argument("--limit", default=-1, type=int, help="Limit dataset")
parser.add_argument("--decay_freq", default=None, type=int, help="Decay freq")
parser.add_argument("--embed_size", default=256, type=int, help="Size of embedding")
parser.add_argument("--model", default="ti", type=str, help="Model type")
parser.add_argument("--dropout", default=0.4, type=float, help="Dropout before")
parser.add_argument("--decay_value", default = 0.95, type = float, help = "decay by value")
parser.add_argument("--margin", default = 0.1, type = float, help = "decay by value")
parser.add_argument("--save_embeds", default = True, type = bool, help = "decay by value")
parser.add_argument("--maxperclass", default = 30, type = int, help="maximum items per class")
parser.add_argument("--itemsperclass", default = 10, type = int, help="maximum items per class")
parser.add_argument("--soft_positive", default = True, type = bool, help = "Soft positive mining")
parser.add_argument("--l2loss", default = True, type = bool, help = "l2 distance between files")
parser.add_argument("--softplus", default = False, type = bool, help = "softplus loss")
parser.add_argument("--load", default = None, type = str, help = "load file")



args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open(args.inpickle, "rb") as a:
    [klass, words_sparse, words, jpgs, enc, modes] = pickle.load(a)


if not args.limit == -1:
    klass = klass[:args.limit]
    words_sparse = words_sparse[:args.limit]
    words = words[:args.limit]
    jpgs = jpgs[:args.limit]
    modes = modes[:args.limit]
    print("Limited inputs of length {} with total classes {}".format(len(klass), klass[-1]))
else:
    print("Original inputs of length {} with total classes {}".format(len(klass), klass[-1]))


def seq_klass(klass):
    x = -1
    arr = []
    klass2 = []
    for i in klass:
        if i not in arr:
            x+=1
            arr.append(i)
        klass2.append(x)
    return klass2

print("Processing inputs")

def limitklass(klas, word, word_sparse, jpg):
    klas = np.array(klas)
    klass2, klass2v = [], []
    word2, word2v = [], []
    word_sparse2, word_sparse2v = [] ,[]
    jpgs2, jpgs2v = [], []
    label = -1
    for j in list(set(klas)):
        x = np.sum(np.array(klas)==j)
        if x >= args.maxperclass:
            label+=1
            indices = random.sample(list(np.argwhere(np.array(klas)==j).squeeze()), args.maxperclass)
            v_indices = list(set(np.argwhere(np.asarray(klass)==j).squeeze()).difference(set(indices)))
            if len(v_indices) >args.itemsperclass:
                v_indices = random.sample(v_indices, args.itemsperclass)
            for i in v_indices:
                klass2v.append(label)
                word2v.append(word[i])
                word_sparse2v.append(word_sparse[i])
                jpgs2v.append(jpg[i])
        elif x >= args.itemsperclass:
            label+=1
            indices = random.sample(list(np.argwhere(np.array(klas)==j).squeeze()), args.itemsperclass)
            v_indices = list(set(np.argwhere(np.asarray(klass)==j).squeeze()).difference(set(indices)))
            if len(v_indices) >args.itemsperclass:
                v_indices = random.sample(v_indices, args.itemsperclass)
            for i in v_indices:
                klass2v.append(label)
                word2v.append(word[i])
                word_sparse2v.append(word_sparse[i])
                jpgs2v.append(jpg[i])
        else:
            continue
        for i in indices:
            klass2.append(label)
            word2.append(word[i])
            word_sparse2.append(word_sparse[i])
            jpgs2.append(jpg[i])
    return klass2, word2, word_sparse2, jpgs2, klass2v, word2v, word_sparse2v, jpgs2v

klass, words, words_sparse, jpgs, klass_v, words_v, words_sparse_v, jpgs_v= limitklass(klass, words, words_sparse, jpgs)
# klass = seq_klass(klass)


print("Processed inputs of length {} with total classes {}".format(len(klass), klass[-1]))

#
# klass, valid_klass, words, valid_words, words_sparse, valid_sparse, jpgs, valid_jpgs = train_test_split(klass, words, words_sparse, jpgs, test_size=0.025)
print("length of train is {}, test is {}".format(len(klass), len(klass_v)))

if args.model == 'ti':
    Model = p_embed_net
elif args.model == 't':
    Model = ModelT
elif args.model == 'i':
    Model = ModelI
else:
    raise ("UnIdentified Model specified")

if args.load is None:
    Inter = Model(embedding= args.embed_size, do = args.dropout).float().to(device)
    Network = TripletNet(Inter).float().to(device)
    print("Created model")
else:
    Network = torch.load(args.load)
    print("Loading model")

criterion = TripletLoss(margin= args.margin, l2= args.l2loss, softplus= args.softplus).to(device)

logging.basicConfig(filename='models_bh/' + args.logid + args.model + '.log', filemode='w', format='%(message)s')
logger = logging.getLogger('dummy')
logger.addHandler(logging.FileHandler("testing.log"))

complete_dataset = image_word_triplet_loader_batchhard(jpgs, words, words_sparse, klass, args.impath, args.itemsperclass)

train_loader = DataLoader(complete_dataset, batch_size=args.batch, shuffle=True)

print("Dataloaders done")
if args.write:
    Writer = SummaryWriter("models_bh/tbx/" + args.logid + args.model)
    Writer.add_scalars("Metadata" + args.model, {"Batch_size": args.batch,
                                                 "learning_rate": args.lr,
                                                 "items per class": args.itemsperclass,
                                                 "max per class": args.maxperclass,
                                                 "No_of_classes": klass[-1],
                                                 "dropout": args.dropout,
                                                 "embed_size": args.embed_size,
                                                 })

valid_class = validation(klass_v, words_v, words_sparse_v, jpgs_v, args.impath, Network, Writer)

optimizer = optim.Adam(Network.parameters(), lr=args.lr)
epochs = args.epoch
train_accuracy = []
train_loss = []
validation_accuracy = []
validation_loss = []

# early_stop = EarlyStopping(patience=100, verbose=False, name=args.logid, path='model_saves')

batches = ceil(len(klass) / args.batch)
trainingcounter = 0

for epoch in range(1, epochs + 1):
    Network.train()


    for batch_idx, data in enumerate(train_loader):
        im, patch, label = data
        im = im.to(device)
        patch = patch.to(device)
        embeds = Network.get_embedding(im.reshape([-1,3,128,256]), patch.reshape([-1,62,12])).cpu().detach().numpy()
        optimizer.zero_grad()
        loss = 0
        pdis_all = []
        ndis_all = []
        pind_all = []
        nind_all = []
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                positive = list(range(10*(i),10*(i)+10))
                negative = list(range(embeds.shape[0]))
                negative = list(set(negative).difference(set(positive)))
                # positive = embeds[10*(i):10*(i)+10]
                anchor_im = im[i,j,...]
                anchor_patch = patch[i,j,...]
                sorted_indices = np.argsort(np.linalg.norm(embeds[10*(i)+j] - embeds, ord = 2, axis=1))
                for n_ind, q in enumerate(sorted_indices):
                    if q not in positive:
                        negative_indice = q
                        nind_all.append(n_ind)
                        break
                for p_ind, q in enumerate(sorted_indices[::-1]):
                    if q in positive:
                        positive_indice = q
                        pind_all.append(p_ind)
                        break
                n_i = list(np.unravel_index(negative_indice, [args.batch, args.itemsperclass]))
                p_i = list(np.unravel_index(positive_indice, [args.batch, args.itemsperclass]))
                a_embed, p_embed, n_embed = Network([im[i,j,...].unsqueeze(0), patch[i,j,...].unsqueeze(0),
                                                     im[p_i[0],p_i[1],...].unsqueeze(0), patch[p_i[0], p_i[1],...].unsqueeze(0),
                                                     im[n_i[0],n_i[1],...].unsqueeze(0), patch[n_i[0], n_i[1],...].unsqueeze(0)])
                loss1, p1, n1 = criterion(a_embed.to(device), p_embed.to(device), n_embed.to(device))
                loss += loss1
                pdis_all.append(p1.item())
                ndis_all.append(n1.item())



        loss = loss/(im.shape[0]*im.shape[1] )
        loss.backward()
        optimizer.step()
        print("epoch {}, batch {}/{}, train_loss {:.5f}, p_dis_mean {:.5f}, n_dis_mean {:.5f}".format(epoch,
                                                                                                       batch_idx,
                                                                                                       train_loader.__len__(),
                                                                                                       loss.item(),
                                                                                                       sum(pdis_all) / len(pdis_all),
                                                                                                       sum(ndis_all) / len(ndis_all)))
        label = np.asarray([[i] * args.itemsperclass for i in label.tolist()]).reshape(-1)
        try:
            embedings_total = np.concatenate((embedings_total, embeds), axis=0)
            labels_total = np.concatenate((labels_total, label), axis=0)
        except:
             embedings_total = embeds
             labels_total = label


        Writer.add_scalars("Training_data_batch_hard",{"batch_loss": loss.item(),
                                            "batch_pdis": sum(pdis_all) / len(pdis_all),
                                            "batch_ndis": sum(ndis_all) / len(ndis_all)
                                            }, trainingcounter)

        Writer.add_scalars("Indices", {"average_n_location": sum(nind_all)/len(nind_all),
                                       "average_p_location": sum(pind_all)/len(pind_all)},
                           trainingcounter)
        trainingcounter+=1



    ps, ns = valid_class.evaluate(embedings_total,labels_total, device, size = 1000, margin = args.margin)

    if args.save_embeds:
        with open("models_bh/"+args.logid+"embeds.pickle", "wb") as q:
            pickle.dump([embedings_total, labels_total], q)

    torch.save(Network.state_dict(), "models_bh/"+args.logid+"timodeldict.pt")
    torch.save(Network, "models_bh/"+args.logid+"timodelcom.pt")



    if args.decay_freq is not None:
        if epoch % args.decay_freq == 0:
            for g in optimizer.param_groups:
                g['lr'] = g["lr"] * args.decay_value

    del embedings_total
    del labels_total