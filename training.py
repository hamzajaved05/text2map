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
import os

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
parser.add_argument("--decay_freq", default=None, type=int, help="Decay freq")
parser.add_argument("--embed_size", default=256, type=int, help="Size of embedding")
parser.add_argument("--model", default="ti", type=str, help="Model type")
parser.add_argument("--dropout", default=0.4, type=float, help="Dropout before")
parser.add_argument("--ld", default = 3, type = int, help = "lev distance for negative mining")
parser.add_argument("--decay_value", default = 0.95, type = float, help = "decay by value")
parser.add_argument("--margin", default = 0.1, type = float, help = "decay by value")


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

def filter_klass(klas, word, word_sparse, jpg):
    klas = np.array(klas)
    klass2 = []
    word2 = []
    word_sparse2 = []
    jpgs2 = []
    for itera, i in enumerate(klas):
        if np.sum(i == klas) < 5:
            pass
        else:
            klass2.append(i)
            word2.append(word[itera])
            word_sparse2.append(word_sparse[itera])
            jpgs2.append(jpg[itera])

    return klass2, word2, word_sparse2, jpg
print("Processing inputs")
# klass, words, words_sparse, jpgs = filter_klass(klass, words, words_sparse, jpgs)
klass = preprocess_klass(klass)
print("Processed inputs")
if not args.limit == -1:
    klass = klass[:args.limit]
    words_sparse = words_sparse[:args.limit]
    words = words[:args.limit]
    jpgs = jpgs[:args.limit]
    modes = modes[:args.limit]

train_size = args.ratio
no_classes = klass[-1] + 1
data_size = len(klass)

if args.model == 'ti':
    Model = p_embed_net
elif args.model == 't':
    Model = ModelT
elif args.model == 'i':
    Model = ModelI
else:
    raise ("UnIdentified Model specified")


Inter = Model(embedding= args.embed_size, do = args.dropout).float().to(device)
Network = TripletNet(Inter).float().to(device)

criterion = TripletLoss(margin= args.margin).to(device)

complete_dataset = image_word_triplet_loader(jpgs, words, words_sparse, klass, args.impath, args.ld)
train_dataset, val_dataset = data.random_split(complete_dataset, [int(data_size * (train_size)),
                                                                  data_size - int(data_size * (train_size))])
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)

print("Dataloaders done")
if args.write:
    Writer = SummaryWriter("ti/tbx/" + args.logid + args.model)
    Writer.add_scalars("Metadata" + args.model, {"Batch_size": args.batch,
                                                  "learning_rate": args.lr,
                                                  "logid": int(args.logid),
                                                  "training_size": train_dataset.__len__(),
                                                  "Validation_size": val_dataset.__len__(),
                                                  "No_of_classes": no_classes,
                                                  "lev_distance": args.ld,
                                                  "dropout": args.dropout,
                                                  "embed_size": args.embed_size,
                                                  })


optimizer = optim.Adam(Network.parameters(), lr=args.lr)
epochs = args.epoch
train_accuracy = []
train_loss = []
validation_accuracy = []
validation_loss = []

# early_stop = EarlyStopping(patience=100, verbose=False, name=args.logid, path='model_saves')

batches = ceil(len(klass) / args.batch)

epoch_metric = {"Training_loss": [], "Training_acc": [], "Validation_loss": [], "Validation_acc": []}

logs = {}
logs["training_batch_pdis"] = []
logs["training_batch_ndis"] = []
logs["training_batch_loss"] = []
logs["validation_batch_pdis"] = []
logs["validation_batch_loss"] = []
logs["validation_batch_ndis"] = []
trainingcounter = 0
validationcounter = 0
for epoch in range(1, epochs + 1):
    print("epoch {}".format(epoch))
    Network.train()
    for batch_idx, data in enumerate(train_loader):
        print("batch {}".format(batch_idx))
        ai, ap, aw, pi, pp, pw, ni, np, nw = data

        optimizer.zero_grad()
        ao, po, no = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni.to(device), np.to(device)])
        # print('e')
        loss, p, n = criterion(ao.to(device), po.to(device), no.to(device))
        print(p, n)
        loss.backward()
        print(loss)
        optimizer.step()
        logs["training_batch_pdis"].append(p.item())
        logs["training_batch_ndis"].append(n.item())
        logs["training_batch_loss"].append(loss.item())
        Writer.add_scalars("Training_data",{"batch_loss": loss.item(),
                                            "batch_pdis": p.item(),
                                            "batch_ndis": n.item()
                                            },
                           trainingcounter)
        trainingcounter+=1


    Network.eval()
    for batch_idx, data in enumerate(val_loader):
        print("batch {}".format(batch_idx))
        ai, ap, aw, pi, pp, pw, ni, np, nw = data
        ao, po, no = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni.to(device), np.to(device)])
        loss, p, n = criterion(ao.to(device), po.to(device), no.to(device))
        logs["validation_batch_pdis"].append(p.item())
        logs["validation_batch_loss"].append(loss.item())
        logs["validation_batch_ndis"].append(n.item())
        Writer.add_scalars("Validation_data",{"batch_loss": loss.item(),
                                            "batch_pdis": p.item(),
                                            "batch_ndis": n.item()
                                            },
                           validationcounter)
        validationcounter+=1

    with open('logs/' + args.logid + 'logfile.pickle', 'wb') as q:
        pickle.dump([logs, args], q)

    torch.save(Network.state_dict(), "models/"+args.logid+"timodeldict.pt")
    torch.save(Network, "models/"+args.logid+"timodelcom.pt")

    if args.decay_freq is not None:
        if epoch % args.decay_freq == 0:
            for g in optimizer.param_groups:
                g['lr'] = g["lr"] * args.decay_value

