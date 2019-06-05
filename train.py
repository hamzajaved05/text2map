import argparse

import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from util.loaders import *
from util.models import *

# from collections import namedtuple

parser = argparse.ArgumentParser(description='Combined Runs')
parser.add_argument("--epoch", type=int, default=500, help="no of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--logid", type=str, help="logid")
parser.add_argument("--decay_freq", default=None, type=int, help="Decay by half after number of epochs")
parser.add_argument("--embed_size", default=256, type=int, help="Size of embedding")
parser.add_argument("--dt", default=False, type=bool, help="dynamic_triplets")
parser.add_argument("--path", type=str, help="path to vlad_results")

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open("Dataset_processing/preprocess.pickle", "rb") as q:
    [jpg_word_dict, lib_ids, lib, triplet] = pickle.load(q)

netvladids = pd.read_csv("final/data_68.csv")["imagesource"].to_numpy()
path = args.path
Writer = SummaryWriter("final/tbx/" + args.logid)


def filter1(triplets, ids):
    y = []
    for i in triplets:
        # print(i)
        if i[1] in ids:
            if i[2] in ids:
                if i[3] in ids:
                    y.append(i)
    return y


print("filtering")
triplet = filter1(triplet, netvladids)
print(len(triplet))
dynamic_triplets = args.dt

complete_dataset = Triplet_loader(triplet, jpg_word_dict, lib_ids, lib, path, netvladids, rand=dynamic_triplets)
train_size = 0.8
data_size = complete_dataset.__len__()
train_dataset, val_dataset = data.random_split(complete_dataset, [int(data_size * (train_size)),
                                                                  data_size - int(data_size * (train_size))])

embed_net = Embedding_net(c_embed=args.embed_size).float().to(device)
model = TripletNet(embed_net).float().to(device)
lossfn = TripletLoss().to(device)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

trash = 0
good = 0
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
print(type(data_size * (train_size)))
print(type(data_size))

Writer.add_scalars("Metadata_" + args.logid, {"Batch_size": args.batch_size,
                                              "learning_rate": args.lr,
                                              "Epochs": args.epoch,
                                              # "Decay_freq": args.decay_freq,
                                              "training_size": int(data_size * train_size),
                                              "Validation_size": data_size - int(data_size * train_size),
                                              })

itert = 0
iterv = 0
for i_epoch in range(1, epochs + 1):
    train_loss = []
    train_pdis = []
    train_ndis = []
    val_loss = []
    val_pdis = []
    val_ndis = []

    model.train()
    for id, data in enumerate(train_loader):
        print("here")
        optimizer.zero_grad()
        if data is not None:
            ou1, ou2, ou3 = model([data[0].float().to(device), data[1].float().to(device), data[2].float().to(device),
                                   data[3].float().to(device), data[4].float().to(device), data[5].float().to(device)])
            loss, p_dis, n_dis = lossfn(ou1, ou2, ou3)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_pdis.append(p_dis.item())
            train_ndis.append(n_dis.item())
            print('here')
            Writer.add_scalars("Train_log", {"Train_loss": loss.item(),
                                             "Train_Same_class_dist": p_dis.mean().item(),
                                             "Train_dif_class_dist": n_dis.mean().item(),
                                             },
                               itert)
            itert += 1
    model.eval()
    for id, data in enumerate(valid_loader):
        if data is not None:
            ou1, ou2, ou3 = model([data[0].float().to(device), data[1].float().to(device), data[2].float().to(device),
                                   data[3].float().to(device), data[4].float().to(device), data[5].float().to(device)])
            loss, p_dis, n_dis = lossfn(ou1, ou2, ou3)
            val_loss.append(loss.item())
            val_pdis.append(p_dis.item())
            val_ndis.append(n_dis.item())
            Writer.add_scalars("val_log", {"val_loss": loss.item(),
                                             "val_Same_class_dist": p_dis.mean().item(),
                                             "val_dif_class_dist": n_dis.mean().item(),
                                             },
                               iterv)
            iterv += 1

    Writer.add_scalars("Training_log", {"Train_loss": sum(train_loss) * 100 / len(train_loss),
                                        "Val_loss": sum(val_loss) / len(val_loss),
                                        "Val_Same_class_dist": sum(val_pdis) / len(val_pdis),
                                        "Train_Same_class_dist": sum(train_pdis) / len(train_pdis),
                                        "Val_dif_class_dist": sum(val_ndis) / len(val_ndis),
                                        "Train_dif_class_dist": sum(train_ndis) / len(train_ndis),
                                        "lr": optimizer.param_groups[0]["lr"]
                                        },
                       i_epoch)
    torch.save(model.state_dict(), "final/" + args.logid + "dict.pt")
    torch.save(model, "final/" + args.logid + "modelcom.pt")
