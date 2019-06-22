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
parser.add_argument("--save_embeds", default = True, type = bool, help = "decay by value")
parser.add_argument("--max_perklass", default = 30, type = int, help="maximum items per class")


args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open(args.inpickle, "rb") as a:
    [klass, words_sparse, words, jpgs, enc, modes] = pickle.load(a)

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
    klass2 = []
    word2 = []
    word_sparse2 = []
    jpgs2 = []
    for itera, i in enumerate(klas):
        x = np.sum(np.array(klass2) == i).item()
        if np.sum(klas == i)<10:
            continue
        elif x<=5:
            klass2.append(i)
            word2.append(word[itera])
            word_sparse2.append(word_sparse[itera])
            jpgs2.append(jpg[itera])
    return klass2, word2, word_sparse2, jpgs2


# klass = preprocess_klass(klass)
# x = pd.read_csv("sparse/68_data_sparse_5.csv")["imagesouce"].to_numpy()
# klass, words, words_sparse, jpgs = limitklass2(klass, words, words_sparse, jpgs, x)
klass, words, words_sparse, jpgs = limitklass(klass, words, words_sparse, jpgs)
klass = seq_klass(klass)

print("Processed inputs of length {}".format(len(klass)))


if not args.limit == -1:
    klass = klass[:args.limit]
    words_sparse = words_sparse[:args.limit]
    words = words[:args.limit]
    jpgs = jpgs[:args.limit]
    modes = modes[:args.limit]
    print("Limited to {} length.".format(args.limit))
else:
    print("Using complete dataset of {}".format(len(klass)))

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
# train_dataset, val_dataset = data.random_split(complete_dataset, [int(data_size * (train_size)),
                                                                  # data_size - int(data_size * (train_size))])


train_loader = DataLoader(complete_dataset, batch_size=args.batch, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)

print("Dataloaders done")
if args.write:
    Writer = SummaryWriter("models/tbx/" + args.logid + args.model)
    Writer.add_scalars("Metadata" + args.model, {"Batch_size": args.batch,
                                                  "learning_rate": args.lr,
                                                  "logid": int(args.logid),
                                                  # "training_size": train_dataset.__len__(),
                                                  # "Validation_size": val_dataset.__len__(),
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
    # print("epoch {}".format(epoch))
    Network.train()
    for batch_idx, data in enumerate(train_loader):
        print("epoch {}, batch {}/ {}".format(epoch, batch_idx, train_loader.__len__()))
        ai, ap, aw, a_index,\
        pi, pp, pw, \
        ni1, np1, nw1,\
        ni2, np2, nw2,\
        ni3, np3, nw3,\
        ni4, np4, nw4,\
        ni5, np5, nw5,\
        ni6, np6, nw6,\
        ni7, np7, nw7,\
        ni8, np8, nw8,\
        ni9, np9, nw9,\
        ni10, np10, nw10, x, pos_ind = data

        optimizer.zero_grad()
        ao, po, no1 = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni1.to(device), np1.to(device)])
        _, _, no2 = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni2.to(device), np2.to(device)])
        _, _, no3 = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni3.to(device), np3.to(device)])
        _, _, no4 = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni4.to(device), np4.to(device)])
        _, _, no5 = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni5.to(device), np5.to(device)])
        _, _, no6 = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni6.to(device), np6.to(device)])
        _, _, no7 = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni7.to(device), np7.to(device)])
        _, _, no8 = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni8.to(device), np8.to(device)])
        _, _, no9 = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni9.to(device), np9.to(device)])
        _, _, no10 = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni10.to(device), np10.to(device)])

        # print('e')
        loss1, p1, n1 = criterion(ao.to(device), po.to(device), no1.to(device))
        loss2, _, n2 = criterion(ao.to(device), po.to(device), no2.to(device))
        loss3, _, n3 = criterion(ao.to(device), po.to(device), no3.to(device))
        loss4, _, n4 = criterion(ao.to(device), po.to(device), no4.to(device))
        loss5, _, n5 = criterion(ao.to(device), po.to(device), no5.to(device))
        loss6, _, n6 = criterion(ao.to(device), po.to(device), no6.to(device))
        loss7, _, n7 = criterion(ao.to(device), po.to(device), no7.to(device))
        loss8, _, n8 = criterion(ao.to(device), po.to(device), no8.to(device))
        loss9, _, n9 = criterion(ao.to(device), po.to(device), no9.to(device))
        loss10, _, n10 = criterion(ao.to(device), po.to(device), no10.to(device))

        # print(p, n)
        loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10
        n = (n1.item()+n2.item()+n3.item()+n4.item()+n5.item()
             +n6.item()+n7.item()+n8.item()+n9.item()+n10.item())/10
        loss.backward()
        # print(loss)
        optimizer.step()

        complete_dataset.result_update(ao.cpu().detach().numpy(), a_index.cpu().detach().numpy())

        logs["training_batch_pdis"].append(p1.item())
        logs["training_batch_ndis"].append(n)
        logs["training_batch_loss"].append(loss.item())
        Writer.add_scalars("Training_data",{"batch_loss": loss.item(),
                                            "batch_pdis": p1.item(),
                                            "batch_ndis": n
                                            },
                           trainingcounter)
        if len(pos_ind) == 1:
            pos_ind = [0,0]
        Writer.add_scalars("Indices", {"n_indices": sum(x) / len(x),
                                       "p_indices": sum(pos_ind) / len(pos_ind),
                                       "max_p_index": pos_ind[-1],
                                       "min_p_index": pos_ind[1],
                                       "len_indices": len(pos_ind)-1
                                       },
                           trainingcounter)

        trainingcounter+=1
        if not epoch == 1:
            complete_dataset.update()
    if epoch ==1:
        complete_dataset.update()

    if args.save_embeds:
        with open("models/"+args.logid+"embeds.pickle", "wb") as q:
            pickle.dump([complete_dataset.libs, complete_dataset.labels], q)


    # torch.save(Network.state_dict(), "models/"+args.logid+"timodeldictpre.pt")
    # torch.save(Network, "models/"+args.logid+"timodelcompre.pt")
    #
    # Network.eval()
    # for batch_idx, data in enumerate(val_loader):
    #     print("batch {}/ {}".format(batch_idx, val_loader.__len__()))
    #     ai, ap, aw, a_index, pi, pp, pw, ni, np, nw  = data
    #
    #     complete_dataset.result_update(ao.detach().numpy(), a_index.detach().numpy())
    #
    #     ao, po, no = Network([ai.to(device), ap.to(device), pi.to(device), pp.to(device), ni.to(device), np.to(device)])
    #     loss, p, n = criterion(ao.to(device), po.to(device), no.to(device))
    #     logs["validation_batch_pdis"].append(p.item())
    #     logs["validation_batch_loss"].append(loss.item())
    #     logs["validation_batch_ndis"].append(n.item())
    #     Writer.add_scalars("Validation_data",{"batch_loss": loss.item(),
    #                                         "batch_pdis": p.item(),
    #                                         "batch_ndis": n.item()
    #                                         },
    #                        validationcounter)
    #     validationcounter+=1

    with open('models/' + args.logid + 'logfile.pickle', 'wb') as q:
        pickle.dump([logs, args], q)

    torch.save(Network.state_dict(), "models/"+args.logid+"timodeldict.pt")
    torch.save(Network, "models/"+args.logid+"timodelcom.pt")

    if args.decay_freq is not None:
        if epoch % args.decay_freq == 0:
            for g in optimizer.param_groups:
                g['lr'] = g["lr"] * args.decay_value