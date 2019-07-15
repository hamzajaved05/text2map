import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from util.loaders import *
from util.models import *
import pickle
import pandas as pd
from util.utilities import getdistance
from random import sample
import numpy as np

parser = argparse.ArgumentParser(description='Combined Runs')
parser.add_argument("--epoch", type=int, default=500, help="no of epochs")
parser.add_argument("--impath", type=str, help="Path for Image patches")
parser.add_argument("--batch", type=int, default=32, help="batch_size")
parser.add_argument("--inpickle", type=str, help="Path of pickle file")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--logid", type=str, help="logid")
parser.add_argument("--decay_freq", default=None, type=int, help="Decay by half after number of epochs")
parser.add_argument("--path", type=str, help="path to vlad_results")
parser.add_argument("--itemsperclass", default = 5, type = int, help="maximum items per class")
parser.add_argument("--load", type = str, help = "load file")
parser.add_argument("--embed", default = 4096, type = int, help = "embed size")
parser.add_argument("--write", default = True, type = bool, help = "write")
parser.add_argument("--nvtxt", default = 'nv_txt/', type = str, help = "nv text folder /")
parser.add_argument("--l2loss", default = True, type = bool, help = "l2 distance between files")
parser.add_argument("--softplus", default = False, type = bool, help = "softplus loss")
parser.add_argument("--margin", default = 0.1, type = float, help = "margin")
parser.add_argument("--randmine", default = True, type = bool, help = "posrand")
parser.add_argument("--switch", default = 20, type = int, help = "switch between random")
parser.add_argument("--csvpath", default = "68_data.csv", type= str, help = "Path for csvfile")
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open(args.inpickle, "rb") as a:
    [_, _, _, _, enc, _, jpgklassraw, jpg2words] = pickle.load(a)

csvfile = pd.read_csv(args.csvpath,
                      usecols= ['imagesource', 'lat', 'long'],
                      skipinitialspace=True).drop_duplicates(['imagesource']).set_index('imagesource').to_dict()

def train_test_split(dict, maxim):
    dictnew = {}
    count = 0
    for i in dict.keys():
        if len(dict[i]) == np.sum(np.array(['jpg' in string for string in dict[i]])):
            if len(dict[i]) >= maxim:
                train = sample(list(dict[i]), maxim)
                dictnew[count] = list(train)
                count+=1

    for itera, i in enumerate(dict.keys()):
        if itera != 0:
            alljpgs = np.concatenate([alljpgs, dict[i].reshape(1, -1)], axis=1)
        if itera == 0:
            alljpgs = dict[i].reshape(1, -1)
    alljpgs = alljpgs.reshape(-1).tolist()

    for itera, i in enumerate(dictnew.keys()):
        if itera != 0:
            libjpgs = np.concatenate([libjpgs, np.asarray(dictnew[i]).reshape(1, -1)], axis=1)
        if itera == 0:
            libjpgs = np.asarray(dictnew[i]).reshape(1, -1)
    libjpgs = libjpgs.reshape(-1).tolist()

    validjpgs = list(set(alljpgs).difference(set(libjpgs)))
    return dictnew, validjpgs


jpgklass_train, jpg_valid = train_test_split(jpgklassraw, args.itemsperclass)


# def reverse_dict(dict):
#     new_dic = {}
#     for k, v in dict.items():
#         for x in np.asarray(v).reshape(-1).tolist():
#             new_dic.setdefault(x, []).append(k)
#     return new_dic
#
# revdict = reverse_dict(jpgklass)

for iter, i in enumerate(list(jpgklass_train.values())):
    assert len(i) == np.sum(np.asarray(['jpg' in string for string in i]))

try:
    Network_patch = torch.load(args.load)
    print("Loading model gpu")
except:
    Network_patch = torch.load(args.load, map_location='cpu')
    print("Loading model cpu")

complete_dataset = Triplet_loaderbh_Textvlad(jpgklass_train, jpg2words, args.itemsperclass, args.nvtxt, args.impath, Network_patch, enc)
jpg_valid_subset = sample(jpg_valid, 1000)
testing_dataset = Triplet_loaderbh_Textvlad_testing(jpg_valid_subset, jpg2words, args.nvtxt, args.impath, Network_patch, enc)

Embed_net = Embedding_net(c_embed=args.embed).float().to(device)
TextVlad = TripletNet(Embed_net).float().to(device)
criterion = TripletLoss(l2= args.l2loss, softplus= args.softplus, margin=args.margin).to(device)
train_loader = DataLoader(complete_dataset, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(testing_dataset, batch_size=1, shuffle=True)

optimizer = optim.Adam(TextVlad.parameters(), lr=args.lr)
epochs = args.epoch
if args.write:
    Writer = SummaryWriter("TVmodels_bh/tbx/" + args.logid)
    Writer.add_scalars("Metadata" , {"Batch_size": args.batch,
                                     "learning_rate": args.lr,
                                     "items per class": args.itemsperclass,
                                     "Number of classes": len(jpgklass_train),
                                     "randmine": int(args.randmine),
                                     "switch": args.switch,
                                     "softplus": int(args.softplus),
                                     "l2loss": int(args.l2loss),
                                     "margin": args.margin})


randmine = args.randmine
names = []
trainingcounter= 0
for epoch in range(1, epochs + 1):
    TextVlad.train()
    for batch_idx, data in enumerate(train_loader):
        textual, nv, klass_batch, indices = data
        klass_batch = klass_batch.numpy()
        textual = textual.to(device)
        nv = nv.to(device)
        embeds = TextVlad.get_embedding(textual.reshape([-1, 1280]), nv.reshape([-1, 4096])).cpu().detach().numpy()
        optimizer.zero_grad()
        loss = 0
        pdis_all = []
        ndis_all = []
        pind_all = []
        nind_all = []

        for i in range(textual.shape[0]):
            current_klass = klass_batch[i]
            all_pos_diff_classes = np.argwhere([len(set(jpgklass_train[i]).intersection(set(jpgklass_train[current_klass])))
                                                for i in jpgklass_train.keys()]).squeeze()

            for j in range(textual.shape[1]):
                positive = list(range(args.itemsperclass*(i),args.itemsperclass*(i)+args.itemsperclass))
                negative = list(range(embeds.shape[0]))
                negative = list(set(negative).difference(set(positive)))

                norm = np.linalg.norm(embeds[args.itemsperclass*(i)+j] - embeds, ord = 2, axis=1)
                sorted_indices = np.argsort(norm)

                if randmine:
                    while True:
                        positive_indice = sample(positive, 1)
                        if positive_indice != (args.itemsperclass*i + j): break
                    p_ind = 0
                    pind_all.append(p_ind)
                    p_i = list(np.unravel_index(positive_indice[0], [args.batch, args.itemsperclass]))
                    positive_distance = norm[positive_indice]
                else:
                    for p_ind, q in enumerate(sorted_indices[::-1]):
                        if q in positive:
                            positive_indice = q
                            pind_all.append(p_ind)
                            p_i = list(np.unravel_index(positive_indice, [args.batch, args.itemsperclass]))
                            positive_distance = norm[positive_indice]
                            break

                if randmine:
                    while True:
                        negative_indice = sample(negative, 1)
                        n_ind = 0
                        nind_all.append(n_ind)
                        n_i = list(np.unravel_index(negative_indice[0], [args.batch, args.itemsperclass]))
                        if klass_batch[n_i[0]] in all_pos_diff_classes: continue
                        break
                else:
                    for n_ind, q in enumerate(sorted_indices):
                        if q not in positive:
                            negative_indice = q
                            nind_all.append(n_ind)
                            n_i = list(np.unravel_index(negative_indice, [args.batch, args.itemsperclass]))
                            if klass_batch[n_i[0]] in all_pos_diff_classes:
                                continue
                            break


                anchor_textual = textual[i,j,...]
                anchor_nv = nv[i,j,...]
                a_embed, p_embed, n_embed = TextVlad([textual[i, j, ...].unsqueeze(0), nv[i, j, ...].unsqueeze(0),
                                                      textual[p_i[0],p_i[1],...].unsqueeze(0), nv[p_i[0], p_i[1],...].unsqueeze(0),
                                                      textual[n_i[0],n_i[1],...].unsqueeze(0), nv[n_i[0], n_i[1],...].unsqueeze(0)])
                loss1, p1, n1 = criterion(a_embed.to(device), p_embed.to(device), n_embed.to(device))
                loss += loss1
                pdis_all.append(p1.item())
                ndis_all.append(n1.item())
            names.append(jpgklass_train[klass_batch[i].item()][indices[i, 4].item()])
            try:
                libtensors = torch.cat([libtensors, a_embed] , dim = 0)
            except:
                libtensors = a_embed

        loss = loss / (textual.shape[0] * textual.shape[1])
        loss.backward()
        optimizer.step()
        print("epoch {}, batch {}/{}, train_loss {:.5f}, p_dis_mean {:.5f}, n_dis_mean {:.5f}".format(epoch,
                                                                                                      batch_idx,
                                                                                                      train_loader.__len__(),
                                                                                                      loss.item(),
                                                                                                      sum(
                                                                                                          pdis_all) / len(
                                                                                                          pdis_all),
                                                                                                      sum(
                                                                                                          ndis_all) / len(
                                                                                                          ndis_all)))
        Writer.add_scalars("Training_data_batch_hard",{"batch_loss": loss.item(),
                                            "batch_pdis": sum(pdis_all) / len(pdis_all),
                                            "batch_ndis": sum(ndis_all) / len(ndis_all)
                                            }, trainingcounter)

        Writer.add_scalars("Indices", {"average_n_location": sum(nind_all)/len(nind_all),
                                       "average_p_location": sum(pind_all)/len(pind_all),},
                           trainingcounter)
        trainingcounter+= 1
        torch.save(TextVlad.state_dict(), "TVmodels_bh/" + args.logid + "timodeldict.pt")
        torch.save(TextVlad, "TVmodels_bh/" + args.logid + "timodelcom.pt")
        libtensors = libtensors.detach().cpu().numpy()

        if ((epoch+1) == args.switch):
            print("Switching random behaviour")
            posrand = False
            negrand = False

    dist = []
    for test_idx, data in enumerate(test_loader):
        TextVlad.eval()
        Network_patch.eval()
        textual, nv, index = data
        embeds = TextVlad.get_embedding(textual.reshape([-1, 1280]), nv.reshape([-1, 4096])).cpu().detach().numpy()
        norm = np.linalg.norm(embeds - libtensors, axis=1)
        closest_image = names[np.argmin(norm)]

        lat = csvfile['lat'][closest_image]
        long = csvfile['long'][closest_image]
        lat2 = csvfile['lat'][jpg_valid_subset[index.item()]]
        long2 = csvfile['lat'][jpg_valid_subset[index.item()]]
        dist.append(getdistance([long, lat], [long2, lat2]))

        Writer.add_scalars("Validation",{"distance_mean": np.mean(dist),
                                         "distance less than 50": np.sum(np.argwhere(np.asarray(dist)<50)),
                                         "distance of zero": np.sum(np.argwhere(np.asarray(dist)<50))}, trainingcounter)


        # if (test_idx + 1) % 1 == 0:
    print("Validation Done")


    with open("TVmodels_bh/"+ args.logid+'inputdata.pickle', "wb") as q:
        pickle.dump([jpgklass_train, jpg_valid, jpg2words, libtensors, names, dist], q)


    del libtensors, dist
