import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from util.loaders import *
from util.models import *
import pickle
import logging
from random import sample


parser = argparse.ArgumentParser(description='Combined Runs')
parser.add_argument("--epoch", type=int, default=500, help="no of epochs")
parser.add_argument("--batch", type=int, default=32, help="batch_size")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--logid", type=str, help="logid")
parser.add_argument("--decay_freq", default=None, type=int, help="Decay by half after number of epochs")
parser.add_argument("--path", type=str, help="path to vlad_results")
parser.add_argument("--itemsperclass", default = 5, type = int, help="maximum items per class")
parser.add_argument("--load", type = str, help = "load file")
parser.add_argument("--embed", default = 4096, type = int, help = "embed size")
parser.add_argument("--write", default = True, type = bool, help = "write")
parser.add_argument("--nvtxt", default = 'nv_txt/', type = str, help = "nv text folder /")
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('training_data_pytorch06.pickle', "rb") as a:
    [_, _, _, _, enc, _, jpgklassraw, jpg2words] = pickle.load(a)

def filterklass(dict, maxim):
    dictnew = {}
    count = 0
    valdict = {}
    for i in dict.keys():
        if len(dict[i]) == np.sum(np.array(['jpg' in string for string in dict[i]])):
            if len(dict[i]) > 2*maxim:
                val = sample(list(dict[i]), int(maxim/2))
                train = set(dict[i]).difference(set(val))
                dictnew[count] = list(train)
                valdict[count] = list(val)
                count+=1
            elif len(dict[i]) >= maxim:
                dictnew[count] = list(dict[i])
                count += 1
    return dictnew, valdict
jpgklass, jpgklass_v = filterklass(jpgklassraw, args.itemsperclass)

def reverse_dict(dict):
    new_dic = {}
    for k, v in dict.items():
        for x in np.asarray(v).reshape(-1).tolist():
            new_dic.setdefault(x, []).append(k)
    return new_dic

revdict = reverse_dict(jpgklass)



with open("TVmodels_bh/" + args.logid + "inputdata.pickle", "wb") as q:
    pickle.dump([jpgklass, jpgklass_v], q)

for iter, i in enumerate(list(jpgklass.values())):
    assert len(i) == np.sum(np.array(['jpg' in string for string in i]))

try:
    Network = torch.load(args.load)
    print("Loading model gpu")
except:
    Network = torch.load(args.load, map_location='cpu')
    print("Loading model cpu")

complete_dataset = Triplet_loaderbh_Textvlad(jpgklass, jpg2words, args.itemsperclass, "nv_txt/", '/media/presage3/Windows-SSD/images/jpeg_patch/', Network, enc)
embed_net = Embedding_net(c_embed=5120).float().to(device)
model = TripletNet(embed_net).float().to(device)
criterion = TripletLoss().to(device)
train_loader = DataLoader(complete_dataset, batch_size=args.batch, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
epochs = 10
if args.write:
    Writer = SummaryWriter("TVmodels_bh/tbx/" + args.logid)
    Writer.add_scalars("Metadata" , {"Batch_size": args.batch,
                                     "learning_rate": args.lr,
                                     "items per class": args.itemsperclass,
                                     "Number of classes": len(jpgklass)})


trainingcounter= 0
for epoch in range(1, epochs + 1):
    for batch_idx, data in enumerate(train_loader):
        textual, nv, klass_batch = data
        klass_batch = klass_batch.numpy()
        textual = textual.to(device)
        nv = nv.to(device)
        embeds = model.get_embedding(textual.reshape([-1,1280]), nv.reshape([-1,4096])).cpu().detach().numpy()
        optimizer.zero_grad()
        loss = 0
        pdis_all = []
        ndis_all = []
        pind_all = []
        nind_all = []

        for i in range(textual.shape[0]):
            current_klass = klass_batch[i]
            all_pos_diff_classes = np.argwhere([len(set(jpgklass[i]).intersection(set(jpgklass[current_klass])))
                                                for i in jpgklass.keys()]).squeeze()

            for j in range(textual.shape[1]):
                positive = list(range(args.itemsperclass*(i),args.itemsperclass*(i)+args.itemsperclass))
                negative = list(range(embeds.shape[0]))
                negative = list(set(negative).difference(set(positive)))
                anchor_textual = textual[i,j,...]
                anchor_nv = nv[i,j,...]
                sorted_indices = np.argsort(np.linalg.norm(embeds[args.itemsperclass*(i)+j] - embeds, ord = 2, axis=1))

                for n_ind, q in enumerate(sorted_indices):
                    if q not in positive:
                        negative_indice = q
                        nind_all.append(n_ind)
                        n_i = list(np.unravel_index(negative_indice, [args.batch, args.itemsperclass]))
                        if klass_batch[n_i[0]] in all_pos_diff_classes:
                            continue
                        break
                for p_ind, q in enumerate(sorted_indices[::-1]):
                    if q in positive:
                        positive_indice = q
                        pind_all.append(p_ind)
                        p_i = list(np.unravel_index(positive_indice, [args.batch, args.itemsperclass]))
                        break


                a_embed, p_embed, n_embed = model([textual[i,j,...].unsqueeze(0), nv[i,j,...].unsqueeze(0),
                                                     textual[p_i[0],p_i[1],...].unsqueeze(0), nv[p_i[0], p_i[1],...].unsqueeze(0),
                                                     textual[n_i[0],n_i[1],...].unsqueeze(0), nv[n_i[0], n_i[1],...].unsqueeze(0)])
                loss1, p1, n1 = criterion(a_embed.to(device), p_embed.to(device), n_embed.to(device))
                loss += loss1
                pdis_all.append(p1.item())
                ndis_all.append(n1.item())

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
                                       "average_p_location": sum(pind_all)/len(pind_all)},
                           trainingcounter)
        trainingcounter+= 1
        torch.save(model.state_dict(), "TVmodels_bh/" + args.logid + "timodeldict.pt")
        torch.save(model, "TVmodels_bh/" + args.logid + "timodelcom.pt")