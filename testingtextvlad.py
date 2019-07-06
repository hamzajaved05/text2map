import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from util.loaders import *
from util.models import *
import pickle
import logging
from random import sample
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open("/home/presage3/tv_com/TVmodels_bh/01inputdata.pickle", "rb") as a:
    [jpgklass, jpgklassv] = pickle.load(a)

with open("training_data_pytorch06.pickle", "rb") as a:
    [_, _, _, _, enc, _, jpgklassraw, jpg2words] = pickle.load(a)



try:
    Network = torch.load("/home/presage3/bh_test/models_bh/01timodelcom.pt")
    print("Loading model small gpu")
except:
    Network = torch.load("/home/presage3/bh_test/models_bh/01timodelcom.pt", map_location='cpu')
    print("Loading model small cpu")

try:
    model = torch.load("/home/presage3/tv_com/TVmodels_bh/01timodelcom.pt")
    print("Loading model small gpu")
except:
    model = torch.load("/home/presage3/tv_com/TVmodels_bh/01timodelcom.pt", map_location='cpu')
    print("Loading model small cpu")
model.eval()
Network.eval()

complete_dataset = Triplet_loaderbh_Textvlad(jpgklass, jpg2words, 2, "nv_txt/", '/media/presage3/Windows-SSD/images/jpeg_patch/', Network, enc)
complete_dataset.testing = True
train_loader = DataLoader(complete_dataset, batch_size=32, shuffle=False)

dict = {"names":[], "embeds":np.array([])}
for ids, data in enumerate(train_loader):
    textual, nv , klass, first, second  = data
    embeds = model.get_embedding(textual.reshape([-1, 1280]), nv.reshape([-1, 4096])).cpu().detach().numpy()
    try:
        dict['embeds'] = np.concatenate([dict['embeds'], embeds], axis=0)
    except:
        dict['embeds'] = embeds

    for itera, u in enumerate(klass):
        dict["names"].append(jpgklass[u.item()][first[itera].item()])
        dict["names"].append(jpgklass[u.item()][second[itera].item()])

    if (ids+1)%1 ==0:
        print("Done {} / {} ".format(ids + 1, train_loader.__len__()))


with open("/home/presage3/tv_com/TVmodels_bh/01embeds.pickle", "wb") as q:
    pickle.dump(dict, q)