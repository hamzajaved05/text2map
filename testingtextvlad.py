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
from util.buildtextvladlib import buildlibrary
from util.utilities import getdistance
import pandas as pd

csvfile = pd.read_csv("/media/presage3/Windows-SSD/images/netvlad/68_data.csv",
                      usecols= ['imagesource', 'lat', 'long'],
                      skipinitialspace=True).drop_duplicates(['imagesource']).set_index('imagesource').to_dict()
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

loadlib = True
if loadlib:
    with open("/home/presage3/tv_com/TVmodels_bh/01embeds.pickle", "rb") as q:
        dict = pickle.load(q)
else:
    dict = buildlibrary(jpgklass, jpg2words, Network, model, enc,savepath="/home/presage3/tv_com/TVmodels_bh/01embeds.pickle", path = "nv_txt/", elements = 2)


for itera, i in enumerate(jpgklassraw.keys()):
    if itera != 0:
        alljpgs = np.concatenate([alljpgs, jpgklassraw[i].reshape(1,-1)], axis = 1)
    if itera == 0:
        alljpgs = jpgklassraw[i].reshape(1,-1)
alljpgs = alljpgs.reshape(-1).tolist()

for itera, i in enumerate(jpgklass.keys()):
    if itera != 0:
        libjpgs = np.concatenate([libjpgs, np.asarray(jpgklass[i]).reshape(1,-1)], axis = 1)
    if itera == 0:
        libjpgs = np.asarray(jpgklass[i]).reshape(1,-1)
libjpgs = libjpgs.reshape(-1).tolist()

validjpgs = list(set(alljpgs).difference(set(libjpgs)))
# indices = sample(range(len(validjpgs)), 3000)


testing_dataset = Triplet_loaderbh_Textvlad_testing(validjpgs, jpg2words, 2, "nv_txt/",
                                                 '/media/presage3/Windows-SSD/images/jpeg_patch/', Network, enc)
test_loader = DataLoader(testing_dataset, batch_size=32, shuffle=False)
testdict = {"names": [], "embeds": np.array([])}
for ids, data in enumerate(test_loader):
    model.eval()
    Network.eval()
    textual, nv, index = data
    embeds = model.get_embedding(textual.reshape([-1, 1280]), nv.reshape([-1, 4096])).cpu().detach().numpy()
    if ids == 0:
        testdict['embeds'] = embeds
    else:
        testdict['embeds'] = np.concatenate([dict['embeds'], embeds], axis=0)

    for itera, u in enumerate(index):
        testdict["names"].append(validjpgs[u.item()])

    if (ids + 1) % 1 == 0:
        print("Done {} / {} ".format(ids + 1, test_loader.__len__()))


dist = []
for itera, i in enumerate(testdict['names']):
    norm = np.linalg.norm(testdict['embeds'][itera] - dict['embeds'], axis = 1)
    closest_image = dict['names'][np.argmin(norm)]
    lat = csvfile.loc[closest_image].get('lat').to_numpy()[0]
    long = csvfile.loc[closest_image].get('long').to_numpy()[0]
    lat2 = csvfile.loc[i].get('lat').to_numpy()[0]
    long2 = csvfile.loc[i].get('long').to_numpy()[0]
    dist.append(getdistance([long, lat], [long2, lat2]))

    print("{}/{}".format(itera, len(testdict['names'])))