import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from util.loaders import *
# from util.models import *
import pickle
import logging
from random import sample
import numpy as np
from util.buildtextvladlib import buildlibrary
from util.utilities import getdistance
import pandas as pd
import random
import matplotlib.pyplot as plt

csvfile = pd.read_csv("/media/presage3/Windows-SSD/images/netvlad/68_data.csv",
                      usecols= ['imagesource', 'lat', 'long'],
                      skipinitialspace=True).drop_duplicates(['imagesource']).set_index('imagesource').to_dict()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open("/home/presage3/tv_com2/TVmodels_bh/03inputdata.pickle", "rb") as a:
    [jpgklass, jpgklassv] = pickle.load(a)

with open("training_data_pytorch06.pickle", "rb") as a:
    [_, _, _, _, enc, _, jpgklassraw, jpg2words] = pickle.load(a)

try:
    Network = torch.load("/home/presage3/bh_test2/models_bh/0101timodelcom.pt")
    print("Loading model small gpu")
except:
    Network = torch.load("/home/presage3/bh_test2/models_bh/0101timodelcom.pt", map_location='cpu')
    print("Loading model small cpu")

try:
    model = torch.load("/home/presage3/tv_com2/TVmodels_bh/03timodelcom.pt")
    print("Loading model small gpu")
except:
    model = torch.load("/home/presage3/tv_com2/TVmodels_bh/03timodelcom.pt", map_location='cpu')
    print("Loading model small cpu")
model.eval()
Network.eval()

loadlib = False
if loadlib:
    with open("/home/presage3/tv_com/TVmodels_bh/01embeds.pickle", "rb") as q:
        dict = pickle.load(q)
else:
    dict = buildlibrary(jpgklass, jpg2words, Network, model, enc,savepath="/home/presage3/tv_com2/TVmodels_bh/03embeds.pickle", path = "nv_txt/", elements = 2)


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


testing_dataset = Triplet_loaderbh_Textvlad_testing(validjpgs, jpg2words, "nv_txt/",
                                                 '/media/presage3/Windows-SSD/images/jpeg_patch/', Network, enc)
test_loader = DataLoader(testing_dataset, batch_size=32, shuffle=False)
testdict = {"names": [], "embeds": np.array([])}
for ids, data in enumerate(test_loader):
    model.eval()
    Network.eval()
    textual, nv, index = data
    embeds = model.get_embedding(textual.reshape([-1, 1280]), nv.reshape([-1, 4096])).cpu().detach().numpy()
    nv = nv.detach().cpu().numpy()
    if ids == 0:
        testdict['embeds'] = embeds
        testdict['netvlad'] = nv
    else:
        testdict['embeds'] = np.concatenate([testdict['embeds'], embeds], axis=0)
        testdict['netvlad'] = np.concatenate([testdict['netvlad'], nv], axis=0)

    for itera, u in enumerate(index):
        testdict["names"].append(validjpgs[u.item()])

    assert len(testdict['names']) == testdict['embeds'].shape[0]
    if (ids + 1) % 1 == 0:
        print("Getting test embeddings: {} / {} ".format(ids + 1, test_loader.__len__()))

assert len(set(testdict['names']).intersection(set(dict['names']))) == 0
dist_tv = []
dist_nv = []

testdictold = testdict
dictold = dict
testdict['netvlad'] = testdict['netvlad'].reshape(-1,4096)

indices = random.sample(list(range(len(dictold['names']))), 1500)
del dict
dict = {}
dict['netvlad'] = np.asarray(dictold['netvlad'].reshape(-1,4096))[indices]
dict['embeds'] = np.asarray(dictold['embeds'].reshape(-1,4096))[indices]
dict['names'] = np.asarray(dictold['names'])[indices]
results = {"dist_tv":[], "dist_nv":[], 'query': [], 'tv_match':[], 'nv_match':[]}
for itera, i in enumerate(testdict['names']):
    norm_tv = np.linalg.norm(testdict['embeds'][itera] - dict['embeds'], axis = 1)
    closest_image_tv = dict['names'][np.argmin(norm_tv)]

    norm_nv = np.linalg.norm(testdict['netvlad'][itera] - dict['netvlad'], axis = 1)
    closest_image_nv = dict['names'][np.argmin(norm_nv)]
    results['query'].append(i)
    results['tv_match'].append(closest_image_tv)
    results['nv_match'].append(closest_image_nv)
    tv_lat = csvfile['lat'][closest_image_tv]
    tv_long = csvfile['long'][closest_image_tv]
    nv_lat = csvfile['lat'][closest_image_nv]
    nv_long = csvfile['long'][closest_image_nv]
    Query_lat = csvfile['lat'][i]
    Query_long = csvfile['long'][i]
    results['dist_tv'].append(getdistance([tv_long, tv_lat], [Query_long, Query_lat]))
    results['dist_nv'].append(getdistance([nv_long, nv_lat], [Query_long, Query_lat]))
    print("Distance evaluations: {}/{}".format(itera, len(testdict['names'])))
    if results['dist_tv'][-1] > 70:
        print(i, closest_image_tv, closest_image_nv)
plt.hist([results['dist_nv'], results['dist_tv']], cumulative=1, bins= 1000, density=1, label=['NetVlad', "TextVlad (ours)"], range = [0,100], histtype='step')
plt.legend()