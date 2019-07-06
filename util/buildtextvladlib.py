import numpy as np
import torch
from torch.utils.data import DataLoader
from util.loaders import *
import pickle


def buildlibrary(jpgklass, jpg2words, Network, model, enc, savepath, path = "nv_txt/", elements = 2):
    complete_dataset = Triplet_loaderbh_Textvlad(jpgklass, jpg2words, elements, path,
                                                 '/media/presage3/Windows-SSD/images/jpeg_patch/', Network, enc)
    complete_dataset.testing = True
    train_loader = DataLoader(complete_dataset, batch_size=32, shuffle=False)

    dict = {"names": [], "embeds": np.array([])}
    for ids, data in enumerate(train_loader):
        textual, nv, klass, first, second = data
        embeds = model.get_embedding(textual.reshape([-1, 1280]), nv.reshape([-1, 4096])).cpu().detach().numpy()
        try:
            dict['embeds'] = np.concatenate([dict['embeds'], embeds], axis=0)
        except:
            dict['embeds'] = embeds

        for itera, u in enumerate(klass):
            dict["names"].append(jpgklass[u.item()][first[itera].item()])
            dict["names"].append(jpgklass[u.item()][second[itera].item()])

        if (ids + 1) % 1 == 0:
            print("Done {} / {} ".format(ids + 1, train_loader.__len__()))

    with open(savepath, "wb") as q:
        pickle.dump(dict, q)

    return dict