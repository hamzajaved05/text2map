import cv2
from torch.utils import data
import numpy as np
from random import sample
import torch
import torch.nn.functional as F
import pandas as pd
import pickle
from collections import Counter
import random
import os

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def im_triplet(jpg, image_lib, labels):
    dumm = jpg
    ind = np.argwhere(dumm == image_lib).reshape(-1)
    classes = labels[ind]
    indi = np.array([])
    for i in classes:
        indi = np.concatenate((indi, image_lib[np.argwhere(i == labels).reshape(-1)]), axis=0)
    while True:
        guess = sample(list(image_lib), 1)[0]
        if guess not in indi:
            break
    return [1 if len(classes) > 1 else 0, jpg, Counter(indi).most_common(1)[0][0], guess]

def check(triple, ids):
    if triple[0] in ids:
        if triple[1] in ids:
            if triple[2] in ids:
                return True
    return False

class image_word_training_loader(data.Dataset):
    def __init__(self, jpeg, words, words_sparse, labels, path):
        self.labels = np.array(labels)
        self.words = words
        self.im_path = path
        self.jpeg = np.array(jpeg)
        self.words_sparse = words_sparse

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        word_indexed = torch.from_numpy(self.words_sparse[index].todense())
        y = torch.tensor(self.labels[index])
        im = torch.tensor(cv2.imread(self.im_path + self.jpeg[index][:-4] + "_" + self.words[index] + ".jpg")).permute(
            2, 0, 1)
        return torch.div(im.float(), 255), word_indexed.float(), self.jpeg[index], self.words[index]


class Triplet_loader(data.Dataset):
    def __init__(self, triplet, jpg_word_dict, lib_ids, lib_embeds, path, netvladids, rand):
        self.triplet = np.array(triplet)
        self.jpg_word_dict = jpg_word_dict
        self.lib_ids = np.array(lib_ids)
        self.lib_embeds = lib_embeds
        self.path = path
        self.netvladids= netvladids
        self.rand = rand

        if self.rand:
            with open("training_data_pytorch04.pickle", "rb") as a:
                [klass, _, words, jpgs, _, _] = pickle.load(a)
            self.jpgs = np.array(jpgs)
            self.klass = np.array(klass)


    def __len__(self):
        return self.triplet.shape[0]

    def __getitem__(self, index):
        patch_out = []
        vlad_out = []
        if self.rand:
            our_triplet = im_triplet(self.triplet[index, 1], image_lib=self.jpgs, labels=self.klass)
            our_triplet = our_triplet[1:]
            print(our_triplet)
            if not check(our_triplet, self.netvladids):
                print("check failed")
                return None
        else:
            our_triplet = self.triplet[index, 1:]

        for itera, i in enumerate(our_triplet):
            words = self.jpg_word_dict[i]
            # array =[for instance in words]
            image_embed =  np.array([])
            for instance in words:
                instance_c = i + instance
                # print(instance_c)
                if instance_c in self.lib_ids:
                    # print("yes")
                    # print(instance_c)
                    indix = np.argwhere(self.lib_ids == instance_c)
                    # if isinstance(indix[0],int):
                    embedding_inst = self.lib_embeds[indix,:].reshape(1, -1)
                    # print(embedding_inst.shape)
                    image_embed=np.concatenate((image_embed, embedding_inst[0]), axis=0)
            emb = torch.tensor(image_embed).view([-1, 128])
            # print(emb.size())
            x = F.pad(emb, (0, 0, 0, 10-emb.size()[0]), "constant", 0)
            _,_, v = torch.svd(x)
            patch_out.append(v.t().reshape(-1))
            # print(i)
            if i in self.netvladids:
                x = np.loadtxt(self.path + i, delimiter=", ")
                vlad_out.append(x.reshape(-1))
            else:
                return None

        return patch_out[0], patch_out[1], patch_out[2], vlad_out[0], vlad_out[1], vlad_out[2], our_triplet



class image_word_triplet_loader(data.Dataset):
    def __init__(self, jpeg, words, words_sparse, labels, path, ld):
        self.labels = np.array(labels)
        self.words = words
        self.im_path = path
        self.jpeg = np.array(jpeg)
        self.words_sparse = words_sparse
        self.ld = ld
        self.firsttime = True
        self.libs={}
        self.libclass = {}
        for itera, i in enumerate(jpeg):
            self.libs[itera] = []
        self.newklass = []
        self.newlib = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #Get anchor
        anchor_word_indexed = torch.from_numpy(self.words_sparse[index].todense())
        # print(self.im_path + self.jpeg[index][:-4] + "_" + self.words[index] + ".jpg")
        im = torch.tensor(cv2.imread(self.im_path + self.jpeg[index][:-4] + "_" + self.words[index] + ".jpg")).permute(
            2, 0, 1)
        anchor_im = torch.div(im.float(), 255)
        anchor_patch_name = self.im_path + self.jpeg[index][:-4] + "_" + self.words[index] + ".jpg"

        #Get positive

        if self.firsttime:
            a_label = self.labels[index]
            high = np.max(np.argwhere(self.labels == a_label))
            low = np.min(np.argwhere(self.labels == a_label))
            positive_index = random.randint(low, high)
        else:
            a_label = self.labels[index]
            # dists = self.labels==a_label
            a_lib = self.values[index]
            dists = np.linalg.norm(a_lib - self.values, axis=1) * (self.labels == a_label)
            positive_index = np.argwhere(dists == np.max(dists))[0].item()

        positive_word_indexed = torch.from_numpy(self.words_sparse[positive_index].todense())
        # print(self.im_path + self.jpeg[positive_index][:-4] + "_" + self.words[positive_index] + ".jpg")

        im = torch.tensor(cv2.imread(self.im_path + self.jpeg[positive_index][:-4] + "_" + self.words[positive_index] + ".jpg")).permute(
            2, 0, 1)
        positive_im = torch.div(im.float(), 255)
        positive_patch_name = self.im_path + self.jpeg[positive_index][:-4] + "_" + self.words[positive_index] + ".jpg"

        #Get negative
        neg_ind = []
        pos_ind = []
        if self.firsttime:
            random_index = random.randint(0, len(self.labels) - 1)
            while len(neg_ind) <10:
                while (self.labels[index] == self.labels[random_index]) or (
                    levenshteinDistance(self.words[index], self.words[random_index]) > self.ld):
                # print("searching")
                    random_index = random.randint(0, len(self.labels) - 1)
                neg_ind.append(random_index)
                x = 0
                first_pos = 0
                last_pos = 0
                len_pos = 0

        else:
            a_lib = self.values[index]
            scores = np.linalg.norm(a_lib - self.values, axis = 1)
            ind = np.argpartition(scores, 40)[:40]
            min_elements = scores[ind]
            min_elements_order = np.argsort(min_elements)
            ordered_indices = ind[min_elements_order]
            for it, i in enumerate(ordered_indices):
                if not self.labels[i] == a_label:
                    neg_ind.append(i)
                    if len(neg_ind)==10:
                        x = it
                        break
                else:
                    pos_ind.append(it)

            first_pos = pos_ind[1]
            last_pos = pos_ind[-1]
            len_pos = len(pos_ind)

        negative_word_indexed = []
        negative_im = []
        negative_patch_name = []
        for negative_index in neg_ind:
            negative_word_indexed.append(torch.from_numpy(self.words_sparse[negative_index].todense()))
            im = torch.tensor(cv2.imread(self.im_path + self.jpeg[negative_index][:-4] + "_" + self.words[negative_index] + ".jpg")).permute(
                2, 0, 1)
            negative_im.append(torch.div(im.float(), 255))
            negative_patch_name.append(self.im_path + self.jpeg[negative_index][:-4] + "_" + self.words[negative_index] + ".jpg")

        return anchor_im, anchor_word_indexed.float(), anchor_patch_name, index, \
               positive_im, positive_word_indexed.float(), positive_patch_name, \
               negative_im[0], negative_word_indexed[0].float(), negative_patch_name[0], \
               negative_im[1], negative_word_indexed[1].float(), negative_patch_name[1], \
               negative_im[2], negative_word_indexed[2].float(), negative_patch_name[2], \
               negative_im[3], negative_word_indexed[3].float(), negative_patch_name[3], \
               negative_im[4], negative_word_indexed[4].float(), negative_patch_name[4], \
               negative_im[5], negative_word_indexed[5].float(), negative_patch_name[5], \
               negative_im[6], negative_word_indexed[6].float(), negative_patch_name[6], \
               negative_im[7], negative_word_indexed[7].float(), negative_patch_name[7], \
               negative_im[8], negative_word_indexed[8].float(), negative_patch_name[8], \
               negative_im[9], negative_word_indexed[9].float(), negative_patch_name[9], \
               x, first_pos, last_pos, len_pos

    def result_update(self, values, indices):
        assert isinstance(values, (np.ndarray))
        for itera, i in enumerate(indices):
            self.libs[i] = values[itera]

    def update(self):
        self.keys = list(self.libs.keys())
        self.values = list(self.libs.values())
        self.firsttime = False
        assert len(self.keys) == len(self.jpeg)

    def summary_writer(self, writer):
        self.writer = writer