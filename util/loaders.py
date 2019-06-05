import cv2
from torch.utils import data
import numpy as np
from random import sample
from random import randint
import torch
import torch.nn.functional as F
import pandas as pd
import pickle
from collections import Counter
from Levenshtein import levenshteinDistance

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
        self.labels = labels#np.array(labels)
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
        return torch.div(im.float(), 255), word_indexed.float(), y.float()
    
    def __gettriplet__(self, index):
        #Get anchor
        anchor_word_indexed = torch.from_numpy(self.words_sparse[index].todense())
        im = torch.tensor(cv2.imread(self.im_path + self.jpeg[index][:-4] + "_" + self.words[index] + ".jpg")).permute(
            2, 0, 1)
        anchor_im = torch.div(im.float(), 255)
        anchor_patch_name = self.im_path + self.jpeg[index][:-4] + "_" + self.words[index] + ".jpg"
        
        #Get positive
        random_index = (index+randint(0,10)) 
        if (labels[index]==labels[positive_index]):
            positive_index=random_index
        elif (labels[index]==labels[index+1] and index<(len(self.labels)-1)):
            positive_index=index+1
        elif (index>1 and labels[index]==labels[index-1]):
            positive_index=index-1
        else:
            positive_index=index
        positive_word_indexed = torch.from_numpy(self.words_sparse[positive_index].todense()) 
        im = torch.tensor(cv2.imread(self.im_path + self.jpeg[positive_index][:-4] + "_" + self.words[positive_index] + ".jpg")).permute(
            2, 0, 1)
        positive_im = torch.div(im.float(), 255)
        positive_patch_name = self.im_path + self.jpeg[positive_index][:-4] + "_" + self.words[positive_index] + ".jpg"
            
        #Get negative
        if (levenshtein_mining):
            random_index = randint(0,len(self.labels)-1)
            while (labels[index]==labels[random_index] and levenshteinDistance(words[index],words[random_index])>5 ):
                random_index = randint(0,len(self.labels)-1)
            negative_index=random_index
        else:
            random_index = randint(0,len(self.labels)-1)
            while (labels[index]==labels[random_index]):
                random_index = randint(0,len(self.labels)-1)
            negative_index=random_index
        
        negative_word_indexed = torch.from_numpy(self.words_sparse[negative_index].todense()) 
        im = torch.tensor(cv2.imread(self.im_path + self.jpeg[negative_index][:-4] + "_" + self.words[negative_index] + ".jpg")).permute(
            2, 0, 1)
        negative_im = torch.div(im.float(), 255)
        negative_patch_name = self.im_path + self.jpeg[negative_index][:-4] + "_" + self.words[negative_index] + ".jpg"
        
        return anchor_word_indexed, anchor_im, anchor_patch_name, positive_word_indexed, positive_im, positive_patch_name, negative_word_indexed, negative_im, negative_patch_name 
        

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
                loc = np.argwhere(i == self.netvladids).item()-1
                x = pd.read_csv(self.path, header=None, skiprows=loc, nrows=1).to_numpy()
                vlad_out.append(x.reshape(-1))
            else:
                return None

        return patch_out[0], patch_out[1], patch_out[2], vlad_out[0], vlad_out[1], vlad_out[2]
