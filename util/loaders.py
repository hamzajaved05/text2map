import cv2
from torch.utils import data
import numpy as np
from random import sample
import torch
import torch.nn.functional as F
from collections import Counter
import logging
from util.utilities import word2encodedword
from util.utilities import wordskip

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

class Triplet_loaderbh_Textvlad(data.Dataset):
    def __init__(self, jpgklass, jpgdict, itemsperclass, pathnv, path_patches, model, enc):
        self.jpgklass = jpgklass
        self.jpgdict = jpgdict
        self.itemsperclass = itemsperclass
        self.path_netvlad = pathnv
        self.model = model
        self.pathpatch = path_patches
        self.enc = enc
        self.model.eval()
        self.model.to(device)
        self.testing = False

    def __len__(self):
        return len(self.jpgklass.keys())

    def __getitem__(self, index):
        no_elements = len(self.jpgklass[index])

        indices = sample(list(np.arange(no_elements)), self.itemsperclass)
        jpgnames = np.asarray(self.jpgklass[index])[indices]
        for single_jpg in jpgnames:
            netvlad = self.readnetvlad(single_jpg).unsqueeze(0)
            embedding = self.get_latent_embedding(str(single_jpg)).unsqueeze(0)
            try:
                patch_embeds = torch.cat([patch_embeds, embedding], dim = 0)
                netvlad_embeds = torch.cat([netvlad_embeds, netvlad], dim = 0)
            except:
                patch_embeds = embedding
                netvlad_embeds = netvlad
        assert patch_embeds.shape[0] == self.itemsperclass, "Patch size issue"
        assert netvlad_embeds.shape[0] == self.itemsperclass, "netvlad size issue"

        return patch_embeds.float().to(device), netvlad_embeds.float().to(device), index, torch.tensor(indices)

    def readnetvlad(self, name):
        return  torch.tensor(np.loadtxt(self.path_netvlad+name))

    def get_latent_embedding(self, jpg):
        count = 0
        for i in self.jpgdict[jpg]:
            if self.word_filter(i):
                count+=1
                encoded_word = word2encodedword(self.enc, i, 12)
                text, image = self.convert2inputs(i, encoded_word, jpg)
                assert self.model.training is False
                embeds = self.model.get_embedding(image.to(device), text.to(device)).reshape([1,-1])
                try:
                    embeddings = torch.cat([embeddings, embeds], dim = 0)
                except:
                    embeddings = embeds
        if count ==0:
            embeddings = torch.zeros(10, 128).to(device)
        else:
            embeddings = F.pad(embeddings, (0, 0, 0, 10 - embeddings.shape[0]))
        return torch.svd(embeddings)[2].t().reshape(-1)

    def word_filter(self, word):
        if len(word) <= 12 and (not wordskip(word, 3, 3)):
            return 1
        else: return 0

    def convert2inputs(self, word, encoded_word, jpg):
        text_i = torch.from_numpy(encoded_word.todense()).unsqueeze(0)
        im = torch.tensor(cv2.imread(self.pathpatch + jpg[:-4] + "_" + word + ".jpg")).permute(
            2, 0, 1)
        image_i = torch.div(im.float(), 255).unsqueeze(0)
        return text_i, image_i

class Triplet_loaderbh_Textvlad_testing(data.Dataset):
    def __init__(self, jpgklass, jpgdict, itemsperclass, pathnv, path_patches, model, enc):
        self.jpgklass = jpgklass
        self.jpgdict = jpgdict
        self.itemsperclass = itemsperclass
        self.path_netvlad = pathnv
        self.model = model
        self.pathpatch = path_patches
        self.enc = enc
        self.model.eval()
        self.model.to(device)

    def __len__(self):
        return len(self.jpgklass)

    def __getitem__(self, index):
        single_jpg = self.jpgklass[index]
        netvlad = self.readnetvlad(single_jpg).unsqueeze(0)
        embedding = self.get_latent_embedding(str(single_jpg)).unsqueeze(0)
        patch_embeds = embedding
        netvlad_embeds = netvlad
        return patch_embeds.float().to(device), netvlad_embeds.float().to(device), index

    def readnetvlad(self, name):
        return  torch.tensor(np.loadtxt(self.path_netvlad+name))

    def get_latent_embedding(self, jpg):
        count = 0
        for i in self.jpgdict[jpg]:
            if self.word_filter(i):
                count+=1
                encoded_word = word2encodedword(self.enc, i, 12)
                text, image = self.convert2inputs(i, encoded_word, jpg)
                assert self.model.training is False
                embeds = self.model.get_embedding(image.to(device), text.to(device)).reshape([1,-1])
                try:
                    embeddings = torch.cat([embeddings, embeds], dim = 0)
                except:
                    embeddings = embeds
        if count ==0:
            embeddings = torch.zeros(10, 128).to(device)
        else:
            embeddings = F.pad(embeddings, (0, 0, 0, 10 - embeddings.shape[0]))
        return torch.svd(embeddings)[2].t().reshape(-1)

    def word_filter(self, word):
        if len(word) <= 12 and (not wordskip(word, 3, 3)):
            return 1
        else: return 0

    def convert2inputs(self, word, encoded_word, jpg):
        text_i = torch.from_numpy(encoded_word.todense()).unsqueeze(0)
        im = torch.tensor(cv2.imread(self.pathpatch + jpg[:-4] + "_" + word + ".jpg")).permute(
            2, 0, 1)
        image_i = torch.div(im.float(), 255).unsqueeze(0)
        return text_i, image_i



class image_word_triplet_loader_allhard(data.Dataset):
    def __init__(self, jpeg, words, words_sparse, labels, path, ld, soft_positive):
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
        self.message = "Ep {}, Btch {},\n" \
                       "A_I {}, A_W '{}', \n" \
                       "P_I {}, P_W {}, s {:.5f}, \n" \
                       "N_i1 {}, N_w1 {}, s1 {:.3f}, \n" \
                       "N_i2 {}, N_w2 {}, s2 {:.3f}, \n" \
                      "N_i3 {}, N_w3 {}, s3 {:.3f}, \n" \
                      "N_i4 {}, N_w4 {}, s4 {:.3f}, \n" \
                      "N_i5 {}, N_w5 {}, s5 {:.3f}, \n" \
                      "N_i6 {}, N_w6 {}, s6 {:.3f}, \n" \
                      "N_i7 {}, N_w7 {}, s7 {:.3f}, \n" \
                      "N_i8 {}, N_w8 {}, s8 {:.3f}, \n" \
                      "N_i9 {}, N_w9 {}, s9 {:.3f}, \n" \
                      "N_i10 {}, N_w10 {}, s10 {:.3f} \n\n"
        self.epoch = 0
        self.logger = logging.getLogger('dummy')
        self.soft_positive = soft_positive


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
            pos_dist = 0
        else:
            a_label = self.labels[index]
            # dists = self.labels==a_label
            a_lib = self.values[index]
            dists = np.linalg.norm(a_lib - self.values, axis=1) * (self.labels == a_label)
            positive_index = np.argwhere(dists == np.max(dists))[0].item()
            pos_dist = np.max(dists)
        positive_word_indexed = torch.from_numpy(self.words_sparse[positive_index].todense())
        # print(self.im_path + self.jpeg[positive_index][:-4] + "_" + self.words[positive_index] + ".jpg")

        im = torch.tensor(cv2.imread(self.im_path + self.jpeg[positive_index][:-4] + "_" + self.words[positive_index] + ".jpg")).permute(
            2, 0, 1)
        positive_im = torch.div(im.float(), 255)
        positive_patch_name = self.im_path + self.jpeg[positive_index][:-4] + "_" + self.words[positive_index] + ".jpg"

        #Get negative
        neg_ind = []
        pos_ind = []


        if not self.firsttime:
            a_lib = self.values[index]
            scores = np.linalg.norm(a_lib - self.values, axis = 1)
            ind = np.argpartition(scores, 100)[:100]
            min_elements = scores[ind]
            min_elements_order = np.argsort(min_elements)
            ordered_indices = ind[min_elements_order]
            for it, i in enumerate(ordered_indices):
                if self.soft_positive:
                    cond = (self.labels[i] != a_label) and (scores[i] > pos_dist)
                else:
                    cond = self.labels[i] != a_label
                if cond:
                    neg_ind.append(i)
                    if len(neg_ind)==10:
                        x = it
                        break
                else:
                    pos_ind.append(it)
            try:
                first_pos = pos_ind[1]
                last_pos = pos_ind[-1]
                len_pos = len(pos_ind)
            except:
                first_pos = 50
                last_pos = 50
                len_pos = 0

        if len(neg_ind) < 10:
            random_index = random.randint(0, len(self.labels) - 1)
            while len(neg_ind) < 10:
                while (self.labels[index] == self.labels[random_index]) or (
                        levenshteinDistance(self.words[index], self.words[random_index]) > self.ld):
                    # print("searching")
                    random_index = random.randint(0, len(self.labels) - 1)
                neg_ind.append(random_index)
                x = 0
                first_pos = 0
                last_pos = 0
                len_pos = 0

        negative_word_indexed = []
        negative_im = []
        negative_patch_name = []
        for negative_index in neg_ind:
            negative_word_indexed.append(torch.from_numpy(self.words_sparse[negative_index].todense()))
            im = torch.tensor(cv2.imread(self.im_path + self.jpeg[negative_index][:-4] + "_" + self.words[negative_index] + ".jpg")).permute(
                2, 0, 1)
            negative_im.append(torch.div(im.float(), 255))
            negative_patch_name.append(self.im_path + self.jpeg[negative_index][:-4] + "_" + self.words[negative_index] + ".jpg")

        if not self.firsttime:
            self.logger.warning(self.message.format(self.epoch, self.batch, index, self.words[index],
                                                positive_index, self.words[positive_index], pos_dist,
                                                neg_ind[0], self.words[neg_ind[0]], scores[neg_ind[0]],
                                                neg_ind[1], self.words[neg_ind[1]], scores[neg_ind[1]],
                                                neg_ind[2], self.words[neg_ind[2]], scores[neg_ind[2]],
                                                neg_ind[3], self.words[neg_ind[3]], scores[neg_ind[3]],
                                                neg_ind[4], self.words[neg_ind[4]], scores[neg_ind[4]],
                                                neg_ind[5], self.words[neg_ind[5]], scores[neg_ind[5]],
                                                neg_ind[6], self.words[neg_ind[6]], scores[neg_ind[6]],
                                                neg_ind[7], self.words[neg_ind[7]], scores[neg_ind[7]],
                                                neg_ind[8], self.words[neg_ind[8]], scores[neg_ind[8]],
                                                neg_ind[9], self.words[neg_ind[9]], scores[neg_ind[9]],
                                                ))


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

    def increaseepoch(self):
        self.epoch += 1
        self.batch = 0

    def increasebatch(self):
        self.batch +=1

class image_word_triplet_loader_batchhard(data.Dataset):
    def __init__(self, jpeg, words, words_sparse, labels, path, itemsperclass):
        self.labels = np.array(labels)
        self.words = words
        self.im_path = path
        self.jpeg = np.array(jpeg)
        self.words_sparse = words_sparse
        self.epoch = 0
        self.logger = logging.getLogger('dummy')
        self.itemsperclass = itemsperclass

    def __len__(self):
        return self.labels[-1]

    def __getitem__(self, index):
        indices = np.argwhere(self.labels == index).squeeze()
        indices_per_batch = sample(list(indices), self.itemsperclass)
        if len(indices_per_batch)<self.itemsperclass:
            raise ValueError("items less than items per class")

        for i in indices_per_batch:
            patch_i = torch.from_numpy(self.words_sparse[i].todense()).unsqueeze(0)
            im = torch.tensor(cv2.imread(self.im_path + self.jpeg[i][:-4] + "_" + self.words[i] + ".jpg")).permute(
                2, 0, 1)
            image_i = torch.div(im.float(), 255).unsqueeze(0)
            try:
                image = torch.cat([image, image_i], dim = 0)
                text = torch.cat([text, patch_i], dim = 0)
                words.append(self.words[i])
            except:
                image = image_i
                text = patch_i
                words = [self.words[i]]


        return image.float(), text.float(), index, words

def textvlad_mining(current_klass, all_positive, embeds, positive, negative, norms):
     return None