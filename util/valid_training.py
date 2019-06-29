import torch
import cv2
import numpy as np
import random


class validation:
    def __init__(self, klass2v, word2v, word_sparse2v, jpgs2v, path,  model, writer):
        self.klass = np.array(klass2v)
        self.words = word2v
        self.sparse = word_sparse2v
        self.jpeg = np.array(jpgs2v)
        self.model = model
        self.im_path = path
        self.writer = writer
        self.step = 0
    #     self.remove_singles()
    #
    # def remove_singles(self):
    #     klass_list = list(set(self.klass))
    #     sums = [np.sum(i==self.klass) for i in klass_list]
    #     for itera, i in enumerate(sums):
    #         if i<2:
    #             indice = np.argwhere(klass_list[itera] == self.klass).item()
    #             del self.klass[indice], self.words[indice], self.sparse[indice], self.jpeg[indice]


    def getinputs(self, index):
        wordi = torch.from_numpy(self.sparse[index].todense())
        im = torch.tensor(cv2.imread(self.im_path + self.jpeg[index][:-4] + "_" + self.words[index] + ".jpg")).permute(
            2, 0, 1)
        anchor_im = torch.div(im.float(), 255)
        return anchor_im.unsqueeze(0), wordi.unsqueeze(0)

    def evaluate(self, lib_embeds, klass_embeds, device, size = None, margin = 0.1):
        print("Running evaluations on size {} !!". format(size))
        self.model.eval()
        if size == None or size > len(self.klass):
            size = len(self.klass)
        indices = random.sample(list(range(len(self.klass))), size)
        ps = []
        ns = []
        nslimit = []
        for i in indices:
            self.step += 1
            p_indices = np.argwhere(klass_embeds == self.klass[i]).squeeze()
            n_indices = np.argwhere(klass_embeds != self.klass[i]).squeeze()
            im, wor = self.getinputs(i)
            embedding = self.model.get_embedding(im.to(device), wor.to(device)).squeeze().detach().cpu().numpy()
            if len(p_indices) == 0 or len(n_indices) == 0:
                continue
            p_norms = np.linalg.norm(lib_embeds[p_indices] - embedding, ord = 2, axis = 1)
            n_norms = np.linalg.norm(lib_embeds[n_indices] - embedding, ord = 2, axis = 1)
            ps.append(p_norms.mean())
            nslimit.append(np.partition(n_norms, 10)[:10].mean())
            ns.append(n_norms.mean())
            self.writer.add_scalars("Validation", {"P_distance_mean": ps[-1],
                                                   "N_distance_mean_all": ns[-1],
                                                   "N_distance_mean_10": nslimit[-1],
                                                   "loss": np.max([0, margin-ns[-1]+ps[-1]]),
                                                   "loss_10": np.max([0, margin - nslimit[-1] + ps[-1]]),
                                                   },
                                    self.step)
        print("Evaluations done !! mean_distance between positive {:.5f} and negative {:.5f}".format(sum(ps)/len(ps),
                                                                                                     sum(ns)/len(ns)))
        return ps, ns

