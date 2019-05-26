from torch.utils import data
import torch
import cv2

class image_word_training_loader(data.Dataset):
  def __init__(self, jpeg, words, words_sparse, labels, path):
        self.labels = labels
        self. words = words
        self.im_path = path
        self.jpeg = jpeg
        self.words_sparse = words_sparse

  def __len__(self):
        return len(self.labels)

  def __getitem__(self, index):
        word_indexed = torch.from_numpy(self.words_sparse[index].todense())
        y = torch.tensor(self.labels[index])
        im = torch.tensor(cv2.imread(self.im_path+self.jpeg[index][:-4]+"_"+self.words[index] + ".jpg")).permute(2,0,1)
        return torch.div(im.float(),255), word_indexed.float(), y.float()