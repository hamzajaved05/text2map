"""
Author: Hamza
Dated: 29.04.2019
Project: texttomap

"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from util.utilities import encoding, word2encdict
from util.utilities import getwords
from util.word_encoding import getklass

print("<< Preprocessing data >>")
reload = False
# if reload:
#     enc_dict, klasses, wordarray = getklass()
#     with open("plogs/pytorch_data_03", 'wb') as f:
#         pickle.dump([enc_dict, klasses, wordarray], f)
# else:
#     with open("Dataset_processing/split/pytorch_data_03", 'rb') as f:
#         [enc_dict, klasses, wordarray] = pickle.load(f)

if reload:
    klasses = getklass()
    with open("Dataset_processing/split/pytorch_data_03", 'wb') as f:
        pickle.dump(klasses, f)
else:
    with open("Dataset_processing/split/pytorch_data_03", 'rb') as f:
        klasses = pickle.load(f)

klasses2 = {}
keys = [key for key in klasses.keys()]
for i in range(0, len(klasses)):
    klasses2[str(i)] = klasses.pop(keys[i])
del klasses, keys

enc, letters = encoding(klasses2, sparsity=True, lowercase=False)

wordsarray = getwords(klasses2)

encodeddicttrain = word2encdict(enc, wordsarray, length=12, lowercase=False)

words = []
klass = []
jpgs = []
for klass_iter in klasses2.keys():
    if len(klasses2[klass_iter]) > 20:
        for word_index in range(1, len(klasses2[klass_iter]), 2):
            if len(klasses2[klass_iter][word_index]) < 12:
                klass.append(int(klass_iter))
                words.append(klasses2[klass_iter][word_index])
                jpgs.append(klasses2[klass_iter][word_index - 1])

modes = [max(set(klasses2[i]), key=klasses2[i].count) for i in klasses2.keys()]

wordssparse = [encodeddicttrain[word_iter] for word_iter in words]

with open("training_data_pytorch04.pickle", "wb") as F:
    pickle.dump([klass, wordssparse, words, jpgs, enc, modes], F)

plot = False
if plot:
    while True:
        fig = plt.figure(2)
        plt.clf()
        indice = np.random.randint(0, len(klasses2["549"]) / 2) * 2
        indices2 = np.random.randint(0, len(klasses2["549"][indice]))
        plt.plot()
        plt.imshow(Image.open("Dataset_processing/jpegs/0068/" + klasses2["549"][indice]))

        plt.draw()
        plt.pause(1e-6)
        input()

print("<< Done >>")
# //////////////////////////////////////
