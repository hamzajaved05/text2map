"""
Author: Hamza
Dated: 05.04.2019
Project: texttomap

"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from util.utilities import getproximal, getproximalwords, readcsv2jpgdict, convertjpg2word, long_lat_dict
import argparse
import os


parser = argparse.ArgumentParser(description="Generating proximal")
parser.add_argument("--csvpath", type=str, help="Path for Image patches")
parser.add_argument("--writepath", type=str, help="Path of pickle file")
args = parser.parse_args()

try:
    csvpath = args.csvpath
except:
    csvpath = "68_data.csv"
    writepath = "fin_temp/"

index = csvpath[:2]

print("Accessing file for jpg dictionary update!!")
jpg_dict = readcsv2jpgdict(csvpath)
word_dict = convertjpg2word(jpg_dict)
loadproximaljpg = False
loadproximalword = False
plot_proximal = False

long_dict, lat_dict = long_lat_dict(csvpath)

if not loadproximaljpg:
    writepath = os.path.join(args.writepath, index+"_proximaljpgs.pickle")
    Proximaljpgs = getproximal(jpg_dict, True, writepath, long_dict, lat_dict, index)
else:
    with open(os.path.join(writepath, "proximaljpgs.pickle"), "rb") as f:
        Proximaljpgs = pickle.load(f)

if plot_proximal:
    wordarray = [i for i in Proximaljpgs.keys()]
    while True:
        fig = plt.figure(2)
        plt.clf()
        indices = np.random.randint(0, len(wordarray))
        indices2 = np.random.randint(0, len(Proximaljpgs[wordarray[indices]]))
        indices3 = np.random.randint(0, len(Proximaljpgs[wordarray[indices]]))
        indices4 = np.random.randint(0, len(Proximaljpgs[wordarray[indices]]))
        indices5 = np.random.randint(0, len(Proximaljpgs[wordarray[indices]]))
        indices6 = np.random.randint(0, len(Proximaljpgs[wordarray[indices]]))
        plt.subplot(231)
        plt.imshow(Image.open("Dataset_processing/jpegs/0068/" + wordarray[indices]))
        plt.subplot(232)
        plt.imshow(Image.open("Dataset_processing/jpegs/0068/" + Proximaljpgs[wordarray[indices]][indices2]))
        plt.subplot(233)
        plt.imshow(Image.open("Dataset_processing/jpegs/0068/" + Proximaljpgs[wordarray[indices]][indices3]))
        plt.subplot(234)
        plt.imshow(Image.open("Dataset_processing/jpegs/0068/" + Proximaljpgs[wordarray[indices]][indices4]))
        plt.subplot(235)
        plt.imshow(Image.open("Dataset_processing/jpegs/0068/" + Proximaljpgs[wordarray[indices]][indices5]))
        plt.subplot(236)
        plt.imshow(Image.open("Dataset_processing/jpegs/0068/" + Proximaljpgs[wordarray[indices]][indices6]))

        plt.draw()
        plt.pause(1e-6)
        input()

if not loadproximalword:
    writepath = os.path.join(args.writepath, index+"_proximalwords.pickle")
    proximal_word_dict = getproximalwords(Proximaljpgs, jpg_dict, True, writepath)
else:
    with open(os.path.join(writepath, "proximalwords.pickle"), "rb") as f:
        proximal_word_dict = pickle.load(f)

klass = {}
count = 0
lev = Levenshtein()
batch = 0
for itera, jpg_source in enumerate(jpg_dict.keys()):
    word_source_array = [i for i in jpg_dict[jpg_source]]
    for itera2, word_source in enumerate(word_source_array):
        klass[str(count)] = []
        for word_target in proximal_word_dict[jpg_source]:
            assert isinstance(word_target, str)
            if (lev.get_raw_score(word_source, word_target) < 2):
                dummy = list(set(word_dict[word_target]).intersection(Proximaljpgs[jpg_source]))
                for i in dummy:
                    klass[str(count)].append(jpg_source)
                    klass[str(count)].append(word_source)
                    klass[str(count)].append(i)
                    klass[str(count)].append(word_target)
                    try:
                        jpg_dict[i].remove(word_target)
                    except:
                        pass

        count = count + 1
        try:
            jpg_dict[jpg_source].remove(word_source)
        except:
            pass
    if (itera+1)%1000 == 0:
        print("Building classes ... >> Done " + str(itera) + " / " + str(len(jpg_dict.keys())))


keys = [i for i in klass.keys()]
for i in keys:
    if len(klass[i]) == 0:
        del klass[i]

with open(os.path.join(writepath, index + "_klasses"), "wb") as F:
    pickle.dump(klass, F)

# keys = [i for i in klass.keys()]
# for i in keys:
#     if len(klass[i]) == 0:
#         del klass[i]
#
# with open("fin_temp/finals.pickle", "rb") as f:
#     klass = pickle.load(f)



# with open("temp2.pickle","rb") as f:
# 	proximal_word_dict = pickle.load(f)
# ite = 0
# for jpg in jpg_dict.keys():
# 	for word in word_dict[jpg]:
# 		klass[str(ite)] = lps(word,proximal_word_dict[jpg])
