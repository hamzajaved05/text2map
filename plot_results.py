import pickle

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util.updatelibrary import jpg_dict_lib as Reader

# matplotlib.use("TkAgg")

import cv2

with open('util/dl_logs/03_test_result_confidenceTriplet.pickle', 'rb') as e:
    results = pickle.load(e)

with open("util/hc_logs/log01.pickle", "rb") as f:
    [testjpg1, libjpg1, confidence1, displacement1] = pickle.load(f)

with open("util/hc_logs/log02.pickle", "rb") as f:
    [testjpg2, libjpg2, confidence2, displacement2] = pickle.load(f)

with open("util/hc_logs/log03.pickle", "rb") as f:
    [testjpg3, libjpg3, confidence3, displacement3] = pickle.load(f)

with open("util/hc_logs/log04.pickle", "rb") as f:
    [testjpg4, libjpg4, confidence4, displacement4] = pickle.load(f)

with open("util/hc_logs/log05.pickle", "rb") as f:
    [testjpg5, libjpg5, confidence5, displacement5] = pickle.load(f)

with open("util/hc_logs/log06.pickle", "rb") as f:
    [testjpg6, libjpg6, confidence6, displacement6] = pickle.load(f)

keys = list(results.keys())

jpg_dict_test = Reader(path='Dataset_processing/test03.txt')

displacement = []
RAD = 0.000008998719243599958;
for i in keys:
    # if results[i][0][1] - results[i][1][1] > 0.01:
    testfile = open("Dataset_processing/raw_image_folder/" + i[:-3] + "txt").read().split()[11:14]
    libfile = open("Dataset_processing/raw_image_folder/" + results[i][0][0][:-3] + "txt").read().split()[11:14]
    hor = math.sqrt(math.pow(float(testfile[0]) - float(libfile[0]), 2)
                    + math.pow(float(testfile[1]) - float(libfile[1]), 2)) / RAD;
    displacement.append(math.sqrt(math.pow(hor, 2) + math.pow(float(testfile[2]) - float(libfile[2]), 2)))

print(len(displacement))
plt.figure(0)
lists, bins, patches = plt.hist([displacement, displacement1, displacement2, displacement3, displacement4, displacement5,
                                 displacement6], bins=1000, range=(0, 500), cumulative=True, histtype="step", normed=True ,
                                color=['r','b','g', 'y', 'c', 'm', 'k'])

while True:
    index = np.random.randint(0, len(keys))
    plt.figure(1)
    plt.clf()
    plt.subplot(221)
    plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + keys[index]))
    plt.axis('off')
    plt.title('Query Image\n' + jpg_dict_test[keys[index]].__str__(), fontsize=8)

    plt.subplot(222)
    plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + results[keys[index]][0][0]))
    plt.axis('off')
    plt.title('Best Match\n' + str(results[keys[index]][0][1] * 100), fontsize=8)
    plt.subplot(223)
    plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + results[keys[index]][1][0]))
    plt.axis('off')
    plt.title('Runner up\n' + str(results[keys[index]][1][1] * 100), fontsize=8)
    plt.subplot(224)
    plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + results[keys[index]][2][0]))
    plt.axis('off')
    plt.title('Bronze\n' + str(results[keys[index]][2][1] * 100), fontsize=8)
    # fig.suptitle(str(plotteddist[indices]))
    plt.waitforbuttonpress()
# plt.draw()
# plt.pause(1e-4)
# input()
