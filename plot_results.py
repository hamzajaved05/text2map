import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from util.updatelibrary import jpg_dict_lib as Reader

matplotlib.use("TkAgg")

import cv2

with open('util/dl_logs/03_test_result_confidenceT.pickle', 'rb') as e:
	results = pickle.load(e)
keys = list(results.keys())

jpg_dict_test = Reader(path = 'Dataset_processing/test03.txt')


displacement = []
RAD = 0.000008998719243599958;
for i in keys:
	if results[i][0][1]-results[i][1][1]>0.01:
		testfile = open("Dataset_processing/jpegs/0068/" + i[:-3] + "txt").read().split()[11:14]
		libfile = open("Dataset_processing/jpegs/0068/" + results[i][0][0][:-3] + "txt").read().split()[11:14]
		hor = math.sqrt(math.pow(float(testfile[0]) - float(libfile[0]), 2)
						+ math.pow(float(testfile[1]) - float(libfile[1]), 2)) / RAD;
		displacement.append(math.sqrt(math.pow(hor, 2) + math.pow(float(testfile[2]) - float(libfile[2]), 2)))

print(len(displacement))
plt.figure(0)
lists, bins, patches = plt.hist(displacement, bins=500,range = (0,200), cumulative=True,histtype = "step",normed = True)



while True:
	index = np.random.randint(0, len(keys))
	plt.figure(1)
	plt.clf()
	plt.subplot(221)
	plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + keys[index]))
	plt.axis('off')
	plt.title('Query Image\n'+jpg_dict_test[keys[index]].__str__(), fontsize = 8)

	plt.subplot(222)
	plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + results[keys[index]][0][0]))
	plt.axis('off')
	plt.title('Best Match\n'+str(results[keys[index]][0][1]*100), fontsize = 8)
	plt.subplot(223)
	plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + results[keys[index]][1][0]))
	plt.axis('off')
	plt.title('Runner up\n'+str(results[keys[index]][1][1]*100), fontsize = 8)
	plt.subplot(224)
	plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + results[keys[index]][2][0]))
	plt.axis('off')
	plt.title('Bronze\n'+str(results[keys[index]][2][1]*100), fontsize = 8)
	# fig.suptitle(str(plotteddist[indices]))
	plt.waitforbuttonpress()
	# plt.draw()
	# plt.pause(1e-4)
	# input()
