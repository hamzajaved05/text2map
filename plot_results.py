import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2

with open('util/dl_logs/resultsmatched_train.pickle', 'rb') as e:
	results = pickle.load(e)

while True:
	index = np.random.randint(0, len(results))
	plt.figure(0)
	plt.clf()
	plt.subplot(121)
	plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + results[index][0]))
	plt.subplot(122)
	plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + results[index][1]))


	# fig.suptitle(str(plotteddist[indices]))

	plt.draw()
	plt.pause(1e-6)
	input()
