import pickle
import matplotlib.pyplot as plt
import numpy as np


with open("plogs/log01.pickle","rb") as f:
	[testjpg1, libjpg1, confidence1, displacement1] = pickle.load(f)

with open("plogs/log02.pickle","rb") as f:
	[testjpg2, libjpg2, confidence2, displacement2] = pickle.load(f)

with open("plogs/log03.pickle","rb") as f:
	[testjpg3, libjpg3, confidence3, displacement3] = pickle.load(f)

with open("plogs/log04.pickle","rb") as f:
	[testjpg4, libjpg4, confidence4, displacement4] = pickle.load(f)

with open("plogs/log05.pickle","rb") as f:
	[testjpg5, libjpg5, confidence5, displacement5] = pickle.load(f)

with open("plogs/log06.pickle","rb") as f:
	[testjpg6, libjpg6, confidence6, displacement6] = pickle.load(f)
fig= plt.figure(1)
lists, bins,patches = plt.hist([displacement4, displacement5], label = ["1/lev_distance","1/lev_distance^2"], bins=500,cumulative=True,histtype = "step",normed = True, color = ["r","g"])
fig.legend()
# fig.suptitle("Reduced dataset vs Complete dataset")
# plt.title("Accuracy vs distance threshold")
plt.title("Cost function")
plt.xlabel("distance (m)")
plt.ylabel("Percentage Accuracy (%)")


with open("plogs/log03.pickle","rb") as f:
	[testjpg, libjpg, confidence, displacement] = pickle.load(f)

import numpy as np
disp = np.array(displacement)
import matplotlib.pyplot as plt
from PIL import Image
import sys
import pickle
from matching import matching
from updatelibrary import jpg_dict_lib
import math



lines = open("../Dataset_processing/test00.txt").read().splitlines()
test_dict = {};
jpg = []
jpg_dict = jpg_dict_lib()
for string in lines:
    if ".jpg" in string:
        jpg = string;
        test_dict[string] = []
    else:
        test_dict[jpg].append(string)


# confidencethreshold = 0.4

# Z = [x for _,x in sorted(zip(confidence,disp))]
np.corrcoef(np.array([confidence,disp]))

while True:
	plt.figure(1)
	plt.clf()
	disp = np.array(disp)
	plt.hist(disp,bins=100,cumulative=True)
	fig = plt.figure(2)
	plt.clf()
	lowerlimit = 800
	upperlimit = 2000
	plotid = 0
	plottedlib = [libjpg[i] for i,q in enumerate(disp) if (q>lowerlimit and q<upperlimit)]
	plottedtest = [testjpg[i] for i,q in enumerate(disp) if (q>lowerlimit and q<upperlimit)]
	plotteddist = [disp[i] for i,q in enumerate(disp) if (q>lowerlimit and q<upperlimit)]
	plottedconfidence = [confidence[i] for i,q in enumerate(disp) if (q>lowerlimit and q<upperlimit)]

	indices = np.random.randint(0, len(plottedlib))

	plt.subplot(121)
	plt.imshow(Image.open("../Dataset_processing/jpegs/0068/" + plottedtest[indices]))
	print(plottedtest[indices])
	print(test_dict[plottedtest[indices]])
	plt.subplot(122)
	plt.imshow(Image.open("../Dataset_processing/jpegs/0068/" + plottedlib[indices]))
	print(plottedlib[indices])
	print(jpg_dict[plottedlib[indices]])
	print(plottedconfidence[indices])

	fig.suptitle(str(plotteddist[indices]))

	plt.draw()
	plt.pause(1e-6)
	input()

# def onclick(event):
# 	print('press', event.key)
# 	# sys.stdout.flush()
# 	if event.key == 'x':
# 		indices = np.random.randint(0, len(plottedlib))
# 		axs[0].imshow(Image.open("../Dataset_processing/jpegs/0068/" + plottedtest[indices]))
# 		axs[1].imshow(Image.open("../Dataset_processing/jpegs/0068/" + plottedlib[indices]))
#
#
# cid = fig.canvas.mpl_connect('key_press_event', onclick)