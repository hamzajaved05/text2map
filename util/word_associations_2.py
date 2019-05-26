"""
Author: Hamza
Dated: 05.04.2019
Project: texttomap

"""

from util.wordlib import word_dict_lib
import numpy as np
import pickle
import matplotlib.pyplot as plt
from py_stringmatching.similarity_measure.overlap_coefficient import OverlapCoefficient
from util.utilities import write_dict_to_txt as WJTT,lev_proximal_strings as lps,getproximal, getproximalwords,jpg2word
from util.updatelibrary import jpg_dict_lib
from PIL import Image
from py_stringmatching.similarity_measure.levenshtein import Levenshtein



inputfilename = "train03"
jpg_dict = jpg_dict_lib(path ="Dataset_processing/" + inputfilename+".txt")
word_dict = word_dict_lib(path ="Dataset_processing/" + inputfilename+".txt")

loadproximaljpg = True
loadproximalword = True
plot_proximal = False

if not loadproximaljpg:
	Proximaljpgs = getproximal(jpg_dict,True,"proximaljpgs03")
else :
	with open("Dataset_processing/split/proximaljpgs03.pickle","rb") as f:
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
	proximal_word_dict = getproximalwords(Proximaljpgs,jpg_dict,True,"proximalwords03")
else:
	# proximal_word_dict = jpg_dict_lib("Dataset_processing/split/proximalwords03.txt")
	with open("Dataset_processing/split/proximalwords03.pickle", "rb") as f:
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
			if (lev.get_raw_score(word_source,word_target) < 2):
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

		count = count+1
		try:
			jpg_dict[jpg_source].remove(word_source)
		except:
			pass



	# if itera == 5000:
	# 	with open("../Dataset_processing/split/word_association_batch_00.pickle", "wb") as F:
	# 		pickle.dump(klass, F)
	# 	klass = {}
	# 	batch+=1
	print("Building classes ... >> Done "+str(itera) +" / "+ str(len(jpg_dict.keys())))

with open("Dataset_processing/split/word_association_batch_03.pickle", "wb") as F:
	pickle.dump(klass, F)


with open("Dataset_processing/split/word_association_batch_03.pickle","rb") as f:
	klass = pickle.load(f)

keys = [i for i in klass.keys()]
for i in keys:
	if len(klass[i]) == 0:
		del klass[i]



# with open("temp2.pickle","rb") as f:
# 	proximal_word_dict = pickle.load(f)
# ite = 0
# for jpg in jpg_dict.keys():
# 	for word in word_dict[jpg]:
# 		klass[str(ite)] = lps(word,proximal_word_dict[jpg])