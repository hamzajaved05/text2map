"""
Author: Hamza
Dated: 15.04.2019
Project: texttomap

"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
import pickle
from scipy.sparse import csc_matrix,hstack



def write_dict_to_txt(dict, name):
	trainfile = open("../Dataset_processing/"+name+".txt", 'w')
	for jpg in dict.keys():
		trainfile.write(str(jpg) + "\n")
		for word in dict[jpg]:
			if isinstance(word, str):
				trainfile.write(str(word) + "\n")
	trainfile.close()

def removeemptykeys(dict):
	for jpg in dict.copy():
		if len(dict[jpg]) == 0:
			del dict[jpg]


def learn_encoding(wordsarray):
	letters = []
	lettercount = 0
	for word in wordsarray:
		for alphabs in word.lower():
			if alphabs not in letters:
				letters.append(alphabs)
				lettercount+=1
	letters = sorted(letters)
	letters = np.array(letters).reshape(-1,1)
	enc = OneHotEncoder(sparse=True)
	enc.fit(letters)
	return enc, lettercount

def word2enc(wordsarray):
	enc, letters = learn_encoding(wordsarray)
	dict = {}

	for word in wordsarray:
		temp = list(word.lower())
		dummy = enc.transform(np.array(temp).reshape(-1,1)).transpose()
		x = np.concatenate((dummy.todense(), np.zeros((dummy.shape[0], 12-dummy.shape[1]))),axis = 1)
		dict[word] = csc_matrix(x)
	return dict

def scoring_words(dict,threshold, lower_value, higher_value):
	dict = jpg2word(dict)
	wordrarray = [y for y in dict.keys()]
	sd = np.empty((len(dict), 2))
	sdthreshold = 0.002
	for itera, words in enumerate(dict.keys()):
		jpgs = []
		jpg_angle_2 = []
		for jpgs in dict[words]:
			jpg_angle_2.append(open("../Dataset_processing/jpegs/" + jpgs[:-3] + "txt").read().split()[11:13])
		jpg_angle_2 = np.array(jpg_angle_2).astype(float)
	sd[itera] = np.std(jpg_angle_2, axis=0)
	# np.count_nonzero(sd[:, 0] == 0)
	freq = [len(dict[i]) for i in dict.keys()]
	sd_scalar = np.sum(sd, axis=1)
	high_score = np.argwhere(sd_scalar < threshold)
	score = [(higher_value if i in high_score else lower_value) for i, j in enumerate(wordrarray)]
	dict2 = {}
	for i,j in enumerate(wordrarray):
		dict2[j] = score[i]
	return dict2

def jpg2word(jpgdict):
	worddict = {}
	for jpgname in jpgdict.keys():
		for word in jpgdict[jpgname]:
			if not word in worddict:
				worddict[word] = [jpgname]
			else:
				worddict[word].append(jpgname)
	return worddict

def lev_proximal_strings(string,stringarray):
	levenshteins = Levenshtein()
	proximalarray = [string]
	for i in stringarray:
		ld = levenshteins.get_raw_score(string,i)
		if i<=1:
			proximalarray.append(i)
	if len(proximalarray == 1):
		proximalarray = []
	return proximalarray


def getproximal(jpg_dict,write,writenamed = "00"):
	sums = []
	Proximaljpgs = {}
	jpgarray = [y for y in jpg_dict.keys()]
	jpg_angle = np.empty((len(jpgarray), 2))
	for itera, jpgname in enumerate(jpgarray):
		jpg_angle[itera] = np.array(open("../Dataset_processing/jpegs/" + jpgname[:-3] + "txt").read().split()[11:13])

	for itera, jpgname in enumerate(jpgarray):
		dif = np.abs(jpg_angle - [jpg_angle[itera]])
		dist = np.linalg.norm(dif, axis=1)
		arg = np.argwhere(dist < 0.0003)
		# sums.append(np.sum(dist<0.00005))
		Proximaljpgs[jpgname] = [jpgarray[int(i)] for i in arg if not (i == itera)]
		# Proximaljpgs[jpgname] = [jpgarray[int(i)] for i in arg if not i==itera]

		if (itera + 1) % 1000 == 0:
			print("%d / %d Done for Proximal JPGS !" % (itera + 1, len(jpgarray)))

	if write:
		with open("../Dataset_processing/split/" + writenamed + ".pickle","wb") as F:
			pickle.dump(Proximaljpgs,F)
	return Proximaljpgs


def getproximalwords(Proximaljpgs, jpg_dict, write, writenamed = "00"):
	proximal_word_dict = {}
	jpgarray = Proximaljpgs.keys()
	for itera, jpgname in enumerate(jpgarray):
		proximal_word_dict[jpgname] = []
		for jpgs in Proximaljpgs[jpgname]:
			for word in jpg_dict[jpgs]:
				if word in proximal_word_dict[jpgname]:
					continue
				else:
					proximal_word_dict[jpgname].append(word)

		print("Done = " + str(itera) + " / " + str(len(jpgarray)))

	if write:
		with open("../Dataset_processing/split/" + writenamed + ".pickle", "wb") as F:
			pickle.dump(proximal_word_dict, F)

		write_dict_to_txt()
	return proximal_word_dict(proximal_word_dict,writenamed)


# def getproximalwords(Proximaljpgs, jpg_dict, write, writenamed = "00"):
# 	proximal_word_dict = {}
# 	jpgarray = Proximaljpgs.keys()
# 	for itera, jpgname in enumerate(jpgarray):
# 		proximal_word_dict[jpgname] = []
# 		for jpgs in Proximaljpgs[jpgname]:
# 			for word in jpg_dict[jpgs]:
# 				if word in proximal_word_dict[jpgname]:
# 					continue
# 				else:
# 					proximal_word_dict[jpgname].append(word)
#
# 		print("Done = " + str(itera) + " / " + str(len(jpgarray)))
#
# 	if write:
# 		with open("../Dataset_processing/split/" + writenamed + ".pickle", "wb") as F:
# 			pickle.dump(proximal_word_dict, F)
#
# 		write_dict_to_txt()
# 	return proximal_word_dict(proximal_word_dict,writenamed)