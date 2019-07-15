"""
Author: Hamza
Dated: 15.04.2019
Project: texttomap

"""
import collections
import pickle
import math
import numpy as np
# from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from scipy.sparse import csc_matrix
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def write_dict_to_txt(dict, path):
    trainfile = open(path, 'w')
    for jpg in dict.keys():
        trainfile.write(str(jpg) + "\n")
        for word in dict[jpg]:
            if isinstance(word, str):
                trainfile.write(str(word) + "\n")
    trainfile.close()


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def removeemptykeys(dict):
    for jpg in dict.copy():
        if len(dict[jpg]) == 0:
            del dict[jpg]


# def learn_encoding(wordsarray):
# 	letters = []
# 	lettercount = 0
# 	for word in wordsarray:
# 		for alphabs in word.lower():
# 			if alphabs not in letters:
# 				letters.append(alphabs)
# 				lettercount+=1
# 	letters = sorted(letters)
# 	letters = np.array(letters).reshape(-1,1)
# 	enc = OneHotEncoder(sparse=True)
# 	enc.fit(letters)
# 	return enc, lettercount
def encoding(klasses, sparsity=True, lowercase=True):
    letters = []
    for klass_ids in klasses.keys():
        for itera in range(1, len(klasses[klass_ids]), 2):
            if lowercase:
                for alphabet in klasses[klass_ids][itera].lower():
                    if not alphabet in letters:
                        letters.append(alphabet)
            else:
                for alphabet in klasses[klass_ids][itera]:
                    if not alphabet in letters:
                        letters.append(alphabet)
    letters = sorted(letters)
    letters = np.array(letters).reshape(-1, 1)
    enc = OneHotEncoder(sparse=sparsity)
    enc.fit(letters)
    return enc, letters


def word2encdict(enc, wordsarray, length, lowercase=True):
    dict = {}
    for word in wordsarray:
        if lowercase:
            temp = list(word.lower())
        else:
            temp = list(word)
        dummy = enc.transform(np.array(temp).reshape(-1, 1)).transpose()
        x = np.concatenate((dummy.todense(), np.zeros((dummy.shape[0], length - dummy.shape[1]))), axis=1)
        dict[word] = csc_matrix(x)
    return dict


def getwords(klasses):
    words = []
    for itera, key in enumerate(klasses.keys()):
        for id in range(1, len(klasses[key]), 2):
            if len(klasses[key][id]) < 12:
                if not klasses[key][id] in words: words.append(klasses[key][id])
    return words


def scoring_words(dict, threshold, lower_value, higher_value):
    dict = jpg2word(dict)
    wordrarray = [y for y in dict.keys()]
    sd = np.empty((len(dict), 2))
    sdthreshold = 0.002
    for itera, words in enumerate(dict.keys()):
        jpgs = []
        jpg_angle_2 = []
        for jpgs in dict[words]:
            jpg_angle_2.append(open("Dataset_processing/jpegs/" + jpgs[:-3] + "txt").read().split()[11:13])
        jpg_angle_2 = np.array(jpg_angle_2).astype(float)
    sd[itera] = np.std(jpg_angle_2, axis=0)
    # np.count_nonzero(sd[:, 0] == 0)
    freq = [len(dict[i]) for i in dict.keys()]
    sd_scalar = np.sum(sd, axis=1)
    high_score = np.argwhere(sd_scalar < threshold)
    score = [(higher_value if i in high_score else lower_value) for i, j in enumerate(wordrarray)]
    dict2 = {}
    for i, j in enumerate(wordrarray):
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


def lev_proximal_strings(string, stringarray):
    levenshteins = Levenshtein()
    proximalarray = [string]
    for i in stringarray:
        ld = levenshteins.get_raw_score(string, i)
        if i <= 1:
            proximalarray.append(i)
    if len(proximalarray == 1):
        proximalarray = []
    return proximalarray


def getproximal(jpg_dict, write, writenamed="00"):
    sums = []
    Proximaljpgs = {}
    jpgarray = [y for y in jpg_dict.keys()]
    jpg_angle = np.empty((len(jpgarray), 2))
    for itera, jpgname in enumerate(jpgarray):
        jpg_angle[itera] = np.array(open("Dataset_processing/jpegs/0068/" + jpgname[:-3] + "txt").read().split()[11:13])

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
        with open("Dataset_processing/split/" + writenamed + ".pickle", "wb") as F:
            pickle.dump(Proximaljpgs, F)
    return Proximaljpgs


def getproximalwords(Proximaljpgs, jpg_dict, write, writenamed="00"):
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
        with open("Dataset_processing/split/" + writenamed + ".pickle", "wb") as F:
            pickle.dump(proximal_word_dict, F)
        write_dict_to_txt(proximal_word_dict, "Dataset_processing/split/" + writenamed + ".txt")


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

def word2encodedword(enc, word, length):
    if len(word)==0:
        x = np.zeros([62,12])
        x = csc_matrix(x)
    else:
        temp = list(word)
        dummy = enc.transform(np.array(temp).reshape(-1, 1)).transpose()
        x = np.concatenate((dummy.todense(), np.zeros((dummy.shape[0], length - dummy.shape[1]))), axis=1)
        x = csc_matrix(x)
    return x

def readtext2worddict(path="../Dataset_processing/train.txt"):
    print("Accessing file for word dictionary update >>" + path + " /")
    lines = open(path).read().splitlines()
    word_dict = {};
    jpg_string = []
    jpgcounter = 0
    for somestr in lines:
        if "jpg" in somestr:
            # if jpgcounter == 500:
            # 	break
            jpg_string = somestr
            jpgcounter += 1
        elif somestr in word_dict.keys():
            word_dict[somestr].append(jpg_string)
        else:
            word_dict[somestr] = [jpg_string]
    print("Word dictionary Updated !!")
    return word_dict

def readtext2jpgdict(path, filt = None):
    print("Accessing file for jpg dictionary update >>" + path + " /")
    lines = open(path).read().splitlines()
    jpg_dict = {};
    jpg_string = []
    for somestr in lines:
        if "jpg" in somestr:
            jpg_dict[somestr] = []
            jpg_string = somestr
        else:
            if filter is not None:
                if not wordskip(somestr, filt[0], filt[1]):
                    jpg_dict[jpg_string].append(somestr)
            else:
                jpg_dict[jpg_string].append(somestr)

    print("JPG dictionary Updated !!")
    return jpg_dict

def readcsv2jpgdict(path):
    print("Accessing file for jpg dictionary update >>" + path + " /")
    data = pd.read_csv(path, skipinitialspace=True, usecols=['imagesource', 'text']).to_numpy()
    jpg_dict = {};
    for comb in data:
        if str(comb[0]) in jpg_dict.keys():
            jpg_dict[str(comb[0])].append(str(comb[1]))
        else:
            jpg_dict[str(comb[0])] = [str(comb[1])]
    print("JPG dictionary Updated !!")
    return jpg_dict

def wordskip(string, length = 3, setlength = 3):
    if len(string)<length:
        return True
    elif len(set(string))<setlength:
        return True
    else:
        return False

def getdistance(a, b):
    RAD = 0.000008998719243599958;
    hor = math.sqrt(math.pow(float(a[0]) - float(b[0]), 2)
                    + math.pow(float(a[1]) - float(b[1]), 2)) / RAD;
    return hor