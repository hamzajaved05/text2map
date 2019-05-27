"""
Author: Hamza
Dated: 05.04.2019
Project: texttomap

"""
import pickle

import numpy as np
from py_stringmatching.similarity_measure.overlap_coefficient import OverlapCoefficient
from updatelibrary import jpg_dict_lib
from utilities import write_dict_to_txt as WJTT
from wordlib import word_dict_lib

inputfilename = "train00"
refresh = False
remove_empty = True
write_removed = False
write_named = "train0"
error_account_proximal = True

jpg_dict = jpg_dict_lib(path="../Dataset_processing/" + inputfilename + ".txt")
jpgarray = [y for y in jpg_dict.keys()]
word_dict = word_dict_lib(path="../Dataset_processing/" + inputfilename + ".txt")

# if error_account_proximal:
# 	wordsizes = np.array([len(i) for i in word_dict.keys()])
# 	for wordsize in wordsizes:
# 		temp = wordsizes == wordsize


if refresh:
    sums = []
    Proximaljpgs = {}
    jpg_angle = np.empty((len(jpgarray), 2))
    for itera, jpgname in enumerate(jpgarray):
        jpg_angle[itera] = np.array(open("../Dataset_processing/jpegs/" + jpgname[:-3] + "txt").read().split()[11:13])

    for itera, jpgname in enumerate(jpgarray):
        dif = np.abs(jpg_angle - [jpg_angle[itera]])
        dist = np.linalg.norm(dif, axis=1)
        arg = np.argwhere(dist < 0.002)
        # sums.append(np.sum(dist<0.00005))
        Proximaljpgs[jpgname] = [jpgarray[int(i)] for i in arg if not (i == itera)]
        # Proximaljpgs[jpgname] = [jpgarray[int(i)] for i in arg if not i==itera]

        if (itera + 1) % 5000 == 0:
            print("%d / %d Done " % (itera + 1, len(jpgarray)))

    jpg_dict1 = jpg_dict
    countremoved = 0
    oc = OverlapCoefficient()
    # limiter = 0
    removedwords = []
    for word in word_dict.keys():
        # limiter+=1
        # if limiter == 5:
        # 	break
        for jpg in word_dict[word]:
            if oc.get_raw_score(word_dict[word], Proximaljpgs[jpg]) == 0:
                # print(word_dict[word],Proximaljpgs[jpg])
                # print(jpg_dict1[jpg], word)
                jpg_dict1[jpg].remove(word)
                removedwords.append(word)
                print(jpg)
                countremoved += 1

    if remove_empty:
        for jpg in jpg_dict1.copy():
            if len(jpg_dict1[jpg]) == 0:
                del jpg_dict1[jpg]
        print("Images remaining = " + str(len(jpg_dict1)))

    if write_removed:
        WJTT(jpg_dict1, write_named)

    with open("Storeddata/word_dict_removed.pickle", "wb") as f:
        pickle.dump([jpg_dict1, countremoved], f)
