#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:22:29 2019

@author: presag3l
"""

import numpy as np
from py_stringmatching.similarity_measure.levenshtein import Levenshtein


def matching(jpg_dict, stringarray, scoring_dict):
    # jpgs = list(jpg_dict.keys())
    # stringid = 0
    # jpgid = 0
    # score = np.ndarray((len(jpg_dict.keys()),len(stringarray)))
    # for querystring in stringarray:
    #     jpgid = 0
    #     for iterator in jpg_dict.keys():
    #         intraimagescore = np.array([])
    #         for iterator2 in jpg_dict[iterator]:
    #             intraimagescore = np.append(intraimagescore, 1 / (1 +levenshteinDistance(iterator2,querystring))**2)
    #         imagescore = np.array(max(intraimagescore))
    #         score[jpgid,stringid] = imagescore
    #         jpgid += 1
    #     stringid +=1

    # jpg = 0
    # score = 0
    # ME = MongeElkan(sim_func=Levenshtein().get_raw_score)
    # for trainimages in jpg_dict.keys():
    #     dummy = 1/(1+ME.get_raw_score(jpg_dict[trainimages],stringarray))**2
    #     if dummy>score:
    #         score = dummy
    #         jpg = trainimages
    lev = Levenshtein()
    jpgs = list(jpg_dict.keys())
    stringid = 0
    jpgid = 0
    lengthstringarray = len(stringarray)
    score = np.ndarray((len(jpg_dict.keys()), len(stringarray)))
    normalizer = len(stringarray)
    for trainimages in jpg_dict.keys():
        stringid = 0
        for querystrings in stringarray:
            dummyscore = []
            for trainstrings in jpg_dict[trainimages]:
                ld = lev.get_raw_score(trainstrings, querystrings) * scoring_dict[trainstrings]
                if not (ld == 1 and (querystrings in trainstrings or trainstrings in querystrings)):
                    dummyscore.append(1 / (1 + ld / 2))
                    # dummyscore.append(ld*min(len(querystrings),len(trainstrings)))
                else:
                    dummyscore.append(1 / (1 + ld))
                    # dummyscore.append(ld*min(len(querystrings),len(trainstrings)))
            score[jpgid, stringid] = max(np.array(dummyscore))
            # del jpg_dict[trainimages][np.argmax(np.array(dummyscore))]
            stringid += 1
        jpgid += 1
    return jpgs[np.argmax(np.sum(score, axis=1))], ((np.max(np.sum(score, axis=1))) / normalizer)
    # return jpg
