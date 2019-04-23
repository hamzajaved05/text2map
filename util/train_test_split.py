#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:26:13 2019

@author: presag3l
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from utilities import write_dict_to_txt as WDTT
def traintestsplit(path = "../Dataset_processing/"):
    lines = open(path+"Log_0068.txt").read().splitlines()
    testlines = open(path + "testjpgnames.txt").read().splitlines()
    jpg_dict = {};
    for line in lines:
        if 'jpg'in line:
            jpg_dict[line]= [];
            dummy = line;
        else:
            jpg_dict[dummy].append(line)

    # dummy = pd.DataFrame.from_dict(jpg_dict,orient='index')
    # print("here1")

    # train, test= train_test_split(dummy,test_size=0.01)
#    traindict = train.to_dict(orient= 'index')
#    testdict = test.to_dict(orient= 'index')

    traindict = {}
    testdict = {}

    for i in jpg_dict.keys():
        if i in testlines:
            testdict[i] = jpg_dict[i]
        else:
            traindict[i] = jpg_dict[i]

    WDTT(traindict,"train02")
    WDTT(testdict,"test02")

    # trainfile = open(path+"train02.txt",'w')
    # testfile = open(path+"test02.txt",'w')
    # # print("here")
    #
    # for jpg, wordstuple in train.iterrows():
    #     trainfile.write(str(jpg)+"\n")
    #     for word in wordstuple:
    #         if isinstance(word,str):
    #             trainfile.write(str(word)+"\n")
    #
    # for jpg, wordstuple in test.iterrows():
    #     testfile.write(str(jpg)+"\n")
    #     for word in wordstuple:
    #         if isinstance(word,str):
    #             testfile.write(str(word)+"\n")
    # trainfile.close()
    # testfile.close()
    return True