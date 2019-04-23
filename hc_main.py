#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 02:19:09 2019

@author: presag3l
"""
from matching import matching
from updatelibrary import jpg_dict_lib
import math
import argparse
import pickle
from utilities import scoring_words

parser = argparse.ArgumentParser(description='Text to map - classical method')
parser.add_argument("trainfile",type = str, help = "Name of train file")
parser.add_argument("testfile",type = str, help = "Name of test file")
parser.add_argument("logfile",type = str, help = "Name of log files")
args = parser.parse_args()

jpg_dict = jpg_dict_lib(path ="../Dataset_processing/" + args.trainfile)
test_dict = jpg_dict_lib(path  = "../Dataset_processing/"+args.testfile)
print("No of Jpegs : "+str(len(jpg_dict))+"\nNo of Test queries : "+str(len(test_dict)))
libjpg = []
testjpg = []
confidences = []
N = len(test_dict.keys())
progress = 0
threshold = 1
lowervalue = 0.5
highervalue = 1

scoring_dict = scoring_words(jpg_dict,threshold=threshold, lower_value=lowervalue, higher_value= highervalue)
print("Scoring dict with threshold of {}, lower than threshold {}, higher than threshold {}" .format(threshold,lowervalue,highervalue))
for jpgname in test_dict.keys():
    bestmatch, confidence = matching(jpg_dict, stringarray=test_dict[jpgname], scoring_dict=scoring_dict)
    libjpg.append(bestmatch)
    confidences.append(confidence)
    testjpg.append(jpgname);
    progress += 1;
    if progress%5 ==0:
        print("Progress --> %d/%d !" % (progress,N))
    # if len(jpgname) == 200:
    #     break
print("Step #1 DONE --> %d/%d !" % (progress,N))
displacement = [];
RAD = 0.000008998719243599958;
for i in range(len(testjpg)):
    testfile = open("../Dataset_processing/jpegs/"+testjpg[i][:-3]+"txt").read().split()[11:14]
    libfile = open("../Dataset_processing/jpegs/"+libjpg[i][:-3]+"txt").read().split()[11:14]
    hor = math.sqrt(math.pow(float(testfile[0]) - float(libfile[0]), 2) + math.pow(float(testfile[1]) - float(libfile[1]), 2)) / RAD;
    displacement.append(math.sqrt (math.pow(hor,2) + math.pow(float(testfile[2])-float(libfile[2]) , 2)))
print("Completed!!\nExporting file with names" + args.logfile)
with open("plogs/"+ args.logfile , 'wb') as f:
    pickle.dump([testjpg,libjpg,confidences,displacement], f)
with open("plogs/keep.txt", "a") as file:
    file.write(args.trainfile +" ... "+args.testfile+" ... "+ args.logfile+" !\n\n")
