#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:26:13 2019

@author: presag3l
"""
import pandas as pd
from util.utilities import write_dict_to_txt as WDTT


testlines = open("Dataset_processing/testjpgnames.txt").read().splitlines()
data = data = pd.read_csv("Dataset_processing/68_data.csv", sep = ", ")
jpg_dict = {};
for index, row in data.iterrows():
    if row["imagesource"] in jpg_dict:
        jpg_dict[row["imagesource"]].append(row["text"])
    else:
        jpg_dict[row["imagesource"]] = [row["text"]]

traindict = {}
testdict = {}

for i in jpg_dict.keys():
    if i in testlines:
        testdict[i] = jpg_dict[i]
    else:
        traindict[i] = jpg_dict[i]

WDTT(traindict,"train03")
WDTT(testdict,"test03")
