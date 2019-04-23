#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:47:14 2019

@author: presag3l
"""
def jpg_dict_lib(path ="../Dataset_processing/train00.txt"):
    print("Accessing file for jpg dictionary update >>" + path + " /")
    lines = open(path).read().splitlines()
    jpg_dict = {};
    jpg_string = []
    for somestr in lines:
        if "jpg" in somestr:
            jpg_dict[somestr] = []
            jpg_string = somestr        
        else:
            jpg_dict[jpg_string].append(somestr)
    print("JPG dictionary Updated !!")
    return jpg_dict