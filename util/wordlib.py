"""
Author: Hamza
Dated: 07.04.2019
Project: texttomap
"""


def word_dict_lib(path="../Dataset_processing/train.txt"):
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
