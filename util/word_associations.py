"""
Author: Hamza
Dated: 04.04.2019
Project: Texttomap
"""
from Levenshtein import levenshteinDistance
import numpy as np
from wordlib import word_dict_lib

path = "../Dataset_processing/train.txt"
lines = open(path).read().splitlines()

word_dict = word_dict_lib()
wordsarray = [y for y in word_dict.keys()]
score = np.full((len(wordsarray), len(wordsarray)), 1)

for iter1, word1 in enumerate(wordsarray):
	for iter2, word2 in enumerate(wordsarray):
		score[iter1, iter2] = levenshteinDistance(word1, word2)
	if iter1 % 50 == 0:
		print("%d/%d done" % (iter1, len(wordsarray)))

wordpairs = []
for i, j in enumerate(wordsarray):
	pos = np.argwhere([score[1, :] == 1])

from updatelibrary import jpg_dict_lib
import math

lines = open("../Dataset_processing/test.txt").read().splitlines()
test_dict = {};
jpg = []
jpg_dict = jpg_dict_lib()
RAD = 0.000008998719243599958;

Angles = []
limit = 500
for i in jpg_dict.keys():
	locations = open("../Dataset_processing/jpegs/" + i[:-3] + "txt").read().split()[11:13]
	Angles.append(locations)
	limit -= 1
	if limit <= 0:
		break

hor = (math.pow(float(testfile[0]) - float(libfile[0]), 2) + math.pow(float(testfile[1]) - float(libfile[1]), 2))