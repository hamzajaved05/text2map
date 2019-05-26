"""
Author: Hamza
Dated: 01.05.2019
Project: texttomap

"""
import os
import argparse


parser = argparse.ArgumentParser(description='Text to map - Training with image patches and text')
parser.add_argument("patch_path",type = str, help = "Path for Image patches")
args= parser.parse_args()
path = args.patch_path
count = 0
for filename in os.listdir(path):
	if filename[-6] == "_":
		edited = path+filename[:-6]+".jpg"
		if not os.path.isfile(edited):
			os.rename(path+filename, edited)
			count+=1
	if filename[-5] == "_":
		edited = path+filename[:-5]+".jpg"
		if not os.path.isfile(edited):
			os.rename(path + filename, edited)
			count+=1
print("Files renamed : "+ str(count) + " ! ")
