"""
Author: Hamza
Dated: 01.05.2019
Project: texttomap

"""
import os
import cv2
import h5py
from sys import getsizeof

# path = "Dataset_processing/jpeg_patch/"
# for filename in os.listdir(path):
# 	if filename[-6] == "_":
# 		edited = path+filename[:-6]+".jpg"
# 		if not os.path.isfile(edited):
# 			os.rename(path+filename, edited)
# 	if filename[-5] == "_":
# 		edited = path+filename[:-5]+".jpg"
# 		if not os.path.isfile(edited):
# 			os.rename(path + filename, edited)
#

dict = {}

path = "Dataset_processing/jpeg_patch/"
count = 0
for itera, filename in enumerate(os.listdir(path)):
	dict[filename] = cv2.imread(path+filename)
	if itera+1%100 ==0:
		print(itera)
	if itera+1 % 2500== 0:
		with h5py.File('Dataset_processing/patches/data'+str(count)+'.h5', 'w') as f:
			dset = f.create_dataset("dataset", data=dict)
		print("saved "+str(count))
		count+=1
		del dict
		dict=  {}
