import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import  sample
from collections import Counter
# from collections import namedtuple

# encoding = namedtuple("encoding", ["id", "embedding"])

with open("training_data_pytorch04.pickle", "rb") as a:
    [klass, _, words, jpgs, _, _] = pickle.load(a)
jpgs = np.array(jpgs)
klass = np.array(klass)

#
# while True:
#     clas = np.argwhere(np.array(klass) == sample(klass, 1))
#     jpeg = list(jpgs[sample(list(clas.reshape(-1)), 4)])
#     plt.figure(1)
#     plt.clf()
#     plt.subplot(221)
#     plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + jpeg[0]))
#     plt.axis('off')
#     # plt.title('Query Image\n' + jpg_dict_test[keys[index]].__str__(), fontsize=8)
#
#     plt.subplot(222)
#     plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + jpeg[1]))
#     plt.subplot(223)
#     plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + jpeg[2]))
#     plt.axis('off')
#     plt.subplot(224)
#     plt.imshow(cv2.imread("Dataset_processing/jpegs/0068/" + jpeg[3]))
#     plt.axis('off')
#     # fig.suptitle(str(plotteddist[indices]))
#     plt.waitforbuttonpress()

# def same_class_jpgs()

from util.updatelibrary import jpg_dict_lib as Reader
jpg_dict_train = Reader(path='Dataset_processing/train03.txt')

def im_triplet(jpg, image_lib, labels):
    dumm =jpg
    ind = np.argwhere(dumm == image_lib).reshape(-1)
    classes = labels[ind]
    indi = np.array([])
    for i in classes:
        indi = np.concatenate((indi, image_lib[np.argwhere(i == labels).reshape(-1)]), axis=0)
    while True:
        guess = sample(list(image_lib), 1)[0]
        if guess not in indi:
            break
    return [1 if len(classes) > 1 else 0, jpg, Counter(indi).most_common(1)[0][0], guess]

jpg_list = list(set(jpgs))
triplet = []

for iter, anc in enumerate(jpg_list):
    trip = im_triplet(anc, image_lib=jpgs, labels=klass)
    triplet.append(trip)
    if (iter+1) % 1000 == 0:
        print("{} / {}".format(iter+1, len(jpg_list)))

with open("triplet_pairs.pickle","wb") as q:
    pickle.dump(triplet, q)

with open('lib/final_processed' + str(1).zfill(3) + '.pickle', 'rb') as q:
    [lib, filenames, indice_dict, words0] = pickle.load(q)

with open("training_data_pytorch04.pickle", "rb") as a:
    [klass, words_sparse, words1, jpgs, enc, modes] = pickle.load(a)

def to_jpg_dict(jpgs, words):
    dict = {}
    for itera, name in enumerate(jpgs):
        if not name in dict:
            dict[name] = [words[itera]]
        else:
            dict[name].append(words[itera])
    return dict

def get_ids(jpeg_names, words_string):
    lib_dict = []
    for itera, name in enumerate(jpeg_names):
        lib_dict.append(name+words_string[itera])
    return lib_dict

jpg_word_dict = to_jpg_dict(jpgs, words1)
lib_ids = get_ids(filenames.tolist(), words0.tolist())

with open("preprocess.pickle","wb") as q:
    pickle.dump([jpg_word_dict, lib_ids, lib, triplet], q)
