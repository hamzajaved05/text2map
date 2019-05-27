"""
Author: Hamza
Dated: 05.04.2019
Project: texttomap

"""

# from updatelibrary import jpg_dict_lib
import pickle


# import matplotlib.pyplot as plt
# from py_stringmatching.similarity_measure.overlap_coefficient import OverlapCoefficient
# from utilities import write_dict_to_txt as WJTT
# from utilities import lev_proximal_strings as lps
# from utilities import getproximal, getproximalwords,jpg2word
# from updatelibrary import jpg_dict_lib
# from PIL import Image
# from py_stringmatching.similarity_measure.levenshtein import Levenshtein
# from util.utilities import word2enc,learn_encoding

def getklass(path="Dataset_processing/split/word_association_batch_03.pickle", text_net=True):
    with open(path, "rb") as f:
        klass = pickle.load(f)

    keys = [i for i in klass.keys()]
    for i in keys:
        if len(klass[i]) < 5:
            del klass[i]

    wordarray = []
    klasses = []
    klass_length = 0

    klassfiltered = {}
    # filteredcounter = 0
    occurence_array = []
    for itera, i in enumerate(klass.keys()):
        klassfiltered[str(i)] = []
        # indicator = False
        for j in range(0, len(klass[i]), 2):
            dummy = klass[i][j] + klass[i][j + 1]
            if not dummy in occurence_array:
                occurence_array.append(dummy)
                klassfiltered[str(i)].append(klass[i][j])
                klassfiltered[str(i)].append(klass[i][j + 1])
            # indicator = True
        # if indicator:
        # 	filteredcounter+=1
        if (itera + 1) % 500 == 0:
            print("filtering : {}".format(100 * itera / len(klass)))

    keys = [i for i in klassfiltered.keys()]
    for i in keys:
        if len(klassfiltered[i]) < 3:
            del klassfiltered[i]

    # with open("Dataset_processing/split/klassfiltered.pickle" ,"wb") as F:
    # 	pickle.dump(klassfiltered,F)

    # count = 0
    for itera, i in enumerate(klassfiltered.keys()):
        # sign = False
        for words in klassfiltered[i]:
            if not "jpg" in words:
                if not words in wordarray:
                    if len(words) <= 12:
                        wordarray.append(words)
                        klasses.append(int(i))
                    # sign = True
        klass_length += 1
        if (itera + 1) % 5000 == 0:
            print("Getting words " + str(itera + 1) + " / " + str(len(klassfiltered.keys())))

    # enc_dict = word2enc(wordarray)
    # print("Created dict encoded")
    # batcher = np.array([])
    # text_net = True
    # if text_net:
    # 	for itera, word in wordarray:
    # 		batcher = np.concatenate(batcher, enc_dict[word],axis = 2)

    return klassfiltered
