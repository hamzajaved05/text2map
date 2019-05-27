"""
Author: Hamza
Dated: 03.05.2019
Project: texttomap

"""
import pickle

with open("pickle03_training_val", "rb") as F:
    [train_loss, train_accuracy, los, acc, batch_size, lr, epochs] = pickle.load(F)
