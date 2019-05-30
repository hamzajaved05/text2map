import pandas as pd
import numpy as np
import pickle
from numpy import linalg as LA
vlad = pd.read_csv("Dataset_processing/nv/nv_68.csv")
jpg = pd.read_csv("Dataset_processing/nv/data_68.csv", usecols = ['imagesource'])
