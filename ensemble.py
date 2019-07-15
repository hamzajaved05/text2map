import pandas as pd
import numpy as np
import torch
import pickle
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

df = pd.read_csv("/home/presage3/vlads/gintingt/68_data_sorted_new.csv")
vlad = pd.read_csv("/home/presage3/vlads/gintingt/68_vladmatched_new.csv", header = None).to_numpy()

ids = df.shape[0]
np.random.seed(912)

dumm = df["imagesouce"].to_numpy()
with open("util/dl_logs/03_test_result_confidenceT.pickle","rb") as q:
    results = pickle.load(q)

# test_ind = np.random.choice(range(ids), 1000)
test_ind = [np.argwhere(i == dumm).item() for i in results.keys()]
test = {"ind": test_ind, "name":df["imagesouce"].to_numpy()[test_ind],
        "lat": df["lat"].to_numpy()[test_ind], "lon":df["long"].to_numpy()[test_ind]}
rest = list(set(range(ids)) - set(test_ind))
lib = np.random.choice(rest, 1100)
library = {"ind": lib, "name":df["imagesouce"].to_numpy()[lib],
        "lat": df["lat"].to_numpy()[lib], "lon":df["long"].to_numpy()[lib]}

library_values = vlad[lib]
test_values = vlad[test_ind]
del vlad

pred = []
mins = []
distan = []
dif = []
def getdistance(a, b):
    RAD = 0.000008998719243599958;
    hor = math.sqrt(math.pow(float(a[0]) - float(b[0]), 2)
                    + math.pow(float(a[1]) - float(b[1]), 2)) / RAD;
    return hor

for itera, i in enumerate(test["ind"]):
    dist = np.linalg.norm(library_values - test_values[itera], axis = 1)
    dif.append(np.diff(np.partition(dist, 1)[:2]).item())
    arg = np.argmin(dist)
    mins.append(np.min(dist))
    testcoord = [test["lat"][itera],test["lon"][itera]]
    libcoord  = [library["lat"][arg], library["lon"][arg]]
    distan.append(getdistance(testcoord, libcoord))
    pred.append(library["name"][arg])
    if itera%20 == 0:
        print("{}/{}".format(itera, len(test["ind"])))


distan2 = []
for itera, i in enumerate(results.keys()):
    testcoord = [test["lat"][itera], test["lon"][itera]]
    matcharg = np.argwhere(results[i][0][0] == dumm).item()
    libcoord  = [df["lat"][matcharg], df["long"][matcharg]]
    distan2.append(getdistance(testcoord, libcoord))

x = (np.asarray(distan)>100) * (np.asarray(mins) <1.4) *( np.asarray(mins) >1.2) * ( np.asarray(dif) < 0.03)
findist = []
for itera, i in enumerate(x):
    if i:
        findist.append(distan2[itera])
    else:
        findist.append(distan[itera])
plt.hist([findist, distan], cumulative = True, histtype = "step", color = ["k", 'b'], bins = 1000, density = True)

def plots(x, t = False):
    q = x
    hist, bin_edges = np.histogram(q, bins = 3000)
    # hist = np.insert(hist, 0 ,0)
    hist2 = np.cumsum(hist)/30
    if t:
        xnew = np.linspace(0, 100 , 5000)
        spl = make_interp_spline( bin_edges[:-1], hist2, k=2) #BSpline object
        power_smooth = spl(xnew)
    else:
        xnew = bin_edges[:-1]
        power_smooth = hist2
    return xnew, power_smooth

x, y = plots(findist)
plt.figure(1)
plt.clf()
# plt.plot([50,50],[0,120], linestyle = ":")
plt.plot(x,y, linestyle = "-", label = "Text enforced NetVlad")
# plt.annotate("{:1f}".format(y[105]), (x[108], y[30]))

x, y = plots(distan)
plt.plot(x,y, label = "NetVlad", linestyle = "-.")
plt.xlim([0, 75])

# plt.annotate()
plt.legend()