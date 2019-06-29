import pickle
import numpy as np
import hypertools as hyp

with open("models/05bhembeds.pickle", "rb") as q:
    [embeds, klass] = pickle.load(q)

with open("/home/presage3/bh_test/models_bh/01embeds.pickle", "rb") as q:
    [embeds1, klass1] = pickle.load(q)

with open("/home/presage3/bh_test/models_bh/02embeds.pickle", "rb") as q:
    [embeds2, klass2] = pickle.load(q)

with open("/home/presage3/bh_test/models_bh/03embeds.pickle", "rb") as q:
    [embeds3, klass3] = pickle.load(q)

with open("/home/presage3/bh_test/models_bh/04embeds.pickle", "rb") as q:
    [embeds4, klass4] = pickle.load(q)

def get_accuracy(embeds, klass):
    correct = [0,0]
    for itera, i in enumerate(embeds):
        normaal = np.linalg.norm(embeds-i, axis = 1, ord = 2)
        sorted = np.argsort(normaal)
        if klass[sorted[1]] == klass[itera]:
            correct[1] +=1
        if klass[itera] in klass[sorted[1:10]]:
            correct[0]+=1

        if (itera+1)%1000 == 0:
            print("{}/{} done".format(itera+1, embeds.shape[0]))

    print("Accuracy is {}/{} i.e {} %".format(correct, embeds.shape[0],100* correct[0]/embeds.shape[0]))

    return correct, embeds.shape[0]

correct1, size1= get_accuracy(embeds1, klass1)
correct2, size2= get_accuracy(embeds2, klass2)
correct3, size3= get_accuracy(embeds3, klass3)
correct4, size4= get_accuracy(embeds4, klass4)
correct, size = get_accuracy(embeds, klass)
print( correct1[0]/size1, correct2[0]/size2, correct3[0]/size3, correct4[0]/size4)
print( correct1[1]/size1, correct2[1]/size2, correct3[1]/size3, correct4[1]/size4)




