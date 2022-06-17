from sklearn import datasets
import matplotlib.pyplot as plt
import heapq
import numpy as np


def moons(N,noise,k):
    moon = datasets.make_moons(N,noise=noise)
    point_sites = moon[0]
    index_list = []
    for i in point_sites:
        tmp = -1*np.sqrt(np.sum(np.square(i-point_sites),axis = 1))
        index_list.append(heapq.nlargest(k,range(N),tmp.take))
    W = np.zeros((64,64))
    for i in index_list:
        for j in range(1,k):
            if i[0] in index_list[i[j]]:
                W[i[0]][i[j]] = 1
    
    return W , moon

if __name__ == "main":
    print ("complete")
    W,moon=moons(64,0.05,4)
    print ((W == W.T).all())
    print ("complete")