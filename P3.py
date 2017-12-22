from __future__ import print_function
from Tools import *
import pandas as pd
import numpy as np

from scipy.io import loadmat
import os


def load_data():
    while 'Hopkins155' not in os.listdir("."):
        os.chdir('..')
    os.chdir('Hopkins155')
    d = os.listdir(".")

    X_global = {}
    labels_global = {}
    n_global = {}

    for i in range(len(d)):
        name = d[i]
        if name not in ['.DS_Store','README.txt']:       
            os.chdir(name)

            mat = loadmat(name+"_truth.mat")

            N = mat["x"].shape[1]
            F = mat["x"].shape[2]
            D = 2*F;

            n_global[name] = np.max(mat["s"])
            labels_global[name] = mat["s"].reshape(N)
            labels_global[name] -= 1

            X = np.zeros((2,F,N))
            X[0] = np.transpose(mat["x"][0])
            X[1] = np.transpose(mat["x"][1])

            X_global[name] = X.reshape(D,N)

            os.chdir('..')
    return X_global, labels_global, n_global


def clustering(X, labels, n):
    # Spectral Clustering

    sigmas = [0.1, 0.3, 0.5, 1, 2, 5, 10, 30, 100, 10000, 1000000]
    Ks = [2, 3, 4, 5, 10, 50, 100, 200, 300, 400]

    error_sp = []

    for K in Ks:
        print('.', end='')
        error_sp.append([])
        for sigma in sigmas:
            W = affinity(X, K, sigma)
            clus, _ = SpectralClustering(W, n)
            error_sp[-1].append(clustering_error(clus, labels, n))
    error_sp = pd.DataFrame(error_sp,index = Ks, columns = sigmas)

    K_min = error_sp.min(1).idxmin()
    sigma_min = error_sp.T[K_min].idxmin()
    print("Minimal error (spectral clustering) = {:.2f}%, (K_min:{}, sigma_min:{})".format(
        np.round(error_sp[sigma_min][K_min]*100,3),K_min,sigma_min)
    )
    show_error_table(np.array(error_sp), sigmas, Ks)

    # Subspace clustering 
    tau=0.005
    mu2=1000
    epsilon=0.1
    clus, _ = SSC(X, n, tau, mu2, epsilon, verbose=False)
    print("Minimal error (subspace clustering) = {:.2f}%, (tau:{})".format(np.round(100*clustering_error(clus, labels, n)),tau))

    # K subspace
    error = []
    for i in range(10):  # multiple trials for K-subspace
        di = [3]*n # Subspaces of dimension 3 ?
        clus, sq_dist, _, _ = K_Subspaces(X, n, di, restart=1)
        error.append(clustering_error(clus, labels, n))
    print('Minimal error (K-subspaces) = {:.2f}%'.format(np.min(error)*100))

# Load the data    

X_global, labels_global, n_global = load_data()

# Example

key = '1R2RC'
    
X = X_global[key]
labels = labels_global[key]
n = n_global[key]
print("{} points in {} frames, {} groups".format(X.shape[1], X.shape[0]/2, n))
clustering(X, labels, n)
