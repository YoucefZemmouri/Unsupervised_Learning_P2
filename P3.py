from Tools import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
import os
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors

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

def affinity(X, K, sigma):
    NN = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(X.T)
    dist, neigh = NN.kneighbors(X.T, K)
    W = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(K):
            w = np.exp(- dist[i][j] ** 2 / (2 * sigma**2))
            W[i, neigh[i][j]] = w
            W[neigh[i][j], i] = w
    return W

def clustering(X, labels, n):
    # Spectral Clustering

    sigmas = [0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50]
    Ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    error_sp = []

    for K in Ks:
        error_sp.append([])
        for sigma in sigmas:
            W = affinity(X, K, sigma)
            clus, _ = SpectralClustering(W, n)
            error_sp[-1].append(clustering_error(clus, labels, n))
    error_sp = pd.DataFrame(error_sp,index = Ks, columns = sigmas)

    K_min = error_sp.min(1).idxmin()
    sigma_min = error_sp.T[K_min].idxmin()
    print("Minimal error (spectral clustering) :",np.round(error_sp[sigma_min][K_min]*100,3),"% (K :",K_min,", sigma:",sigma_min,")")

    # Subspace clustering 
    tau=0.005
    mu2=1000
    epsilon=0.1
    clus, _ = SSC(X, n, tau, mu2, epsilon, verbose=False)
    print("Minimal error (subspace clustering) :",np.round(100*clustering_error(clus, labels, n)),"% (tau :",tau,")")

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
clustering(X, labels, n)