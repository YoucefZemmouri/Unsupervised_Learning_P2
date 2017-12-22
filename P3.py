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
    errorSp = 1
    sigmas = [0.1, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10, 30, 100, 10000]
    Ks = [i for i in [2,5,10,20,50,100,200,500,1000] if i <= X.shape[1]]

    for K in Ks:
        for sigma in sigmas:
            W = affinity(X, K, sigma)
            clus, _ = SpectralClustering(W, n)
            error_K_sigma = clustering_error(clus, labels, n)
            if error_K_sigma < errorSp:
                K_min = K
                sigma_min = sigma
                errorSp = error_K_sigma

    print("Minimal error (spectral clustering) = {:.2f}%, (K_min:{}, sigma_min:{})".format(
            errorSp*100,
            K_min,
            sigma_min
        )
    )

    # Subspace clustering 
    epsilon = 0.1
    errorSSC = 1
    taus = [10, 100, 1000, 10000]
    mu2s = [1000, 10000, 100000, 1000000]

    for tau in taus:
        for mu2 in mu2s:
            clus, _ = SSC(X, n, tau, mu2, epsilon, verbose=False)
            error_K_mu = clustering_error(clus, labels, n)
            if errorSSC > error_K_mu:
                errorSSC = error_K_mu
                tau_opt = tau
                mu2_opt = mu2
    print(
        "Minimal error (subspace clustering) = {:.3f}%, (tau:{}, mu2:{})".format(
            100*errorSSC,
            tau_opt,
            mu2_opt
        )
    )
    
    # K subspace
    errorKS = 1
    for dim in range(2,6):
        di = [dim]*n # Dimension of the trajectories subspaces
        clus, sq_dist, _, _ = K_Subspaces(X, n, di, restart=10, verbose=False)
        error_dim = clustering_error(clus, labels, n)
        if errorKS > np.min(error_dim):
            dim_opt = dim
            errorKS = np.min(error_dim)
    print('Minimal error (K-subspaces) = {:.2f}%, (dimension of subspaces : {})'.format(
            np.min(errorKS)*100,
            dim_opt
        )
    )
    
    return errorSp, errorSSC, errorKS

# Load the data    

X_global, labels_global, n_global = load_data()

Keys_1 = ["1R2RC", "1R2TCR", "1R2TCRT", "1RT2TC", "2T3RCR"]
Keys_2 = ["cars1", "cars2", "cars3", "cars4", "cars5"]
Keys_3 = ["head", "kanatani1", "arm", "articulated", "people1"]
Keys = Keys_1+Keys_2+Keys_3

# Test of the algorithms
errors = []
for key in Keys:
    print(key)
    X = X_global[key]
    labels = labels_global[key]
    n = n_global[key]
    print("Infos : {} points in {} frames, {} groups".format(X.shape[1], X.shape[0]/2, n))
    errors.append(clustering(X, labels, n))

means_1 = np.mean(errors[:5],0)
means_2 = np.mean(errors[5:10],0)
means_3 = np.mean(errors[10:],0)

print("Means for group 1 :",np.round(100*means_1,3),"(Spectral Clustering, SSC and K-Subspaces)")
print("Means for group 2 :",np.round(100*means_2,3),"(Spectral Clustering, SSC and K-Subspaces)")
print("Means for group 3 :",np.round(100*means_3,3),"(Spectral Clustering, SSC and K-Subspaces)")
