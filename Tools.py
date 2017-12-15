import numpy as np
from sklearn.cluster import KMeans
from numpy import linalg as LA
import time
from scipy.stats import ortho_group


def SoftThreshold(epsilon, x):
    """
    :param epsilon: Margin
    :param x: Value
    :return: Soft thresholding operator
    """
    return np.sign(x) * np.maximum(np.abs(x) - epsilon, 0)


# Spectral Clustering
def spectralClustering(W, n):
    """
    Spectral Clustering (Algorithm 4.5)
    :param W: Affinity matrix
    :param n: Number of clusters
    :return: The segmentation of the data into n groups
    """
    N = W.shape[0]
    L = np.diag(W.dot(np.ones(N))) - W
    evalue, evector = LA.eig(L)
    Y = np.matrix(evector[0:n])
    Clus = KMeans(n_clusters=n, random_state=0).fit(Y.T)
    return Clus.labels_, Y.T


# Sparse Subspace Clustering for Noisy Data
def LASSO(X, mu2=0.1, tau=0.1, epsilon=0.1):
    """
    Matrix LASSO Minimization by ADMM (Algorithm 8.5)
    :param X: Data matrix
    :param mu2: Algorithm parameter
    :param tau: step gradient ascent
    :return: Sparse representation C
    """
    C = np.zeros(X.shape)
    L = np.zeros(X.shape)
    I = np.identity(X.shape[1])
    A = np.linalg.inv(tau * (X.T).dot(X) + mu2 * I)
    XtX = tau * (X.T).dot(X)

    count = 0
    t = time.time()

    while True:
        Z = A.dot(XtX + mu2 * C - L)
        C = SoftThreshold(1 / mu2, Z + L / mu2)
        C = C - np.diag(np.diag(C))
        L = L + mu2 * (Z - C)
        max_norm = np.max(np.abs(Z))
        if count % 10 == 0:
            print('i=', count, ' step = ', max_norm)
        if max_norm < epsilon:
            break
        count += 1
    elapsed = time.time() - t
    print(count, ' iterations used, ', elapsed, ' seconds')
    return C


def NormalizedSpectralClustering(n, W):
    """
    Normalized Spectral Clustering (Algorithm 4.7)
    :param n: Number of clusters
    :param W: Affinity matrix
    :return: Segmentation of the data in n groups
    """
    N = W.shape[0]
    D = np.diag(W.dot(np.ones(N)))
    D_12 = np.linalg.inv(np.sqrt(D))
    L = D - W
    M = D_12.dot(L).dot(D_12)
    evalue, evector = LA.eig(M)
    Y = np.matrix(evector[0:n])
    Clus = KMeans(n_clusters=n, random_state=0).fit(Y.T)
    return Clus.labels_, Y.T


def SSC(X, n, mu2=0.1, tau=0.1):
    """
    SSC for Noisy Data (Algorithm 8.6)
    :param X: Data
    :param n: number of linear subspaces
    :param mu2: Algorithm parameter
    :param tau: step gradient ascent
    :return: Segmentation of the data into n groups
    """
    C = LASSO(X, mu2, tau)
    W = np.abs(C) + (np.abs(C)).T
    return NormalizedSpectralClustering(n, W)


# K-Subspaces
def K_Subspaces(X, n, d):
    """
    K-Subspaces
    :param X: Data
    :param n: Number of subspaces
    :param d: List of dimensions of the subspaces
    :return: Subspace parametrs, low dimension representation and segmentation of the data
    """
    Mus = np.random.rand((X.shape[0], len(d)))
    U = []
    for i in range(len(d)):
        temp = ortho_group(X.shape[0])
        temp = U[:,0:d[i]]
        U.append(temp)

    I = np.identity(X.shape[0])
    dist = np.zeros(len(d))

    while True:
        for j in range(X.shape[1]):
            for l in range(len(d)):
                dist[l] = (I - (U[l].T).dot(U[l])).dot(X[:,j]-Mus[l])
            i = np.argmin(dist)
    return
