import numpy as np
from sklearn.cluster import KMeans
from numpy import linalg as LA


def SoftThreshold(epsilon, x):
    """
    :param epsilon: Margin
    :param x: Value
    :return: Soft thresholding operator
    """
    return np.sign(x) * np.maximum(np.abs(x) - epsilon, 0)


# Spectral Clustering
def spectralClustering(W, n):
    '''
    Spectral Clustering (Algorithm 4.5)
    :param W: Affinity matrix
    :param n: Number of clusters
    :return: The segmentation of the data into n groups
    '''
    N = W.shape[0]
    L = np.diag(W.dot(np.ones(N))) - W
    evalue, evector = LA.eig(L)
    Y = np.matrix(evector[0:n])
    KMeans(n_clusters=n, random_state=0).fit(Y)
    return


# Sparse Subspace Clustering for Noisy Data
def LASSO(X, mu2=0.1, tau=0.1):
    '''
    Matrix LASSO Minimization by ADMM (Algorithm 8.5)
    :param X: Data matrix
    :param mu2:
    :param tau:
    :return: Sparse representation C
    '''
    C = np.zeros(X.shape)
    L = np.zeros(X.shape)
    I = np.identity(X.shape[1])
    A = np.linalg.inv(tau * (X.T).dot(X) + mu2 * I)
    XtX = tau * (X.T).dot(X)
    while True:
        Z = A.dot(XtX + mu2 * C - L)
        C = SoftThreshold(1 / mu2, Z + L / mu2)
        C = C - np.diag(np.diag(C))
        L = L + mu2 * (Z - C)
    return C


def NormalizedSpectralClustering(n, W):
    '''
    Normalized Spectral Clustering (Algorithm 4.7)
    :param n:
    :param W:
    :return:
    '''
    N = W.shape[0]
    D = np.diag(W.dot(np.ones(N)))
    D_12 = np.sqrt(D)
    L = D - W
    M = D_12.dot(L).dot(D_12)
    evalue, evector = LA.eig(M)
    Y = np.matrix(evector[0:n])
    KMeans(n_clusters=n, random_state=0).fit(Y)
    return


def SSC(X, n, mu2=0.1, tau=0.1):
    """
    SSC for Noisy Data (Algorithm 8.6)
    :param X:
    :param n:
    :param mu2:
    :param tau:
    :return:
    """
    C = LASSO(X, mu2, tau)
    W = np.abs(C) + (np.abs(C)).T
    NormalizedSpectralClustering(n, W)
    return


# K-Subspaces
def K_Subspaces(X, n, d):
    '''
    K-Subspaces
    :param X: Data
    :param n: Number of subspaces
    :param d: List of dimensions of the subspaces
    :return:
    '''
    return
