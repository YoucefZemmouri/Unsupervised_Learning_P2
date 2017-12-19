from __future__ import print_function
import numpy as np
from sklearn.cluster import KMeans
from numpy import linalg as LA
from scipy import linalg as sLA
import time
from scipy.stats import ortho_group
from sklearn.decomposition import PCA

def SoftThreshold(epsilon, x):
    """
    :param epsilon: Margin
    :param x: Value
    :return: Soft thresholding operator
    """
    return np.sign(x) * np.maximum(np.abs(x) - epsilon, 0)


def SpectralClustering(W, n):
    """
    Spectral Clustering (Algorithm 4.5)
    :param W: Affinity matrix
    :param n: Number of clusters
    :return: The segmentation of the data into n groups
    """
    N = W.shape[0]
    L = np.diag(W.dot(np.ones(N))) - W
    evalue, evector = LA.eigh(L)
    Y = evector[:, 0:n]
    Clus = KMeans(n_clusters=n, random_state=0).fit(Y)
    return Clus.labels_, Y


# Sparse Subspace Clustering for Noisy Data
def LASSO(X, mu2=0.1, tau=0.1, epsilon=0.01,verbose=True):
    """
    Matrix LASSO Minimization by ADMM (Algorithm 8.5)
    :param X: Data matrix
    :param mu2: Algorithm parameter
    :param tau: step gradient ascent
    :return: Sparse representation C
    """
    N = X.shape[1]
    D = X.shape[0]
    C = np.zeros((N, N))
    L = np.zeros((N, N))
    I = np.identity(N)
    tauXtX = tau * (X.T).dot(X)
    A = np.linalg.inv(tauXtX + mu2 * I)

    count = 0
    t = time.time()
    while True:
        Z = A.dot(tauXtX + mu2 * C - L)
        C = SoftThreshold(1 / mu2, Z + L / mu2)
        np.fill_diagonal(C, 0)
        # C = C - np.diag(np.diag(C))
        L = L + mu2 * (Z - C)
        max_norm = np.max(np.abs(Z - C))
        if count % 10 == 0:
            print('i=', count, ' step = ', max_norm)
        if max_norm < epsilon:
            break
        old_C=C
        count += 1
    elapsed = time.time() - t
    if verbose==True:
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
    D = W.dot(np.ones(N))
    if np.any(D == 0):
        print("Warning: at least one sample is completely isolated from others")
    D_12 = np.sqrt(D)  # D is diagonal, inv(D) = 1/D
    D_12[D_12 != 0] = 1/D_12[D_12 != 0]
    D = np.diag(D)
    D_12 = np.diag(D_12)
    L = D - W
    M = D_12.dot(L).dot(D_12)
    evalue, evector = LA.eig(M)
    Y = evector[:, 0:n]
    Clus = KMeans(n_clusters=n, random_state=0).fit(Y)
    return Clus.labels_, Y


def SSC(X, n, mu2=0.1, tau=0.1, epsilon=0.01, verbose=True):
    """
    SSC for Noisy Data (Algorithm 8.6)
    :param X: Data
    :param n: number of linear subspaces
    :param mu2: Algorithm parameter
    :param tau: step gradient ascent
    :return: Segmentation of the data into n groups
    """
    C = LASSO(X, mu2, tau, epsilon, verbose)
    W = np.abs(C) + (np.abs(C)).T
    return NormalizedSpectralClustering(n, W)


def K_Subspaces(X, n, d, restart):
    """
    K-Subspaces
    :param X: Data
    :param n: Number of subspaces
    :param d: List of dimensions of the subspaces
    :param restart: number of restarts
    :return: assignments, subspaces represented by U & mu
    """
    print('K-subspace...')
    assert(len(d) == n)
    N = X.shape[1]
    D = X.shape[0]
    minimum_sum_sq_dist = 1e10

    for r in range(restart):
        print('Repeat {}: '.format(r), end='')
        Mus = np.zeros((D, n))
        U = [np.array([]) for _ in range(n)]

        # generate random ortho matrix is very slow for large dimension.
        # instead, we want to initialize by random assignments to subspaces
        # for l in range(n):
        #     print(1)
        #     temp = ortho_group.rvs(D) # so long...
        #     temp = temp[:, 0:d[l]]
        #     U.append(temp)

        # subspace_has[l] is the list of point indices that belongs to Subspace_l
        subspace_has = [[] for _ in range(n)]
        # start with a random assignment
        for j in range(N):
            subspace_has[np.random.randint(n)].append(j)
        subspace_has_old = subspace_has

        count = 0
        while True:
            print(str(count)+' ', end='')
            count += 1
            # Do PCA for each subspace
            for l in range(n):
                points = subspace_has[l]
                X_in_l = X[:, points]
                Mus[:, l] = X_in_l.sum(1).T / len(points)

                # eigen decomposition in sLA is now fast enough
                # x_mu = X_in_l - Mus[:, l, None]
                # M = x_mu.dot(x_mu.T)
                # evalue, evector = sLA.eigh(M,eigvals=(D-d[l],D-1))  # d[l] largest eigen
                # U[l] = evector  # last(largest) d[l] evector

                # so instead, we use a optimized PCA module in sklearn
                pca = PCA(n_components=d[l])
                pca.fit(X_in_l.T)
                U[l] = pca.components_.T

            # Calculate distance
            I = np.identity(D)
            sqdist = np.zeros((n, N))  # dist[l,j] means distance of X_j to Subspace_l
            for l in range(n):
                A = (I - U[l].dot(U[l].T))
                for j in range(N):
                    v = A.dot(X[:, j]-Mus[:, l])
                    sqdist[l, j] = np.sum(v*v)

            # Assign point to nearest subspace
            subspace_has = [[] for _ in range(n)]
            for j in range(N):
                l_best_for_j = np.argmin(sqdist[:, j])
                subspace_has[l_best_for_j].append(j)
            if subspace_has_old == subspace_has:  # stop when no updates for assignments
                break
            subspace_has_old = subspace_has

        sum_sq_dist = sqdist.sum()
        print('sum_sq_dist = {}'.format(sum_sq_dist))
        if sum_sq_dist < minimum_sum_sq_dist:
            minimum_sum_sq_dist = sum_sq_dist
            best_assignments = np.zeros(N).astype(int)  # assignments[j] means X_j assigned to Subspace_{assignments[j]}
            for l in range(n):
                best_assignments[subspace_has[l]] = l

    print('Sum of sq distance = {}'.format(minimum_sum_sq_dist))
    return best_assignments, minimum_sum_sq_dist, U, Mus
