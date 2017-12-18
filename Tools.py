import numpy as np
from sklearn.cluster import KMeans
from numpy import linalg as LA
from scipy import linalg as sLA
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
    evalue, evector = LA.eigh(L)
    Y = evector[:,0:n]
    Clus = KMeans(n_clusters=n, random_state=0).fit(Y)
    return Clus.labels_, Y


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
    :return: assignments, subspaces represented by U & mu
    """
    print('K-subspace...')
    assert(len(d) == n)
    N = X.shape[1]
    D = X.shape[0]
    Mus = np.random.rand(D, n)
    U = [np.array([]) for _ in range(n)]
    # generate random ortho matrix is very slow for large dimension
    # for l in range(n):
    #     print(1)
    #     temp = special_ortho_group.rvs(D)
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
        print(count)
        # Do PCA for each subspace
        for l in range(n):
            points = subspace_has[l]
            X_in_l = X[:, points]
            Mus[:, l] = X_in_l.sum(1).T / len(points)
            x_mu = X_in_l - Mus[:, l, None]
            M = x_mu.dot(x_mu.T)
            evalue, evector = sLA.eigh(M,eigvals=(D-d[l],D-1))  # d[l] largest eigen
            U[l] = evector # last(largest) d[l] evector

        # Calculate distance
        I = np.identity(D)
        dist = np.zeros((n, N))  # dist[l,j] means distance of X_j to Subspace_l
        for l in range(n):
            A = (I - U[l].dot(U[l].T))
            for j in range(N):
                v = A.dot(X[:, j]-Mus[:, l])
                dist[l, j] = np.sum(v*v)

        # Assign point to nearest subspace
        subspace_has = [[] for _ in range(n)]
        for j in range(N):
            l_best_for_j = np.argmin(dist[:, j])
            subspace_has[l_best_for_j].append(j)
        if subspace_has_old == subspace_has:  # no updates for assignments
            break
        subspace_has_old = subspace_has
        count += 1

    assignments = np.zeros(N).astype(int)  # assignments[j] means X_j assigned to Subspace_{assignments[j]}
    for l in range(n):
        assignments[subspace_has[l]] = l
    print('{} iterations used for K-subspace'.format(count))
    return assignments, U, Mus
