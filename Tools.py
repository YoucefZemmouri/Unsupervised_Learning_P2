from __future__ import print_function
import numpy as np
from sklearn.cluster import KMeans
from numpy import linalg as LA
import time
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import matplotlib.colors

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
        if (count % 10 == 0) and verbose:
            print('i=', count, ' step = ', max_norm)
        if max_norm < epsilon:
            break
        count += 1
    elapsed = time.time() - t
    print('LASSO: {} iteration used, {} seconds'.format(count, elapsed))
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
    Sparse Subspace Clustering for Noisy Data (Algorithm 8.6)
    :param X: Data
    :param n: number of linear subspaces
    :param mu2: Algorithm parameter
    :param tau: step gradient ascent
    :return: Segmentation of the data into n groups
    """
    C = LASSO(X, mu2, tau, epsilon, verbose)
    W = np.abs(C) + (np.abs(C)).T
    return NormalizedSpectralClustering(n, W)


def K_Subspaces(X, n, d, restart, verbose=True):
    """
    K-Subspaces
    :param X: Data
    :param n: Number of subspaces
    :param d: List of dimensions of the subspaces
    :param restart: number of restarts
    :return: assignments, subspaces represented by U & mu
    """
    if verbose:
        print('K-subspace...')
    assert(len(d) == n)
    N = X.shape[1]
    D = X.shape[0]
    minimum_sum_sq_dist = 1e10

    for r in range(restart):
        if verbose:
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
            if verbose:
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
            sum_sq_dist = 0
            for l in range(n):
                # A = (I - U[l].dot(U[l].T)) # don't do this , U[l].dot(U[l].T) too slow
                for j in range(N):
                    xc = X[:, j]-Mus[:, l]
                    # v = A.dot(X[:, j]-Mus[:, l])
                    v = xc - U[l].dot(U[l].T.dot(xc))
                    sqdist[l, j] = np.sum(v*v)
            # Assign point to nearest subspace
            subspace_has = [[] for _ in range(n)]
            for j in range(N):
                l_best_for_j = np.argmin(sqdist[:, j])
                subspace_has[l_best_for_j].append(j)
                sum_sq_dist += sqdist[l_best_for_j, j]
            if subspace_has_old == subspace_has:  # stop when no updates for assignments
                break
            subspace_has_old = subspace_has

        if verbose:
            print('sum_sq_dist = {}'.format(sum_sq_dist))
        if sum_sq_dist < minimum_sum_sq_dist:
            minimum_sum_sq_dist = sum_sq_dist
            best_assignments = np.zeros(N).astype(int)  # assignments[j] means X_j assigned to Subspace_{assignments[j]}
            for l in range(n):
                best_assignments[subspace_has[l]] = l

    if verbose:
        print('Sum of sq distance = {}'.format(minimum_sum_sq_dist))
    return best_assignments, minimum_sum_sq_dist, U, Mus


def clustering_error(clus, labels, n):
    """
    Evaluation of Clustering with Hungarian algorithm
    :param clus: 1-d array of labels to evaluate, range in {0, 1, ..., n-1}
    :param labels: 1-d array of ground truth labels, range in {0, 1, ..., n-1}
    :param n: number of groups
    :return: error: minimum error percentage obtained among all permutations

    """
    assert(len(clus) == len(labels))
    N = len(clus)
    A = np.zeros((n,n))  # A[i,j] means number of samples in group i which are classified in group j by 'clus'
    for k in range(len(labels)):
        A[labels[k], clus[k]] += 1
    # we want to maximize the number of correctly classified samples
    # while linear_sum_assignment will minimize
    W = np.max(A) - A
    row_ind, col_ind = linear_sum_assignment(W)
    correctly_classified_samples = A[row_ind, col_ind].sum()
    return 1-correctly_classified_samples/N


def affinity(X, K, sigma):
    """
    Build affinity matrix for data X, with K-NN and Gaussian distance
    :param X: Data, each column is a sample
    :param K: K for K-NN
    :param sigma: sigma for Gaussian
    :return: W: symmetric affinity matrix
    """
    NN = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(X.T)
    dist, neigh = NN.kneighbors(X.T, K)
    N = X.shape[1]
    W = np.zeros((N, N))
    sqsigma2 = 2 * sigma ** 2
    for i in range(N):
        for j in range(K):
            w = np.exp(- dist[i][j] ** 2 / sqsigma2)
            nb = neigh[i][j]
            W[i, nb] = W[nb, i] = w
    return W


def show_error_table(errors, sigmas, Ks):
    """
    Show error table of clustering errors with different choices of sigma and K
    :param errors: numpy array, error[K_i, sigma_j] is in [0, 1]
    :param sigmas: list of sigma
    :param Ks: list of K
    """
    errors = errors.round(3)
    Ks = [str(x)+'-NN' for x in Ks]
    fig = plt.figure()
    ax = fig.gca()
    the_table = ax.table(cellText=errors*100,
                         rowLabels=Ks,
                         colLabels=sigmas,
                         loc='center',
                         cellColours=plt.cm.cool(errors),
                         bbox=(0,0,1,1))
    ax.axis('tight')
    ax.axis('off')
    # fig.tight_layout()
    fig.set_size_inches(w=7, h=3)
    # plt.title('Clustering error percentage with choice of sigma and k-NN')
    plt.show()
