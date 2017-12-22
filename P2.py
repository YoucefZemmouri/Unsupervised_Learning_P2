from __future__ import print_function
from Tools import *
import scipy.io



def find_best_K_and_sigma_for_SC(X, labels, n):

    K_ranges = np.array([2,3,4,5,6,8,10,15,30,50])
    sigma_ranges = np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 9, 13, 20, 50])

    errors = np.zeros((len(K_ranges), len(sigma_ranges)))
    for ki in range(len(K_ranges)):
        print('.',end='')
        K = K_ranges[ki]
        for sigma_i in range(len(sigma_ranges)):
            W = affinity(X, K, sigma_ranges[sigma_i])

            clus, _ = SpectralClustering(W, n)
            errors[ki, sigma_i] = clustering_error(clus, labels, n)

    show_error_table(errors, sigma_ranges, K_ranges)


# load faces
mat = scipy.io.loadmat('ExtendedYaleB.mat')

X_global = mat['EYALEB_DATA'] / 255.  # rescale data to [0,1]
labels_global = mat['EYALEB_LABEL'].flatten()
labels_global -= 1  # to make range in {0, 1, ..., n-1}

illum_num = 64  # number of faces for each person


# select n first individuals
n = 10
selections = []
for i in range(n):
    for j in range(len(labels_global)):
        if labels_global[j] == i:
            selections.append(j)

X = X_global[:, selections]
labels = labels_global[selections]

print('{} groups, {} faces in total'.format(n,X.shape[1]))


# SpectralClustering
find_best_K_and_sigma_for_SC(X, labels, n)
# W = affinity(X, K=4, sigma=4.0)
# clus, _ = SpectralClustering(W, n)
# e = clustering_error(clus, labels, n);
# print('SC clustering error = {:.2f}%, K={},s={}'.format(e*100,4,4.0))

# K subspace
for i in range(5):  # multiple trials for K-subspace
    di = [3]*n  # 3 dimension subspace for faces
    clus, sq_dist, _, _ = K_Subspaces(X, n, di, restart=1)
    error = clustering_error(clus, labels, n)
    print('K-subspace clustering error = {:.2f}%'.format(error*100))

# SSC
clus, _ = SSC(X, n, tau=0.05, mu2=1000, epsilon=0.0000001)
error = clustering_error(clus, labels, n)
print('SSC, clustering error = {:.2f}%'.format(error*100))
