from Tools import *
import scipy.io
from sklearn.neighbors import NearestNeighbors



def find_best_K_and_sigma_for_SC(X, labels, n):

    K_ranges = np.arange(2, 20, 1)
    sigma_ranges = np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 9, 13, 20, 30])

    errors = np.zeros((len(K_ranges), len(sigma_ranges)))
    for ki in range(len(K_ranges)):
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

im_shape = (42, 48)
illum_num = 64  # number of faces for each person

# show one image
# plt.imshow(X[:, 2].reshape(im_shape).T)
# plt.colorbar()
# plt.show()

# select n first individuals
n = 2
X = X_global[:, 0:illum_num * n]
labels = labels_global[0:illum_num * n]




# SpectralClustering
find_best_K_and_sigma_for_SC(X, labels, n)

# K subspace
for i in range(5):  # multiple trials for K-subspace
    di = [3]*n  # 3 dimension subspace for faces
    clus, sq_dist, _, _ = K_Subspaces(X, n, di, restart=1)
    error = clustering_error(clus, labels, n)
    print('clustering error = {:.2f}%'.format(error*100))

# SSC
clus, _ = SSC(X, n, tau=0.05, mu2=1000, epsilon=0.0000001)
error = clustering_error(clus, labels, n)
print('SSC, clustering error = {:.2f}%'.format(error*100))
