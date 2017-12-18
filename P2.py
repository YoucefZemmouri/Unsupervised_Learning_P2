from Tools import *
import scipy.io
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors


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


def show_error_table(cells, sigmas, Ks):
    cells = cells.round(3)
    Ks = [str(x)+'-NN' for x in Ks]
    fig = plt.figure()
    ax = fig.gca()
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=cells,
                         rowLabels=Ks,
                         colLabels=sigmas,
                         loc='center',
                         cellColours=plt.cm.cool(cells))
    plt.title('Clustering error with choice of sigma and k-NN')
    plt.show()


def find_best_K_and_sigma(X_small, labels_small, n):
    K_largest = 30
    nbrs = NearestNeighbors(n_neighbors=K_largest, algorithm='ball_tree').fit(X_small.T)

    K_ranges = np.arange(2, 20, 1)
    sigma_ranges = np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 9, 13, 20, 30])

    errors = np.zeros((len(K_ranges), len(sigma_ranges)))
    for ki in range(len(K_ranges)):
        K = K_ranges[ki]
        distances, indices = nbrs.kneighbors(X_small.T, K)
        for sigma_i in range(len(sigma_ranges)):
            sq_sigma = sigma_ranges[sigma_i] ** 2
            # build affinity matrix W
            W = np.zeros((X_small.shape[1], X_small.shape[1]))
            for i in range(X_small.shape[1]):
                for j in range(len(indices[i])):
                    index = indices[i][j]
                    dist = distances[i][j]
                    w = np.exp(- dist ** 2 / 2 / sq_sigma)
                    W[i, index] = W[index, i] = w

            clus, _ = SpectralClustering(W, n)
            error = clustering_error(clus, labels_small, n)
            errors[ki, sigma_i] = error

    show_error_table(errors, sigma_ranges, K_ranges)


# load faces
mat = scipy.io.loadmat('ExtendedYaleB.mat')

X = mat['EYALEB_DATA'] / 255.  # rescale data to [0,1]
labels = mat['EYALEB_LABEL'].flatten()
labels -= 1  # to make range in {0, 1, ..., n-1}

im_shape = (42, 48)
illum_num = 64  # number of faces for each person

n = 2
X_small = X[:, 0:illum_num*n]
labels_small = labels[0:illum_num*n]

# show one image
# plt.imshow(X[:, 2].reshape(im_shape).T)
# plt.colorbar()
# plt.show()

# SSC
clus, _ = SSC(X_small, n, tau=0.05, mu2=1000, epsilon=0.0000001)
error = clustering_error(clus, labels_small, n)
print('SSC, clustering error = {:.2f}%'.format(error*100))

# SpectralClustering
find_best_K_and_sigma(X_small, labels_small, n)

# K subspace
for i in range(5):  # multiple trials for K-subspace
    di = [3,3]  # 3 dimension subspace for faces
    clus, sq_dist, _, _ = K_Subspaces(X_small, n, di, restart=1)
    error = clustering_error(clus, labels_small, n)
    print('clustering error = {:.2f}%'.format(error*100))
