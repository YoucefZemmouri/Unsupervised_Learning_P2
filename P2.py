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
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=0, vmax=1))
    # sm._A = []
    # plt.colorbar(sm, fraction=0.026, pad=0.04)
    plt.show()




mat = scipy.io.loadmat('ExtendedYaleB.mat')

X = mat['EYALEB_DATA'] / 255.
labels = mat['EYALEB_LABEL'].flatten()
labels -= 1  # to make range in {0, 1, ..., n-1}

imshape=(48,42)
illum_num = 64




n = 2
X_small = X[:, 0:illum_num*n]
labels_small = labels[0:illum_num*n]

K_largest = 30
nbrs = NearestNeighbors(n_neighbors=K_largest, algorithm='ball_tree').fit(X_small.T)

K_ranges = np.arange(2, 10, 1)
sigma_ranges = np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5])

Kcoord, sigma_coord = np.meshgrid(K_ranges, sigma_ranges, indexing='ij')
errors = np.zeros(Kcoord.shape)
for ki in range(len(K_ranges)):
    K = K_ranges[ki]
    distances, indices = nbrs.kneighbors(X_small.T, K)
    for sigma_i in range(len(sigma_ranges)):
        sq_sigma = sigma_ranges[sigma_i]**2
        # build affinity matrix W
        W = np.zeros((X_small.shape[1], X_small.shape[1]))
        for i in range(X_small.shape[1]):
            for j in range(len(indices[i])):
                index = indices[i][j]
                dist = distances[i][j]
                w = np.exp(- dist**2 / 2 / sq_sigma)
                W[i, index] = W[index, i] = w

        clus, _ = spectralClustering(W, n)
        error = clustering_error(clus, labels_small, n)
        errors[ki, sigma_i] = error
        # print(clus)
        # print(labels_small)

show_error_table(errors, sigma_ranges, K_ranges)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(Kcoord, sigma_coord, errors)
# plt.show()