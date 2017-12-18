from Tools import *
import scipy.io
from sklearn.neighbors import NearestNeighbors

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('ExtendedYaleB.mat')

X = mat['EYALEB_DATA'] / 255.
labels = mat['EYALEB_LABEL'].flatten()
labels -= 1  # to make range in {0, 1, ..., n-1}

imshape=(48,42)
illum_num = 64



def evaluation_clustering(clus, labels, n):
    """
    Evaluation of Clustering by 'Jaccard Coefficient'
    reference 1 : http://syllabus.cs.manchester.ac.uk/ugt/2017/COMP24111/materials/slides/Cluster-Validation.pdf
    reference 2 : http://www.lsi.upc.edu/~bejar/amlt/material/04-Validation.pdf
    :param clus: 1-d array of labels to evaluate, range in {0, 1, ..., n-1}
    :param labels: 1-d array of ground truth labels, range in {0, 1, ..., n-1}
    :param n: number of groups
    :return: rand_index: rand index, see reference

    """
    assert(len(clus)==len(labels))
    N = len(clus)
    A = np.zeros((n,n))  # A[i,j] means number of samples in group i which are classified in group j by 'clus'
    for k in range(len(labels)):
        A[labels[k], clus[k]] += 1
    ni = np.sum(A,0)
    nj = np.sum(A,1)
    sqA = np.sum(A*A)
    nini = np.sum(ni*ni)
    njnj = np.sum(nj*nj)
    a = (sqA - N)/2
    b = (njnj-sqA)/2
    c = (nini-sqA)/2
    d = (N*N + sqA - nini - njnj)/2
    return (a)/(a+b+c)

n = 5
X_small = X[:, 0:illum_num*n]
labels_small = labels[0:illum_num*n]

K_largest = 30
nbrs = NearestNeighbors(n_neighbors=K_largest, algorithm='ball_tree').fit(X_small.T)

K_ranges = np.arange(2, 20, 1)
sigma_ranges = np.array([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10])
Kcoord, sigma_coord = np.meshgrid(K_ranges, sigma_ranges, indexing='ij')
rand_index = np.zeros(Kcoord.shape)
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
        rate = evaluation_clustering(clus, labels_small, n)
        rand_index[ki, sigma_i] = rate



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Kcoord, sigma_coord, rand_index)
plt.show()