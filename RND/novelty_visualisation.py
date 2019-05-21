import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.neighbors import BallTree

# MNIST data
(X, Y), (X_test, Y_test) = mnist.load_data('/tmp/mnist.npz')
# Add channel count
X.shape += (1,)
X = X.astype(np.float64) / 256
X_test.shape += (1,)
X_vect = X.reshape((-1, 28 * 28))
small_inds = np.random.choice(np.arange(X_vect.shape[0]), size = 500,
                              replace = False)
X_small = X_vect[small_inds, :]
Y_small = Y[small_inds]

#  Functions {{{1 #
def tsne(X):
    return TSNE(n_components = 2,
                init = 'random',
                perplexity = 10,
                random_state = 0).fit_transform(X)

def plot_embedding(X, Y):
    plt.figure()
    markers = [c + m for c in 'rbgkcm' for m in '.xo']
    for i in range(10):
        inds = Y == i
        plt.plot(X[inds, 0], X[inds, 1], markers[i], label = str(i))
    plt.legend()
    plt.show()

def calculate_variances(X, base_class = None):
    """
    # Arguments:
    X: list of numpy arrays for the different classes
    base_class: index of the base class; None meaning take the average for all
                classes
    """
    assert len(X[0].shape) == 2
    # TODO base class
    class_variances = [np.var(Xc, axis = 0) for Xc in X]
    num = [len(Xc) for Xc in X]
    intra_variance = np.sum([n * cv for n, cv in zip(num, class_variances)])
    total_num = np.sum(num)
    intra_variance /= total_num
    means = [np.mean(Xc, axis = 0) for Xc in X]
    total_mean = np.sum([n * m for n, m in zip(num, means)])
    total_mean /= total_num
    inter_variance = np.sum([n * (m - total_mean)**2
                             for n, m in zip(num, means)])
    inter_variance /= total_num
    # average of squared distances between each pair of points from separate
    # classes
    # might not make sense
    pairwise_variance = 0
    for i, Xi in enumerate(X):
        for j, Xj in enumerate(X):
            if j == i:
                continue
            ll, kk = np.meshgrid(np.arange(len(Xi)), np.arange(len(Xj)),
                                 indexing = 'ij')
            Xl = Xi[ll.flatten(), :]
            Xk = Xj[kk.flatten(), :]
            pairwise_variance += np.sum((Xl - Xk)**2) / (2 * total_num)
    return intra_variance, inter_variance, pairwise_variance

def dist(X, a, b):
    return (np.sum(np.square(X[a, :] - X[b, :]), axis = -1))**0.5

def calculate_distance_matrix(X):
    """
    Complete graph
    """
    i = np.arange(len(X))
    ii, jj = np.meshgrid(i, i, indexing = 'ij')
    return squared_dist(X, ii.flatten(), jj.flatten()).reshape(ii.shape)

def calculate_sparse_distance_matrix(X, tree, k = 10):
    distances = []
    N = len(X)
    for x in X:
        neigbours = tree.query(x.reshape((1, -1)), k = k)
        d = np.zeros((N,))
        d[neighbours] = np.sum(np.square(x - X[neighbours, :]), axis = -1)
        distances.append(d)
    return np.array(distances)

def gen_gdf(X, Y, tree, fname, k = 10):
    with open(fname, 'w') as f:
        f.write('nodedef>name VARCHAR,class VARCHAR\n')
        f.write('\n'.join('{},{}'.format(i, y)
                          for i, y in enumerate(Y.flatten())))
        f.write('\nedgedef>node1 VARCHAR,node2 VARCHAR,weight DOUBLE\n')
        for node1, x in enumerate(X):
            weights, neighbours = tree.query(x.reshape((1, -1)), k = k)
            weights.shape = (-1,)
            neighbours.shape = (-1,)
            f.write('\n'.join(
                '{},{},{}'.format(node1, neighbour, 1 / weight)
                for neighbour, weight in zip(neighbours, weights)
                if neighbour != node1))
            f.write('\n')

def gen_novelty_gdf(novelties, Y, fname):
    with open(fname, 'w') as f:
        f.write('nodedef>name VARCHAR,class VARCHAR\n')
        f.write('\n'.join('{},{}'.format(i, y)
                          for i, y in enumerate(Y.flatten())))
        base_ind = len(Y)
        f.write('\n{},base\n'.format(base_ind))
        f.write('edgedef>node1 VARCHAR,node2 VARCHAR,weight DOUBLE\n')
        f.write('\n'.join(
            '{},{},{}'.format(len(base_ind, i, novelty)
            for i, novelty in enumerate(novelties))))
        f.write('\n')

#  1}}} #

X_sne = tsne(X_small)
plot_embedding(X_sne, Y_small)
plt.title('pixel space')

X_feat = shared.predict(X[small_inds, :, :, :])
X_feat_sne = tsne(X_feat)
plot_embedding(X_feat_sne, Y_small)
plt.title('feature space')

X_rand = target.predict(X[small_inds, :, :, :])

tree_vec = BallTree(X_small)
tree_feat = BallTree(X_feat)
tree_rand = BallTree(X_rand)

gen_gdf(X_small, Y_small, tree_vec, '/tmp/pixels.gdf', k = 8)
gen_gdf(X_feat, Y_small, tree_feat, '/tmp/features.gdf', k = 8)
gen_gdf(X_rand, Y_small, tree_rand, '/tmp/random_feat.gdf', k = 8)

# vim:set et sw=4 ts=4 fdm=marker:
