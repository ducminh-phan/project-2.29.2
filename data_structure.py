import random
from functools import wraps
from math import log

import pylab
from matplotlib import pyplot as plt

from utils import euclidean_distance, get_colors


class FullyDynClus:
    def __init__(self, eps, n_clusters, window, distance_func=euclidean_distance):
        self.eps = eps              # epsilon in the approximation ratio of (2 + ε)
        self.n_clusters = n_clusters    # the number of clusters k
        self.window = window        # the size of the sliding window
        self.d = distance_func      # the distance function

        self.gamma = []             # the list Γ of guesses β
        self.structs = []           # the container of the data structures L_β
        self.points = []            # the list of points ever inserted
        self.dmin = float('inf')    # the minimum distance between any two points in the window
        self.dmax = 0               # the maximum distance

    def delete(self, x):
        """Delete the point x from the data structures."""
        for struct in self.structs:
            struct.delete(x)

    def insert(self, x):
        """
        Insert a point x int the data structures. We will first find the new dmin, dmax,
        then update gamma and the list of data structures if necessary, then insert x
        into each data structure. We need extra care for the first points ever inserted.
        """
        new_dmin, new_dmax = self.find_new_dmin_dmax(x)
        # [self.structs[i].visualize(i) for i in range(len(self.gamma))]

        if len(self.points) >= 2:
            self.update(new_dmin, new_dmax)

        # Update dmin and dmax
        self.dmin = min(new_dmin, self.dmin)
        self.dmax = max(new_dmax, self.dmax)

        # Finally, x is inserted to the list of points and the data structures
        self.points.append(x)

        for struct in self.structs:
            struct.insert(x)

    def update(self, new_dmin, new_dmax):
        """
        Check if we need to update Γ and the list of data structures when
        a new point x is inserted, then perform the update if necessary.
        """
        eps = self.eps

        # The current exponents of (1 + ε) corresponding to dmax and dmin
        i_min = int(log(self.dmin, 1 + eps)) + 1
        i_max = int(log(self.dmax, 1 + eps))

        # The new exponents of (1 + ε)
        new_i_min = int(log(new_dmin, 1 + eps)) + 1
        new_i_max = int(log(new_dmax, 1 + eps))

        # If the list Γ is currently empty, and there exists i such that
        # dmin <= (1 + ε) ** i <= dmax, we update Γ and the list of data structures

        if not self.gamma and new_i_max >= new_i_min:
            self.gamma = [(1 + eps) ** i
                          for i in range(new_i_min, new_i_max + 1)]

            self.structs = [L(self.n_clusters, beta, self.d).fit(self.points[-self.window:])
                            for beta in self.gamma]

            return

        # Add new β and the corresponding data structures if necessary

        if new_i_max > i_max:
            for i in range(i_max + 1, new_i_max + 1):
                beta = (1 + eps) ** i
                self.gamma.append(beta)
                self.structs.append(
                    L(self.n_clusters, beta, self.d).fit(self.points[-self.window:])
                )

        if new_i_min < i_min:
            for i in range(i_min - 1, new_i_min - 1, -1):
                beta = (1 + eps) ** i
                self.gamma.insert(0, beta)
                self.structs.insert(
                    0, L(self.n_clusters, beta, self.d).fit(self.points[-self.window:])
                )

    def find_new_dmin_dmax(self, x):
        """
        Find the min and max distance from a new point x
        to the current points in the window.
        """
        new_dmin = float('inf')
        new_dmax = 0

        for point in self.points:
            dx = self.d(x, point)
            if dx < new_dmin:
                new_dmin = dx
            if dx > new_dmax:
                new_dmax = dx

        return new_dmin, new_dmax

    def get_result(self):
        for struct in self.structs:
            if not struct.unclustered_points:
                return struct

    @property
    def op_count(self):
        return sum(struct.op_count for struct in self.structs)


def count_operation(func):
    """A decorator to count the number of update operations."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        self.op_count += 1

        return func(*args, **kwargs)

    return wrapper


class L:
    def __init__(self, n_clusters=8, beta=16, distance_func=euclidean_distance):
        self.n_clusters = n_clusters
        self.beta = beta
        self.d = distance_func

        self.labels = {}            # maps a point to the id of the cluster it belongs to
        self.centers = list()       # the list of centers such that the pairwise distances > 2β
        self.collection = list()    # the collection of disjoint clusters
        self.unclustered_points = set()         # the set of unclustered points
        self.colors = get_colors(n_clusters)    # the colors used to visualize the clusters
        self.op_count = 0           # the counter of update operations

    def __repr__(self):
        return '<L(β={})>'.format(self.beta)

    def fit(self, X):
        # self.random_recluster(X, self.n_clusters)
        for point in X:
            self.insert(point)
        return self

    def _get_closest_points(self, point_list, center):
        """
        Find the points in point_list within the radius of 2β from the center.
        """
        return {point
                for point in point_list
                if self.d(point, center) <= 2 * self.beta}

    def random_recluster(self, X, n_clusters):
        self.unclustered_points = set(X)
        k = -1

        # Remove the label associated to the points to be reclustered
        for data_point in X:
            self.labels.pop(data_point, None)

        while self.unclustered_points and k < n_clusters - 1:
            k += 1
            center = random.choice(tuple(self.unclustered_points))
            kth_cluster = self._get_closest_points(
                self.unclustered_points,
                center
            )

            self.unclustered_points -= kth_cluster
            for point in kth_cluster:
                self.labels[point] = len(self.centers)

            self.centers.append(center)
            self.collection.append(kth_cluster)

    @count_operation
    def delete(self, x):
        # Find the id of the cluster containing x, if any
        cluster_id = self.labels.pop(x, None)

        if cluster_id is not None:
            self.collection[cluster_id].remove(x)
        else:
            self.unclustered_points.remove(x)

        if x not in self.centers:
            return

        k = len(self.centers)

        x_hat = self.unclustered_points.copy()
        for cluster in self.collection[cluster_id:]:
            x_hat |= cluster

        self.centers = self.centers[:cluster_id]
        self.collection = self.collection[:cluster_id]

        self.random_recluster(x_hat, k - cluster_id)

    @count_operation
    def insert(self, x):
        for center in self.centers:
            if self.d(center, x) <= 2 * self.beta:
                # There exists a center c_i such that d(x, c_i) <= 2β:
                # insert x to the cluster C_i

                cluster_id = self.labels[center]
                self.labels[x] = cluster_id
                self.collection[cluster_id].add(x)

                return

        # Now d(x, c_i) > 2β for all centers c_i

        if len(self.centers) < self.n_clusters:
            # if we can still add another cluster:
            # make x the center of a new cluster

            self.labels[x] = len(self.centers)
            self.centers.append(x)
            self.collection.append({x})

            return

        # We have enough clusters and x cannot be added to any cluster
        self.unclustered_points.add(x)

    def visualize(self, num):
        k = len(self.collection)

        plt.figure(figsize=(12, 9))
        plt.xlim(-180, 180)
        plt.ylim(-180, 180)

        for i in range(k):
            plt.scatter(*zip(*self.collection[i]), s=2, c=self.colors[i])
            for x, y in self.collection[i]:
                plt.annotate(i, (x, y))

        if self.unclustered_points:
            plt.scatter(*zip(*self.unclustered_points), s=2, c='b')
            for x, y in self.unclustered_points:
                plt.annotate(-1, (x, y))

        # plot the centers
        for i in range(k):
            plt.scatter(*self.centers[i], marker='x', c=self.colors[i])

        file_name = 'figs/{}.png'.format(str(num).zfill(5))
        pylab.savefig(file_name)

        plt.close()
