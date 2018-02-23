from math import sqrt

import numpy as np
import pylab
from scipy import spatial

DATASET_PATH = 'dataset/twitter_1000000.txt'


def read_data(limit):
    """
    Read all the points into a numpy array, limited to the first `limit` points.
    If a limit of 0 provided, all points are returned.
    """
    data = np.fromfile(DATASET_PATH, sep=' ')
    data = np.reshape(data, (1000000, 3))

    data = data[:limit] if limit else data
    data = np.delete(data, 0, axis=1)  # delete the timestamp column

    return data


def data_stream(limit):
    """
    Return a generator to get the first `limit` points one by one.
    """
    with open(DATASET_PATH) as f:
        for _ in range(limit):
            p = f.readline()
            if not p:
                break

            p = tuple(map(float, p.split()[1:]))
            yield p


def min_dist(point_list):
    """
    Find the minimum distance between two points in a point list using
    Delaunay triangulation.
    """

    # Find the Delaunay triangulation
    mesh = spatial.Delaunay(point_list)

    # Get the edges of the triangulation
    edges = np.vstack((mesh.simplices[:, :2], mesh.simplices[:, -2:]))

    # The x- and y- coordinates of the edges
    x = mesh.points[edges[:, 0]]
    y = mesh.points[edges[:, 1]]

    # Calculate the lengths of the Delaunay edges, which are candidates for minimum distance
    dists = np.sqrt(np.sum((x - y) ** 2, axis=1))

    idx = np.argmin(dists)
    idx1, idx2 = edges[idx]
    p1, p2 = point_list[edges[idx]]

    print('The closest points are:')
    print('Index: {}, coordinates: {}'.format(idx1, tuple(p1)))
    print('Index: {}, coordinates: {}'.format(idx2, tuple(p2)))

    return np.min(dists)


def max_dist(point_list):
    """
    Find the maximum distance between two points in a point list using convex hull.
    """

    # Find the convex hull of the point set
    conv_hull = spatial.ConvexHull(point_list)
    
    # Find the vertices on the convex hull
    vertices = np.vstack(point_list[conv_hull.vertices])

    # Find the pairwise distances
    dists = spatial.distance.pdist(vertices)
    dists = spatial.distance.squareform(dists)

    # Find the indices of the furthest points
    idx = np.unravel_index(np.argmax(dists), dists.shape)
    idx = list(idx)

    idx1, idx2 = conv_hull.vertices[idx]
    p1, p2 = vertices[idx]

    print('The furthest points are:')
    print('Index: {}, coordinates: {}'.format(idx1, tuple(p1)))
    print('Index: {}, coordinates: {}'.format(idx2, tuple(p2)))

    return np.max(dists)


def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_colors(num_colors):
    color_map = pylab.get_cmap('tab20')  # other choice: hsv

    return [color_map(i / num_colors) for i in range(num_colors)]
