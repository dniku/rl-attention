import random
import numpy as np
from itertools import chain
from matplotlib import pyplot as plt
from sklearn import cluster as skl

testcase = np.random.uniform(low=0, high=15, size = (7,7,4))


def get_surrounding_points(j):
    points = []
    points.append((int(np.floor(j[0])), int(np.floor(j[1]))))
    points.append((int(np.floor(j[0])), int(np.ceil(j[1]))))
    points.append((int(np.ceil(j[0])), int(np.floor(j[1]))))
    points.append((int(np.ceil(j[0])), int(np.ceil(j[1]))))
    return points


def filter_k(input_tensor, grid=False, order_method='max'):
    '''
    :param input_tensor: A h x w x filters np.array
    :param grid: Set to true if need integer valued coordinates.

    Returns a list of tuples (filter_number, [filter_coords])

    '''
    centroids = []
    for filter in range(np.shape(input_tensor)[2]):
        centroids.append(weighted_k(input_tensor[..., filter]))
    if grid:
        int_centroids = []
        for i in range(len(centroids)):
            best_points = []
            for j in centroids[i]:
                points = get_surrounding_points(j)
                point_values = [(input_tensor[k[0], k[1], i]) ** 2 for k in points]
                if order_method == 'max':
                    centroid_value = max(point_values)
                best_points.append(points[np.argmax(point_values)])
            int_centroids.append((best_points, centroid_value))
        return_values = list(enumerate(int_centroids))
        return_values = sorted(return_values, key=lambda x: -x[1][1])
        return return_values
    return_values = list(enumerate(centroids))
    return_values = sorted(return_values, key=lambda x: -x[1][1])
    return return_values


def weighted_k(testcase, k=2):
    x,y = np.shape(testcase)
    coords = [(a, b) for a in range(x) for b in range(y)]
    coords = np.array(coords)
    kmean = skl.KMeans(n_clusters=k)
    testcase = np.reshape(testcase, [-1])
    means =  kmean.fit(coords, sample_weight=testcase).cluster_centers_
    return means


filters = filter_k(testcase, grid=True)
print(filters)
