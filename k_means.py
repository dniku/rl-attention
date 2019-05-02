import numpy as np
from sklearn import cluster as skl


def get_surrounding_points(j):
    # Gets the four points surrounding coordinate j

    points = [(int(np.floor(j[0])), int(np.floor(j[1]))),
              (int(np.floor(j[0])), int(np.ceil(j[1]))),
              (int(np.ceil(j[0])), int(np.floor(j[1]))),
              (int(np.ceil(j[0])), int(np.ceil(j[1])))]
    return points


def filter_k(input_tensor, order_method='max', k=2):
    """
    :param input_tensor: A h x w x filters np.array
    :param order_method: 'max' or 'sum'
    :param k: number of clusters per input

    Returns a list of tuples (y, x, filter_number) sorted by decreasing importance of peak.
    """

    peaks = []
    centroids = []
    for filter_n in range(np.shape(input_tensor)[2]):
        centroids.append(weighted_k(input_tensor[..., filter_n], k=k))

    for i in range(len(centroids)):
        for j in centroids[i]:
            points = get_surrounding_points(j)
            point_values = [(input_tensor[k[0], k[1], i]) ** 2 for k in points]
            if order_method == 'max':
                centroid_value = max(point_values)
            elif order_method == 'square':
                centroid_value = sum(point_values)
            else:
                raise KeyError
            best_point = points[int(np.argmax(point_values))]
            peaks.append((best_point[1], best_point[0], i, centroid_value))
    sorted_peaks = sorted(peaks, key=lambda x: -x[3])
    return [peak[:3] for peak in sorted_peaks[:2]]


def weighted_k(input_tensor, k=2):
    # Returns k points via k-means clustering on input_tensor

    x, y = np.shape(input_tensor)
    coordinates = [(a, b) for a in range(x) for b in range(y)]
    coordinates = np.array(coordinates)
    k_mean = skl.KMeans(n_clusters=k)
    input_tensor = np.reshape(input_tensor, [-1])
    means = k_mean.fit(coordinates, sample_weight=input_tensor).cluster_centers_
    return means


# test_case = np.random.uniform(low=0, high=15, size=(20, 20, 4))
# print(filter_k(test_case))
