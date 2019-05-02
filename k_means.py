import numpy as np
from sklearn import cluster as skl


def get_surrounding_points(j):
    # Gets the four points surrounding coordinate j

    points = [(int(np.floor(j[0])), int(np.floor(j[1]))),
              (int(np.floor(j[0])), int(np.ceil(j[1]))),
              (int(np.ceil(j[0])), int(np.floor(j[1]))),
              (int(np.ceil(j[0])), int(np.ceil(j[1])))]
    return points


def filter_k(input_tensor, order_method='max', k=2, top_x=2):
    """
    :param input_tensor: A h x w x filters np.array
    :param order_method: 'max' or 'sum'
    :param k: number of clusters per input filter
    :param top_x: number of peaks to return

    Returns a list of tuples (y, x, filter_number) of length top_x, sorted by decreasing importance of peak.
    """
    input_tensor[input_tensor == 0] = np.finfo(float).eps

    peaks = []
    centroids = []
    for filter_n in range(np.shape(input_tensor)[2]):
        centroids.append(weighted_k(input_tensor[..., filter_n], k=k))

    for i in range(len(centroids)):
        for j in centroids[i]:
            points = get_surrounding_points(j)
            points = np.clip(points, 0, np.shape(input_tensor)[0]-1)
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
    return [peak[:3] for peak in sorted_peaks[:top_x]]


def weighted_k(input_tensor, k=2):
    # Returns k points via k-means clustering on input_tensor

    x, y = np.shape(input_tensor)
    coordinates = [(a, b) for a in range(x) for b in range(y)]
    coordinates = np.array(coordinates)
    k_mean = skl.KMeans(n_clusters=k, tol=1e-2, algorithm='elkan', n_init=2)
    input_tensor = np.reshape(input_tensor, [-1])
    means = k_mean.fit(coordinates, sample_weight=input_tensor).cluster_centers_
    return means


# test_case = np.random.uniform(low=0, high=10, size=(7, 7, 32))
# for i in range(100):
#     print(i)
#     filter_k(test_case)
