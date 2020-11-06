from sklearn import datasets as datasets
from sklearn.neighbors import KDTree as KDTree
from sklearn.cluster import DBSCAN
import numpy as np

num_samples = 1000
num_features = 300
data_point, data_class = datasets.make_blobs(num_samples, num_features)
data_minimum_distance = [0] * num_samples
# what distance metric should be used in KDTree?
# what value should we set leaf_size? does this perform well on massive data sets?
k_d_tree = KDTree(data_point, leaf_size=40, metric='minkowski')
for i in range(num_samples):
    dist, index_of_near_point = k_d_tree.query([data_point[i]], k=2)
    data_minimum_distance[i] = dist[0][1]

target_distance = np.mean(data_minimum_distance)

# DBSCAN metric algorithm should be the same with KDTree
clustering = DBSCAN(eps=target_distance, min_samples=5, metric='minkowski', metric_params={"p": 2}).fit(data_point)
result = clustering._labels

