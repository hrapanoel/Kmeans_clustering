# Compute k-means clustering

import numpy as np

def dist(a,b):
	'''Compute the eucledean distance between two points a and b
	Args: 
		2 points
	Returns: 
		distance between a and b
	'''
	return np.linalg.norm(a-b)

def k_means(input_data, K, eps=1e-15):
	'''Cluster the data points in input_data into K clusters
	Args:
		input_data: array of shape (n_samples, n_features)
		K: int, number of clusters
		eps: float, tolerance
	Returns:
		clusters: 1D array of length n_samples, cluster id that each point belongs to
		centroids: array of shape (K, n_features), coordinates of centroids
	'''

	n_samples, n_features = input_data.shape

	# Check K <= n_samples
	if K > n_samples:
		raise ValueError("The number of samples %d should be greater than the number of clusters %d" % (n_samples, K))

	clusters = np.zeros(n_samples)
	#random initialization of centroids
	centroids_new = input_data[np.random.randint(n_samples, size=K)]
	centroids_old = np.zeros(centroids_new.shape)
	#distance between centroids
	centroids_dist = dist(centroids_new, centroids_old) #???

	while centroids_dist>eps: 

		# cluster assignment step
		for i in range(n_samples):
			x = input_data[i]
			#Compute distance between x and the centroids
			d = np.zeros(K)
			for k in range(K):
				d[k] = dist(x, centroids_new[k])
			# cluster assignment
			clusters[i] = np.argmin(d)

		# move centroid step
		centroids = np.zeros(centroids_new.shape)
		for k in range(K):
			data_cluster_k = input_data[np.where(clusters==k)]
			centroids[k] = np.mean(data_cluster_k, axis=0)

		centroids_old = centroids_new
		centroids_new = centroids

		centroids_dist = dist(centroids_new, centroids_old)

	return clusters, centroids_new

class Kmeans():
	'''K-means clustering. Cluster the data points in input_data into K clusters
	Args:
		input_data: array of shape (n_samples, n_features)
		K: int, number of clusters
		eps: float, tolerance
	Returns:
		clusters: 1D array of length n_samples, cluster id that each point belongs to
		centroids: array of shape (K, n_features), coordinates of centroids
	'''
	def __init__(self, K, eps=1e-15):
		self.K = K
		self.eps = eps


	def fit(self, X):
		'''K-means clustering
		Args: 
			X: array of shape (n_samples, n_features)
		Returns:
			cluster_id: 1D array of length n_samples, cluster id that each point belongs to
			centroids: array of shape (K, n_features), coordinates of centroids
			'''

		self.cluster_id, self.centroids = k_means(X, K=self.K, eps=self.eps)
		return self
	