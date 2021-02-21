import torch
import math
import matplotlib.pyplot as plt

class KMeans:
	"""
	Implementation of KMeans clustering in pytorch
	Parameters:
		n_clusters	: number of clusters (int)
		max_iter	: maximum number of iteration (int)
		tol			: tolerance (float)
	Attributes:
		centroids	: cluster centroids (torch.Tensor)
	"""
	def __init__(self,
				 n_clusters,
				 max_iter=100,
				 tol=0.0001,
				 verbose=0):

		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.tol = tol
		self.centroids = None
		self.verbose = verbose

	@staticmethod
	def euclidean_similarity(a, b):
		"""
		Compute euclidean similarity of 2 sets of vectors
		Parameters:
			a: torch.Tensor, shape: [m, n_features]
			b: torch.Tensor, shape: [n, n_features]
		"""

		return 2 * a @ b.transpose(-2,-1) - (a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[...,None,:]

	def remaining_memory(self):
		"""
		Get the remaining memory in GPU
		"""
		torch.cuda.synchronize()
		torch.cuda.empty_cache()
		remaining = torch.cuda.memory_allocated()
		return remaining

	def maximum_similarity(self, a, b):
		"""
		Compute maximum similarity or minimum distance of each vector
		in 'a' with all vectors in 'b'
		Parameters:
			a: torch.Tensor, shape: [m, n_features]
			b: torch.Tensor, shape: [n, n_features]
		"""

		device = a.device.type
		batchsize = a.shape[0]
		
		similarity_function = self.euclidean_similarity

		if device == 'cpu':
			sim = similarity_function(a, b)
			max_sim_v, max_sim_i = sim.max(dim=-1)
			return max_sim_v, max_sim_i
		else:
			if a.dtype == torch.float:
				expected = a.shape[0] * a.shape[1] * b.shape[0] * 4
			elif a.dtype == torch.half:
				expected = a.shape[0] * a.shape[1] * b.shape[0] * 2
			ratio = math.ceil(expected / self.remaining_memory())
			sub_batchsize = math.ceil(batchsize/ratio)
			msv, msi = [], []
			for i in range(ratio):
				if i*sub_batchsize >= batchsize:
					continue
				sub_x = a[i*sub_batchsize: (i+1)*sub_batchsize]
				sub_sim = similarity_function(sub_x, b)
				sub_max_sim_v, sub_max_sim_i = sub_sim.max(dim=-1)
				del sub_sim
				msv.append(sub_max_sim_v)
				msi.append(sub_max_sim_i)
			if ratio == 1:
				max_sim_v, max_sim_i = msv[0], msi[0]
			else:
				max_sim_v = torch.cat(msv, dim=0)
				max_sim_i = torch.cat(msi, dim=0)

			return max_sim_v, max_sim_i

	def fit_predict(self, X, centroids=None):
		"""
		Fitting the data
		Parameters:
			X: torch.Tensor, shape: [n_samples, n_features]

			centroids: {torch.Tensor, None}, default: None
				if given, centroids will be initilized with given tensor
				if None, centroids will be randomly chosen from X

		Return:
			labels: torch.Tensor, shape: [n_samples]	
		"""

		batchsize, emb_dim = X.shape
		device = X.device.type

		if centroids is None:
			self.centroids = X[torch.randint(low=0,high=99,size=(self.n_clusters,),device=device)]
		
		else:
			self.centroids = centroids

		num_points_in_clusters = torch.ones(self.n_clusters, device=device)

		closest = None

		for i in range(self.max_iter):
			x = X

			closest = self.maximum_similarity(a=x, b=self.centroids)[1]
			matched_clusters, counts = closest.unique(return_counts=True)

			c_grad = torch.zeros_like(self.centroids)
			expanded_closest = closest[None].expand(self.n_clusters, -1)
			mask = (expanded_closest==torch.arange(self.n_clusters, device=device)[:, None]).float()
			c_grad = mask @ x /mask.sum(-1)[..., :, None]
			c_grad [c_grad!=c_grad] = 0

			error = (c_grad - self.centroids).pow(2).sum()
			lr = 1

			num_points_in_clusters[matched_clusters] += counts
			self.centroids = self.centroids * (1-lr) + c_grad * lr
			if self.verbose >= 2:
				print('iter:', i, 'error:', error.item())
			if error <= self.tol:
				break

		return closest

	def inertia_(self, X, labels):
		"""
		Calculating the mean squared distance of X from the centroids
		Parameters:
			X: torch.Tensor, shape: [n_samples, n_features]
			labels: torch.Tensor, shape: [n_samples]
		Return:
			inertia: the mean squared distance
		"""
		device = X.device.type
		inertia = torch.tensor(0.0).to(device)
		for i in range(len(X)):
			inertia += torch.sqrt(torch.sum(torch.pow((X[i] - self.centroids[labels[i]]),2)))

		return inertia/len(X)