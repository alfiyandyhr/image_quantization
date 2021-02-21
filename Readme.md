Image Quantization using KMeans Clustering
==============================================================================

[Quantization](https://en.wikipedia.org/wiki/Quantization_(image_processing)) in image processing is a compression technique achieved by compressing a range of values of colors to a single quantum value. In this code, the quantization process is done using KMeans clustering algorithm. The theory behind this code is derived from the reference book listed at the bottom of the page, the difference is that the code is manually implemented in PyTorch instead of using sklearn library.

# Dependencies
PyTorch, matplotlib

# Usage
********************************************************************************
```python

        from matplotlib.image import imread
	import matplotlib.pyplot as plt
	from kmeans import KMeans
	import torch

	#Importing the image file
	image = imread('images/IMG_0015.jpg')
	X = image.reshape(-1, 3)
	X_t = torch.from_numpy(X).float()

	#Clustering the image colors
	kmeans = KMeans(n_clusters=5)
	labels = kmeans.fit_predict(X_t)
	segmented_img = kmeans.centroids[labels]
	segmented_img = segmented_img.view(image.shape)

	#Converting tensors to numpy ndarrays
	new_img = segmented_img.numpy()

	plt.imsave('images/5.jpg',new_img.astype('uint8'))
```
# Results

### Original Image
<p float="center">
  <img src="/images/IMG_0015.jpg" width="49%"/>
</p>

### Clustered into 8 clusters
<p float="center">
  <img src="/images/8.jpg" width="49%"/>
</p>

### Clustered into 5 clusters
<p float="center">
  <img src="/images/5.jpg" width="49%"/>
</p>

### Clustered into 2 clusters
<p float="center">
  <img src="/images/2.jpg" width="49%"/>
</p>

# Reference
Aurélien Géron. 2017. Hands-on machine learning with Scikit-Learn and TensorFlow: concepts, tools, and techniques to build intelligent systems. " O'Reilly Media, Inc.".
