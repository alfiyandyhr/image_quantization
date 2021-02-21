from matplotlib.image import imread
import matplotlib.pyplot as plt
from kmeans import KMeans
import torch

image = imread('images/IMG_0015.jpg')
X = image.reshape(-1, 3)
X_t = torch.from_numpy(X).float()

kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(X_t)
segmented_img = kmeans.centroids[labels]
segmented_img = segmented_img.view(image.shape)

new_img = segmented_img.numpy()

plt.imsave('images/5.jpg',new_img.astype('uint8'))