import numpy as np
from sklearn.cluster import MeanShift 
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

centers = [[1,1],[5,5]]

x,y = make_blobs(n_samples=200, centers= centers, cluster_std=1) # fas les mostres i lo juntes que estaras
#plt.scatter(x[:,0],x[:,1])
#plt.show()

ms = MeanShift()

ms.fit(x)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)

n_clusters_ = len(np.unique(labels))
print("Number of estimated cluster:", n_clusters_)

colors = 10*['r.','g.','b.','c.','k.','y.','m.']

print(colors)
print(labels)

for i in range(len(x)):
	plt.plot(x[i][0],x[i][1], colors[labels[i]], markersize=10)

plt.scatter(cluster_centers[:,0],cluster_centers[:,1], marker="x", s=150, linewidths=5,zorder=10)
plt.show()