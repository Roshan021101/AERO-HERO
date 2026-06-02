import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = np.array([
    [30,50000],[35,60000],[40,80000],[25,300000],
    [45,100000],[20,200000],[50,1200000],[50,1200000],
    [55,1500000],[60,1400000],[28,40000000]
])

X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X_scaled)

print("Cluster Labels:")
print(kmeans.labels_)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0], X_scaled[:, 1],
    c=kmeans.labels_, cmap='viridis', s=50, alpha=0.8
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c='red', s=200, marker='X', label='Centroids'
)

plt.xlabel('Age (Scaled)')
plt.ylabel('Income (Scaled)')
plt.title('K-Means Clustering')
plt.legend()
plt.grid(True)
plt.show()
