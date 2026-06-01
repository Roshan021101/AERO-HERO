import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample Dataset
X = np.array([
    [30, 50000],
    [35, 60000],
    [40, 80000],
    [25, 300000],
    [45, 100000],
    [20, 200000],
    [50, 1200000],
    [50, 1200000],
    [55, 1500000],
    [60, 1400000],
    [28, 40000000]
])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and Train K-Means Model
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)

# Cluster Labels and Centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centroids)

# Visualize Clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap='viridis',
    s=50,
    alpha=0.8
)

# Plot Centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='red',
    s=200,
    marker='X',
    label='Centroids'
)

plt.xlabel('Age (Scaled)')
plt.ylabel('Income (Scaled)')
plt.title('K-Means Clustering')
plt.legend()
plt.grid(True)
plt.show()