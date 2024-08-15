import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_normalized)
centroids = kmeans.cluster_centers_

# 2D Visualization of Clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=y_kmeans, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
plt.title('K-Means Clustering (2D)')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

# 3D Visualization of Clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_normalized[:, 0], X_normalized[:, 1], X_normalized[:, 2], c=y_kmeans, cmap='viridis', marker='o')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=300, c='red', marker='x')
ax.set_title('K-Means Clustering (3D)')
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[2])
plt.show()

# Compare cluster assignments with actual class labels
conf_matrix = confusion_matrix(y, y_kmeans)
print("Confusion Matrix:\n", conf_matrix)

# Silhouette Score Evaluation
silhouette_avg = silhouette_score(X_normalized, y_kmeans)
print("Average Silhouette Score: ", silhouette_avg)

# Plot Silhouette Score
plt.figure(figsize=(10, 7))
plt.bar(range(len(kmeans.labels_)), silhouette_score(X_normalized, y_kmeans, sample_size=len(X_normalized)))
plt.title('Silhouette Scores for Each Sample')
plt.xlabel('Sample Index')
plt.ylabel('Silhouette Score')
plt.show()
