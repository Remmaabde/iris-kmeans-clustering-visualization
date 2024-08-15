# iris-kmeans-clustering-visualization
# Iris K-Means Clustering Analysis

## Overview

This project applies K-Means clustering to the well-known Iris dataset. The main objectives are to normalize the dataset, perform clustering with k=3, visualize the results, and evaluate the clustering quality.

## Features

- Data Normalization: Standardized the Iris dataset to have a mean of 0 and a standard deviation of 1.
- K-Means Clustering: Applied K-Means algorithm with k=3 to group the dataset into three clusters.
- Visualization:
  - 2D and 3D plots of the clusters and centroids.
  - Comparison of cluster assignments with actual class labels.
- Performance Evaluation: 
  - Computed and visualized silhouette scores to assess the quality of the clusters.

## Setup Instructions

1. # Clone the repository:
     bash
   git clone https://github.com/your-username/iris-kmeans-clustering-analysis.git
   cd iris-kmeans-clustering-analysis
2. # Install Required Dependencies
    # pip install -r requirements.txt
3. Run the code:
    # python kmeans_iris.py
# Project Structure
kmeans_iris.py - Main script to perform K-Means clustering and visualizations.
requirements.txt - List of dependencies required to run the project.
README.md - Project documentation.
# Results
2D and 3D Visualizations: The clusters and centroids are visualized using Matplotlib and Seaborn.
Cluster vs Actual Labels: Compared cluster assignments with actual labels to evaluate clustering accuracy.
Silhouette Scores: Plotted silhouette scores to analyze the quality of the clusters formed.
