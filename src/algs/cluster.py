import hdbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

def cluster_by_agglomerative(M, distance_threshold=0.5, linkage='complete'):
    """
    Cluster data using Agglomerative Clustering on a precomputed distance matrix.

    Parameters:
    - M (numpy array): A precomputed distance matrix.

    Returns:
    - numpy array: Cluster labels.
    """
    clusterer = AgglomerativeClustering(
        n_clusters = None,
        distance_threshold = distance_threshold,
        metric = 'precomputed',
        linkage = linkage)
    
    return clusterer.fit_predict(M)

def cluster_by_dbscan(M, eps=0.18, min_samples=2):
    """
    Cluster data using DBSCAN on a precomputed distance matrix.

    Parameters:
    - M (numpy array): A precomputed distance matrix.

    Returns:
    - numpy array: Cluster labels.
    """
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clusterer.fit(M)
    return clusterer.labels_

def cluster_by_hdbscan(M, min_cluster_size=2):
    """
    Cluster data using HDBSCAN on a precomputed distance matrix.

    Parameters:
    - M (numpy array): A precomputed distance matrix.

    Returns:
    - numpy array: Cluster labels.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size, 
        gen_min_span_tree = True,
        metric = 'precomputed')
    clusterer.fit(M)
    
    return clusterer.labels_

def find_cluster_centers(labels, dist_mat, mode='similarity'):
    """
    Find the most central member of each cluster based on intra-cluster similarity or distance.

    Parameters:
    - labels (numpy array): Array of cluster labels for each data point.
    - dist_mat (numpy array): Precomputed distance or similarity matrix for all data points.
    - mode (str): 'similarity' to pick the point with the highest sum of similarities,
                  'distance' to pick the point with the lowest sum of distances.

    Returns:
    - centers (list): List of indices of the central members of each cluster.
    """
    unique_labels = np.unique(labels)
    centers = {}

    for label in unique_labels:
        if label == -1:
            continue
        
        indices = np.where(labels == label)[0]
        sub_mat = dist_mat[np.ix_(indices, indices)]

        if mode == 'distance':
            sum_distances = np.sum(sub_mat, axis=1)
            center_index = np.argmin(sum_distances)
        elif mode == 'similarity':
            sum_similarities = np.sum(sub_mat, axis=1)
            center_index = np.argmax(sum_similarities)

        centers[label] = {'center': indices[center_index], 'lst': indices}

    return centers


class Clusterer:
    def __init__(self, method='hdbscan', **kwargs):
        """
        Initialize the Clusterer class with the chosen clustering method and parameters.

        Parameters:
        - method (str): The clustering method to use ('agglomerative', 'dbscan', 'hdbscan').
        - **kwargs: Additional parameters for the clustering algorithm.
        """
        self.method = method
        self.params = kwargs

    def fit_predict(self, mat):
        """
        Fit the chosen clustering algorithm and predict the cluster labels.

        Parameters:
        - M (numpy array): A precomputed distance matrix.

        Returns:
        - numpy array: Cluster labels.
        """
        if self.method == 'agglomerative':
            # Default parameters for AgglomerativeClustering
            default_params = {'n_clusters': None, 'distance_threshold': 0.5, 'metric': 'precomputed', 'linkage': 'complete'}
            # Update default parameters with any user-provided parameters
            default_params.update(self.params)
            clusterer = AgglomerativeClustering(**default_params)

        elif self.method == 'dbscan':
            # Default parameters for DBSCAN
            default_params = {'eps': 0.18, 'min_samples': 1, 'metric': 'precomputed'}
            # Update default parameters with any user-provided parameters
            default_params.update(self.params)
            clusterer = DBSCAN(**default_params)

        elif self.method == 'hdbscan':
            # Default parameters for HDBSCAN
            default_params = {'min_cluster_size': 2, 'metric': 'precomputed'}
            # Update default parameters with any user-provided parameters
            default_params.update(self.params)
            clusterer = hdbscan.HDBSCAN(**default_params)
            clusterer.fit(mat)
            return clusterer.labels_

        else:
            raise ValueError("Unsupported clustering method")

        labels = clusterer.fit_predict(mat)
        
        return labels

    def sort_cluster_members(self, mat, labels):
        """
        Sorts members within each cluster based on average distance or similarity.

        Parameters:
        - M (numpy array): The precomputed distance or similarity matrix.
        - labels (numpy array): Cluster labels returned from fit_predict.

        Returns:
        - dict: A dictionary where keys are cluster labels and values are sorted lists of indices.
        """
        cluster_indices = {}
        for label in np.unique(labels):
            if label == -1:  # Skipping noise if using DBSCAN or similar
                continue
            
            # Extract indices for current cluster
            indices = np.where(labels == label)[0]
            if len(indices) > 1:
                # Calculate mean distance or similarity for each member to others in the cluster
                sub_matrix = mat[np.ix_(indices, indices)]
            
                if self.method in ['dbscan', 'hdbscan']:  # Assuming distance matrix
                    scores = np.mean(sub_matrix, axis=1)
                    sorted_indices = indices[np.argsort(scores)]
                else:  # Assuming similarity matrix for others
                    scores = np.mean(sub_matrix, axis=1)
                    sorted_indices = indices[np.argsort(-scores)]
            else:
                sorted_indices = indices
            
            cluster_indices[label] = sorted_indices
        
        return cluster_indices
    

def display_cluster_images(cluster_indices, crops_series, n=5):
    """
    Display images from clusters.

    Parameters:
    - cluster_indices (dict): Dictionary of sorted cluster indices from sort_cluster_members.
    - crops_series (pd.Series): Pandas Series with frame IDs as index and crop image paths as values.
    - n (int): Maximum number of images per row to display.
    """
    # 创建一个足够大的figure来显示所有的图片
    num_clusters = len(cluster_indices)
    fig, axes = plt.subplots(nrows=num_clusters, ncols=n, figsize=(n * 4, num_clusters * 4))
    
    for i, (label, indices) in enumerate(cluster_indices.items()):
        if num_clusters == 1:  # 如果只有一个类别，确保axes是一维数组
            ax_row = axes
        else:
            ax_row = axes[i]
        
        for j, idx in enumerate(indices[:n]):  # 只展示每个类的前n个图片
            if j < len(indices):  # 确保当前索引在范围内
                img_path = crops_series.loc[crops_series.index[idx]]
                image = Image.open(img_path)
                ax_row[j].imshow(image)
                ax_row[j].set_title(f'Frame ID: {crops_series.index[idx]}', fontsize=12)
                ax_row[j].axis('off')  # 不显示坐标轴
            
            if j <= n - 1:
                for k in range(j, n):
                    ax_row[k].axis('off')  # 超出部分留空

    plt.tight_layout(pad=.1)
    plt.show()
