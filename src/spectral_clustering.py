# src/clustering.py

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

def spectral_clustering_segmentation(
    img: np.ndarray,
    K: int,
    sigma: float = 10.0,
    metric: str = 'chebyshev'
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs spectral clustering-based image segmentation.

    Parameters:
    -----------
    img : np.ndarray
        Input image as a 3D array (height x width x channels).
    K : int
        Number of clusters for segmentation.
    sigma : float
        Sigma value for RBF kernel (controls similarity sensitivity).
    metric : str
        Distance metric used to compute pairwise pixel distances (default: 'chebyshev').

    Returns:
    --------
    segmented_img : np.ndarray
        Segmented image with cluster-averaged colors.
    labels : np.ndarray
        Cluster label for each pixel.
    cluster_means : np.ndarray
        Mean RGB value of each cluster.
    """
    height, width, channel = img.shape
    N = height * width
    pixels = img.reshape(N, channel).astype(float)

    # Compute RBF similarity matrix using Chebyshev distance [Default Case]
    dists = cdist(pixels, pixels, metric = metric)
    connection = np.exp(-(dists**2) / (2 * sigma**2))

    # Construct the normalized Laplacian
    D = np.diag(connection.sum(axis=1))
    eps = 1e-10
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + eps))
    L_sym = np.eye(N) - D_inv_sqrt @ connection @ D_inv_sqrt

    # Compute the first K eigenvectors of L_sym
    eigvals, eigvecs = eigh(L_sym)
    eigvecs_k = eigvecs[:, :K]

    # Normalize rows
    normed_eigvecs = eigvecs_k / np.linalg.norm(eigvecs_k, axis=1, keepdims=True)

    # Apply KMeans to the eigenvector representation
    kmeans = KMeans(n_clusters=K, random_state=40)
    labels = kmeans.fit_predict(normed_eigvecs)

    # Map cluster labels back to pixel colors
    cluster_means = np.array([
        pixels[labels == i].mean(axis=0) if np.any(labels == i) else np.zeros(3)
        for i in range(K)
    ])
    segmented_img = cluster_means[labels].reshape(height, width, channel).astype(np.uint8)

    return segmented_img, labels, cluster_means
