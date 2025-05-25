# src/kmeans_clustering.py

import numpy as np

def kmeans(img, K=4, it=10, distance_metric='chebyshev'):
    """
    Applies K-Means clustering to an image using the specified distance metric.

    Parameters:
    -----------
    img : np.ndarray
        Input image (grayscale or RGB).
    K : int
        Number of clusters.
    it : int
        Number of iterations.
    distance_metric : str
        Distance metric to use ('euclidean' or 'chebyshev').

    Returns:
    --------
    segmented_img : np.ndarray
        Segmented image with clustered pixel values.
    """
    if img.ndim == 2:
        h, w = img.shape
        pixels = img.reshape(-1, 1).astype(np.float32)
    else:
        h, w, c = img.shape
        pixels = img.reshape(-1, c).astype(np.float32)

    np.random.seed(40)
    centroids = pixels[np.random.choice(pixels.shape[0], K, replace=False)]

    for _ in range(it):
        if distance_metric == 'chebyshev':
            distances = np.max(np.abs(pixels[:, None] - centroids[None, :]), axis=2)
        elif distance_metric == 'euclidean':
            distances = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
        else:
            raise ValueError("Unsupported distance metric. Use 'euclidean' or 'chebyshev'.")

        labels = np.argmin(distances, axis=1)

        for k in range(K):
            if np.any(labels == k):
                centroids[k] = pixels[labels == k].mean(axis=0)

    segmented_pixels = centroids[labels].reshape((h, w, -1))
    if segmented_pixels.shape[2] == 1:
        segmented_pixels = segmented_pixels.squeeze(-1)

    return segmented_pixels.astype(np.uint8)
