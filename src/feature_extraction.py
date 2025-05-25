# src/feature_extraction.py

import numpy as np
import cv2
from skimage.filters import gabor
from skimage.feature import local_binary_pattern

def compute_sobel_magnitude(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    return normalize(edge_mag)

def normalize(img):
    return (img - img.min()) / (img.ptp() + 1e-5)

def extract_features(img_rgb, flat_color, pos_weight=0.1, color_weight=1.0):
    """
    Extracts combined color, spatial, and Sobel texture features for each pixel.

    Parameters:
    -----------
    img_rgb : np.ndarray
        Original RGB image of shape (H, W, 3).
    flat_color : np.ndarray
        Flattened color feature array of shape (H*W, 3).
    pos_weight : float
        Weight for spatial coordinates.
    color_weight : float
        Weight for color features.

    Returns:
    --------
    features : np.ndarray
        Combined feature matrix of shape (H*W, 6).
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Color
    color = flat_color.reshape(h, w, 3).astype(np.float32) / 255.0
    color = color.reshape(-1, 3) * color_weight

    # Spatial
    coords = np.indices((h, w)).transpose(1, 2, 0).reshape(-1, 2)
    coords = coords / np.array([[h, w]]) * pos_weight

    # Texture (Sobel)
    sobel = compute_sobel_magnitude(gray).reshape(-1, 1)

    return np.hstack([color, coords, sobel])
