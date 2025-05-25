# scripts/run_kmeans.py

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from src.feature_extraction import compute_sobel_magnitude
from src.kmeans_clustering import kmeans  # fallback if not using combined features

def extract_combined_features(img_rgb, color_weight=1.0, pos_weight=0.1):
    """
    Extracts color + spatial + sobel texture features from the image.
    
    Parameters:
    -----------
    img_rgb : np.ndarray
        Input image in RGB format.
    color_weight : float
        Weight for color features.
    pos_weight : float
        Weight for spatial features.

    Returns:
    --------
    features : np.ndarray
        Combined feature vector for K-means.
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # --- Color Features ---
    color = img_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    color *= color_weight

    # --- Spatial Features ---
    coords = np.indices((h, w)).transpose(1, 2, 0).reshape(-1, 2)
    coords = coords / np.array([[h, w]]) * pos_weight

    # --- Sobel Texture Feature ---
    sobel = compute_sobel_magnitude(gray).reshape(-1, 1)

    return np.hstack([color, coords, sobel])


def main(args):
    # Load image using OpenCV (BGR by default)
    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {args.image_path}")

    # Resize and convert to RGB
    img_bgr = cv2.resize(img_bgr, (args.resize, args.resize))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # === Feature Extraction ===
    if args.use_combined_features:
        features = extract_combined_features(
            img_rgb,
            color_weight=args.color_weight,
            pos_weight=args.pos_weight
        )
        kmeans = KMeans(n_clusters=args.K, max_iter=args.iterations, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_[:, :3]
        segmented = centers[labels].reshape(args.resize, args.resize, 3)
        segmented_rgb = (segmented * 255).astype(np.uint8)
    else:
        # Default path using color-only (YCrCb)
        img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        segmented_ycrcb = kmeans(
            img=img_ycrcb,
            K=args.K,
            it=args.iterations,
            distance_metric=args.metric
        )
        segmented_bgr = cv2.cvtColor(segmented_ycrcb, cv2.COLOR_YCrCb2BGR)
        segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

    # === Plotting Results ===
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image (RGB)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_rgb)
    plt.title(f"K-Means Segmentation (K={args.K})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run K-Means with color + position + texture (Sobel) features")

    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--K", type=int, default=4, help="Number of clusters")
    parser.add_argument("--iterations", type=int, default=10, help="KMeans iterations")
    parser.add_argument("--metric", type=str, default='chebyshev', help="Distance metric for basic KMeans")
    parser.add_argument("--resize", type=int, default=32, help="Resize image to (resize x resize)")

    parser.add_argument("--use_combined_features", action="store_true",
                        help="Enable to use combined [color + position + sobel texture] features")
    parser.add_argument("--pos_weight", type=float, default=0.1, help="Weight for spatial coordinates")
    parser.add_argument("--color_weight", type=float, default=1.0, help="Weight for color features")

    args = parser.parse_args()
    main(args)
