# scripts/run_spectral.py

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.spectral_clustering import spectral_clustering_segmentation

def main(args):
    # Load image using OpenCV (BGR by default)
    img_bgr = cv2.imread(args.image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {args.image_path}")
    
    # Resize and convert BGR to RGB for display
    img_bgr = cv2.resize(img_bgr, (args.resize, args.resize))
    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert image to YCrCb for clustering
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # Apply spectral clustering in YCrCb space
    segmented_ycrcb, labels, centers = spectral_clustering_segmentation(
        img_ycrcb, K=args.K, sigma=args.sigma, metric=args.metric
    )

    # Convert segmented image back to RGB for display
    segmented_bgr = cv2.cvtColor(segmented_ycrcb, cv2.COLOR_YCrCb2BGR)
    segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb_original)
    plt.title("Original Image (RGB)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_rgb)
    plt.title(f"Spectral Clustering (K={args.K})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spectral Clustering on an Image in YCrCb Space")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--K", type=int, default=4, help="Number of clusters")
    parser.add_argument("--sigma", type=float, default=10.0, help="Sigma value for RBF kernel")
    parser.add_argument("--metric", type=str, default='chebyshev', help="Distance metric (e.g., chebyshev, euclidean)")
    parser.add_argument("--resize", type=int, default=32, help="Resize image to (resize x resize) - !!Not Suggested for ease of computation!!")

    args = parser.parse_args()
    main(args)
