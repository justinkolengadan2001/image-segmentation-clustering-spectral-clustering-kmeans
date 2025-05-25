import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score,
)
from collections import defaultdict

# ------------------------ Texture Feature ------------------------

def compute_sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """
    Computes normalized Sobel edge magnitude from grayscale image.
    """
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    return (edge_mag - edge_mag.min()) / (edge_mag.ptp() + 1e-5)

# ------------------------ Feature Engineering ------------------------

def get_combined_features_sobel(
    img: np.ndarray, flat_color: np.ndarray, pos_weight: float = 0.1, color_weight: float = 1.0
) -> np.ndarray:
    """
    Combines color + spatial + sobel texture features into a single vector per pixel.
    Assumes image is 32x32.
    """
    h, w = 32, 32
    color = flat_color.reshape(h, w, 3).astype(np.float32) / 255.0
    color = color.reshape(-1, 3) * color_weight

    coords = np.indices((h, w)).transpose(1, 2, 0).reshape(-1, 2)
    coords = coords / np.array([[h, w]]) * pos_weight

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = compute_sobel_magnitude(gray).reshape(-1, 1)

    return np.hstack([color, coords, sobel])

# ------------------------ Metric Evaluation ------------------------

def evaluate_k_range_all_30_images(
    bicubic_imgs, X_rgb, X_hsv, X_lab, X_yCrCb, k_range=range(2, 11),
    pos_weight: float = 0.1, color_weight: float = 1.0
) -> pd.DataFrame:
    """
    Evaluates clustering metrics (Silhouette, DBI, CHI, Inertia) for K=2..10
    over the first 30 images and 4 color spaces using KMeans on
    color + spatial + Sobel texture features.
    """
    results = defaultdict(list)
    color_spaces = {
        "RGB": X_rgb,
        "HSV": X_hsv,
        "LAB": X_lab,
        "YCrCb": X_yCrCb
    }

    for color_name, X_space in color_spaces.items():
        for idx in range(30):
            img = bicubic_imgs[idx]
            flat_color = X_space[idx]
            features = get_combined_features_sobel(img, flat_color, pos_weight, color_weight)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
                labels = kmeans.fit_predict(features)

                if len(set(labels)) < 2:
                    continue  # skip degenerate cluster

                results["ColorSpace"].append(color_name)
                results["ImageIndex"].append(idx)
                results["K"].append(k)
                results["Inertia"].append(kmeans.inertia_)
                results["DaviesBouldin"].append(davies_bouldin_score(features, labels))
                results["CalinskiHarabasz"].append(calinski_harabasz_score(features, labels))
                results["Silhouette"].append(silhouette_score(features, labels))

    return pd.DataFrame(results)

# ------------------------ Best K Selection ------------------------

def find_best_k_per_colorspace(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each color space, finds the best K based on voting from top-3 K values
    for each metric: Davies-Bouldin (min), Calinski-Harabasz (max), 
    Silhouette (max), Inertia (min).
    """
    best_k_summary = {}

    for color in df["ColorSpace"].unique():
        subset = df[df["ColorSpace"] == color]

        # Metric-based rankings
        db_rank = subset.groupby("K")["DaviesBouldin"].mean().sort_values().head(3).index
        chi_rank = subset.groupby("K")["CalinskiHarabasz"].mean().sort_values(ascending=False).head(3).index
        sil_rank = subset.groupby("K")["Silhouette"].mean().sort_values(ascending=False).head(3).index
        inertia_rank = subset.groupby("K")["Inertia"].mean().sort_values().head(3).index

        # Voting mechanism
        combined = list(db_rank) + list(chi_rank) + list(sil_rank) + list(inertia_rank)
        vote_counts = pd.Series(combined).value_counts().sort_values(ascending=False)

        best_k_summary[color] = {
            "Best_K": vote_counts.index[0],
            "Top_Ks_by_Count": dict(vote_counts)
        }

    return pd.DataFrame(best_k_summary).T
