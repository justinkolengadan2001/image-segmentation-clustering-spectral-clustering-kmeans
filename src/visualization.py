import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.patches import Patch
from sklearn.cluster import KMeans

from src.spectral_clustering import spectral_clustering_segmentation
from src.kmeans_clustering import kmeans


def plot_spectral_segmentation(images, labels, K, cifar10_labels):
    """
    Plots original and spectral-clustered versions of images.

    Parameters:
    -----------
    images : list of np.ndarray
        List of images in BGR format.
    labels : list or np.ndarray
        Corresponding CIFAR-10 class labels.
    K : int
        Number of clusters.
    cifar10_labels : list
        List of CIFAR-10 class names indexed by label.
    """
    num_images = len(images)
    fig, axes = plt.subplots(2, num_images, figsize=(5 * num_images, 8))

    for i, (img, label) in enumerate(zip(images, labels)):
        class_name = cifar10_labels[int(label)]

        # Convert to YCrCb for clustering
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        segmented_img, _, centers = spectral_clustering_segmentation(img_ycrcb, K)
        centers = np.uint8(centers)

        # Convert for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segmented_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_YCrCb2RGB)

        axes[0, i].imshow(img_rgb)
        axes[0, i].set_title(f"Original ({class_name})", fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(segmented_rgb)
        axes[1, i].set_title(f"Spectral Clustering (K={K})", fontsize=10)
        axes[1, i].axis('off')

        # Add cluster legend
        legend_elements = [Patch(facecolor=np.array(c) / 255.0, label=f'Cluster {j+1}')
                           for j, c in enumerate(centers)]
        axes[1, i].legend(handles=legend_elements,
                          loc='upper center',
                          bbox_to_anchor=(0.5, -0.15),
                          ncol=2,
                          fontsize=6,
                          frameon=False)

    plt.tight_layout()
    plt.show()


def plot_kmeans_segmentation(image_label_pairs, K=4, metric='chebyshev'):
    """
    Plots original and K-Means clustered versions of images.

    Parameters:
    -----------
    image_label_pairs : list of (np.ndarray, str)
        List of (image, label) tuples in BGR format.
    K : int
        Number of clusters.
    metric : str
        Distance metric for K-Means.
    """
    n = len(image_label_pairs)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6))

    for i, (img, label) in enumerate(image_label_pairs):
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        segmented_img = kmeans(img_ycrcb, K=K, distance_metric=metric)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segmented_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_YCrCb2RGB)

        axes[0, i].imshow(img_rgb)
        axes[0, i].set_title(f"{label} Original", fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(segmented_rgb)
        axes[1, i].set_title(f"{label} K-Means (K={K})", fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_texture_ablation(img_rgb, flat_color, texture_funcs, space_name="RGB"):
    """
    Visualizes the effect of different texture features added to color + position.

    Parameters:
    -----------
    img_rgb : np.ndarray
        Original image in RGB.
    flat_color : np.ndarray
        Flattened color features of shape (H*W, 3).
    texture_funcs : dict[str, Callable]
        Dictionary of texture function name to feature extractor.
    space_name : str
        Name of the color space used (for the plot title).
    """
    fig, axs = plt.subplots(1, len(texture_funcs) + 1, figsize=(22, 4))
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original", fontsize=14)
    axs[0].axis("off")

    for i, (name, func) in enumerate(texture_funcs.items()):
        features = func(img_rgb, flat_color)
        model = KMeans(n_clusters=4, random_state=42)
        labels = model.fit_predict(features)
        centers = model.cluster_centers_[:, :3]
        segmented = centers[labels].reshape(32, 32, 3)

        axs[i + 1].imshow((segmented * 255).astype(np.uint8))
        axs[i + 1].set_title(name, fontsize=14)
        axs[i + 1].axis("off")

    plt.suptitle(f"{space_name}: KMeans with Color + Position + Texture", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_k_metrics_grid(df, save_path="results/kmeans_metrics_plot.png"):
    """
    Plots a grid of evaluation metrics (Silhouette, Davies-Bouldin, 
    Calinski-Harabasz, Inertia) vs K for each color space across multiple images.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with clustering results. Must contain columns:
        'ColorSpace', 'ImageIndex', 'K', 'Silhouette', 'DaviesBouldin', 
        'CalinskiHarabasz', 'Inertia'.
    save_path : str
        Path to save the final figure as PNG.
    """
    metrics = ['Silhouette', 'DaviesBouldin', 'CalinskiHarabasz', 'Inertia']
    color_spaces = df['ColorSpace'].unique()
    k_values = sorted(df['K'].unique())

    fig, axs = plt.subplots(len(color_spaces), len(metrics), figsize=(24, 18), sharex=True)
    fig.suptitle("K-Means Evaluation Metrics vs K for Each Color Space", fontsize=18)

    for i, color in enumerate(color_spaces):
        df_color = df[df["ColorSpace"] == color]
        for j, metric in enumerate(metrics):
            ax = axs[i, j]
            for idx in sorted(df_color['ImageIndex'].unique()):
                df_img = df_color[df_color["ImageIndex"] == idx]
                ax.plot(df_img['K'], df_img[metric], linewidth=1.8, alpha=0.75)

            if i == 0:
                ax.set_title(metric, fontsize=16)
            if j == 0:
                ax.set_ylabel(color, fontsize=16)

            ax.grid(True)
            if i == len(color_spaces) - 1:
                ax.set_xlabel("K", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_dataframe_as_image(df, filename="results/kmeans_table.png"):
    """
    Saves a Pandas DataFrame as a styled table image (PNG) using matplotlib.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to save as an image.
    filename : str
        Output path for saving the image.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
