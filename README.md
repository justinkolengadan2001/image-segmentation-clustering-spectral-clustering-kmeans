# Image Segmentation using K-Means and Spectral Clustering

This project implements unsupervised image segmentation on CIFAR-10 images using:

- **K-Means Clustering**
- **Spectral Clustering**
- Feature combinations of **color**, **spatial coordinates**, and **texture (Sobel edge)**

---

## Project Structure

```text
image-segmentation-clustering/
├── data/                      # Processed images & features (not versioned)
│   ├── bicubic_imgs.pkl
│   ├── X_rgb.npy / X_hsv.npy / X_lab.npy / X_ycrcb.npy
│   ├── selected_labels.npy
│
├── results/                   # Output plots & evaluation CSVs
│   ├── kmeans_metrics_plot.png
│   ├── kmeans_metrics_table.png
│   ├── kmeans_metrics_scores.csv
│
├── scripts/                   # Executable scripts
│   ├── preprocess_cifar10_features.py
│   ├── run_kmeans.py
│   ├── run_spectral.py
│   └── analyze_kmeans_metrics.py
│
├── src/                       # Reusable logic modules
│   ├── evaluation.py
│   ├── feature_extraction.py
│   ├── kmeans_clustering.py
│   ├── spectral_clustering.py
│   └── visualization.py
│
├── notebooks/                 
│   ├── KMeans_Ablation_Study.ipynb
│   ├── Our_Implementation_KMeans.ipynb
│   ├── Our_Implementation_Spectral_Clustering_Ablation_Study.ipynb
│   └── Preprocess_CIFAR-10_features.ipynb
│
├── .gitignore
├── requirements.txt
├── README.md
└── Project_Report.pdf
```

---

## Quick Start

### 1. Clone the repository

* git clone https://github.com/your-username/image-segmentation-clustering-spectral-clustering-kmeans.git
* cd image-segmentation-clustering-spectral-clustering-kmeans

### 2. Install dependencies

pip install -r requirements.txt

### 3. Preprocess CIFAR-10 images

python scripts/preprocess_cifar10_features.py

This will:
- Select 10 images each from the classes airplane, horse, and deer
- Resize them to 32×32 using bicubic interpolation
- Extract color features in 4 color spaces
- Save all processed data in data/

### 4. Run Image Segmentation Algorithms (with texture + spatial)

### Run K-Means with full feature set (color + position + texture):

python scripts/run_kmeans.py --image_path data/raw/test1.jpg --K 4 --use_combined_features --pos_weight 0.1

### Run Spectral Clustering on a single image:

python scripts/run_spectral.py --image_path data/raw/test1.jpg --K 4

### Run K-Means using color-only + chebyshev distance (YCrCb):

python scripts/run_kmeans.py --image_path data/raw/test1.jpg --K 4 --metric chebyshev

### 5. Analyze clustering metrics across K and color spaces

python scripts/analyze_kmeans_metrics.py \
    --image_path data/bicubic_imgs.pkl \
    --rgb_features data/X_rgb.npy \
    --hsv_features data/X_hsv.npy \
    --lab_features data/X_lab.npy \
    --ycrcb_features data/X_ycrcb.npy

### 6. Clustering Analysis (Programmatic Use)

You can also import and use this project inside a notebook:

### Visualizing segmentations:

* from src.visualization import plot_spectral_segmentation, plot_kmeans_segmentation

- For spectral clustering (on multiple CIFAR-10 images)
* plot_spectral_segmentation(images, labels, K=8, cifar10_labels=cifar10_labels)

- For k-means clustering
* plot_kmeans_segmentation([(img1, "Dog"), (img2, "Airplane")], K=6)

### Evaluation Metrics Used

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Inertia (K-Means cost)

### Metric Evaluation and Plots

Automatically evaluate metrics for K = 2 to 10 and color spaces to find the optimal configuration.

* from src.evaluation import evaluate_k_range_all_30_images, find_best_k_per_colorspace
* from src.visualization import plot_k_metrics_grid, save_dataframe_as_image

* df_scores = evaluate_k_range_all_30_images(bicubic_imgs, X_rgb, X_hsv, X_lab, X_yCrCb)

- Line plots of metrics vs. K
* plot_k_metrics_grid(df_scores)

- Summary table image
* summary_df = (df_scores.groupby(["ColorSpace", "K"])
	     .mean(numeric_only=True)
             .drop(columns=["ImageIndex"])
             .round(3)
             .reset_index()
)

* save_dataframe_as_image(summary_df, "results/kmeans_table.png")

### Highlights

- Supports both K-Means and Spectral Clustering
- Modular design: all logic is reusable via src/
- Texture extracted via Sobel magnitude

### Future Ideas

- Extend to larger datasets (e.g., ImageNet subsamples)
- Add GUI 