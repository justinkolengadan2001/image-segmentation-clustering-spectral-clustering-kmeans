# scripts/analyze_kmeans_metrics.py

import os
import argparse
import pandas as pd
import numpy as np
import cv2
import pickle  # or use np.load if data is in .npz or .npy

from src.evaluation import evaluate_k_range_all_30_images, find_best_k_per_colorspace
from src.visualization import plot_k_metrics_grid, save_dataframe_as_image

def main(args):
    # === Step 1: Ensure results folder exists ===
    os.makedirs("results", exist_ok=True)

    # === Step 2: Load data ===
    print("Loading image and color feature arrays...")
    with open(args.image_path, "rb") as f:
        bicubic_imgs = pickle.load(f)  # list of 32x32x3 RGB images

    X_rgb = np.load(args.rgb_features)
    X_hsv = np.load(args.hsv_features)
    X_lab = np.load(args.lab_features)
    X_ycrcb = np.load(args.ycrcb_features)

    # === Step 3: Run evaluation ===
    print("Running clustering metric evaluation for K = 2 to 10...")
    df_scores = evaluate_k_range_all_30_images(
        bicubic_imgs,
        X_rgb,
        X_hsv,
        X_lab,
        X_ycrCb=X_ycrcb,
        k_range=range(2, 11),
        pos_weight=args.pos_weight,
        color_weight=args.color_weight
    )

    # === Step 4: Save CSV output ===
    csv_path = "results/kmeans_metrics_scores.csv"
    df_scores.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

    # === Step 5: Create and save summary table as image ===
    print("Generating table summary...")
    summary_df = (
        df_scores
        .groupby(["ColorSpace", "K"])
        .mean(numeric_only=True)
        .drop(columns=["ImageIndex"])
        .round(3)
        .reset_index()
    )
    table_path = "results/kmeans_metrics_table.png"
    save_dataframe_as_image(summary_df, table_path)
    print(f"Table image saved to {table_path}")

    # === Step 6: Save grid of metric-vs-K plots ===
    print("Plotting metric grid...")
    plot_k_metrics_grid(df_scores)
    print("Plot saved as results/kmeans_metrics_plot.png")

    # === Step 7: Print best-K summary ===
    print("\nBest K values based on metric voting:")
    best_k_df = find_best_k_per_colorspace(df_scores)
    print(best_k_df.to_string())
    best_k_df.to_csv("results/best_k_per_colorspace.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze KMeans metrics for different color spaces and K values.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to pickled 32x32 RGB image list")
    parser.add_argument("--rgb_features", type=str, required=True, help="Path to RGB color feature array")
    parser.add_argument("--hsv_features", type=str, required=True, help="Path to HSV color feature array")
    parser.add_argument("--lab_features", type=str, required=True, help="Path to LAB color feature array")
    parser.add_argument("--ycrcb_features", type=str, required=True, help="Path to YCrCb color feature array")
    parser.add_argument("--pos_weight", type=float, default=0.1, help="Weight for spatial features")
    parser.add_argument("--color_weight", type=float, default=1.0, help="Weight for color features")

    args = parser.parse_args()
    main(args)
