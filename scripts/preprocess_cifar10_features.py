# scripts/preprocess_cifar10_features.py

import os
import numpy as np
import pickle
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from PIL import Image
import cv2

def load_cifar10_images(target_classes, max_per_class=10):
    """
    Loads CIFAR-10 images and labels for specified classes.
    """
    transform = transforms.ToTensor()
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    class_indices = {name: idx for idx, name in enumerate(dataset.classes)}
    selected_indices = {name: [] for name in target_classes}

    for i, (img, label) in enumerate(dataset):
        for cls in target_classes:
            if label == class_indices[cls] and len(selected_indices[cls]) < max_per_class:
                selected_indices[cls].append(i)
        if all(len(lst) == max_per_class for lst in selected_indices.values()):
            break

    selected_imgs = []
    selected_labels = []

    for cls, indices in selected_indices.items():
        for idx in indices:
            img, label = dataset[idx]
            selected_imgs.append(img.numpy().transpose(1, 2, 0))  # CHW → HWC
            selected_labels.append(label)

    return selected_imgs, selected_labels


def resize_and_convert_color_spaces(imgs, size=(32, 32)):
    """
    Resize images using bicubic interpolation and convert them into
    RGB, HSV, LAB, YCrCb flattened feature arrays.
    """
    bicubic_imgs = []

    for img in imgs:
        img_resized = Image.fromarray((img * 255).astype(np.uint8)).resize(size, Image.BICUBIC)
        bicubic_imgs.append(np.array(img_resized))

    X_rgb = np.array(bicubic_imgs).reshape(len(bicubic_imgs), -1)

    hsv_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in bicubic_imgs]
    X_hsv = np.array(hsv_imgs).reshape(len(hsv_imgs), -1)

    lab_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2LAB) for img in bicubic_imgs]
    X_lab = np.array(lab_imgs).reshape(len(lab_imgs), -1)

    ycrcb_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb) for img in bicubic_imgs]
    X_ycrcb = np.array(ycrcb_imgs).reshape(len(ycrcb_imgs), -1)

    return bicubic_imgs, X_rgb, X_hsv, X_lab, X_ycrcb


def save_data(bicubic_imgs, labels, X_rgb, X_hsv, X_lab, X_ycrcb, out_dir="data"):
    """
    Saves processed data to disk.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save resized RGB images
    with open(os.path.join(out_dir, "bicubic_imgs.pkl"), "wb") as f:
        pickle.dump(bicubic_imgs, f)

    # Save image class labels
    np.save(os.path.join(out_dir, "selected_labels.npy"), np.array(labels))

    # Save flattened features
    np.save(os.path.join(out_dir, "X_rgb.npy"), X_rgb)
    np.save(os.path.join(out_dir, "X_hsv.npy"), X_hsv)
    np.save(os.path.join(out_dir, "X_lab.npy"), X_lab)
    np.save(os.path.join(out_dir, "X_ycrcb.npy"), X_ycrcb)


def main():
    print("Starting preprocessing of CIFAR-10 images...")
    target_classes = ['horse', 'deer', 'airplane']
    selected_imgs, selected_labels = load_cifar10_images(target_classes, max_per_class=10)

    print("Resizing and converting color spaces...")
    bicubic_imgs, X_rgb, X_hsv, X_lab, X_ycrcb = resize_and_convert_color_spaces(selected_imgs)

    print("Saving processed arrays and labels to 'data/' folder...")
    save_data(bicubic_imgs, selected_labels, X_rgb, X_hsv, X_lab, X_ycrcb)

    print("All preprocessing complete and data saved.")


if __name__ == "__main__":
    main()
