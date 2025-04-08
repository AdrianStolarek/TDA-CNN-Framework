from data_enrichment_pipelines import *
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.util import random_noise
from sklearn.utils import shuffle
import pickle
import os
import cv2
import pandas as pd
import itertools
from pathlib import Path


def load_cifar10_batch(batch_path):

    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    # Reshape the data to (num_samples, 3, 32, 32) and then to (num_samples, 32, 32, 3) for RGB images
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, labels


def load_from_folder(dataset_path, dataset_size_limit=None, grayscale=False):
    class_folders = sorted([f for f in os.listdir(
        dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    class_iterators = []

    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        images = []
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append((img, class_folder))
        class_iterators.append(itertools.cycle(images))

    images = []
    labels = []
    total_samples = 0
    while True:
        for it in class_iterators:
            try:
                img, label = next(it)
                images.append(img)
                labels.append(label)
                total_samples += 1
                if dataset_size_limit and total_samples >= dataset_size_limit:
                    return np.array(images), np.array(labels)
            except StopIteration:
                return np.array(images), np.array(labels)


def load_from_csv(csv_path, images_dir, test_size=0.2, random_state=42, grayscale=False):
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, row['image'])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)

            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(row['label'])

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def save_in_batches(pipeline, X, y, batch_size=32, output_dir="tds_processed"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        processed_batch = pipeline.fit_transform(batch_X)

        np.savez(os.path.join(output_dir, f"batch_{i//batch_size}.npz"),
                 X=processed_batch, y=batch_y)


def load_processed_batches(output_dir="tds_processed", batch_size=None):
    all_X = []
    all_y = []

    for batch_file in sorted(Path(output_dir).glob("*.npz")):
        with np.load(batch_file) as data:
            all_X.append(data['X'])
            all_y.append(data['y'])

    X_combined = np.concatenate(all_X)
    y_combined = np.concatenate(all_y)

    if batch_size is not None:
        X_combined = X_combined[:batch_size]
        y_combined = y_combined[:batch_size]

    return X_combined, y_combined


def detect_image_size(images):
    if len(images) == 0:
        raise ValueError("No images found in the dataset")
    return images[0].shape[0], images[0].shape[1]
