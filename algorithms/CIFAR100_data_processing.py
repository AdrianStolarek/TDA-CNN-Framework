import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets
from sklearn.preprocessing import LabelEncoder
from data_enrichment_pipelines import VECTOR_STITCHING_PI_Pipeline_RGB, VECTOR_STITCHING_PI_Pipeline_RGB_pure
from data_preprocessing_pipelines import save_in_batches


RAW_DIR = "E:\\ZoryaDyn_Benchmarks\\CIFAR100\\processed_RAW"
TDA_DIR = "E:\\ZoryaDyn_Benchmarks\\CIFAR100\\processed_TDA"
TDA_DIR_PURE = "E:\\ZoryaDyn_Benchmarks\\CIFAR100\\processed_TDA_PURE"
BATCH_SIZE = 1000
DATASET_LIMIT = 5000
RANDOM_STATE = 42

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(TDA_DIR_PURE, exist_ok=True)


def process_cifar100(training_samples_limit=None):

    print(f"[INFO] Loading CIFAR-100 dataset...")

    (X_train, y_train), (X_test, y_test) = datasets.cifar100.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    if training_samples_limit is not None:
        print(
            f"[INFO] Limiting training dataset to {training_samples_limit} samples")
        X_train = X_train[:training_samples_limit]
        y_train = y_train[:training_samples_limit]

    print("=" * 50)
    print("CIFAR-100 Dataset Information:")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Number of training classes: {len(np.unique(y_train))}")
    print(f"Number of testing classes: {len(np.unique(y_test))}")
    print("=" * 50)

    plt.figure(figsize=(12, 6))
    for i in range(min(5, len(X_train))):
        plt.subplot(1, 5, i+1)
        plt.imshow(X_train[i])
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(RAW_DIR, "sample_images.png"))
    plt.close()

    print(f"[INFO] Saving raw training images to {RAW_DIR}...")
    np.savez(os.path.join(RAW_DIR, "raw_train_dataset.npz"),
             X=X_train, y=y_train)

    print(f"[INFO] Saving raw testing images to {RAW_DIR}...")
    np.savez(os.path.join(RAW_DIR, "raw_test_dataset.npz"),
             X=X_test, y=y_test)

    INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS = X_train[0].shape
    print(
        f"[INFO] Image dimensions: {INPUT_HEIGHT}x{INPUT_WIDTH}x{INPUT_CHANNELS}")

    rgb_pipeline = VECTOR_STITCHING_PI_Pipeline_RGB_pure(
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH
    )

    print("[INFO] Processing training data through TDA pipeline...")
    save_in_batches(
        rgb_pipeline,
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        output_dir=os.path.join(TDA_DIR_PURE, "train")
    )

    print("[INFO] Processing testing data through TDA pipeline...")
    save_in_batches(
        rgb_pipeline,
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        output_dir=os.path.join(TDA_DIR_PURE, "test")
    )

    print("\n[INFO] CIFAR-100 dataset processing complete!")
    print(f"Raw data saved to: {RAW_DIR}")
    print(f"TDA processed data saved to: {TDA_DIR_PURE}")
    print("Dataset statistics:")
    print(f"- Training images: {len(X_train)}")
    print(f"- Testing images: {len(X_test)}")
    print(f"- Image size: {INPUT_HEIGHT}x{INPUT_WIDTH}")
    print(f"- Number of classes: {len(np.unique(y_train))}")


if __name__ == "__main__":
    process_cifar100(training_samples_limit=DATASET_LIMIT)
