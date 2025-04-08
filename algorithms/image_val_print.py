import os
import numpy as np
import matplotlib.pyplot as plt
from image_val import (
    visualize_dataset_samples,
    compare_raw_and_tda,
    visualize_tda_channels,
    display_noisy_images,
    check_class_distribution,
    visualize_tsne_embeddings,
    check_processed_data
)
from data_preprocessing_pipelines import load_processed_batches
from sklearn.preprocessing import LabelEncoder

# Define directories
RAW_DIR = "E:\\ZoryaDyn_Benchmarks\\WAR_TECH\\processed_RAW"
TDA_DIR = "E:\\ZoryaDyn_Benchmarks\\WAR_TECH\\processed_TDA"
VALIDATION_DIR = "E:\\ZoryaDyn_Benchmarks\\WAR_TECH\\validation_results"

# Create validation directory if it doesn't exist
os.makedirs(VALIDATION_DIR, exist_ok=True)

def main():
    print("="*50)
    print("STARTING IMAGE PROCESSING VALIDATION")
    print("="*50)

    # Load raw data
    print("\nLoading raw dataset...")
    raw_data = np.load(os.path.join(RAW_DIR, "raw_dataset.npz"))
    X_raw = raw_data['X']
    y_raw = raw_data['y']

    # Load TDA processed data
    print("\nLoading TDA processed dataset...")
    X_tda, y_tda = load_processed_batches(output_dir=TDA_DIR)

    # Basic integrity checks
    print("\nPerforming basic integrity checks...")
    print(f"Raw dataset shape: {X_raw.shape}")
    print(f"TDA dataset shape: {X_tda.shape}")
    print(f"Raw labels shape: {y_raw.shape}")
    print(f"TDA labels shape: {y_tda.shape}")

    # Check if labels match
    labels_match = np.array_equal(y_raw, y_tda)
    print(f"Labels match between raw and TDA datasets: {labels_match}")

    if not labels_match:
        print("WARNING: Labels do not match between datasets!")
        print(f"Raw unique labels: {np.unique(y_raw)}")
        print(f"TDA unique labels: {np.unique(y_tda)}")

    # Map numeric labels to class names if available (example)
    # If you have a mapping of indices to actual class names, add it here
    # labels_map = {0: "Tank", 1: "Aircraft", 2: "Ship", ...}
    # If not, we'll use None and display numeric labels
    labels_map = None

    print("\nGenerating validation visualizations...")

    # 1. Visualize samples from raw dataset
    plt_raw = visualize_dataset_samples(
        X_raw, y_raw, labels_map=labels_map, n_samples=5,
        title="Raw Dataset Samples"
    )
    plt_raw.savefig(os.path.join(VALIDATION_DIR, "raw_samples.png"))
    plt_raw.close()

    # 2. Visualize samples from TDA dataset
    plt_tda = visualize_dataset_samples(
        X_tda, y_tda, labels_map=labels_map, n_samples=5,
        title="TDA Dataset Samples"
    )
    plt_tda.savefig(os.path.join(VALIDATION_DIR, "tda_samples.png"))
    plt_tda.close()

    # 3. Compare raw and TDA images
    plt_compare = compare_raw_and_tda(X_raw, X_tda, y_raw, n_samples=4)
    plt_compare.savefig(os.path.join(
        VALIDATION_DIR, "raw_vs_tda_comparison.png"))
    plt_compare.close()

    # 4. Visualize TDA channels for one sample
    plt_channels = visualize_tda_channels(X_tda, index=0)
    plt_channels.savefig(os.path.join(VALIDATION_DIR, "tda_channels.png"))
    plt_channels.close()

    # 5. Check class distribution
    plt_dist = check_class_distribution(y_raw, labels_map=labels_map)
    plt_dist.savefig(os.path.join(VALIDATION_DIR, "class_distribution.png"))
    plt_dist.close()

    # 6. Check for Gaussian noise effect (if available)
    # You would need the original images before noise was added
    # If you have access to them, uncomment this section
    """
    # Load original images before noise
    original_data = np.load("path_to_original_data.npz")
    X_original = original_data['X']
    
    # Compare original vs noisy
    plt_noise = display_noisy_images(X_original, X_raw, n_samples=3)
    plt_noise.savefig(os.path.join(VALIDATION_DIR, "noise_comparison.png"))
    plt_noise.close()
    """

    # 7. Value range analysis
    print("\nData value range analysis:")
    print(
        f"Raw data - Min: {X_raw.min()}, Max: {X_raw.max()}, Mean: {X_raw.mean():.4f}, Std: {X_raw.std():.4f}")
    print(
        f"TDA data - Min: {X_tda.min()}, Max: {X_tda.max()}, Mean: {X_tda.mean():.4f}, Std: {X_tda.std():.4f}")

    # Check for NaN or Inf values
    nan_raw = np.isnan(X_raw).sum()
    inf_raw = np.isinf(X_raw).sum()
    nan_tda = np.isnan(X_tda).sum()
    inf_tda = np.isinf(X_tda).sum()

    print(f"Raw data - NaN values: {nan_raw}, Inf values: {inf_raw}")
    print(f"TDA data - NaN values: {nan_tda}, Inf values: {inf_tda}")

    if nan_raw > 0 or inf_raw > 0:
        print("WARNING: Raw data contains NaN or Inf values!")

    if nan_tda > 0 or inf_tda > 0:
        print("WARNING: TDA data contains NaN or Inf values!")

    # 8. Optional: Compute t-SNE visualization for dataset exploration
    # This can be computationally expensive, so use a subset of data if needed
    try:
        print("\nComputing t-SNE visualization (this may take a while)...")

        # Use a subset of data to speed up t-SNE computation
        sample_size = min(500, len(X_raw))
        indices = np.random.choice(len(X_raw), sample_size, replace=False)

        X_raw_subset = X_raw[indices]
        y_raw_subset = y_raw[indices]
        X_tda_subset = X_tda[indices]

        # Compute t-SNE for raw data
        plt_tsne_raw = visualize_tsne_embeddings(
            X_raw_subset, y_raw_subset,
            perplexity=min(30, sample_size//5),
            n_iter=1000,
            figsize=(10, 8)
        )
        plt_tsne_raw.suptitle("t-SNE Embeddings of Raw Data")
        plt_tsne_raw.savefig(os.path.join(VALIDATION_DIR, "tsne_raw.png"))
        plt_tsne_raw.close()

        # Compute t-SNE for TDA data
        plt_tsne_tda = visualize_tsne_embeddings(
            X_tda_subset, y_raw_subset,
            perplexity=min(30, sample_size//5),
            n_iter=1000,
            figsize=(10, 8)
        )
        plt_tsne_tda.suptitle("t-SNE Embeddings of TDA Data")
        plt_tsne_tda.savefig(os.path.join(VALIDATION_DIR, "tsne_tda.png"))
        plt_tsne_tda.close()

        print("t-SNE visualizations completed and saved.")
    except Exception as e:
        print(f"Error computing t-SNE visualizations: {str(e)}")

    print("\nValidation complete! Results saved to:", VALIDATION_DIR)
    print("="*50)


if __name__ == "__main__":
    main()
