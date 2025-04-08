import numpy as np
import matplotlib.pyplot as plt
import os
from data_preprocessing_pipelines import load_processed_batches
import random
from sklearn.manifold import TSNE


def visualize_dataset_samples(X, y, labels_map=None, n_samples=5, figsize=(15, 10), title="Dataset Samples"):
    """
    Visualize random samples from a dataset with their labels.

    Args:
        X: Image data array
        y: Labels array
        labels_map: Optional dictionary mapping label indices to label names
        n_samples: Number of samples to display
        figsize: Figure size
        title: Plot title
    """
    # If dataset is too small, adjust n_samples
    n_samples = min(n_samples, len(X))

    # Select random indices
    indices = random.sample(range(len(X)), n_samples)

    # Create figure
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16)

    # Display each sample
    for i, idx in enumerate(indices):
        plt.subplot(1, n_samples, i + 1)

        # Handle different channel configurations
        img = X[idx]
        if img.shape[-1] == 1:  # Grayscale
            plt.imshow(img[:, :, 0], cmap='gray')
        elif img.shape[-1] == 2:  # TDA with 2 channels
            # Show only the first channel as this is the original image
            plt.imshow(img[:, :, 0], cmap='gray')
        elif img.shape[-1] == 3:  # RGB
            plt.imshow(img)
        elif img.shape[-1] > 3:  # TDA with more channels
            # Show only the RGB channels
            plt.imshow(img[:, :, :3])

        # Display label
        if labels_map is not None and y[idx] in labels_map:
            label_text = labels_map[y[idx]]
        else:
            label_text = f"Class {y[idx]}"

        plt.title(f"{label_text}")
        plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return plt


def compare_raw_and_tda(X_raw, X_tda, y, indices=None, n_samples=3, figsize=(15, 10)):
    """
    Compare raw and TDA-processed images side by side.

    Args:
        X_raw: Raw image data array
        X_tda: TDA-processed image data array
        y: Labels array
        indices: Specific indices to display (if None, random samples are selected)
        n_samples: Number of samples to display if indices is None
        figsize: Figure size
    """
    if indices is None:
        indices = random.sample(range(len(X_raw)), min(n_samples, len(X_raw)))
    else:
        n_samples = len(indices)

    plt.figure(figsize=figsize)
    plt.suptitle("Comparison of Raw vs TDA-Processed Images", fontsize=16)

    for i, idx in enumerate(indices):
        # Display raw image
        plt.subplot(2, n_samples, i + 1)

        # Handle different channel configurations for raw images
        raw_img = X_raw[idx]
        if raw_img.shape[-1] == 1:  # Grayscale
            plt.imshow(raw_img[:, :, 0], cmap='gray')
        else:  # RGB
            plt.imshow(raw_img)

        plt.title(f"Raw - Class {y[idx]}")
        plt.axis('off')

        # Display TDA image
        plt.subplot(2, n_samples, i + 1 + n_samples)

        # For TDA, we need to decide which channels to show
        tda_img = X_tda[idx]
        if tda_img.shape[-1] == 2:  # TDA with 2 channels (original + TDA)
            # Show the TDA channel
            plt.imshow(tda_img[:, :, 1], cmap='viridis')
            plt.title("TDA Channel")
        else:  # Enriched with more channels
            # Show a composite of TDA channels
            tda_channels = tda_img[:, :,
                                   3:] if tda_img.shape[-1] > 3 else tda_img[:, :, 1:]
            if tda_channels.shape[-1] > 0:
                # Average of TDA channels for visualization
                tda_vis = np.mean(tda_channels, axis=2)
                plt.imshow(tda_vis, cmap='viridis')
            else:
                plt.text(0.5, 0.5, "No TDA channels",
                         horizontalalignment='center', verticalalignment='center')

        plt.title(f"TDA - Class {y[idx]}")
        plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return plt


def visualize_tda_channels(X_tda, index=0, figsize=(15, 10)):
    """
    Visualize all channels of a TDA-processed image.

    Args:
        X_tda: TDA-processed image data array
        index: Index of the image to visualize
        figsize: Figure size
    """
    img = X_tda[index]
    n_channels = img.shape[2]

    plt.figure(figsize=figsize)
    plt.suptitle(f"TDA Image Channels (Image #{index})", fontsize=16)

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(n_channels)))

    for i in range(n_channels):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img[:, :, i], cmap='viridis')
        plt.title(f"Channel {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return plt


def display_noisy_images(X_original, X_noisy, indices=None, n_samples=3, figsize=(15, 10)):
    """
    Display original images and their noisy versions side by side.

    Args:
        X_original: Original image data array
        X_noisy: Noisy image data array
        indices: Specific indices to display (if None, random samples are selected)
        n_samples: Number of samples to display if indices is None
        figsize: Figure size
    """
    if indices is None:
        indices = random.sample(range(len(X_original)),
                                min(n_samples, len(X_original)))
    else:
        n_samples = len(indices)

    plt.figure(figsize=figsize)
    plt.suptitle("Comparison of Original vs Noisy Images", fontsize=16)

    for i, idx in enumerate(indices):
        # Display original image
        plt.subplot(2, n_samples, i + 1)

        # Handle different channel configurations
        img = X_original[idx]
        if img.shape[-1] == 1:  # Grayscale
            plt.imshow(img[:, :, 0], cmap='gray')
        else:  # RGB
            plt.imshow(img)

        plt.title(f"Original #{idx}")
        plt.axis('off')

        # Display noisy image
        plt.subplot(2, n_samples, i + 1 + n_samples)

        noisy_img = X_noisy[idx]
        if noisy_img.shape[-1] == 1:  # Grayscale
            plt.imshow(noisy_img[:, :, 0], cmap='gray')
        else:  # RGB
            plt.imshow(noisy_img)

        plt.title(f"Noisy #{idx}")
        plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return plt


def check_class_distribution(y, labels_map=None, figsize=(10, 6)):
    """
    Visualize the class distribution in the dataset.

    Args:
        y: Labels array
        labels_map: Optional dictionary mapping label indices to label names
        figsize: Figure size
    """
    unique_labels, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=figsize)
    plt.bar(range(len(unique_labels)), counts)

    if labels_map is not None:
        labels = [labels_map[label]
                  if label in labels_map else f"Class {label}" for label in unique_labels]
    else:
        labels = [f"Class {label}" for label in unique_labels]

    plt.xticks(range(len(unique_labels)), labels, rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution")
    plt.tight_layout()
    return plt


def visualize_tsne_embeddings(X, y, n_components=2, perplexity=30, n_iter=1000, figsize=(10, 8)):
    """
    Visualize t-SNE embeddings of the dataset.

    Args:
        X: Image data array
        y: Labels array
        n_components: Number of t-SNE components
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations for t-SNE
        figsize: Figure size
    """
    # Flatten images
    X_flat = X.reshape(X.shape[0], -1)

    # Compute t-SNE
    print("Computing t-SNE embeddings...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                n_iter=n_iter, random_state=1)
    X_tsne = tsne.fit_transform(X_flat)

    # Plot embeddings
    plt.figure(figsize=figsize)
    unique_labels = np.unique(y)

    for label in unique_labels:
        indices = y == label
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],
                    label=f"Class {label}", alpha=0.6)

    plt.legend()
    plt.title("t-SNE Embeddings")
    plt.tight_layout()
    return plt


def check_processed_data(raw_dir, tda_dir, n_samples=5):
    """
    Comprehensive check of raw and TDA-processed data.

    Args:
        raw_dir: Directory with raw processed data
        tda_dir: Directory with TDA processed data
        n_samples: Number of samples to display
    """
    # Load raw data
    print(f"Loading raw data from {raw_dir}...")
    raw_data = np.load(os.path.join(raw_dir, "raw_dataset.npz"))
    X_raw = raw_data['X']
    y_raw = raw_data['y']

    # Load TDA data
    print(f"Loading TDA data from {tda_dir}...")
    X_tda, y_tda = load_processed_batches(output_dir=tda_dir)

    # Check array shapes
    print("\nData shapes:")
    print(f"Raw data shape: {X_raw.shape}")
    print(f"TDA data shape: {X_tda.shape}")
    print(f"Raw labels shape: {y_raw.shape}")
    print(f"TDA labels shape: {y_tda.shape}")

    # Check for NaN or infinite values
    print("\nChecking for NaN/Inf values:")
    print(f"Raw data NaN count: {np.isnan(X_raw).sum()}")
    print(f"TDA data NaN count: {np.isnan(X_tda).sum()}")
    print(f"Raw data Inf count: {np.isinf(X_raw).sum()}")
    print(f"TDA data Inf count: {np.isinf(X_tda).sum()}")

    # Check value ranges
    print("\nData value ranges:")
    print(f"Raw data min: {X_raw.min()}, max: {X_raw.max()}")
    print(f"TDA data min: {X_tda.min()}, max: {X_tda.max()}")

    # Verify label consistency
    labels_match = np.array_equal(y_raw, y_tda)
    print(f"\nLabels match between raw and TDA: {labels_match}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Sample raw images
    fig1 = visualize_dataset_samples(
        X_raw, y_raw, n_samples=n_samples, title="Raw Dataset Samples")
    fig1.savefig(os.path.join(raw_dir, "raw_samples.png"))

    # 2. Sample TDA images
    fig2 = visualize_dataset_samples(
        X_tda, y_tda, n_samples=n_samples, title="TDA Dataset Samples")
    fig2.savefig(os.path.join(tda_dir, "tda_samples.png"))

    # 3. Compare raw and TDA
    fig3 = compare_raw_and_tda(X_raw, X_tda, y_raw, n_samples=n_samples)
    fig3.savefig(os.path.join(tda_dir, "raw_vs_tda_comparison.png"))

    # 4. Class distribution
    fig4 = check_class_distribution(y_raw)
    fig4.savefig(os.path.join(raw_dir, "class_distribution.png"))

    # 5. TDA channels for one sample
    if X_tda.shape[-1] > 1:
        fig5 = visualize_tda_channels(X_tda, index=0)
        fig5.savefig(os.path.join(tda_dir, "tda_channels.png"))

    print("\nVisualizations saved to the respective directories.")
    print("Data check complete!")
