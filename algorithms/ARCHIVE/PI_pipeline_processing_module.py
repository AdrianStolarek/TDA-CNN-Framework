import os
import numpy as np
import psutil
from GPUtil import showUtilization
from tda_pipelines_PCA import VECTOR_STITCHING_PI_Pipeline_RGB
from data_preprocessing import load_cifar10_batch
import matplotlib.pyplot as plt


def log_resource_utilization(phase):
    """
    Logs CPU, RAM, and GPU utilization at a specific phase of the process.

    Parameters:
        phase (str): Description of the current phase.

    Returns:
        None
    """
    print(f"[INFO] Resource utilization during {phase}:")
    print(f"CPU usage: {psutil.cpu_percent()}%")
    print(f"RAM usage: {psutil.virtual_memory().percent}%")
    showUtilization()


def process_and_save_data(raw_data, labels, output_dir, batch_size, pipeline):
    """
    Processes data in batches and saves the processed output and labels to the specified directory.

    Parameters:
        raw_data (np.ndarray): Input data to be processed.
        labels (np.ndarray): Corresponding labels for the input data.
        output_dir (str): Directory where processed data will be saved.
        batch_size (int): Number of samples to process in each batch.
        pipeline: Preprocessing pipeline instance.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    num_samples = len(raw_data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        print(f"[INFO] Processing batch {batch_idx + 1}/{num_batches}...")

        # Process the batch data
        batch_data = raw_data[start_idx:end_idx]
        processed_batch = pipeline.fit_transform(batch_data)
        batch_labels = labels[start_idx:end_idx]

        print(f"[SIZE OF THE PROCESSED DATA] {processed_batch.shape}")

        # processed_image = processed_batch[25, :, :, 5]

        # plt.figure(figsize=(5, 5))
        # plt.imshow(processed_image.astype("uint8"))
        # plt.axis("off")
        # plt.show()

        # Save processed batch and labels to file
        batch_file = os.path.join(
            output_dir, f"processed_batch_{batch_idx + 1}.npz"
        )
        np.savez_compressed(
            batch_file, data=processed_batch, labels=batch_labels)
        print(f"[INFO] Saved batch {batch_idx + 1} to {batch_file}")

    print("[INFO] All data processed and saved.")


def save_raw_data(raw_data, output_dir, batch_size):
    """
    Saves raw data in batches to the specified directory without any processing.

    Parameters:
        raw_data (np.ndarray): Input data to be saved.
        output_dir (str): Directory where raw data will be saved.
        batch_size (int): Number of samples per batch.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    num_samples = len(raw_data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        print(f"[INFO] Saving raw batch {batch_idx + 1}/{num_batches}...")

        # Get the batch data
        batch_data = raw_data[start_idx:end_idx]

        # Save the raw batch to file
        batch_file = os.path.join(output_dir, f"raw_batch_{batch_idx + 1}.npy")
        np.save(batch_file, batch_data)
        print(f"[INFO] Saved raw batch {batch_idx + 1} to {batch_file}")

    print("[INFO] All raw data saved.")


if __name__ == "__main__":
    batch_path = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10-batches-py\\data_batch_2"
    output_dir = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10_processed"

    print("[INFO] Loading dataset...")
    raw_data, labels = load_cifar10_batch(batch_path)

    input_height = 32
    input_width = 32
    print("[INFO] Initializing Vector Stitching Pipeline...")
    vector_stitching_pipeline = VECTOR_STITCHING_PI_Pipeline_RGB(
        input_height=input_height, input_width=input_width
    )

    # Process and save data in batches
    batch_size = 5000
    process_and_save_data(raw_data, labels, output_dir,
                          batch_size, vector_stitching_pipeline)
