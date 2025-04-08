import os
import numpy as np
import psutil
from GPUtil import showUtilization
from tda_pipelines_PCA import VECTOR_STITCHING_PI_Pipeline_RGB
from data_preprocessing import load_cifar10_batch
import matplotlib.pyplot as plt


def log_resource_utilization(phase):
    """Logs CPU, RAM, and GPU utilization at a specific phase."""
    print(f"[INFO] Resource utilization during {phase}:")
    print(f"CPU usage: {psutil.cpu_percent()}%")
    print(f"RAM usage: {psutil.virtual_memory().percent}%")
    showUtilization()


def add_gaussian_noise(images, mean=0.1, std=0.3):
    """Dodaje szum Gaussowski do obrazów."""
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0.0, 1.0)  # Przycięcie wartości do [0,1]


def add_salt_and_pepper_noise(images, prob=0.02):
    """Dodaje szum typu "salt and pepper" do obrazów."""
    noisy_images = np.copy(images)
    for img in noisy_images:
        num_pixels = int(prob * img.shape[0] * img.shape[1])

        if num_pixels < 1:  # Zapobieganie braku szumu
            num_pixels = 1

        # Dodajemy sól (białe piksele)
        coords_x = np.random.randint(0, img.shape[0], num_pixels)
        coords_y = np.random.randint(0, img.shape[1], num_pixels)
        img[coords_x, coords_y] = 1.0

        # Dodajemy pieprz (czarne piksele)
        coords_x = np.random.randint(0, img.shape[0], num_pixels)
        coords_y = np.random.randint(0, img.shape[1], num_pixels)
        img[coords_x, coords_y] = 0.0

    return noisy_images


def add_noise_to_data(data):
    """Dodaje kombinację szumu Gaussowskiego i Salt & Pepper do danych."""
    noisy_data = add_gaussian_noise(data, mean=0.2, std=0.1)
    noisy_data = add_salt_and_pepper_noise(noisy_data, prob=0.02)
    return noisy_data


def process_and_save_data(raw_data, labels, output_dir, batch_size, pipeline):
    """Przetwarza i zapisuje dane przy użyciu Vector Stitching."""
    os.makedirs(output_dir, exist_ok=True)

    num_samples = len(raw_data)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        print(f"[INFO] Processing batch {batch_idx + 1}/{num_batches}...")

        # Pobranie batcha i dodanie szumu
        batch_data = raw_data[start_idx:end_idx]
        # **Zapewniamy, że etykiety są poprawne**
        batch_labels = labels[start_idx:end_idx]
        noisy_batch_data = add_noise_to_data(batch_data)

        # Przetworzenie zaszumionych danych przez pipeline
        processed_batch = pipeline.fit_transform(noisy_batch_data)

        print(f"[SIZE OF THE PROCESSED DATA] {processed_batch.shape}")

        # **Zapis przetworzonych danych**
        batch_file = os.path.join(
            output_dir, f"processed_batch_{batch_idx + 1}.npz")
        np.savez_compressed(
            batch_file, data=processed_batch, labels=batch_labels)  # **Etykiety są teraz zapisane!**
        print(f"[INFO] Saved processed batch {batch_idx + 1} to {batch_file}")

        # **Dodatkowo zapisujemy surowe zaszumione dane + etykiety**
        raw_noisy_file = os.path.join(
            output_dir, f"raw_noisy_batch_{batch_idx + 1}.npy")
        np.save(raw_noisy_file, noisy_batch_data)

        raw_noisy_labels_file = os.path.join(
            output_dir, f"raw_noisy_labels_{batch_idx + 1}.npy")  # **Zapisujemy etykiety!**
        np.save(raw_noisy_labels_file, batch_labels)

        print(
            f"[INFO] Saved raw noisy batch {batch_idx + 1} to {raw_noisy_file}")
        print(
            f"[INFO] Saved raw noisy labels batch {batch_idx + 1} to {raw_noisy_labels_file}")

    print("[INFO] All data processed and saved.")


if __name__ == "__main__":
    batch_path = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10-batches-py\\test_batch"
    output_dir = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10_processed"

    print("[INFO] Loading dataset...")
    raw_data, labels = load_cifar10_batch(batch_path)

    input_height = 32
    input_width = 32
    print("[INFO] Initializing Vector Stitching Pipeline...")
    vector_stitching_pipeline = VECTOR_STITCHING_PI_Pipeline_RGB(
        input_height=input_height, input_width=input_width
    )

    # Procesowanie i zapisanie danych
    batch_size = 5000
    process_and_save_data(raw_data, labels, output_dir,
                          batch_size, vector_stitching_pipeline)
