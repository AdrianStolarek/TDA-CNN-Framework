from keras.callbacks import EarlyStopping
from models.raw_models import MiniVGG_Raw_Model2
from models.tda_models import MiniVGG_TDA_Model2
import numpy as np
import tensorflow as tf
from data_preprocessing import load_cifar10_batch

# Define data directories
processed_data_dir = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10_processed"
raw_data_dir = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10-batches-py"

# Define file paths for processed data
processed_files = [
    f"{processed_data_dir}\\BATCH{i}_processed_batch_{j}.npz"
    for i in range(1, 6) for j in range(1, 3)  # Iterujemy po batch_1 i batch_2
] + [
    f"{processed_data_dir}\\TEST_BATCH_processed_batch_{i}.npz" for i in range(1, 3)
]

# Define file paths for raw data
raw_files = [
    f"{raw_data_dir}\\data_batch_{i}" for i in range(1, 6)
] + [f"{raw_data_dir}\\test_batch"]


def load_data_in_batches(files, batch_size, total_limit=None):
    """
    Generator to load data and labels in batches from given .npz files with an optional limit.
    Parameters: kitty says hi
        files (list of str): List of file paths to load data from.
        batch_size (int): Number of samples per batch.
        total_limit (int): Maximum number of samples to load (default: None).
    Yields:
        tuple: Batch of data and labels (both as NumPy arrays).
    """
    total_loaded = 0  # Counter to track total loaded samples

    for file in files:
        print(f"[INFO] Loading data from {file}...")

        # Load data and labels from .npz file
        if file.endswith(".npz"):
            with np.load(file) as data_file:
                # Upewnij się, że dane to tablica NumPy
                data = np.array(data_file["data"])
                # Upewnij się, że etykiety to tablica NumPy
                labels = np.array(data_file["labels"])
        else:  # If not .npz, assume it's a CIFAR-10 batch
            data, labels = load_cifar10_batch(file)
            data = np.array(data)  # Konwersja do tablicy NumPy
            labels = np.array(labels)  # Konwersja do tablicy NumPy

        # Log diagnostyczny dla całego pliku
        print(
            f"[DEBUG] Data shape: {data.shape}, Labels shape: {labels.shape}")
        print(
            f"[DEBUG] Min data value: {data.min()}, Max data value: {data.max()}")
        print(f"[DEBUG] Unique labels: {np.unique(labels)}")

        for i in range(0, len(data), batch_size):
            if total_limit is not None and total_loaded >= total_limit:
                return  # Stop if total_limit is reached

            batch_data = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            # Update the counter
            total_loaded += len(batch_data)

            # Log diagnostyczny dla batchy
            print(
                f"[DEBUG] Batch size: {len(batch_data)}, Total loaded: {total_loaded}")
            print(f"[DEBUG] Batch labels: {np.unique(batch_labels)}")

            # Stop if total_limit is reached
            if total_limit is not None and total_loaded > total_limit:
                batch_data = batch_data[: total_limit -
                                        (total_loaded - len(batch_data))]
                batch_labels = batch_labels[: total_limit -
                                            (total_loaded - len(batch_labels))]
                yield batch_data, batch_labels
                return

            yield batch_data, batch_labels


def train_model_in_batches(model, data_files, batch_size, epochs, callbacks, total_limit=None):
    """
    Trains a model using data loaded in batches and logs metrics for each batch.
    """
    for epoch in range(epochs):
        print(f"[INFO] Starting epoch {epoch + 1}/{epochs}...")
        batch_count = 0
        for batch_data, batch_labels in load_data_in_batches(data_files, batch_size, total_limit=total_limit):
            # Log diagnostyczny dla danych wejściowych
            print(f"[DEBUG] Training batch data shape: {batch_data.shape}")
            print(f"[DEBUG] Training batch labels shape: {batch_labels.shape}")
            print(f"[DEBUG] Unique labels in batch: {np.unique(batch_labels)}")

            metrics = model.train_on_batch(batch_data, batch_labels)
            loss, accuracy = metrics[0], metrics[1]
            batch_count += 1
            print(
                f"[INFO] Epoch {epoch + 1}, Batch {batch_count} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


def evaluate_model_in_batches(model, data_files, batch_size):
    """
    Evaluates a model using data loaded in batches and returns average metrics.
    """
    total_loss, total_accuracy = 0, 0
    batch_count = 0
    for batch_data, batch_labels in load_data_in_batches(data_files, batch_size):
        # Log diagnostyczny dla danych testowych
        print(f"[DEBUG] Testing batch data shape: {batch_data.shape}")
        print(f"[DEBUG] Testing batch labels shape: {batch_labels.shape}")
        print(f"[DEBUG] Unique labels in batch: {np.unique(batch_labels)}")

        metrics = model.test_on_batch(batch_data, batch_labels)
        loss, accuracy = metrics[0], metrics[1]
        total_loss += loss
        total_accuracy += accuracy
        batch_count += 1
        print(
            f"[INFO] Batch {batch_count} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Calculate average loss and accuracy
    average_loss = total_loss / batch_count
    average_accuracy = total_accuracy / batch_count
    print(
        f"[INFO] Evaluation completed. Average Loss: {average_loss:.4f}, Average Accuracy: {average_accuracy:.4f}")
    return average_loss, average_accuracy


# Define input shapes
input_height, input_width = 32, 32
input_shape_raw = (input_height, input_width, 3)
input_shape_tda = (input_height, input_width, 6)

# Initialize models
print("[INFO] Initializing MiniVGG Raw Model...")
raw_model = MiniVGG_Raw_Model2(input_shape=input_shape_raw).model
print("[INFO] Initializing MiniVGG TDA Model...")
tda_model = MiniVGG_TDA_Model2(input_shape=input_shape_tda).model

# Define early stopping callback
callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

# Train and evaluate raw model
print("[INFO] Training MiniVGG Raw Model...")
train_model_in_batches(
    raw_model, raw_files[:-1], batch_size=33, epochs=15, callbacks=callbacks, total_limit=15000
)
print("[INFO] Evaluating MiniVGG Raw Model...")
raw_eval = evaluate_model_in_batches(
    raw_model, raw_files[-1:], batch_size=100)
print(f"[RESULT] Raw Model Evaluation Completed.")

# Train and evaluate TDA model
print("[INFO] Training MiniVGG TDA Model...")
train_model_in_batches(
    tda_model, processed_files[:-2], batch_size=33, epochs=15, callbacks=callbacks, total_limit=15000
)
print("[INFO] Evaluating MiniVGG TDA Model...")
tda_eval = evaluate_model_in_batches(
    tda_model, processed_files[-2:], batch_size=100)
print(f"[RESULT] TDA Model Evaluation Completed.")
