import os
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from models.raw_models import MiniVGG_Raw_Model2
from models.tda_models import MiniVGG_TDA_Model2
from data_preprocessing import load_cifar10_batch

# **1. Ścieżki do przetworzonych danych TDA i surowych RAW**
processed_data_dir = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10_processed"
raw_data_dir = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10-batches-py"

# **Przetworzone dane dla modelu TDA**
processed_files = [
    f"{processed_data_dir}\\BATCH{i}_processed_batch_{j}.npz"
    for i in range(1, 6) for j in range(1, 3)
] + [
    f"{processed_data_dir}\\TEST_BATCH_processed_batch_{i}.npz" for i in range(1, 3)
]

# **Surowe dane dla modelu RAW**
raw_files = [
    f"{raw_data_dir}\\data_batch_{i}" for i in range(1, 6)
] + [f"{raw_data_dir}\\test_batch"]

# **Zaszumione dane testowe dla modelu RAW**
raw_noisy_test_files = [
    f"{processed_data_dir}\\raw_noisy_batch_1.npy",
    f"{processed_data_dir}\\raw_noisy_batch_2.npy"
]

# **Etykiety do zaszumionych testowych danych RAW**
raw_noisy_labels_files = [
    f"{processed_data_dir}\\raw_noisy_labels_1.npy",
    f"{processed_data_dir}\\raw_noisy_labels_2.npy"
]


# **2. Funkcja do ładowania danych w batchach**
def load_data_in_batches(files, label_files=None, batch_size=100, total_limit=None):
    """
    Generator do ładowania danych i etykiet w batchach.
    - Dla plików `.npz` pobiera `data` i `labels`.
    - Dla plików `.npy` ładuje `data` oraz **szuka osobnych plików z etykietami**.
    """
    total_loaded = 0

    for i, file in enumerate(files):
        print(f"[INFO] Loading data from {file}...")

        # **Ładowanie danych**
        if file.endswith(".npz"):
            with np.load(file) as data_file:
                data = np.array(data_file["data"])
                labels = np.array(data_file["labels"])
        elif file.endswith(".npy"):
            data = np.load(file)
            # **Ładowanie etykiet**
            labels = np.load(
                label_files[i]) if label_files is not None else None
        else:
            data, labels = load_cifar10_batch(file)
            data = np.array(data)
            labels = np.array(labels)

        print(f"[DEBUG] Data shape: {data.shape}")
        if labels is not None:
            print(
                f"[DEBUG] Labels shape: {labels.shape}, Unique labels: {np.unique(labels)}")

        for j in range(0, len(data), batch_size):
            if total_limit is not None and total_loaded >= total_limit:
                return

            batch_data = data[j:j + batch_size]
            batch_labels = labels[j:j +
                                  batch_size] if labels is not None else None

            total_loaded += len(batch_data)

            print(
                f"[DEBUG] Batch {j // batch_size + 1} - Size: {len(batch_data)}, Total loaded: {total_loaded}")

            yield batch_data, batch_labels


# **3. Funkcja do trenowania modelu w batchach**
def train_model_in_batches(model, data_files, batch_size, epochs, callbacks, total_limit=None):
    """
    Trenuje model na batchach danych.
    """
    for epoch in range(epochs):
        print(f"[INFO] Starting epoch {epoch + 1}/{epochs}...")
        batch_count = 0
        for batch_data, batch_labels in load_data_in_batches(data_files, batch_size=batch_size, total_limit=total_limit):
            if batch_labels is None:
                raise ValueError("Brak etykiet w danych treningowych!")

            metrics = model.train_on_batch(batch_data, batch_labels)
            loss, accuracy = metrics[0], metrics[1]
            batch_count += 1
            print(
                f"[INFO] Epoch {epoch + 1}, Batch {batch_count} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


# **4. Funkcja do ewaluacji modelu w batchach**
def evaluate_model_in_batches(model, data_files, label_files, batch_size):
    """
    Testuje model na batchach danych testowych.
    """
    total_loss, total_accuracy = 0, 0
    batch_count = 0
    for batch_data, batch_labels in load_data_in_batches(data_files, label_files, batch_size=batch_size):
        if batch_labels is None:
            raise ValueError("Brak etykiet w danych testowych!")

        metrics = model.test_on_batch(batch_data, batch_labels)
        loss, accuracy = metrics[0], metrics[1]
        total_loss += loss
        total_accuracy += accuracy
        batch_count += 1
        print(
            f"[INFO] Batch {batch_count} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    average_loss = total_loss / batch_count
    average_accuracy = total_accuracy / batch_count
    print(
        f"[INFO] Evaluation Completed. Average Loss: {average_loss:.4f}, Average Accuracy: {average_accuracy:.4f}")
    return average_loss, average_accuracy


# **5. Ustawienia modelu**
input_height, input_width = 32, 32
input_shape_raw = (input_height, input_width, 3)
input_shape_tda = (input_height, input_width, 6)

print("[INFO] Initializing MiniVGG Raw Model...")
raw_model = MiniVGG_Raw_Model2(input_shape=input_shape_raw).model
print("[INFO] Initializing MiniVGG TDA Model...")
tda_model = MiniVGG_TDA_Model2(input_shape=input_shape_tda).model

callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

# **6. Trening i testowanie modelu RAW**
print("[INFO] Training MiniVGG Raw Model...")
train_model_in_batches(
    raw_model, raw_files[:-1], batch_size=100, epochs=10, callbacks=callbacks, total_limit=5000
)

print("[INFO] Evaluating MiniVGG Raw Model on noisy test data...")
raw_eval = evaluate_model_in_batches(
    raw_model, raw_noisy_test_files, raw_noisy_labels_files, batch_size=100
)
print(f"[RESULT] Raw Model Evaluation Completed.")

# **7. Trening i testowanie modelu TDA**
print("[INFO] Training MiniVGG TDA Model...")
train_model_in_batches(
    tda_model, processed_files[:-2], batch_size=100, epochs=10, callbacks=callbacks, total_limit=5000
)

print("[INFO] Evaluating MiniVGG TDA Model...")
tda_eval = evaluate_model_in_batches(
    tda_model, processed_files[-2:], None, batch_size=100
)
print(f"[RESULT] TDA Model Evaluation Completed.")
