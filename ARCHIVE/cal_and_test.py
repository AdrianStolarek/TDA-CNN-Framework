from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from models.raw_models import MiniVGG_Raw_Model
from models.tda_models import MiniVGG_TDA_Model
from data_preprocessing import load_cifar10_batch
from tda_pipelines import VECTOR_STITCHING_PI_Pipeline_RGB
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#   Na razie najistotniejsze jest przeprowadzenie sensownych testów na CIFAR-10 i większych sieciach (może być nawet 10 milionów parametrów).
#   Przetwórz wszystkie dane trenignowe i testowe a następnie wytrenuj te sieci i zweryfikuj skuteczność + w jakich sytuacjach to jest skuteczne (przedziały wielkości).
#   W zasadzie wszystko jest gotowe, tylko trzeba powiększyć sieci i masowo przetworzyć obrazy.
#   Dodatkowo przed przetwarzaniem ogarnąłbym skalowanie intensywności kolorów na PI.

# Define the CIFAR-10 batch path
batch_path = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10-batches-py\\data_batch_1"

# Load data
print("[INFO] Loading dataset...")
raw_data, labels = load_cifar10_batch(batch_path)
print(f"Model will train on: {tf.test.gpu_device_name() or 'CPU'}")
print(tf.config.list_physical_devices())


def display_single_image(image, label, title="Raw Image"):
    plt.figure(figsize=(5, 5))
    plt.imshow(image.astype("uint8"))
    plt.title(f"{title} - Label: {label}")
    plt.axis("off")
    plt.show()


print("[INFO] Displaying a single raw image...")
display_single_image(raw_data[200], labels[200])

# Limit dataset size
LIMIT = 2000
TRAIN_SLICE = 0.8
TRAIN_LIMIT = int(LIMIT * TRAIN_SLICE)

raw_data_limited = raw_data[:LIMIT]
labels_limited = labels[:LIMIT]
labels_array = np.array(labels_limited)

raw_data_train = raw_data_limited[:TRAIN_LIMIT]
labels_train = labels_limited[:TRAIN_LIMIT]
labels_train_array = np.array(labels_train)

raw_data_test = raw_data[TRAIN_LIMIT:LIMIT]
labels_test = labels[TRAIN_LIMIT:LIMIT]
labels_test_array = np.array(labels_test)

print(
    f"[INFO] Loaded data shape (limited): {raw_data_limited.shape}, Labels: {len(labels_limited)}")

# Set input size
input_height = 32
input_width = 32
print("[INFO] Setting input dimensions to (32, 32)")

# Initialize pipeline
print("[INFO] Initializing Vector Stitching Pipeline for RGB data...")
vector_stitching_pipeline = VECTOR_STITCHING_PI_Pipeline_RGB(
    input_height=input_height, input_width=input_width)

# Process data
print("[INFO] Processing data through pipeline...")
vector_stitched_data = vector_stitching_pipeline.fit_transform(
    raw_data_limited)
print(
    f"[INFO] Final Vector Stitched Data Shape: {vector_stitched_data.shape} \n")

assert vector_stitched_data.shape[1:] == (input_height, input_width, 6), \
    f"Unexpected Vector Stitched data shape: {vector_stitched_data.shape}"

print("[INFO] Data processed successfully. Splitting into training and testing sets...")


def display_single_processed_image(processed_data, title="Processed Image"):
    sample_image = processed_data[200]
    final1 = sample_image[0]
    final2 = sample_image[14]
    final3 = sample_image[28]
    combined_image = sample_image.mean(axis=0)

    plt.subplot(1, 3, 1)
    plt.imshow(final1)
    plt.title(title)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(final2)
    plt.title(title)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(final3)
    plt.title(title)
    plt.axis("off")
    plt.show()


print("[INFO] Displaying a single processed image...")
display_single_processed_image(vector_stitched_data)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    vector_stitched_data, labels_limited, test_size=0.2, random_state=42)

print("[INFO] Data split into training and testing sets.")
print(
    f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}\n")

# Define input shapes
input_shape_raw = (input_height, input_width, 3)
input_shape_tda = (input_height, input_width, 6)

# Initialize models
print("[INFO] Initializing MiniVGG Raw Model...")
raw_model = MiniVGG_Raw_Model(input_shape=input_shape_raw).model
print("[INFO] Initializing MiniVGG TDA Model...")
tda_model = MiniVGG_TDA_Model(input_shape=input_shape_tda).model

# Define early stopping callback
callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Train and evaluate raw model
print("[INFO] Training MiniVGG Raw Model...")
history_raw = raw_model.fit(
    raw_data_train, labels_train_array, validation_split=0.2, epochs=40, batch_size=30,
    callbacks=callbacks, verbose=1)
print("[INFO] Evaluating MiniVGG Raw Model...")
raw_eval = raw_model.evaluate(raw_data_test, labels_test_array, verbose=1)
print(f"[RESULT] Raw Model Test Accuracy: {raw_eval[1]:.4f}\n")

# Train and evaluate TDA model
print("[INFO] Training MiniVGG TDA Model...")
history_tda = tda_model.fit(
    X_train, y_train, validation_split=0.2, epochs=40, batch_size=30,
    callbacks=callbacks, verbose=1)
print("[INFO] Evaluating MiniVGG TDA Model...")
tda_eval = tda_model.evaluate(X_test, y_test, verbose=1)
print(f"[RESULT] TDA Model Test Accuracy: {tda_eval[1]:.4f}\n")
