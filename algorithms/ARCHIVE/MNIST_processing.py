from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from models.raw_models import MiniVGG_Raw_Model2
from models.tda_models import MiniVGG_TDA_Model2
from MNIST_pipelines import VECTOR_STITCHING_PI_Pipeline_Gray
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist


def add_gaussian_noise(images, mean=0.0, std=0.1):
    """
    Dodaje szum Gaussowski do obrazów.
    """
    noise = np.random.normal(mean, std, images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0.0, 1.0)  # Przycięcie wartości do [0,1]


def add_salt_and_pepper_noise(images, prob=0.02):
    """
    Dodaje szum typu "salt and pepper" do obrazów.
    """
    noisy_images = np.copy(images)
    for img in noisy_images:
        # Ile pikseli zmieniamy
        num_pixels = int(prob * img.shape[0] * img.shape[1])

        if num_pixels < 1:  # Zapewnienie, że nie próbujemy dodać 0 pikseli
            num_pixels = 1

        # Dodajemy sól (białe piksele)
        coords_x = np.random.randint(0, img.shape[0], num_pixels)
        coords_y = np.random.randint(0, img.shape[1], num_pixels)
        img[coords_x, coords_y] = 1.0  # Biały piksel

        # Dodajemy pieprz (czarne piksele)
        coords_x = np.random.randint(0, img.shape[0], num_pixels)
        coords_y = np.random.randint(0, img.shape[1], num_pixels)
        img[coords_x, coords_y] = 0.0  # Czarny piksel

    return noisy_images


# Load MNIST dataset
print("[INFO] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data to match the expected format (height, width, channels)
x_train = np.expand_dims(x_train, axis=-1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)  # (10000, 28, 28, 1)

# Normalize pixel values to the range [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Limit dataset size for faster testing
LIMIT = 1000
TRAIN_SLICE = 0.8
TRAIN_LIMIT = int(LIMIT * TRAIN_SLICE)

x_train_limited = x_train[:LIMIT]
y_train_limited = y_train[:LIMIT]

# Podział danych dla obu modeli
x_train_subset, x_test_subset, y_train_subset, y_test_subset = train_test_split(
    x_train_limited, y_train_limited, test_size=0.2, random_state=42
)

print(
    f"[INFO] Loaded data shape (limited): {x_train_limited.shape}, Labels: {len(y_train_limited)}")
print(
    f"[INFO] Training set: {x_train_subset.shape}, Test set: {x_test_subset.shape}")

# Zaszumienie tylko danych testowych
print("[INFO] Adding noise to test data...")
x_test_noisy = add_gaussian_noise(x_test_subset, mean=0.85, std=0.15)
x_test_noisy = add_salt_and_pepper_noise(x_test_noisy, prob=0.02)

# Wizualizacja kilku zaszumionych obrazów
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    axes[i].imshow(x_test_noisy[i].squeeze(), cmap='gray')
    axes[i].axis('off')
plt.show()

# Set input size
input_height, input_width = 28, 28
print("[INFO] Setting input dimensions to (28, 28)")

# Initialize pipeline
print("[INFO] Initializing Vector Stitching Pipeline for grayscale data...")
vector_stitching_pipeline = VECTOR_STITCHING_PI_Pipeline_Gray(
    input_height=input_height, input_width=input_width
)

# Process data through pipeline for the TDA model
print("[INFO] Processing data through pipeline for TDA model...")
X_train_tda = vector_stitching_pipeline.fit_transform(x_train_subset)
X_test_tda = vector_stitching_pipeline.fit_transform(x_test_noisy)

print(f"[INFO] Final Vector Stitched Data Shape (Train): {X_train_tda.shape}")
print(f"[INFO] Final Vector Stitched Data Shape (Test): {X_test_tda.shape}\n")

assert X_train_tda.shape[1:] == (input_height, input_width, 2), \
    f"Unexpected Vector Stitched data shape: {X_train_tda.shape}"

# Define input shapes
input_shape_raw = (input_height, input_width, 1)
input_shape_tda = (input_height, input_width, 2)

# Initialize models
print("[INFO] Initializing MiniVGG Raw Model...")
raw_model = MiniVGG_Raw_Model2(input_shape=input_shape_raw).model
print("[INFO] Initializing MiniVGG TDA Model...")
tda_model = MiniVGG_TDA_Model2(input_shape=input_shape_tda).model

# Define early stopping callback
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

# Convert data to numpy arrays for compatibility
X_train_tda = np.array(X_train_tda)
X_test_tda = np.array(X_test_tda)
x_train_subset = np.array(x_train_subset)
x_test_noisy = np.array(x_test_noisy)

# Train and evaluate raw model
print("[INFO] Training MiniVGG Raw Model...")
history_raw = raw_model.fit(
    x_train_subset, y_train_subset, validation_split=0.2, epochs=20, batch_size=1,
    callbacks=callbacks, verbose=1
)
print("[INFO] Evaluating MiniVGG Raw Model...")
raw_eval = raw_model.evaluate(x_test_noisy, y_test_subset, verbose=1)
print(f"[RESULT] Raw Model Test Accuracy: {raw_eval[1]:.4f}\n")

# Train and evaluate TDA model
print("[INFO] Training MiniVGG TDA Model...")
history_tda = tda_model.fit(
    X_train_tda, y_train_subset, validation_split=0.2, epochs=20, batch_size=1,
    callbacks=callbacks, verbose=1
)
print("[INFO] Evaluating MiniVGG TDA Model...")
tda_eval = tda_model.evaluate(X_test_tda, y_test_subset, verbose=1)
print(f"[RESULT] TDA Model Test Accuracy: {tda_eval[1]:.4f}\n")
