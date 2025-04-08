import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from data_preprocessing_pipelines import save_in_batches
from data_enrichment_pipelines import VECTOR_STITCHING_PI_Pipeline_RGB
from data_preprocessing_pipelines import load_from_single_folder, add_gaussian_noise_to_subset


DATASET_PATH = "E:\\ZoryaDyn_Benchmarks\\WAR_TECH\\obshaya_papk"
RAW_DIR = "E:\\ZoryaDyn_Benchmarks\\WAR_TECH\\processed_RAW"
TDA_DIR = "E:\\ZoryaDyn_Benchmarks\\WAR_TECH\\processed_TDA"
BATCH_SIZE = 128
DATASET_LIMIT = 2504
TARGET_SIZE = (250, 250)

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(TDA_DIR, exist_ok=True)

print(f"[INFO] Loading images from {DATASET_PATH}...")
X_full, y_full = load_from_single_folder(
    DATASET_PATH,
    target_size=TARGET_SIZE,
    dataset_size_limit=DATASET_LIMIT,
    preserve_channels=True
)

print("Unique labels:", np.unique(y_full))
print("Label counts:", {label: np.sum(y_full == label)
      for label in np.unique(y_full)})
print("Image shape:", X_full[0].shape)

X_full_noisy = add_gaussian_noise_to_subset(
    X_full, proportion=0.2, mean=0.6, sigma=0.2)

label_encoder = LabelEncoder()
y_full_encoded = label_encoder.fit_transform(y_full)

plt.figure(figsize=(12, 6))
for i in range(min(5, len(X_full))):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_full_noisy[i])
    plt.title(f"Label: {y_full[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(RAW_DIR, "sample_images.png"))
plt.close()

print(f"[INFO] Saving raw images to {RAW_DIR}...")
np.savez(os.path.join(RAW_DIR, "raw_dataset.npz"),
         X=X_full_noisy, y=y_full_encoded)

INPUT_HEIGHT, INPUT_WIDTH = X_full[0].shape[0], X_full[0].shape[1]
INPUT_CHANNELS = X_full[0].shape[2]

print(
    f"[INFO] Image dimensions: {INPUT_HEIGHT}x{INPUT_WIDTH}x{INPUT_CHANNELS}")

rgb_pipeline = VECTOR_STITCHING_PI_Pipeline_RGB(
    input_height=INPUT_HEIGHT,
    input_width=INPUT_WIDTH
)

print("[INFO] Processing data through TDA pipeline...")
save_in_batches(
    rgb_pipeline,
    X_full_noisy,
    y_full_encoded,
    batch_size=BATCH_SIZE,
    output_dir=TDA_DIR
)

print("\n[INFO] Dataset processing complete!")
print(f"Raw data saved to: {RAW_DIR}")
print(f"TDA processed data saved to: {TDA_DIR}")
print("Dataset statistics:")
print(f"- Total images: {len(X_full)}")
print(f"- Image size: {INPUT_HEIGHT}x{INPUT_WIDTH}")
print(f"- Labels: {np.unique(y_full)}")
print(
    f"- Class distribution: {[np.sum(y_full == label) for label in np.unique(y_full)]}")
