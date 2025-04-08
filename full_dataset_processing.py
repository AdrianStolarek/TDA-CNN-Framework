from sys import platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from data_preprocessing_pipelines_FUNGI import detect_image_size, load_from_folder, save_in_batches
from data_enrichment_pipelines import VECTOR_STITCHING_PI_Pipeline_RGB, VECTOR_STITCHING_PI_Pipeline_RGB_pure

BATCH_PATH = ""
TEMP_DIR = ""
PURE_DIR = ""

if platform == "win32" or platform == "linux2":
    BATCH_PATH = "E:\\ZoryaDyn_Benchmarks\\FUNGI"
    TEMP_DIR = "E:\\ZoryaDyn_Benchmarks\\FUNGI_processed"
    PURE_DIR = "E:\\ZoryaDyn_Benchmarks\\FUNGI_processed_pure"
else:
    BATCH_PATH = Path(__file__).parent / 'data' / 'DeFungi'
    TEMP_DIR = Path(__file__).parent / 'data' / 'DeFungi-output' / 'processed'
    PURE_DIR = Path(__file__).parent / 'data' / 'DeFungi-output' / 'processed_PURE'

BATCH_SIZE = 10
DATASET_LIMIT = 3000

X_full, y_full = load_from_folder(BATCH_PATH, dataset_size_limit=DATASET_LIMIT)

print("Unique labels:", np.unique(y_full))
print("Label counts:", {label: np.sum(y_full == label)
                        for label in np.unique(y_full)})

INPUT_HEIGHT, INPUT_WIDTH = detect_image_size(X_full)

label_encoder = LabelEncoder()
y_full_encoded = label_encoder.fit_transform(y_full)

vector_stitching_pipeline = VECTOR_STITCHING_PI_Pipeline_RGB(
    input_height=INPUT_HEIGHT,
    input_width=INPUT_WIDTH
)

vector_stitching_pipeline_pure = VECTOR_STITCHING_PI_Pipeline_RGB_pure(
    input_height=INPUT_HEIGHT,
    input_width=INPUT_WIDTH
)

print("[INFO] Processing LIMITED data through TDA pipeline...")
save_in_batches(
    vector_stitching_pipeline,
    X_full,
    y_full_encoded,
    batch_size=BATCH_SIZE,
    output_dir=TEMP_DIR
)

print("[INFO] Processing LIMITED data through TDA pipeline (pure PI version)...")
save_in_batches(
    vector_stitching_pipeline_pure,
    X_full,
    y_full_encoded,
    batch_size=BATCH_SIZE,
    output_dir=PURE_DIR
)

print("[INFO] Data processed!")
