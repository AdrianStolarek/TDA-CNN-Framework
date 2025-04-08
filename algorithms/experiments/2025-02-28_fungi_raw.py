import numpy as np
from sklearn.model_selection import train_test_split
from keras import callbacks
import tensorflow as tf
import mlflow
import optuna
import shutil
import atexit
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from data_preprocessing_pipelines import load_cifar10_batch, detect_image_size, load_from_folder, load_from_csv, process_in_batches
# from data_enrichment_pipelines import VECTOR_STITCHING_PI_Pipeline_RGB
from models.raw_models import MiniVGG_Raw_Model
from models.tda_models import MiniVGG_TDA_Model

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=10*1024)])

BATCH_PATH = Path(__file__).parent / 'data' / 'DeFungi'
TEMP_DIR = Path(__file__).parent / 'data' / 'DeFungi-tmp'
RANDOM_STATE = 2
DATASET_SIZE_LIMIT = 500  # 9100 is the entire dataset
TEST_SIZE = 0.2
BATCH_SIZE = 100  # You can crank it up a bit

mlflow.set_tracking_uri("http://mlflow.zorya-dynamics.cloud")
mlflow.set_experiment(f'2025-02-28_fungi_raw_dataset_size--{DATASET_SIZE_LIMIT}')

X_full, y_full = load_from_folder(BATCH_PATH, DATASET_SIZE_LIMIT)

print("Unique labels:", np.unique(y_full))
print("Label counts:", {label: np.sum(y_full == label)
      for label in np.unique(y_full)})

INPUT_HEIGHT, INPUT_WIDTH = detect_image_size(X_full)
INPUT_SHAPE_RAW = (INPUT_HEIGHT, INPUT_WIDTH, 3)
INPUT_SHAPE_TDA = (INPUT_HEIGHT, INPUT_WIDTH, 6)

label_encoder = LabelEncoder()
y_full_encoded = label_encoder.fit_transform(y_full)

# vector_stitching_pipeline = VECTOR_STITCHING_PI_Pipeline_RGB(
#     input_height=INPUT_HEIGHT,
#     input_width=INPUT_WIDTH
# )

# print("[INFO] Processing LIMITED data through TDA pipeline...")
# X_full_processed, y_full_processed = process_in_batches(
#     vector_stitching_pipeline,
#     X_full,
#     y_full_encoded,
#     batch_size=BATCH_SIZE,
#     output_dir=TEMP_DIR
# )

# Podział danych
X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
    X_full,
    y_full_encoded,
    test_size=TEST_SIZE,
    stratify=y_full,
    random_state=RANDOM_STATE
)

y_raw_train = np.array(y_raw_train)
y_raw_test = np.array(y_raw_test)

# X_tda_train, X_tda_test, y_tda_train, y_tda_test = train_test_split(
#     X_full_processed,
#     y_full_processed,
#     test_size=TEST_SIZE,
#     stratify=y_full_processed,
#     random_state=RANDOM_STATE
# )


# print("="*50)
# print("[DEBUG] Typy danych przed treningiem:")
# print(f"RAW: X_train - {X_raw_train.dtype}, y_train - {y_raw_train.dtype}")
# print(f"TDA: X_train - {X_tda_train.dtype}, y_train - {y_tda_train.dtype}")
# print("Przykładowe etykiety TDA:", y_tda_train[:5])
# print("="*50)


def objective(trial):
    with mlflow.start_run():
        params = {
            # 32, 128
            'block_1_size': trial.suggest_int('block_1_size', 32, 64, step=8),
            # 64, 256
            'block_2_size': trial.suggest_int('block_2_size', 32, 128, step=16),
            # 128, 512
            'block_3_size': trial.suggest_int('block_3_size', 32, 128, step=32),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        }
        raw_model = MiniVGG_Raw_Model(
            input_shape=INPUT_SHAPE_RAW, **params).model
        # tda_model = MiniVGG_TDA_Model(
        #     input_shape=INPUT_SHAPE_TDA, **params).model

        mlflow.log_params(params)

        callbacks_es = [callbacks.EarlyStopping(
            monitor='accuracy', patience=10)]

        raw_history = raw_model.fit(
            X_raw_train.astype('float32'),
            y_raw_train.astype('int32'),
            epochs=40,
            batch_size=32,
            callbacks=callbacks_es,
            verbose=0
        )
        raw_eval = raw_model.evaluate(
            X_raw_test.astype('float32'),
            y_raw_test.astype('int32'),
            verbose=0
        )

        # tda_history = tda_model.fit(
        #     X_tda_train.astype('float32'),
        #     y_tda_train.astype('int32'),
        #     epochs=40,
        #     batch_size=32,
        #     callbacks=callbacks_es,
        #     verbose=0
        # )
        # tda_eval = tda_model.evaluate(
        #     X_tda_test.astype('float32'),
        #     y_tda_test.astype('int32'),
        #     verbose=0
        # )

        mlflow.log_metrics({
            'raw-loss': raw_eval[0],
            'raw-accuracy': raw_eval[1],
            # 'tda-loss': tda_eval[0],
            # 'tda-accuracy': tda_eval[1]
        })

        return raw_eval[1]


def cleanup():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"Cleaned up temporary directory: {TEMP_DIR}")


study = optuna.create_study(direction="maximize", storage='sqlite:///my.db')

study.optimize(objective, n_trials=1200)

print('best params:')
print(study.best_params)
atexit.register(cleanup)
