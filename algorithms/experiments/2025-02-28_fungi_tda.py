import numpy as np
from sklearn.model_selection import train_test_split
from keras import callbacks
import mlflow
import optuna
import shutil
import atexit
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from data_preprocessing_pipelines import detect_image_size, load_from_folder, load_from_csv, load_processed_batches
from models.raw_models import MiniVGG_Raw_Model
from models.tda_models import MiniVGG_TDA_Model

BATCH_PATH = Path(__file__).parent / 'data' / 'DeFungi'
TEMP_DIR = Path(__file__).parent / 'data' / 'DeFungi-tmp'
RANDOM_STATE = 2
DATASET_SIZE_LIMIT = 500  # 9100 is the entire dataset
TEST_SIZE = 0.2

mlflow.set_tracking_uri("http://mlflow.zorya-dynamics.cloud")
mlflow.set_experiment(f'2025-02-28_fungi_tda_dataset_size--{DATASET_SIZE_LIMIT}')

X_full_processed, y_full_processed = load_processed_batches(
    output_dir=TEMP_DIR, batch_size=DATASET_SIZE_LIMIT)

INPUT_HEIGHT, INPUT_WIDTH = detect_image_size(X_full_processed)
INPUT_SHAPE_RAW = (INPUT_HEIGHT, INPUT_WIDTH, 3)
INPUT_SHAPE_TDA = (INPUT_HEIGHT, INPUT_WIDTH, 6)

X_tda_train, X_tda_test, y_tda_train, y_tda_test = train_test_split(
    X_full_processed,
    y_full_processed,
    test_size=TEST_SIZE,
    stratify=y_full_processed,
    random_state=RANDOM_STATE
)

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
        
        tda_model = MiniVGG_TDA_Model(
            input_shape=INPUT_SHAPE_TDA, **params).model

        mlflow.log_param('dataset-size', DATASET_SIZE_LIMIT)
        mlflow.log_param('random_state', RANDOM_STATE)
        mlflow.log_params(params)

        callbacks_es = [callbacks.EarlyStopping(
            monitor='accuracy', patience=10)]

        tda_history = tda_model.fit(
            X_tda_train.astype('float32'),
            y_tda_train.astype('int32'),
            epochs=40,
            batch_size=32,
            callbacks=callbacks_es,
            verbose=0
        )
        tda_eval = tda_model.evaluate(
            X_tda_test.astype('float32'),
            y_tda_test.astype('int32'),
            verbose=0
        )

        mlflow.log_metrics({
            'tda-loss': tda_eval[0],
            'tda-accuracy': tda_eval[1]
        })

        return tda_eval[1]

study = optuna.create_study(direction="maximize", storage='sqlite:///my.db')

study.optimize(objective, n_trials=1000)
