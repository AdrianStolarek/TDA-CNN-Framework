import mlflow
from keras import callbacks
import numpy as np
from pathlib import Path

from optuna_training_framework import OptunaRunner, DataParams, OptunaParams
from ARCHIVE.data_preprocessing import load_cifar10_batch
from models.raw_models import MiniVGG_Raw_Model

INPUT_HEIGHT = 32
INPUT_WIDTH = 32
INPUT_SHAPE_RAW = (INPUT_HEIGHT, INPUT_WIDTH, 3)
DATASET_SIZE_LIMIT = 500
BATCH_PATH = Path(__file__).parent / "data" / "cifar-10" / "data_batch_1"

raw_data, labels = load_cifar10_batch(BATCH_PATH)
raw_data_limited = raw_data[:DATASET_SIZE_LIMIT]
labels_limited = labels[:DATASET_SIZE_LIMIT]

def objective(trial, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        params = {
            # 32, 128
            'block_1_size': trial.suggest_int('block_1_size', 32, 512, step=16),
            # 64, 256
            'block_2_size': trial.suggest_int('block_2_size', 32, 512, step=32),
            # 128, 512
            'block_3_size': trial.suggest_int('block_3_size', 32, 512, step=64),
            'optimizer': trial.suggest_categorical('optimizer', ['adam','sgd'])
        }
        raw_model = MiniVGG_Raw_Model(
            input_shape=INPUT_SHAPE_RAW, **params).model

        mlflow.log_params(params)

        callbacks_es = [callbacks.EarlyStopping(
            monitor='val_loss', patience=10)]

        # Train and evaluate raw model
        history_raw = raw_model.fit(
            np.array(X_train), np.array(y_train), validation_split=0.2, epochs=20, batch_size=30,
            callbacks=callbacks_es, verbose=0)
        raw_eval = raw_model.evaluate(
            np.array(X_test), np.array(y_test), verbose=1)

        mlflow.log_metric('raw-loss', raw_eval[0])
        mlflow.log_metric('raw-accuracy', raw_eval[1])

        # todo, log modle to mlflow

        return raw_eval[1]


data_params = DataParams(data=raw_data_limited, labels=labels_limited, random_state=42, test_size=30)
optuna_params = OptunaParams(direction="maximize", objective=objective)

runner = OptunaRunner("test-run-raw-vgg", data_params=data_params, optuna_params=optuna_params)

runner.run_trial(10)
