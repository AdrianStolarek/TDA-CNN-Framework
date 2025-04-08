import mlflow
from keras import callbacks
import numpy as np
from pathlib import Path
from keras import layers

from optuna_training_framework import OptunaRunner, DataParams, OptunaParams
from ARCHIVE.data_preprocessing import load_cifar10_batch
from classification_model import SklearnClassificationModel

INPUT_HEIGHT = 32
INPUT_WIDTH = 32
INPUT_SHAPE_RAW = (INPUT_HEIGHT, INPUT_WIDTH, 3)
DATASET_SIZE_LIMIT = 4000
BATCH_PATH = Path(__file__).parent / "data" / "cifar-10" / "data_batch_1"

raw_data, labels = load_cifar10_batch(BATCH_PATH)
raw_data_limited = raw_data[:DATASET_SIZE_LIMIT]
labels_limited = labels[:DATASET_SIZE_LIMIT]


def objective(trial, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        params = {
            'filters_in_type_1_block': trial.suggest_int('filters_in_type_1_block', 32, 128, step=32),
            'filters_in_type_2_block': trial.suggest_int('filters_in_type_2_block', 64, 256, step=64),
            'filters_in_type_3_block': trial.suggest_int('filters_in_type_3_block', 128, 512, step=128),
        }

        model_builder = SklearnClassificationModel(layers.Conv2D(
            32, (3, 3), activation='relu', padding='same', input_shape=INPUT_SHAPE_RAW))

        model_builder.add_type1_block(
            filters=params['filters_in_type_1_block'])

        model_builder.add_type2_block(
            filters=params['filters_in_type_2_block'])

        model_builder.add_type3_block(
            filters=params['filters_in_type_3_block'])

        model_builder.add_type4_block()

        model = model_builder.compile_model(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        mlflow.log_params(params)

        callbacks_es = [callbacks.EarlyStopping(
            monitor='val_loss', patience=10)]

        # Train and evaluate raw model
        history = model.fit(
            np.array(X_train), np.array(y_train), validation_split=0.2, epochs=20, batch_size=30,
            callbacks=callbacks_es, verbose=0)
        result = model.evaluate(
            np.array(X_test), np.array(y_test), verbose=1)

        mlflow.log_metric('raw-loss', result[0])
        mlflow.log_metric('raw-accuracy', result[1])

        return result[1]


data_params = DataParams(data=raw_data_limited,
                         labels=labels_limited, random_state=42, test_size=30)
optuna_params = OptunaParams(direction="maximize", objective=objective)

runner = OptunaRunner("michal-sklearn-model-builder-test",
                      data_params=data_params, optuna_params=optuna_params)

runner.run_trial(50)
