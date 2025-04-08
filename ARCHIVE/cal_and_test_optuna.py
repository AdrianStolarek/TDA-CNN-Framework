import numpy as np
from sklearn.model_selection import train_test_split
from keras import callbacks
import mlflow
import optuna

from data_preprocessing import load_cifar10_batch
from tda_pipelines_PCA import VECTOR_STITCHING_PI_Pipeline_RGB
from models.raw_models import MiniVGG_Raw_Model
from models.tda_models import MiniVGG_TDA_Model

mlflow.set_tracking_uri("http://mlflow.zorya-dynamics.cloud")
mlflow.set_experiment('raw-vs-tda-cifar-10-TEST-RUN')

# constants
BATCH_PATH = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10-batches-py\\data_batch_1"
RANDOM_STATE = 1
DATASET_SIZE_LIMIT = 500
TEST_SIZE = 0.2
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
INPUT_SHAPE_RAW = (INPUT_HEIGHT, INPUT_WIDTH, 3)
INPUT_SHAPE_TDA = (INPUT_HEIGHT, INPUT_WIDTH, 6)

# Slicing data
raw_data, labels = load_cifar10_batch(BATCH_PATH)
raw_data_limited = raw_data[:DATASET_SIZE_LIMIT]
labels_limited = labels[:DATASET_SIZE_LIMIT]
labels_array = np.array(labels_limited)

X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
    raw_data_limited, labels_limited, test_size=TEST_SIZE, stratify=labels_limited, random_state=RANDOM_STATE)

# Initialize pipeline
vector_stitching_pipeline = VECTOR_STITCHING_PI_Pipeline_RGB(
    input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH)

# Process data
print("[INFO] Processing data through pipeline...")
vector_stitched_data = vector_stitching_pipeline.fit_transform(
    raw_data_limited)
print(
    f"[INFO] Final Vector Stitched Data Shape: {vector_stitched_data.shape} \n")

X_vector_train, X_vector_test, y_vector_train, y_vector_test = train_test_split(
    vector_stitched_data, labels_limited, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def objective(trial):
    with mlflow.start_run():
        params = {
            # 32, 128
            'block_1_size': trial.suggest_int('block_1_size', 32, 512, step=16),
            # 64, 256
            'block_2_size': trial.suggest_int('block_2_size', 32, 512, step=32),
            # 128, 512
            'block_3_size': trial.suggest_int('block_3_size', 32, 512, step=64),
            'optimizer': trial.suggest_categorical('optimizer', ['adam'])
        }
        raw_model = MiniVGG_Raw_Model(
            input_shape=INPUT_SHAPE_RAW, **params).model
        tda_model = MiniVGG_TDA_Model(
            input_shape=INPUT_SHAPE_TDA, **params).model

        mlflow.log_params(params)

        callbacks_es = [callbacks.EarlyStopping(
            monitor='val_loss', patience=10)]

        # Train and evaluate raw model
        history_raw = raw_model.fit(
            np.array(X_raw_train), np.array(y_raw_train), validation_split=0.2, epochs=20, batch_size=30,
            callbacks=callbacks_es, verbose=0)
        raw_eval = raw_model.evaluate(
            np.array(X_raw_test), np.array(y_raw_test), verbose=1)

        history_tda = tda_model.fit(
            np.array(X_vector_train), np.array(y_vector_train), validation_split=0.2, epochs=20, batch_size=30,
            callbacks=callbacks_es, verbose=0)
        tda_eval = tda_model.evaluate(
            np.array(X_vector_test), np.array(y_vector_test), verbose=1)

        mlflow.log_metric('raw-loss', raw_eval[0])
        mlflow.log_metric('raw-accuracy', raw_eval[1])
        mlflow.log_metric('tda-loss', tda_eval[0])
        mlflow.log_metric('tda-accuracy', tda_eval[1])

        # todo, log modle to mlflow

        return tda_eval[1]


# study = optuna.create_study(direction=["minimize", "maximize"])
study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=10)

print('best params:')
print(study.best_params)
