import numpy as np
import os
import mlflow
import optuna
from sklearn.preprocessing import LabelEncoder
from data_preprocessing_pipelines import detect_image_size
from models.raw_models import MiniVGG_Raw_Model
from keras import mixed_precision

from cross_validation_objective import cross_validation_objective

mlflow.set_tracking_uri("http://mlflow.zorya-dynamics.cloud")
mixed_precision.set_global_policy('mixed_float16')

RAW_DIR = "E:\\ZoryaDyn_Benchmarks\\WAR_TECH\\processed_RAW"
RANDOM_STATE = 1
N_FOLDS = 5  # don't touch xd
EXPERIMENT_NAME = "2025-03-14_war_tech_raw_optimization-TEST"

print(f"[INFO] Loading raw dataset from {RAW_DIR}...")
raw_data = np.load(os.path.join(RAW_DIR, "raw_dataset.npz"))
X_raw = raw_data['X']
y_raw = raw_data['y']

print("=" * 50)
print("Raw Dataset Information:")
print(f"Raw dataset shape: {X_raw.shape}")
print(f"Number of classes: {len(np.unique(y_raw))}")
print(f"Classes: {np.unique(y_raw)}")
print("=" * 50)

INPUT_HEIGHT, INPUT_WIDTH = X_raw.shape[1:3]
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, X_raw.shape[3])

print(f"Raw input shape: {INPUT_SHAPE}")


def objective(trial):
    return cross_validation_objective(
        trial=trial,
        X_data=X_raw,
        y_data=y_raw,
        input_shape=INPUT_SHAPE,
        model_class=MiniVGG_Raw_Model,
        n_folds=N_FOLDS,
        experiment_name=EXPERIMENT_NAME,
        optimize_target='accuracy'
    )


print(
    f"[INFO] Starting Optuna study for RAW data with {N_FOLDS}-fold cross-validation...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

print('='*50)
print('Best parameters for RAW model:')
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"Best accuracy: {study.best_value:.4f}")
print('='*50)

with open(os.path.join(RAW_DIR, "optuna_raw_results.txt"), "w") as f:
    f.write(f"Best parameters for RAW model:\n")
    for key, value in study.best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"Best accuracy: {study.best_value:.4f}\n")

print("[INFO] Raw data optimization complete!")
