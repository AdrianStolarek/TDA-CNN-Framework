import numpy as np
import os
import mlflow
import optuna
from data_preprocessing_pipelines import detect_image_size, load_processed_batches
from models.tda_models import MiniVGG_TDA_Model
from keras import mixed_precision

from cross_validation_objective import cross_validation_objective

mlflow.set_tracking_uri("http://mlflow.zorya-dynamics.cloud")
mixed_precision.set_global_policy('mixed_float16')

TDA_DIR = "E:\\ZoryaDyn_Benchmarks\\WAR_TECH\\processed_TDA"
RANDOM_STATE = 1
N_FOLDS = 5
EXPERIMENT_NAME = "2025-03-14_war_tech_tda_optimization-TEST"

print(f"[INFO] Loading TDA processed dataset from {TDA_DIR}...")
X_tda, y_tda = load_processed_batches(output_dir=TDA_DIR)

print("=" * 50)
print("TDA Dataset Information:")
print(f"TDA dataset shape: {X_tda.shape}")
print(f"Number of classes: {len(np.unique(y_tda))}")
print(f"Classes: {np.unique(y_tda)}")
print("=" * 50)

INPUT_HEIGHT, INPUT_WIDTH = X_tda.shape[1:3]
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, X_tda.shape[3])

print(f"TDA input shape: {INPUT_SHAPE}")


def objective(trial):
    return cross_validation_objective(
        trial=trial,
        X_data=X_tda,
        y_data=y_tda,
        input_shape=INPUT_SHAPE,
        model_class=MiniVGG_TDA_Model,
        n_folds=N_FOLDS,
        experiment_name=EXPERIMENT_NAME,
        optimize_target='accuracy'
    )


print(
    f"[INFO] Starting Optuna study for TDA data with {N_FOLDS}-fold cross-validation...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

print('='*50)
print('Best parameters for TDA model:')
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"Best accuracy: {study.best_value:.4f}")
print('='*50)

with open(os.path.join(TDA_DIR, "optuna_tda_results.txt"), "w") as f:
    f.write(f"Best parameters for TDA model:\n")
    for key, value in study.best_params.items():
        f.write(f"  {key}: {value}\n")
    f.write(f"Best accuracy: {study.best_value:.4f}\n")

print("[INFO] TDA data optimization complete!")
