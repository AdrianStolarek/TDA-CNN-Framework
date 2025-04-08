import numpy as np
import os
import mlflow
import tensorflow as tf
from keras import applications
from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from datetime import datetime
from keras import mixed_precision
import matplotlib.pyplot as plt
from models.raw_models import MiniVGG_Raw_Model
from data_preprocessing_pipelines import detect_image_size


mlflow.set_tracking_uri("http://mlflow.zorya-dynamics.cloud")
mixed_precision.set_global_policy('mixed_float16')

RAW_DIR = "E:\\ZoryaDyn_Benchmarks\\CIFAR100\\processed_RAW"
RANDOM_STATE = 42
TRAINING_SAMPLES = 1000  # Odpal 500, 1000, 10000, 30000
EXPERIMENT_NAME = f"RAW_cifar100_resnet50_{TRAINING_SAMPLES}-ADRIAN"


def build_resnet50_model(input_shape, num_classes):

    inputs = layers.Input(shape=input_shape)

    base_model = applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        input_shape=input_shape
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_and_evaluate():

    print(f"[INFO] Loading raw CIFAR-100 dataset from {RAW_DIR}...")

    train_data = np.load(os.path.join(RAW_DIR, "raw_train_dataset.npz"))
    X_train_full = train_data['X']
    y_train_full = train_data['y']

    INPUT_HEIGHT, INPUT_WIDTH = detect_image_size(X_train_full)
    INPUT_SHAPE_RAW = (INPUT_HEIGHT, INPUT_WIDTH, 3)

    if TRAINING_SAMPLES is not None and TRAINING_SAMPLES < len(X_train_full):
        np.random.seed(RANDOM_STATE)
        indices = np.random.choice(
            len(X_train_full), TRAINING_SAMPLES, replace=False)
        X_train = X_train_full[indices]
        y_train = y_train_full[indices]
        print(
            f"[INFO] Using {TRAINING_SAMPLES} out of {len(X_train_full)} available training samples")
    else:
        X_train = X_train_full
        y_train = y_train_full
        print(f"[INFO] Using all {len(X_train)} available training samples")

    test_data = np.load(os.path.join(RAW_DIR, "raw_test_dataset.npz"))
    X_test = test_data['X']
    y_test = test_data['y']

    # plt.figure(figsize=(8, 6))
    # img = X_test[200, :, :, :]
    # imgplot = plt.imshow(img)
    # .title(y_test[200])
    # plt.show()

    print("=" * 50)
    print("Raw Dataset Information:")
    print(f"Training dataset shape: {X_train.shape}")
    print(f"Testing dataset shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print("=" * 50)

    INPUT_SHAPE = X_train[0].shape
    NUM_CLASSES = len(np.unique(y_train))

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"raw_MobileNetV2_{len(X_train)}_samples"):
        mlflow.log_param("model_type", "MobileNetV2")
        mlflow.log_param("data_type", "raw")
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("testing_samples", len(X_test))
        mlflow.log_param("input_shape", INPUT_SHAPE)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("random_state", RANDOM_STATE)

        model = MiniVGG_Raw_Model(input_shape=INPUT_SHAPE_RAW).model

        callbacks_fin = [
            callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                        patience=10, min_lr=1e-6)
        ]

        print(
            f"[INFO] Training MobileNetV2 model on raw data with {len(X_train)} samples...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,
            batch_size=64,
            callbacks=callbacks_fin,
            verbose=1
        )

        print("[INFO] Evaluating model on test data...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)

        for epoch in range(len(history.history['accuracy'])):
            mlflow.log_metric("train_accuracy",
                              history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric(
                "train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric(
                "val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric(
                "val_loss", history.history['val_loss'][epoch], step=epoch)

        print("=" * 50)
        print(
            f"MobileNetV2 on raw CIFAR-100 (training samples: {len(X_train)}):")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Number of training epochs: {len(history.history['accuracy'])}")
        print("=" * 50)

        with open(os.path.join(RAW_DIR, f"MobileNetV2_raw_results_{len(X_train)}_samples.txt"), "w") as f:
            f.write(
                f"MobileNetV2 on raw CIFAR-100 (training samples: {len(X_train)}):\n")
            f.write(f"Test accuracy: {test_accuracy:.4f}\n")
            f.write(f"Test loss: {test_loss:.4f}\n")
            f.write(
                f"Number of training epochs: {len(history.history['accuracy'])}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            TRAINING_SAMPLES = int(sys.argv[1])
            print(
                f"[INFO] Setting training samples to {TRAINING_SAMPLES} from command line argument")
        except ValueError:
            print(
                f"[ERROR] Invalid training samples value: {sys.argv[1]}. Using default: {TRAINING_SAMPLES}")

    train_and_evaluate()
