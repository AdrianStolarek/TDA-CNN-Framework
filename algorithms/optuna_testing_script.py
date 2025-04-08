import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from data_preprocessing_pipelines import detect_image_size, load_from_folder, load_processed_batches
from models.fungi_model import FungiModel, FungiDataset, get_transforms, train_model, test_model, plot_results

mlflow.set_tracking_uri("http://mlflow.zorya-dynamics.cloud")
mlflow.set_experiment('Classification-FUNGI-Test-PyTorch')

# Parametry
BATCH_PATH = "E:\\ZoryaDyn_Benchmarks\\FUNGI"
TEMP_DIR = "E:\\ZoryaDyn_Benchmarks\\FUNGI_processed"
PURE_DIR = "E:\\ZoryaDyn_Benchmarks\\FUNGI_processed_pure"
DATASET_SIZE_LIMIT = 30  # 9100 to cały dataset
RANDOM_STATE = 1
TEST_SIZE = 0.2
BATCH_SIZE = 30
NUM_EPOCHS = 150

print("[INFO] Ładowanie danych surowych...")
X_full, y_full = load_from_folder(BATCH_PATH, DATASET_SIZE_LIMIT)

print("Unique labels:", np.unique(y_full))
print("Label counts:", {label: np.sum(y_full == label)
                        for label in np.unique(y_full)})

ORIG_HEIGHT, ORIG_WIDTH = detect_image_size(X_full)
print(f"Oryginalne wymiary obrazów: {ORIG_HEIGHT}x{ORIG_WIDTH}")

INPUT_HEIGHT, INPUT_WIDTH = 500, 500
INPUT_SHAPE_RAW = (INPUT_HEIGHT, INPUT_WIDTH, 3)
INPUT_SHAPE_TDA = (INPUT_HEIGHT, INPUT_WIDTH, 6)
print(f"Wymiary wejściowe modelu: {INPUT_HEIGHT}x{INPUT_WIDTH}")

label_encoder = LabelEncoder()
y_full_encoded = label_encoder.fit_transform(y_full)
num_classes = len(np.unique(y_full_encoded))

print("[INFO] Ładowanie danych TDA...")
X_tda_full, y_tda_full = load_processed_batches(
    output_dir=TEMP_DIR, batch_size=DATASET_SIZE_LIMIT)

print("[INFO] Ładowanie danych TDA pure...")
X_tda_pure_full, y_tda_pure_full = load_processed_batches(
    output_dir=PURE_DIR, batch_size=DATASET_SIZE_LIMIT)

X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
    X_full,
    y_full_encoded,
    test_size=TEST_SIZE,
    stratify=y_full,
    random_state=RANDOM_STATE
)

X_tda_train, X_tda_test, y_tda_train, y_tda_test = train_test_split(
    X_tda_full,
    y_tda_full,
    test_size=TEST_SIZE,
    stratify=y_tda_full,
    random_state=RANDOM_STATE
)

X_tda_pure_train, X_tda_pure_test, y_tda_pure_train, y_tda_pure_test = train_test_split(
    X_tda_pure_full,
    y_tda_pure_full,
    test_size=TEST_SIZE,
    stratify=y_tda_pure_full,
    random_state=RANDOM_STATE
)

print("="*50)
print("[DEBUG] Typy danych przed treningiem:")
print(f"RAW: X_train - {X_raw_train.dtype}, y_train - {y_raw_train.dtype}")
print(f"TDA: X_train - {X_tda_train.dtype}, y_train - {y_tda_train.dtype}")
print(
    f"TDA Pure: X_train - {X_tda_pure_train.dtype}, y_train - {y_tda_pure_train.dtype}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")


def run_experiment(data_type, X_train, y_train, X_test, y_test, in_channels):
    with mlflow.start_run(run_name=f"fungi_{data_type}_model"):

        train_transform, test_transform = get_transforms(data_type)

        temp_model = FungiModel(in_channels=in_channels,
                                num_classes=num_classes)
        input_size = temp_model.get_input_size()
        print(
            f"[INFO] Model {data_type} oczekuje wymiarów wejściowych: {input_size}")

        is_tda = data_type != 'raw'
        train_dataset = FungiDataset(
            X_train, y_train, transform=train_transform, is_tda=is_tda)
        test_dataset = FungiDataset(
            X_test, y_test, transform=test_transform, is_tda=is_tda)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        model = FungiModel(in_channels=in_channels,
                           num_classes=num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.0001, weight_decay=0.1)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[7, 14, 21, 28, 35], gamma=0.1)

        mlflow.log_param("data_type", data_type)
        mlflow.log_param("in_channels", in_channels)
        mlflow.log_param("optimizer", "adamw")
        mlflow.log_param("learning_rate", 0.0001)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)

        patience = 10

        logs = train_model(
            model, train_loader, test_loader, criterion, optimizer, scheduler,
            device, num_epochs=NUM_EPOCHS, patience=patience
        )

        test_loss, test_acc = test_model(model, test_loader, criterion, device)

        mlflow.log_metric(f"{data_type}-loss", test_loss)
        mlflow.log_metric(f"{data_type}-accuracy", test_acc)

        for epoch, (train_loss, train_acc, val_loss, val_acc) in enumerate(zip(
            logs['train_loss'], logs['train_acc'], logs['val_loss'], logs['val_acc']
        )):
            mlflow.log_metric(f"{data_type}-train-loss",
                              train_loss, step=epoch)
            mlflow.log_metric(f"{data_type}-train-accuracy",
                              train_acc, step=epoch)
            mlflow.log_metric(f"{data_type}-val-loss", val_loss, step=epoch)
            mlflow.log_metric(f"{data_type}-val-accuracy", val_acc, step=epoch)

        plot_path = plot_results(logs, title_suffix=f" - {data_type.upper()}")
        mlflow.log_artifact(plot_path)

        return test_loss, test_acc, logs


print("="*50)
print("[INFO] Uruchamianie eksperymentu na danych surowych...")
raw_results = run_experiment(
    "raw", X_raw_train, y_raw_train, X_raw_test, y_raw_test, in_channels=3
)

print("="*50)
print("[INFO] Uruchamianie eksperymentu na danych TDA hybrydowych...")
tda_results = run_experiment(
    "tda_hybrid", X_tda_train, y_tda_train, X_tda_test, y_tda_test, in_channels=6
)

print("="*50)
print("[INFO] Uruchamianie eksperymentu na danych TDA czystych...")
tda_pure_results = run_experiment(
    "tda_pure", X_tda_pure_train, y_tda_pure_train, X_tda_pure_test, y_tda_pure_test, in_channels=3
)
print("="*50)
print("[WYNIKI] Podsumowanie eksperymentów:")
print(
    f"RAW - Test Loss: {raw_results[0]:.4f}, Test Accuracy: {raw_results[1]:.4f}")
print(
    f"TDA Hybrid - Test Loss: {tda_results[0]:.4f}, Test Accuracy: {tda_results[1]:.4f}")
print(
    f"TDA Pure - Test Loss: {tda_pure_results[0]:.4f}, Test Accuracy: {tda_pure_results[1]:.4f}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(raw_results[2]['train_loss'], label='RAW')
plt.plot(tda_results[2]['train_loss'], label='TDA Hybrid')
plt.plot(tda_pure_results[2]['train_loss'], label='TDA Pure')
plt.title('Porównanie straty treningowej', fontsize=15)
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(raw_results[2]['val_loss'], label='RAW')
plt.plot(tda_results[2]['val_loss'], label='TDA Hybrid')
plt.plot(tda_pure_results[2]['val_loss'], label='TDA Pure')
plt.title('Porównanie straty walidacyjnej', fontsize=15)
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(raw_results[2]['train_acc'], label='RAW')
plt.plot(tda_results[2]['train_acc'], label='TDA Hybrid')
plt.plot(tda_pure_results[2]['train_acc'], label='TDA Pure')
plt.title('Porównanie dokładności treningowej', fontsize=15)
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(raw_results[2]['val_acc'], label='RAW')
plt.plot(tda_results[2]['val_acc'], label='TDA Hybrid')
plt.plot(tda_pure_results[2]['val_acc'], label='TDA Pure')
plt.title('Porównanie dokładności walidacyjnej', fontsize=15)
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.tight_layout()
plt.savefig('wyniki_porownawcze.png')
mlflow.log_artifact('wyniki_porownawcze.png')
plt.show()
