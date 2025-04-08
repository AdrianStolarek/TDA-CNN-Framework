from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.util import random_noise
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import itertools
from pathlib import Path
import numpy as np
from collections import defaultdict


def load_cifar10_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    # Reshape the data to (num_samples, 3, 32, 32) and then to (num_samples, 32, 32, 3) for RGB images
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, labels


def load_from_folder(dataset_path, dataset_size_limit=None, preserve_channels=True):
    """
    Wczytuje obrazy z folderu, zachowując ich oryginalną liczbę kanałów.

    Args:
        dataset_path: Ścieżka do katalogu z danymi
        dataset_size_limit: Maksymalna liczba obrazów do wczytania
        preserve_channels: Jeśli True, zachowuje oryginalną liczbę kanałów obrazu
                          Jeśli False, konwertuje wszystkie obrazy do RGB (3 kanały)

    Returns:
        Tuple (obrazy, etykiety) jako tablice NumPy
    """
    class_folders = sorted([f for f in os.listdir(
        dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    class_iterators = []

    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        images = []
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)

                # Wczytaj obraz z zachowaniem oryginalnej liczby kanałów
                if preserve_channels:
                    # Wczytaj i sprawdź liczbę kanałów
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue

                    # Sprawdź wymiary i liczbę kanałów
                    if len(img.shape) == 2:  # Obraz w skali szarości
                        # Dodaj wymiar kanału (H, W) -> (H, W, 1)
                        img = img[:, :, np.newaxis]
                    elif img.shape[2] == 3:  # Obraz kolorowy BGR
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Inne przypadki (np. RGBA) można obsłużyć w miarę potrzeb
                else:
                    # Dotychczasowe zachowanie - konwersja do RGB
                    img = cv2.imread(img_path)
                    # Dla pewności, że obrazy są w RGB
                    if img is None:
                        continue
                    if len(img.shape) == 2:  # Skala szarości
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:  # BGR
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                images.append((img, class_folder))

        if images:  # Sprawdź, czy znaleziono jakiekolwiek obrazy
            class_iterators.append(itertools.cycle(images))

    if not class_iterators:
        raise ValueError(f"Nie znaleziono obrazów w katalogu: {dataset_path}")

    images = []
    labels = []
    total_samples = 0
    samples_per_class = {}

    # Inicjalizacja liczników dla każdej klasy
    for class_folder in class_folders:
        samples_per_class[class_folder] = 0

    # Jeśli dataset_size_limit jest określony, oblicz limit na klasę
    if dataset_size_limit:
        limit_per_class = dataset_size_limit // len(class_folders)
    else:
        limit_per_class = float('inf')

    # Zbierz równą liczbę próbek z każdej klasy
    while True:
        for i, it in enumerate(class_iterators):
            class_folder = class_folders[i]

            # Jeśli osiągnięto limit dla tej klasy, przejdź dalej
            if samples_per_class[class_folder] >= limit_per_class:
                continue

            try:
                img, label = next(it)
                images.append(img)
                labels.append(label)
                samples_per_class[label] += 1
                total_samples += 1

                # Jeśli osiągnięto limit dla wszystkich klas, zakończ
                if all(samples_per_class[cls] >= limit_per_class for cls in class_folders):
                    break
            except StopIteration:
                break

        # Warunek zakończenia pętli
        if dataset_size_limit and total_samples >= dataset_size_limit:
            break
        if all(samples_per_class[cls] >= limit_per_class for cls in class_folders):
            break
        # Jeśli wszystkie iteratory są wyczerpane, zakończ
        if all(samples_per_class[cls] == 0 for cls in class_folders):
            break

    # Wyświetl informacje o liczbie próbek na klasę
    print("[INFO] Liczba próbek na klasę:")
    for cls in class_folders:
        print(f"  - {cls}: {samples_per_class[cls]}")

    if not images:
        raise ValueError(
            "Nie wczytano żadnych obrazów. Sprawdź ścieżkę i format plików.")

    return np.array(images), np.array(labels)


def load_from_csv(csv_path, images_dir, test_size=0.2, random_state=42, preserve_channels=True):
    """
    Wczytuje obrazy na podstawie pliku CSV, zachowując ich oryginalną liczbę kanałów.

    Args:
        csv_path: Ścieżka do pliku CSV z danymi
        images_dir: Ścieżka do katalogu z obrazami
        test_size: Rozmiar zbioru testowego (0-1)
        random_state: Ziarno losowości dla podziału
        preserve_channels: Jeśli True, zachowuje oryginalną liczbę kanałów obrazu

    Returns:
        Tuple (X_train, X_test, y_train, y_test) jako tablice NumPy
    """
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, row['image'])
        if os.path.exists(img_path):
            if preserve_channels:
                # Wczytaj z zachowaniem oryginalnej liczby kanałów
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                # Sprawdź wymiary i liczbę kanałów
                if len(img.shape) == 2:  # Obraz w skali szarości
                    img = img[:, :, np.newaxis]  # Dodaj wymiar kanału
                elif img.shape[2] == 3:  # Obraz kolorowy BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # Konwersja do RGB (dotychczasowe zachowanie)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(row['label'])

    if not images:
        raise ValueError(
            f"Nie znaleziono obrazów dla ścieżek w CSV: {csv_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def save_in_batches(pipeline, X, y, batch_size=32, output_dir="tds_processed"):
    """
    Przetwarza dane przez pipeline i zapisuje je w partiach.

    Args:
        pipeline: Pipeline do przetwarzania danych
        X: Dane wejściowe (obrazy)
        y: Etykiety
        batch_size: Rozmiar partii
        output_dir: Katalog wyjściowy
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Sprawdź typ danych wejściowych
    print(f"[DEBUG] Typy danych przed przetwarzaniem:")
    print(f"X - typ: {X.dtype}, kształt: {X.shape}")
    print(f"y - typ: {y.dtype}, kształt: {y.shape}")

    # Sprawdź, czy dane obrazu są w zakresie 0-255 czy 0-1
    if X.dtype == np.uint8:
        print("[INFO] Obrazy są w formacie uint8 (0-255). Konwersja do float32 (0-1)...")
        X = X.astype('float32') / 255.0

    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        try:
            # Przetwórz dane przez pipeline
            processed_batch = pipeline.fit_transform(batch_X)

            # Sprawdź przetworzone dane
            print(f"[DEBUG] Przetworzona partia {i//batch_size}:")
            print(
                f"Kształt: {processed_batch.shape}, Typ: {processed_batch.dtype}")
            print(f"Min: {processed_batch.min()}, Max: {processed_batch.max()}")

            # Sprawdź, czy są wartości NaN lub inf
            nan_count = np.isnan(processed_batch).sum()
            inf_count = np.isinf(processed_batch).sum()
            if nan_count > 0 or inf_count > 0:
                print(
                    f"[WARNING] Znaleziono {nan_count} wartości NaN i {inf_count} wartości nieskończonych")
                # Zastąp NaN i inf zerami
                processed_batch = np.nan_to_num(processed_batch)
                print("[INFO] Zastąpiono wartości NaN i inf zerami")

            # Zapisz przetworzone dane
            np.savez(os.path.join(output_dir, f"batch_{i//batch_size}.npz"),
                     X=processed_batch, y=batch_y)

            print(
                f"[INFO] Zapisano partię {i//batch_size}: {len(batch_X)} próbek")

        except Exception as e:
            print(
                f"[ERROR] Błąd podczas przetwarzania partii {i//batch_size}: {str(e)}")
            # Możesz dodać więcej informacji debugujących
            print(f"Kształt batch_X: {batch_X.shape}, Typ: {batch_X.dtype}")
            print(f"Unikalne wartości y: {np.unique(batch_y)}")
            raise  # Podnieś błąd dalej, aby zatrzymać przetwarzanie


def load_processed_batches(output_dir="tds_processed", batch_size=None):
    """
    Wczytuje przetworzone dane z partii.

    Args:
        output_dir: Katalog z przetworzonymi danymi
        batch_size: Opcjonalne ograniczenie liczby próbek

    Returns:
        Tuple (X, y) jako tablice NumPy
    """
    all_X = []
    all_y = []

    # Sprawdź, czy katalog istnieje
    if not os.path.exists(output_dir):
        raise ValueError(f"Katalog {output_dir} nie istnieje")

    # Znajdź pliki .npz w katalogu
    batch_files = sorted(Path(output_dir).glob("*.npz"))
    if not batch_files:
        raise ValueError(f"Nie znaleziono plików .npz w katalogu {output_dir}")

    print(
        f"[INFO] Znaleziono {len(batch_files)} plików batch w katalogu {output_dir}")

    for batch_file in batch_files:
        try:
            with np.load(batch_file) as data:
                all_X.append(data['X'])
                all_y.append(data['y'])
                print(
                    f"[INFO] Wczytano {batch_file.name}: X.shape={data['X'].shape}, y.shape={data['y'].shape}")
        except Exception as e:
            print(
                f"[ERROR] Nie udało się wczytać pliku {batch_file}: {str(e)}")
            continue

    if not all_X:
        raise ValueError("Nie udało się wczytać żadnych danych z plików batch")

    # Połącz wszystkie partie
    X_combined = np.concatenate(all_X)
    y_combined = np.concatenate(all_y)

    print(
        f"[INFO] Całkowity kształt wczytanych danych: X={X_combined.shape}, y={y_combined.shape}")

    # Ogranicz liczbę próbek, jeśli określono batch_size
    if batch_size is not None:
        if batch_size < len(X_combined):
            X_combined = X_combined[:batch_size]
            y_combined = y_combined[:batch_size]
            print(f"[INFO] Ograniczono do {batch_size} próbek")

    return X_combined, y_combined


def detect_image_size(images):
    """
    Wykrywa rozmiar obrazu na podstawie pierwszego obrazu w zbiorze.

    Args:
        images: Tablica obrazów

    Returns:
        Tuple (wysokość, szerokość)
    """
    if len(images) == 0:
        raise ValueError("Nie znaleziono obrazów w zbiorze danych")

    # Wyświetl informacje o zbiorze danych
    print(f"[INFO] Wykryto rozmiar obrazu: {images[0].shape}")
    print(
        f"[INFO] Liczba kanałów: {images[0].shape[2] if len(images[0].shape) > 2 else 1}")

    return images[0].shape[0], images[0].shape[1]


def check_and_print_dataset_info(X, y):
    """
    Sprawdza i wyświetla informacje o zbiorze danych.

    Args:
        X: Dane obrazów
        y: Etykiety
    """
    print("\n" + "="*50)
    print("INFORMACJE O ZBIORZE DANYCH")
    print("="*50)

    print(f"Kształt X: {X.shape}")
    print(f"Typ danych X: {X.dtype}")
    print(f"Zakres wartości X: min={X.min()}, max={X.max()}")

    print(f"Kształt y: {y.shape}")
    print(f"Typ danych y: {y.dtype}")
    print(f"Unikalne etykiety: {np.unique(y)}")

    # Sprawdź, czy są brakujące wartości
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(
            f"[WARNING] Znaleziono {nan_count} wartości NaN i {inf_count} wartości nieskończonych w danych")
    else:
        print("[INFO] Brak wartości NaN i inf w danych")

    # Wyświetl histogram wartości pikseli
    pixel_values = X.flatten()
    plt.figure(figsize=(10, 4))
    plt.hist(pixel_values, bins=50)
    plt.title("Histogram wartości pikseli")
    plt.xlabel("Wartość piksela")
    plt.ylabel("Liczba wystąpień")
    plt.show()

    print("="*50)


def load_from_single_folder(dataset_path, target_size=(270, 180), dataset_size_limit=None, preserve_channels=True):
    """
    Enhanced function to load images from a single folder with better error handling and diagnostics.

    Args:
        dataset_path: Path to the directory with all images
        target_size: Tuple (width, height) for image resizing
        dataset_size_limit: Maximum number of images to load (optional)
        preserve_channels: If True, preserves original channel count

    Returns:
        Tuple (images, labels) as NumPy arrays
    """
    import os
    import cv2
    import numpy as np
    from collections import defaultdict
    import random

    print(f"[INFO] Attempting to load images from: {dataset_path}")
    print(f"[INFO] Absolute path: {os.path.abspath(dataset_path)}")

    # Verify directory exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"Directory does not exist: {dataset_path}")

    # Check if directory is empty
    dir_contents = os.listdir(dataset_path)
    if not dir_contents:
        raise ValueError(f"Directory is empty: {dataset_path}")

    print(f"[INFO] Directory contains {len(dir_contents)} items")

    # Get file extensions in the directory
    extensions = set()
    for item in dir_contents:
        if os.path.isfile(os.path.join(dataset_path, item)):
            _, ext = os.path.splitext(item.lower())
            extensions.add(ext)

    print(f"[INFO] File extensions found: {extensions}")

    # Get all image files from the directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = []

    for item in dir_contents:
        item_path = os.path.join(dataset_path, item)
        if os.path.isfile(item_path):
            _, ext = os.path.splitext(item.lower())
            if ext in image_extensions:
                image_files.append(item_path)

    print(f"[INFO] Found {len(image_files)} image files")

    if not image_files:
        print(
            "[ERROR] No image files found with extensions: {', '.join(image_extensions)}")
        print("[DEBUG] First 10 items in directory:")
        for i, item in enumerate(dir_contents[:10]):
            print(f"  - {item}")
        raise ValueError(f"No image files found in {dataset_path}")

    # Sample a few image files to check
    sample_files = random.sample(image_files, min(5, len(image_files)))
    print("[INFO] Sample image files:")
    for file in sample_files:
        print(f"  - {os.path.basename(file)}")

    # Extract class prefixes from filenames
    class_prefixes = set()
    for img_path in image_files:
        filename = os.path.basename(img_path)
        # Extract first word before space or underscore or parenthesis
        prefix = filename.split(' ')[0].split('_')[0].split('(')[0]
        class_prefixes.add(prefix)

    print(f"[INFO] Detected class prefixes: {class_prefixes}")

    # Extract labels from filenames and organize by label
    label_to_files = defaultdict(list)

    for img_path in image_files:
        filename = os.path.basename(img_path)
        # Extract first word before space or parenthesis as label
        label = filename.split(' ')[0].split('(')[0]
        label_to_files[label].append(img_path)

    print(
        f"[INFO] Found {len(label_to_files)} unique labels: {list(label_to_files.keys())}")
    print("[INFO] Files per label:")
    for label, files in label_to_files.items():
        print(f"  - {label}: {len(files)} files")

    # Calculate how many images to take per class for balanced dataset
    if dataset_size_limit:
        samples_per_class = dataset_size_limit // len(label_to_files)
    else:
        samples_per_class = max(len(files)
                                for files in label_to_files.values())

    print(f"[INFO] Taking up to {samples_per_class} samples per class")

    # Load and process images
    images = []
    labels = []
    error_count = 0
    successful_count = 0

    for label, files in label_to_files.items():
        print(f"[INFO] Processing label: {label}")
        for i, img_path in enumerate(files[:samples_per_class]):
            try:
                if preserve_channels:
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        print(f"[WARNING] Failed to load image: {img_path}")
                        error_count += 1
                        continue

                    # Resize image to target size
                    img = cv2.resize(img, target_size)

                    # Handle channels
                    if len(img.shape) == 2:  # Grayscale
                        img = img[:, :, np.newaxis]
                    elif img.shape[2] == 3:  # BGR to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"[WARNING] Failed to load image: {img_path}")
                        error_count += 1
                        continue

                    # Resize image to target size
                    img = cv2.resize(img, target_size)

                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                images.append(img)
                labels.append(label)
                successful_count += 1

            except Exception as e:
                print(f"[ERROR] Failed to process image {img_path}: {str(e)}")
                error_count += 1

    print(
        f"[INFO] Successfully loaded {successful_count} images, {error_count} errors")

    if not images:
        raise ValueError(
            "No images were loaded. Check file paths and formats.")

    # Shuffle the data to ensure randomness
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)

    return np.array(images), np.array(labels)


def add_gaussian_noise_to_subset(X, proportion=0.2, mean=0.7, sigma=0.3):
    """
    Add Gaussian noise to a random subset of images.

    Args:
        X: Array of images
        proportion: Proportion of images to add noise to (0-1)
        mean: Mean of Gaussian noise
        sigma: Standard deviation of Gaussian noise

    Returns:
        Array of images with noise added to subset
    """
    import numpy as np
    from skimage.util import random_noise

    # Create a copy to avoid modifying the original
    X_noisy = X.copy()

    # Determine how many images to add noise to
    n_images = X.shape[0]
    n_noisy = int(n_images * proportion)

    # Randomly select indices for noisy images
    noisy_indices = np.random.choice(n_images, n_noisy, replace=False)

    print(
        f"[INFO] Adding Gaussian noise to {n_noisy} images ({proportion*100:.1f}% of the dataset)")

    # Add noise to selected images
    for idx in noisy_indices:
        # Normalize image to 0-1 range if needed
        img = X_noisy[idx].astype('float32')
        if img.max() > 1.0:
            img = img / 255.0

        # Add Gaussian noise
        img_noisy = random_noise(img, mode='gaussian',
                                 mean=mean, var=sigma**2, clip=True)

        # Convert back to original range if needed
        if X.max() > 1.0:
            img_noisy = (img_noisy * 255).astype(X.dtype)

        X_noisy[idx] = img_noisy

    return X_noisy


def create_label_mapping(labels):
    """
    Creates a mapping between label strings and numeric indices.
    Also provides display names for each class.

    Args:
        labels: Array of string labels

    Returns:
        Tuple containing (label_to_index, index_to_label, display_names)
    """
    unique_labels = sorted(set(labels))

    # Create mappings
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for idx, label in enumerate(unique_labels)}

    # Create user-friendly display names (can be customized)
    display_names = {}
    for label in unique_labels:
        if label.lower() == 'btr':
            display_names[label_to_index[label]] = "Armored Personnel Carrier"
        elif label.lower() == 'pehota':
            display_names[label_to_index[label]] = "Infantry"
        elif label.lower() == 'rszo':
            display_names[label_to_index[label]] = "Rocket Artillery"
        elif label.lower() == 'tank':
            display_names[label_to_index[label]] = "Tank"
        else:
            display_names[label_to_index[label]] = label.capitalize()

    return label_to_index, index_to_label, display_names
