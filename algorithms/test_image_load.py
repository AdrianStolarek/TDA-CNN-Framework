import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data_preprocessing_pipelines import load_from_single_folder
from data_preprocessing_pipelines import create_label_mapping

# Define path for debugging
DATASET_PATH = "E:\\ZoryaDyn_Benchmarks\\WAR_TECH\\obshaya_papk"
OUTPUT_DIR = "E:\\ZoryaDyn_Benchmarks\\WAR_TECH\\debug_results"
TARGET_SIZE = (270, 180)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def debug_dataset_folder():
    """
    Debugs the dataset folder structure and attempts to load images.
    """
    print("="*50)
    print("DATASET FOLDER DEBUGGING")
    print("="*50)

    # Check if path exists
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Path does not exist: {DATASET_PATH}")
        return

    # Check if it's a directory
    if not os.path.isdir(DATASET_PATH):
        print(f"[ERROR] Path is not a directory: {DATASET_PATH}")
        return

    # List directory contents
    dir_contents = os.listdir(DATASET_PATH)
    print(f"[INFO] Directory contains {len(dir_contents)} items")

    # Count files and subdirectories
    files = [f for f in dir_contents if os.path.isfile(
        os.path.join(DATASET_PATH, f))]
    subdirs = [d for d in dir_contents if os.path.isdir(
        os.path.join(DATASET_PATH, d))]

    print(f"[INFO] Found {len(files)} files and {len(subdirs)} subdirectories")

    # Check for image files directly in the folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in files if os.path.splitext(f.lower())[
        1] in image_extensions]

    print(f"[INFO] Found {len(image_files)} image files in the main directory")

    if image_files:
        print("[INFO] Sample image files:")
        for i, img_file in enumerate(image_files[:5]):
            print(f"  - {img_file}")

    # Check subdirectories for images
    subdir_images = {}
    for subdir in subdirs:
        subdir_path = os.path.join(DATASET_PATH, subdir)
        subdir_files = os.listdir(subdir_path)
        subdir_image_files = [f for f in subdir_files if os.path.splitext(f.lower())[
            1] in image_extensions]
        subdir_images[subdir] = subdir_image_files
        print(
            f"[INFO] Subdirectory '{subdir}' contains {len(subdir_image_files)} image files")

    # Try to load a few individual image files to check if they're valid
    if image_files:
        print("\n[INFO] Testing individual image loading:")
        for i, img_file in enumerate(image_files[:3]):
            img_path = os.path.join(DATASET_PATH, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(
                        f"[WARNING] Failed to load {img_file} using cv2.imread")
                    try:
                        # Try another method
                        from PIL import Image
                        img_pil = Image.open(img_path)
                        print(
                            f"[INFO] Successfully loaded {img_file} using PIL: {img_pil.size}, {img_pil.mode}")
                    except Exception as e:
                        print(f"[ERROR] PIL also failed: {str(e)}")
                else:
                    print(
                        f"[INFO] Successfully loaded {img_file}: Shape={img.shape}, dtype={img.dtype}")
            except Exception as e:
                print(f"[ERROR] Failed to load {img_file}: {str(e)}")

    # Now try to use the enhanced loading function
    print("\n[INFO] Attempting to load images using enhanced function:")
    try:
        X, y = load_from_single_folder(
            DATASET_PATH,
            target_size=TARGET_SIZE,
            dataset_size_limit=100,  # Limit for testing
            preserve_channels=True
        )

        print(f"[SUCCESS] Loaded {len(X)} images with labels {np.unique(y)}")

        # Create label mapping
        label_to_index, index_to_label, display_names = create_label_mapping(y)
        print(f"[INFO] Label mapping: {label_to_index}")

        # Save dataset diagnostics
        with open(os.path.join(OUTPUT_DIR, "dataset_info.txt"), "w") as f:
            f.write(f"Total images: {len(X)}\n")
            f.write(f"Image shape: {X[0].shape}\n")
            f.write(f"Data type: {X.dtype}\n")
            f.write(f"Unique labels: {np.unique(y)}\n\n")

            # Distribution of labels
            f.write("Label distribution:\n")
            for label in np.unique(y):
                count = np.sum(y == label)
                index = label_to_index[label]
                display_name = display_names[index]
                f.write(f"  - {label} ({display_name}): {count} images\n")

        # Visualize a few images
        plt.figure(figsize=(15, 10))
        for i in range(min(5, len(X))):
            plt.subplot(1, 5, i+1)
            plt.imshow(X[i])
            label_idx = label_to_index[y[i]]
            plt.title(f"{display_names[label_idx]}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"))
        plt.close()

        return X, y, label_to_index, index_to_label, display_names

    except Exception as e:
        print(f"[ERROR] Enhanced loading function failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


if __name__ == "__main__":
    debug_dataset_folder()
