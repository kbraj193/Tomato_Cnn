import os
import shutil
import random

# -----------------------------
# Paths
# -----------------------------
SOURCE_DIR = "Tomato_dataset"       # Original dataset
OUTPUT_DIR = "Tomato_Split"         # New split dataset

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# Create output folders
for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Iterate through each disease class
for class_name in os.listdir(SOURCE_DIR):

    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    # Get image list
    images = [img for img in os.listdir(class_path)
              if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "validation": val_images,
        "test": test_images
    }

    # Copy files
    for split_name, split_images in splits.items():

        dest_class = os.path.join(OUTPUT_DIR, split_name, class_name)
        os.makedirs(dest_class, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(dest_class, img)
            shutil.copy2(src, dst)

    print(f"{class_name}")
    print(f" Train      : {len(train_images)}")
    print(f" Validation : {len(val_images)}")
    print(f" Test       : {len(test_images)}")
    print("-" * 40)

print("\nDataset split completed successfully!")