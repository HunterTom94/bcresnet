import os
import shutil
import random


def split_files(input_folder, output_train, output_test, output_val, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    """
    Splits files from the input folder into three output folders (Train, Test, Validation).
    """
    # Ensure the sum of ratios is 1
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Ensure output directories exist
    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_test, exist_ok=True)
    os.makedirs(output_val, exist_ok=True)

    # Get all files from the input folder
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Shuffle files to ensure randomness
    random.shuffle(files)

    # Calculate split points
    total_files = len(files)
    train_split = int(total_files * train_ratio)
    test_split = int(total_files * (train_ratio + test_ratio))

    # Split files
    train_files = files[:train_split]
    test_files = files[train_split:test_split]
    val_files = files[test_split:]

    # Copy files to respective folders
    for file in train_files:
        shutil.copy(os.path.join(input_folder, file), os.path.join(output_train, file))

    for file in test_files:
        shutil.copy(os.path.join(input_folder, file), os.path.join(output_test, file))

    for file in val_files:
        shutil.copy(os.path.join(input_folder, file), os.path.join(output_val, file))

    print(f"Split {total_files} files into:")
    print(f"  Train: {len(train_files)} ({train_ratio * 100}%)")
    print(f"  Test: {len(test_files)} ({test_ratio * 100}%)")
    print(f"  Validation: {len(val_files)} ({val_ratio * 100}%)")
    print(f"Train files saved in: {output_train}")
    print(f"Test files saved in: {output_test}")
    print(f"Validation files saved in: {output_val}")


# Example usage
input_folder = './beats'  # Replace with the path to your input folder
output_train = './train'  # Replace with the desired path for the Train folder
output_test = './test'  # Replace with the desired path for the Test folder
output_val = './val'  # Replace with the desired path for the Validation folder

# Call the function with split ratios (default: 70% Train, 20% Test, 10% Validation)
split_files(input_folder, output_train, output_test, output_val, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1)
