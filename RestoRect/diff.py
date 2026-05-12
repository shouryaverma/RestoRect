# import os

# def find_missing_files(folder1, folder2):
#     # Get file lists
#     files1 = set(os.listdir(folder1))
#     files2 = set(os.listdir(folder2))
    
#     # Files missing in folder2
#     missing_in_folder2 = files1 - files2
    
#     # Files missing in folder1
#     missing_in_folder1 = files2 - files1
    
#     return missing_in_folder2, missing_in_folder1


# # Example usage
# folder1 = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LSUI/gt_test"
# folder2 = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LSUI/input_test"

# missing_in_folder2, missing_in_folder1 = find_missing_files(folder1, folder2)

# print("Files missing in folder1:", missing_in_folder1)
# print("Files missing in folder2:", missing_in_folder2)


# import os
# import shutil
# import random

# # Paths
# input_folder = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LSUI/input"
# gt_folder = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LSUI/GT"

# input_train = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LSUI/input_train"
# input_test = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LSUI/input_test"
# gt_train = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LSUI/gt_train"
# gt_test = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LSUI/gt_test"

# # Create output dirs if not exist
# for d in [input_train, input_test, gt_train, gt_test]:
#     os.makedirs(d, exist_ok=True)

# # Get all files (assuming matched names between input and gt)
# input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])
# gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".jpg")])

# # Sanity check
# if len(input_files) != len(gt_files):
#     print("⚠️ Warning: input and gt folders have different number of files!")

# # Fix seed for reproducibility
# random.seed(42)

# # Shuffle for randomness (but deterministic now)
# combined = list(zip(input_files, gt_files))
# random.shuffle(combined)

# # Split 80/20
# split_idx = int(0.8 * len(combined))
# train_set = combined[:split_idx]
# test_set = combined[split_idx:]

# # Copy files to new folders
# for inp, gt in train_set:
#     shutil.copy(os.path.join(input_folder, inp), os.path.join(input_train, inp))
#     shutil.copy(os.path.join(gt_folder, gt), os.path.join(gt_train, gt))

# for inp, gt in test_set:
#     shutil.copy(os.path.join(input_folder, inp), os.path.join(input_test, inp))
#     shutil.copy(os.path.join(gt_folder, gt), os.path.join(gt_test, gt))

# print(f"✅ Done! {len(train_set)} training pairs, {len(test_set)} testing pairs.")

import os
import shutil

def unfold_folders(main_dir, output_dir):
    """
    Flattens a directory of folders with images into a single folder.
    Renames images as foldername_imagename.png.

    Args:
        main_dir (str): Path to the main directory with subfolders.
        output_dir (str): Path to save unfolded images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Loop over subfolders
    for folder in os.listdir(main_dir):
        folder_path = os.path.join(main_dir, folder)

        if os.path.isdir(folder_path):
            # Loop over files in subfolder
            for file in os.listdir(folder_path):
                if file.endswith(".png"):
                    src = os.path.join(folder_path, file)
                    new_name = f"{folder}_{file}"
                    dst = os.path.join(output_dir, new_name)
                    
                    shutil.copy2(src, dst)  # use copy2 to preserve metadata

    print(f"Unfolded images are saved in: {output_dir}")


# Example usage
main_dir = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LOL_blur/test/low_blur"
output_dir = "/depot/natallah/data/shourya/Reti-Diff-main/datasets/LOL_blur/test_input"
unfold_folders(main_dir, output_dir)