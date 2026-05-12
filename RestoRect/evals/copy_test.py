import os
import shutil
from pathlib import Path

def copy_files_excluding_removed(source_dir, dest_dir, removed_samples_file):
    """Copy files from source to destination, excluding files listed in removed_samples.txt"""
    
    # Read removed samples and extract filenames
    excluded_files = set()
    if os.path.exists(removed_samples_file):
        with open(removed_samples_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    filename = os.path.basename(line)
                    excluded_files.add(filename)
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy files not in excluded list
    source_path = Path(source_dir)
    if source_path.exists():
        for file_path in source_path.iterdir():
            if file_path.is_file() and file_path.name not in excluded_files:
                shutil.copy2(file_path, dest_dir)

copy_files_excluding_removed(
    "/depot/natallah/data/shourya/Reti-Diff-main/results/SSID/visualization/SSID_ValSet",
    "/depot/natallah/data/shourya/Reti-Diff-main/results/SSID/visualization/SSID_ValSet_new",
    "/depot/natallah/data/shourya/Reti-Diff-main/Reti-Diff/evals/removed_samples.txt"
)