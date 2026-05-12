
import os
from pathlib import Path

def remove_low_prefix():
    folder_path = Path("/depot/natallah/data/shourya/Reti-Diff-main/results/LLIE_Real_3/visualization/Testset")
    
    for file_path in folder_path.glob("low*.png"):
        new_name = file_path.name[3:]  # Remove "low" prefix
        new_path = file_path.parent / new_name
        file_path.rename(new_path)
        print(f"Renamed: {file_path.name} -> {new_name}")

if __name__ == "__main__":
    remove_low_prefix()