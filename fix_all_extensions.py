import os
from pathlib import Path

# === CONFIGURATION ===
# Set this to your main dataset folder
DATASET_ROOT = r"E:/hackathon_dataset_v2" 

def fix_dataset_extensions():
    root_path = Path(DATASET_ROOT)
    
    if not root_path.exists():
        print(f"Error: Dataset path not found: {root_path}")
        return

    print(f"Scanning {root_path} for files without extensions...")
    
    renamed_count = 0
    
    # Walk through every folder and subfolder
    for file_path in root_path.rglob("*"):
        # Check if it is a file and has NO extension (suffix is empty)
        if file_path.is_file() and file_path.suffix == "":
            
            # Create new name with .wav extension
            new_path = file_path.with_suffix(".wav")
            
            try:
                file_path.rename(new_path)
                # Print every 100th file to show progress without spamming
                if renamed_count % 100 == 0:
                    print(f"Renaming: {file_path.name} -> {new_path.name}")
                renamed_count += 1
            except OSError as e:
                print(f"Error renaming {file_path.name}: {e}")

    print("-" * 50)
    print(f"DONE! Added .wav extension to {renamed_count} files.")
    print("You can now run your mp3 conversion script.")

if __name__ == "__main__":
    fix_dataset_extensions()