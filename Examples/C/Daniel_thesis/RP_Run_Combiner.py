import os
import shutil

# Base directory where your folders reside
base_path = "/Users/danielcampos/Desktop"

# List of folder names
folders = [
    os.path.join(base_path, "CODEoutput4"),  # no offset
    os.path.join(base_path, "CODEoutput5"),
    os.path.join(base_path, "CODEoutput6"),
    os.path.join(base_path, "CODEoutput7"),
    os.path.join(base_path, "CODEoutput8"),
    os.path.join(base_path, "CODEoutput9")
]

# Offsets to add for each folder; first folder gets 0, then new values for subsequent folders.
offsets = [0, 419531, 333174, 228366, 346516, 399524]

# Destination folder for the combined data
combined_folder = os.path.join(base_path, "CombinedTrainingData")
os.makedirs(combined_folder, exist_ok=True)

# Initialize cumulative offset
cumulative_offset = 0

# Process each folder with its corresponding offset
for folder, offset in zip(folders, offsets):
    cumulative_offset += offset
    print(f"Processing folder {folder} with cumulative offset {cumulative_offset}")
    
    # Iterate through all files in the current folder
    for filename in os.listdir(folder):
        # Process only .bin files (adjust if necessary)
        if filename.endswith(".bin"):
            name_part, ext = os.path.splitext(filename)
            # Assume files are named like "bin0.bin", "bin1.bin", etc.
            try:
                original_num = int(name_part.replace("bin", ""))
            except ValueError:
                print(f"Skipping file {filename} (unexpected name format)")
                continue
            
            # Calculate the new bin number using the cumulative offset
            new_num = original_num + cumulative_offset
            new_filename = f"bin{new_num}{ext}"
            
            # Construct full paths for the source and destination
            src_path = os.path.join(folder, filename)
            dst_path = os.path.join(combined_folder, new_filename)
            
            # Copy the file from the source to the destination
            shutil.copy(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
            
print("All files have been copied and renamed into", combined_folder)
