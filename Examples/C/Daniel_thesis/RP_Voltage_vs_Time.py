import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
FOLDER_PATH = "/Users/danielcampos/Desktop/IOCODEoutput1/"  # Your folder path
FILE_PATTERN = "stft_*.bin"

# Global variables for navigation
bin_files = []
current_index = 0
fig, ax = None, None

def load_time_data_from_bin(filepath):
    """
    Loads metadata and time-domain data from a single .bin file.
    """
    try:
        with open(filepath, 'rb') as f:
            # Read metadata header (7 uint32_t values)
            # Meta: [SAMPLES_20MS, nperseg, noverlap, num_subwindows, new_freq_bins, effective_sr, time_offset]
            metadata_uint32 = np.fromfile(f, dtype=np.uint32, count=7)
            if len(metadata_uint32) < 7:
                print(f"Warning: Could not read full metadata from {filepath}")
                return None, None, None, None

            samples_in_chunk = metadata_uint32[0]
            # nperseg = metadata_uint32[1]
            # noverlap = metadata_uint32[2]
            # num_subwindows = metadata_uint32[3]
            # new_freq_bins = metadata_uint32[4]
            effective_sr = float(metadata_uint32[5]) # Convert to float for calculations
            # time_offset_ms = metadata_uint32[6]

            if samples_in_chunk == 0 or effective_sr == 0:
                print(f"Warning: Invalid metadata (samples or SR is zero) in {filepath}")
                return None, None, None, None

            # Read time-domain data (samples_in_chunk float values)
            time_data = np.fromfile(f, dtype=np.float32, count=samples_in_chunk)
            if len(time_data) < samples_in_chunk:
                print(f"Warning: Could not read full time data from {filepath}. Expected {samples_in_chunk}, got {len(time_data)}")
                return None, None, None, None

            filename = os.path.basename(filepath)
            return time_data, effective_sr, samples_in_chunk, filename
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None, None, None, None

def update_plot():
    """
    Updates the plot with the data from the current_index file.
    """
    global current_index, bin_files, ax

    if not bin_files:
        ax.set_title("No .bin files found in the specified directory.")
        ax.plot([], []) # Clear plot
        fig.canvas.draw_idle()
        return

    if current_index < 0:
        current_index = 0
    if current_index >= len(bin_files):
        current_index = len(bin_files) - 1

    filepath = bin_files[current_index]
    voltage_data, effective_sr, num_samples, filename = load_time_data_from_bin(filepath)

    ax.clear() # Clear previous plot

    if voltage_data is not None and effective_sr is not None and num_samples is not None:
        # Create time axis
        time_axis_s = np.arange(num_samples) / effective_sr
        
        ax.plot(time_axis_s * 1000, voltage_data) # Time in ms
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"File: {filename}\nChunk {current_index + 1}/{len(bin_files)} (SR: {effective_sr/1000:.2f} kS/s)")
        ax.grid(True)
    else:
        ax.set_title(f"Could not load data for: {os.path.basename(filepath)}\nChunk {current_index + 1}/{len(bin_files)}")

    fig.canvas.draw_idle()

def on_key(event):
    """
    Handles key press events for navigation.
    """
    global current_index

    if event.key == 'right':
        if current_index < len(bin_files) - 1:
            current_index += 1
            update_plot()
    elif event.key == 'left':
        if current_index > 0:
            current_index -= 1
            update_plot()

def main():
    global bin_files, fig, ax, current_index

    # Find and sort .bin files
    search_path = os.path.join(FOLDER_PATH, FILE_PATTERN)
    bin_files = sorted(glob.glob(search_path))

    if not bin_files:
        print(f"No files matching '{FILE_PATTERN}' found in '{FOLDER_PATH}'.")
        # Still create a plot window to show the message
        # return # Optionally exit if no files

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Initial plot
    current_index = 0
    update_plot()

    if bin_files:
         print(f"Loaded {len(bin_files)} files. Use left/right arrow keys to navigate.")
    else:
        print("Displaying empty plot. Check folder path and file pattern.")


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()