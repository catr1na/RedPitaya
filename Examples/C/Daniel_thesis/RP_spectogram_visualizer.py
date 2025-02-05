import numpy as np
import matplotlib.pyplot as plt

def load_spectrogram_file(filename):
    """
    Reads a binary file saved by your C code.
    File format:
      - 7 unsigned 32-bit integers (metadata):
          meta[0] = SAMPLES_20MS (number of time-domain samples)
          meta[1] = nperseg (FFT sub-window size)
          meta[2] = noverlap (overlap count)
          meta[3] = num_subwindows (number of STFT frames, e.g. 38)
          meta[4] = fft_out_size (number of frequency bins, e.g. 513)
          meta[5] = effective sample rate
          meta[6] = time offset (ms)
      - SAMPLES_20MS floats (raw time-domain data)
      - (num_subwindows * fft_out_size) floats (STFT power array in dB, row-major)
    """
    # Read metadata: 7 uint32 values (each 4 bytes)
    meta = np.fromfile(filename, dtype=np.uint32, count=7)
    if meta.size < 7:
        raise ValueError("File too short: cannot read metadata.")
    
    samples_20ms   = meta[0]
    nperseg        = meta[1]
    noverlap       = meta[2]
    num_subwindows = meta[3]
    fft_out_size   = meta[4]
    effective_sr   = meta[5]
    time_offset    = meta[6]
    
    # Calculate byte offsets:
    meta_bytes      = 7 * 4  # 7 uint32's = 28 bytes
    raw_time_bytes  = samples_20ms * 4  # each float is 4 bytes
    stft_offset     = meta_bytes + raw_time_bytes
    
    # Read raw time-domain data (if needed)
    raw_time = np.fromfile(filename, dtype=np.float32, count=samples_20ms, offset=meta_bytes)
    
    # Read STFT data (spectrogram)
    stft_data = np.fromfile(filename, dtype=np.float32, count=num_subwindows * fft_out_size, offset=stft_offset)
    stft_data = stft_data.reshape((num_subwindows, fft_out_size))
    
    return {
        'samples_20ms': samples_20ms,
        'nperseg': nperseg,
        'noverlap': noverlap,
        'num_subwindows': num_subwindows,
        'fft_out_size': fft_out_size,
        'effective_sr': effective_sr,
        'time_offset': time_offset,
        'raw_time': raw_time,
        'stft': stft_data
    }

def plot_spectrogram(spectrogram, nperseg, effective_sr):
    """
    Plot the full 2D spectrogram.
    - The x-axis corresponds to time (each sub-window is nperseg/effective_sr seconds)
    - The y-axis corresponds to frequency (from 0 to effective_sr/2)
    """
    num_subwindows = spectrogram.shape[0]
    fft_out_size   = spectrogram.shape[1]
    dt = nperseg / effective_sr  # time per subwindow in seconds
    time_axis = np.arange(num_subwindows) * dt * 1000  # convert to milliseconds
    freq_axis = np.linspace(0, effective_sr/2, fft_out_size)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower',
               extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram (20 ms chunk)")
    plt.colorbar(label="Magnitude (dB)")
    plt.show()

def plot_spectrogram(spectrogram, nperseg, effective_sr):
    """
    Plot the full 2D spectrogram with a logarithmic frequency scale.
    - The x-axis corresponds to time (each sub-window is nperseg/effective_sr seconds)
    - The y-axis corresponds to frequency (log scale from 0 to effective_sr/2)
    """
    num_subwindows = spectrogram.shape[0]
    fft_out_size   = spectrogram.shape[1]
    dt = nperseg / effective_sr  # time per subwindow in seconds
    time_axis = np.arange(num_subwindows) * dt * 1000  # convert to milliseconds
    freq_axis = np.linspace(0, effective_sr/2, fft_out_size)

    plt.figure(figsize=(10, 6))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower',
               extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    plt.yscale('log')  # Set log scale for the frequency axis
    plt.ylim(freq_axis[1], freq_axis[-1])  # Avoid log(0) issue
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (Hz) [Log Scale]")
    plt.title("Spectrogram (20 ms chunk)")
    plt.colorbar(label="Magnitude (dB)")
    plt.show()

# --- Main code ---

# You can change the filename as needed.

path_folder = "/Users/danielcampos/Desktop/output/output/"

file_number = input("Enter the spectrogram file number (e.g., 100): ").strip()

# Format the filename as 'stft_000XXX.bin'
filename = f"stft_{int(file_number):06d}.bin"  # Ensures zero-padding to 6 digits
full_path = path_folder + filename  # Combine with the directory path

data = load_spectrogram_file(full_path)

stft_data = data['stft']
nperseg = data['nperseg']
effective_sr = data['effective_sr']

print("Metadata:")
print(f"  Samples per 20ms chunk: {data['samples_20ms']}")
print(f"  nperseg: {nperseg}")
print(f"  noverlap: {data['noverlap']}")
print(f"  Number of subwindows: {data['num_subwindows']}")
print(f"  FFT output size (bins): {data['fft_out_size']}")
print(f"  Effective sample rate: {effective_sr}")
print(f"  Time offset (ms): {data['time_offset']}")

plot_spectrogram(stft_data, nperseg, effective_sr)
