#THIS IS FOR OLD LOG NON REBINNED DATA. THIS HAS REBINNING PREPROCESSING

import numpy as np
import matplotlib.pyplot as plt

# Adjust to wherever your .bin files live
path_folder = "/Users/danielcampos/Desktop/CombinedTrainingData/"

current_file_number = None

def load_spectrogram_file(file_number):
    """
    Reads one of your 'stft_000XXX.bin' files and returns a dict with:
      - metadata
      - raw_time (unused here)
      - stft: a (num_subwindows × fft_out_size) array of dB values
    """
    filename = f"stft_{int(file_number):06d}.bin"
    full_path = path_folder + filename
    try:
        # 1) Read the 7× uint32 metadata
        meta = np.fromfile(full_path, dtype=np.uint32, count=7)
        if meta.size < 7:
            raise ValueError("Cannot read full metadata")
        samples_20ms, nperseg, noverlap, num_subwindows, fft_out_size, effective_sr, time_offset = meta

        # 2) Skip past metadata + raw_time to the STFT block
        offset = 7*4 + samples_20ms*4
        stft_data = np.fromfile(full_path,
                                dtype=np.float32,
                                count=num_subwindows * fft_out_size,
                                offset=offset)
        stft_data = stft_data.reshape((num_subwindows, fft_out_size))

        return {
            'nperseg':        nperseg,
            'effective_sr':   effective_sr,
            'fft_out_size':   fft_out_size,
            'num_subwindows': num_subwindows,
            'filename':       filename,
            'stft':           stft_data
        }
    except Exception as e:
        print(f"Error loading {full_path}: {e}")
        return None

def log_scale_spectrogram(spectrogram, new_num_freq_bins=129):
    """
    spectrogram: 2D array (orig_freq_bins, num_time_steps)
    returns   : 2D array (new_num_freq_bins, num_time_steps) with log-spaced freq bins
    """
    orig_bins, num_times = spectrogram.shape
    x_old = np.arange(orig_bins)
    x_new = np.logspace(0, np.log10(orig_bins - 1), new_num_freq_bins)
    new_spec = np.zeros((new_num_freq_bins, num_times), dtype=spectrogram.dtype)
    for t in range(num_times):
        new_spec[:, t] = np.interp(x_new, x_old, spectrogram[:, t])
    return new_spec

def plot_spectrogram(file_number, new_num_freq_bins=129):
    global fig
    plt.clf()
    ax = fig.add_subplot(111)

    data = load_spectrogram_file(file_number)
    if data is None:
        ax.text(.5,.5, f"Failed to load file {file_number}", 
                ha='center', va='center')
        plt.draw()
        return

    stft = data['stft']            # shape (num_subwindows, fft_out_size)
    nperseg = data['nperseg']
    sr_half = data['effective_sr'] / 2.0
    fft_out_size = data['fft_out_size']
    filename = data['filename']

    # time axis (ms)
    dt = nperseg / data['effective_sr']
    time_axis = np.arange(stft.shape[0]) * dt * 1000

    # 1) transpose so we get (freq_bins, time_steps)
    spec = stft.T

    # 2) log-scale it
    log_spec = log_scale_spectrogram(spec, new_num_freq_bins=new_num_freq_bins)

    # 3) build a log-spaced freq axis between the first non-zero bin and Nyquist
    orig_freqs = np.linspace(0, sr_half, fft_out_size)
    f_min = orig_freqs[1]        # avoid zero
    f_max = orig_freqs[-1]
    freq_axis = np.logspace(np.log10(f_min), np.log10(f_max),
                             new_num_freq_bins)

    # 4) plot with pcolormesh so we can set a log y-axis
    X, Y = np.meshgrid(time_axis, freq_axis)
    im = ax.pcolormesh(X, Y, log_spec, shading='auto')
    ax.set_yscale('log')
    ax.set_ylim(f_min, f_max)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz) [log]")
    ax.set_title(f"Log-Scaled Spectrogram: {filename}")
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")

    fig.text(1, 0.01,
             "Use ←/→ to navigate",
             ha='center', va='center', fontsize=10, style='italic')
    plt.tight_layout()
    plt.draw()

def on_key(event):
    global current_file_number
    if event.key == 'right':
        current_file_number += 1
        plot_spectrogram(current_file_number)
    elif event.key == 'left' and current_file_number > 0:
        current_file_number -= 1
        plot_spectrogram(current_file_number)

if __name__ == "__main__":
    try:
        current_file_number = int(input("Enter spectrogram file number (e.g. 100): ").strip())
    except ValueError:
        print("Invalid input—must be an integer.")
        exit(1)

    fig = plt.figure(figsize=(10, 6))
    plot_spectrogram(current_file_number)
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    plt.show()
