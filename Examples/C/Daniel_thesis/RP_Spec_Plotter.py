#THIS IS FOR NEW LOG REBINNED DATA

import numpy as np
import matplotlib.pyplot as plt
import os

# ——— CONFIG ———
path_folder = "/Users/danielcampos/Desktop/IOCODEoutput1/"
file_pattern = "stft_{:06d}.bin"

def load_log_spectrogram(file_number):
    """
    Reads:
      - 7×uint32 metadata:
          [samples20ms, nperseg, noverlap, num_subwindows,
           new_freq_bins, effective_sr, time_offset_ms]
      - samples20ms floats (raw time, skipped)
      - log-scaled STFT floats: new_freq_bins × num_subwindows (row-major)
    Returns dict with metadata + 'spec' as a (time, freq) 2D array.
    """
    fn = os.path.join(path_folder, file_pattern.format(file_number))
    raw = np.fromfile(fn, dtype=np.uint32, count=7)
    if raw.size < 7:
        raise ValueError(f"Metadata corrupt in {fn}")
    samples20ms, nperseg, noverlap, num_sub, new_bins, eff_sr, t0 = raw
    # skip raw time samples
    offset = 7*4 + int(samples20ms)*4
    count = int(num_sub) * int(new_bins)
    data = np.fromfile(fn, dtype=np.float32, count=count, offset=offset)
    if data.size < count:
        raise ValueError(f"STFT data corrupt in {fn}")
    # reshape into (freq_bins, time_steps) then transpose
    spec = data.reshape((new_bins, num_sub)).T  # shape: (time, freq)
    return {
        'nperseg':     int(nperseg),
        'noverlap':    int(noverlap),
        'num_sub':     int(num_sub),
        'new_bins':    int(new_bins),
        'eff_sr':      float(eff_sr),
        't0_ms':       float(t0),
        'spec':        spec
    }

def plot_file(file_number):
    global fig
    plt.clf()
    ax = fig.add_subplot(111)

    meta = load_log_spectrogram(file_number)
    spec = meta['spec']        # (time_steps, log_bins)
    nperseg = meta['nperseg']
    eff_sr  = meta['eff_sr']
    N, M    = spec.shape
    t0      = meta['t0_ms']
    fn      = file_pattern.format(file_number)

    # time axis (ms)
    hop = nperseg - meta['noverlap']
    dt = hop/eff_sr*1000
    t = t0 + np.arange(N)*dt

    # reconstruct freq axis by inverting C code's log mapping:
    orig_bins = nperseg//2 + 1
    log_max   = np.log10(orig_bins-1)
    i         = np.arange(M)
    orig_idx  = 10**(log_max * i/(M-1))
    freq      = orig_idx * (eff_sr/2)/(orig_bins-1)

    # mesh and plot
    T, F = np.meshgrid(t, freq)
    im = ax.pcolormesh(T, F, spec.T, shading='auto')
    ax.set_yscale('log')
    ax.set_ylim(freq[1], freq[-1])

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz) [log]")
    ax.set_title(f"Log-scaled spectrogram: {fn}")
    plt.colorbar(im, ax=ax, label="Power (dB)")
    fig.tight_layout()
    plt.draw()

def on_key(event):
    global idx
    if event.key == 'right':
        idx += 1
        plot_file(idx)
    elif event.key == 'left' and idx > 0:
        idx -= 1
        plot_file(idx)

if __name__ == "__main__":
    try:
        idx = int(input("Enter file number: ").strip())
    except:
        print("Must be integer"); exit(1)

    fig = plt.figure(figsize=(10, 6))
    plot_file(idx)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
