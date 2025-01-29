#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:21:35 2025

@author: danielcampos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

file_path="/Users/danielcampos/Desktop/output/output/"
bin_num=100 #Has to be greater than 1

def read_spectrogram_files(directory, num_files):
    spectrograms = []
    times = []
    
    for i in range(num_files):
        filename = f"{directory}/spectrogram_{i:06d}.bin"
        with open(filename, 'rb') as f:
            # Read metadata
            metadata = np.fromfile(f, dtype=np.uint32, count=4)
            n_samples, n_freq_bins, sample_rate, time_offset = metadata
            
            # Skip time domain and magnitude data
            f.seek(n_samples * 4 + n_freq_bins * 4, 1)  # 4 bytes per float
            
            # Read dB values
            power_db = np.fromfile(f, dtype=np.float32, count=n_freq_bins)
            
            spectrograms.append(power_db)
            times.append(time_offset / 1000.0)  # Convert to seconds
    
    return np.array(spectrograms), np.array(times), sample_rate

# Read 201 frames
spec_data, times, sample_rate = read_spectrogram_files(file_path, bin_num)

# Create original linear frequency axis
orig_freqs = np.linspace(0, sample_rate/2, spec_data.shape[1])

# Define new logarithmically spaced frequency axis
log_freqs = np.geomspace(orig_freqs[1], orig_freqs[-1], spec_data.shape[1])  # Start from orig_freqs[1] to avoid log(0)

# Interpolate spectrogram data onto the log frequency scale
interp_func = interp2d(times, orig_freqs, spec_data.T, kind='linear')
log_spec_data = interp_func(times, log_freqs)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, log_freqs, log_spec_data, shading='gouraud')
plt.colorbar(label='Power (dB)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram (Log Frequency Scale)')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.ylim([log_freqs.min(), log_freqs.max()])  # Ensure correct limits
plt.show()
