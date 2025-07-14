import numpy as np
import matplotlib.pyplot as plt

def create_log_binning_for_your_setup(orig_freq_bins=129, n_output_bins=10):
    """
    Create logarithmic binning strategy for your specific setup:
    - Original: 129 frequency bins from your FFT
    - Target: 10 logarithmic bins
    
    Args:
        orig_freq_bins: Your current FFT output size (129)
        n_output_bins: Target number of output bins (10)
    
    Returns:
        bin_boundaries: Array of bin indices (length n_output_bins + 1)
        bin_ranges: List of (start_idx, end_idx) for each output bin
    """
    # Skip DC bin (index 0), work with bins 1 to 128
    # Create logarithmically spaced positions in the range [1, 128]
    log_positions = np.logspace(np.log10(1), np.log10(orig_freq_bins-1), n_output_bins + 1)
    
    # Convert to integer indices
    bin_boundaries = np.round(log_positions).astype(int)
    
    # Ensure we don't exceed bounds
    bin_boundaries = np.clip(bin_boundaries, 1, orig_freq_bins-1)
    
    # Create ranges for integration
    bin_ranges = []
    for i in range(n_output_bins):
        start_idx = bin_boundaries[i]
        end_idx = bin_boundaries[i+1]
        bin_ranges.append((start_idx, end_idx))
    
    return bin_boundaries, bin_ranges

def new_rebinning_function(spectrogram, n_output_bins=10):
    """
    Replace the current log_scale_spectrogram with integration-based binning
    
    Input: spectrogram shape = (orig_freq_bins=129, num_time_steps=38)
    Output: shape = (n_output_bins=10, num_time_steps=38)
    """
    orig_freq_bins, num_time_steps = spectrogram.shape
    
    # Get the logarithmic bin ranges
    _, bin_ranges = create_log_binning_for_your_setup(orig_freq_bins, n_output_bins)
    
    # Initialize output
    new_spec = np.zeros((n_output_bins, num_time_steps), dtype=spectrogram.dtype)
    
    # Integrate power in each logarithmic bin
    for i, (start_idx, end_idx) in enumerate(bin_ranges):
        if start_idx < end_idx:
            # Sum (integrate) the power across this frequency range
            new_spec[i, :] = np.sum(spectrogram[start_idx:end_idx+1, :], axis=0)
        else:
            # Single bin case
            new_spec[i, :] = spectrogram[start_idx, :]
    
    return new_spec

# Test and visualize the binning strategy
def visualize_binning_strategy(orig_freq_bins=129, n_output_bins=10):
    """
    Show how the 129 original bins map to 10 logarithmic bins
    """
    bin_boundaries, bin_ranges = create_log_binning_for_your_setup(orig_freq_bins, n_output_bins)
    
    print(f"Binning Strategy: {orig_freq_bins} bins → {n_output_bins} bins")
    print("Bin boundaries (indices):", bin_boundaries)
    print("\nBin ranges:")
    
    for i, (start, end) in enumerate(bin_ranges):
        width = end - start + 1
        print(f"Output bin {i+1}: indices {start}-{end} (width: {width} bins)")
    
    return bin_boundaries, bin_ranges

# Example usage matching your current preprocessing
def updated_preprocess_spectrogram(spectrogram, n_output_bins=10):
    """
    Updated version of your preprocess_spectrogram function
    
    Input: spectrogram shape = (num_subwindows=38, fft_out_size=129)
    Steps:
      (1) Transpose → (129, 38)
      (2) Log-bin with integration from 129 → 10 bins → (10, 38)
      (3) Divide by peak value → still (10, 38)
      (4) Expand dims → (10, 38, 1)
    Returns: (10, 38, 1) instead of (513, 38, 1)
    """
    # (1) Transpose (38×129 → 129×38)
    spec_T = spectrogram.T  # shape = (129, 38)
    
    # (2) NEW: Integration-based log binning instead of interpolation
    spec_log = new_rebinning_function(spec_T, n_output_bins)  # shape = (10, 38)
    
    # (3) Normalize by the maximum of *this* spectrogram
    peak = np.max(spec_log)
    if peak > 0:
        spec_log = spec_log / peak
    
    # (4) Add channel dimension → (10, 38, 1)
    spec_final = spec_log[..., np.newaxis]
    
# Test the binning strategy
if __name__ == "__main__":
    # Visualize the binning strategy
    print("=== Logarithmic Binning Strategy ===")
    bin_boundaries, bin_ranges = visualize_binning_strategy(129, 10)
    
    # Create synthetic test data to validate the function
    print("\n=== Testing with Synthetic Data ===")
    
    # Create a synthetic spectrogram (129 freq bins, 38 time steps)
    # Simulate some frequency content
    freq_bins, time_steps = 129, 38
    synthetic_spec = np.random.rand(freq_bins, time_steps) * 0.1  # Base noise
    
    # Add some "bubble signatures" at specific frequencies
    synthetic_spec[10:20, :] += 2.0    # Low freq bubble signature
    synthetic_spec[50:60, :] += 1.5    # Mid freq signature  
    synthetic_spec[100:110, :] += 1.0  # High freq signature
    
    # Test the new rebinning function
    rebinned = new_rebinning_function(synthetic_spec, n_output_bins=10)
    
    print(f"Original shape: {synthetic_spec.shape}")
    print(f"Rebinned shape: {rebinned.shape}")
    print(f"Original total energy: {np.sum(synthetic_spec):.3f}")
    print(f"Rebinned total energy: {np.sum(rebinned):.3f}")
    print(f"Energy conservation: {np.sum(rebinned)/np.sum(synthetic_spec)*100:.1f}%")
    
    # Test the full preprocessing pipeline
    print("\n=== Testing Full Preprocessing Pipeline ===")
    
    # Simulate your file format: (38, 129) - time first, then frequency
    input_spec = synthetic_spec.T  # Shape (38, 129)
    
    # Test both old and new preprocessing
    old_result = preprocess_spectrogram_old(input_spec)  # Would be (513, 38, 1)
    new_result = updated_preprocess_spectrogram(input_spec, n_output_bins=10)  # (10, 38, 1)
    
    print(f"Old preprocessing output shape: {old_result.shape}")
    print(f"New preprocessing output shape: {new_result.shape}")
    print(f"Compression ratio: {old_result.shape[0]/new_result.shape[0]:.1f}x")

# For comparison - your original preprocessing function
def preprocess_spectrogram_old(spectrogram):
    """
    Your original preprocessing function for comparison
    """
    # (1) Transpose (38×129 → 129×38)
    spec_T = spectrogram.T
    
    # (2) Log-scale frequency axis → (513, 38)
    spec_log = log_scale_spectrogram_old(spec_T, new_num_freq_bins=513)
    
    # (3) Normalize by the maximum
    peak = np.max(spec_log)
    if peak > 0:
        spec_log = spec_log / peak
    
    # (4) Add channel dimension → (513, 38, 1)
    spec_final = spec_log[..., np.newaxis]
    
    return spec_final

def log_scale_spectrogram_old(spectrogram, new_num_freq_bins=513):
    """
    Your original log scaling function (interpolation-based)
    """
    orig_num_freq_bins, num_time_steps = spectrogram.shape
    x_old = np.arange(orig_num_freq_bins, dtype=np.float32)
    x_new = np.logspace(0, np.log10(orig_num_freq_bins - 1), new_num_freq_bins, dtype=np.float32)
    new_spec = np.zeros((new_num_freq_bins, num_time_steps), dtype=spectrogram.dtype)

    for t in range(num_time_steps):
        new_spec[:, t] = np.interp(x_new, x_old, spectrogram[:, t])
    return new_spec
