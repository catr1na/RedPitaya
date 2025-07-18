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
    
    # Ensure we don't exceed bounds and handle duplicates
    bin_boundaries = np.clip(bin_boundaries, 1, orig_freq_bins-1)
    
    # Remove duplicates while preserving order
    unique_boundaries = [bin_boundaries[0]]
    for i in range(1, len(bin_boundaries)):
        if bin_boundaries[i] > unique_boundaries[-1]:
            unique_boundaries.append(bin_boundaries[i])
    
    # If we lost boundaries due to duplicates, spread them out
    if len(unique_boundaries) < n_output_bins + 1:
        # Fall back to linear spacing for the remaining bins
        remaining_bins = n_output_bins + 1 - len(unique_boundaries)
        start_idx = unique_boundaries[-1] + 1
        end_idx = orig_freq_bins - 1
        if start_idx <= end_idx:
            linear_boundaries = np.linspace(start_idx, end_idx, remaining_bins + 1)[1:]
            unique_boundaries.extend(np.round(linear_boundaries).astype(int))
    
    bin_boundaries = np.array(unique_boundaries[:n_output_bins + 1])
    
    # Create ranges for integration
    bin_ranges = []
    for i in range(len(bin_boundaries) - 1):
        start_idx = bin_boundaries[i]
        end_idx = bin_boundaries[i+1] - 1  # Make ranges non-overlapping
        bin_ranges.append((start_idx, end_idx))
    
    return bin_boundaries, bin_ranges

def new_rebinning_function(spectrogram, n_output_bins=10):
    """
    Integration-based logarithmic binning
    
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
        if i >= n_output_bins:  # Safety check
            break
            
        if start_idx <= end_idx:
            # Sum (integrate) the power across this frequency range
            new_spec[i, :] = np.sum(spectrogram[start_idx:end_idx+1, :], axis=0)
        else:
            # This shouldn't happen with fixed binning, but safety check
            new_spec[i, :] = 0
    
    return new_spec

def updated_preprocess_spectrogram(spectrogram, n_output_bins=10):
    """
    Updated version of preprocess_spectrogram function
    
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
    
    # (2) Integration-based log binning instead of interpolation
    spec_log = new_rebinning_function(spec_T, n_output_bins)  # shape = (10, 38)
    
    # (3) Normalize by the maximum of *this* spectrogram
    peak = np.max(spec_log)
    if peak > 0:
        spec_log = spec_log / peak
    
    # (4) Add channel dimension → (10, 38, 1)
    spec_final = spec_log[..., np.newaxis]
    
    return spec_final  # This was missing in your original code!

def visualize_binning_strategy(orig_freq_bins=129, n_output_bins=10):
    """
    Show how the 129 original bins map to 10 logarithmic bins
    """
    bin_boundaries, bin_ranges = create_log_binning_for_your_setup(orig_freq_bins, n_output_bins)
    
    print(f"Binning Strategy: {orig_freq_bins} bins → {n_output_bins} bins")
    print("Bin boundaries (indices):", bin_boundaries)
    print("\nBin ranges:")
    
    total_bins_used = 0
    for i, (start, end) in enumerate(bin_ranges):
        width = end - start + 1
        total_bins_used += width
        print(f"Output bin {i+1}: indices {start}-{end} (width: {width} bins)")
    
    print(f"\nTotal input bins used: {total_bins_used} out of {orig_freq_bins-1} (excluding DC bin)")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Show the bin boundaries
    plt.subplot(1, 2, 1)
    x = np.arange(1, orig_freq_bins)  # Frequency bins 1-128
    colors = plt.cm.tab10(np.linspace(0, 1, n_output_bins))
    
    for i, (start, end) in enumerate(bin_ranges):
        mask = (x >= start) & (x <= end)
        plt.bar(x[mask], np.ones(np.sum(mask)), color=colors[i], alpha=0.7, 
                label=f'Bin {i+1}' if i < 5 else '')
    
    plt.xlabel('Original Frequency Bin Index')
    plt.ylabel('Output Bin Assignment')
    plt.title('Logarithmic Binning Strategy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Show bin widths
    plt.subplot(1, 2, 2)
    bin_widths = [end - start + 1 for start, end in bin_ranges]
    plt.bar(range(1, n_output_bins + 1), bin_widths, color='skyblue', alpha=0.7)
    plt.xlabel('Output Bin Number')
    plt.ylabel('Width (number of input bins)')
    plt.title('Bin Widths')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('binning_strategy_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return bin_boundaries, bin_ranges

# Test the functions
if __name__ == "__main__":
    # IMPORTANT: You need to retrain your model with input_shape=(10,38,1) instead of (513,38,1)
    # The current model expects (513,38,1) but the new binning outputs (10,38,1)
    
    # For now, this will fail until you retrain the model
    # model = load_model('bubble_detector_modelNEW1.h5')
    
    # Test the new binning strategy first
    print("=== Testing New Binning Strategy ===")
    visualize_binning_strategy(129, 10)
    
    # Test with dummy data to verify shapes
    print("\n=== Testing Data Pipeline ===")
    dummy_spectrogram = np.random.rand(38, 129)  # Your input format
    processed = updated_preprocess_spectrogram(dummy_spectrogram, n_output_bins=10)
    print(f"Input shape: {dummy_spectrogram.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Output min/max: {processed.min():.4f} / {processed.max():.4f}")
    
    # Once you retrain your model with input_shape=(10,38,1), uncomment below:
    """
    # (1) Load the retrained model with input_shape=(10,38,1)
    model = load_model('bubble_detector_model_10bins.h5')  # New model name
    
    # (2) Single‐file example
    single_bin = '/Users/Catrina/Desktop/CombinedTrainingData2/stft_436851.bin'
    print("=== Single File Prediction ===")
    predict_on_bin(single_bin, model, threshold=0.8)
    
    # (3) Continue with directory evaluation as before...
    """
    
    print("\n=== Next Steps ===")
    print("1. Retrain your model with input_shape=(10, 38, 1)")
    print("2. Save the new model with a different name (e.g., 'bubble_detector_model_10bins.h5')")
    print("3. Update the model loading line in this script")
    print("4. Test the new model with the 10-bin preprocessing")
