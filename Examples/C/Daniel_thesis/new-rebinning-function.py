"""
Inference script for Bubble vs. Background Classification (using old, non‐log‐scaled .bin files).

This script:
  1) Reads each old‐style .bin file (metadata → raw time → STFT power array of shape (38,129)).
  2) Applies a log‐frequency rebinning with integration to 10 bins → (10,38).
  3) Normalizes each spectrogram by its own peak.
  4) Adds a channel dimension so that the final input is (10,38,1).
  5) Calls `model.predict()` on shape (1,10,38,1).
  6) Can loop over a directory subset (all bubble files + 40% of the non‐bubble files),
     compute precision/recall/FPR/accuracy/F1, and plot ROC/PR curves.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score
)

###############################################################################
# Log‐frequency rebinning with integration (NEW APPROACH)
###############################################################################

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

def preprocess_spectrogram(spectrogram, n_output_bins=10):
    """
    Preprocess spectrogram for model input
    
    Input: spectrogram shape = (num_subwindows=38, fft_out_size=129)
    Steps:
      (1) Transpose → (129, 38)
      (2) Log-bin with integration from 129 → 10 bins → (10, 38)
      (3) Divide by peak value → still (10, 38)
      (4) Expand dims → (10, 38, 1)
    Returns: (10, 38, 1)
    """
    # (1) Transpose (38×129 → 129×38)
    spec_T = spectrogram.T  # shape = (129, 38)
    
    # (2) Integration-based log binning
    spec_log = new_rebinning_function(spec_T, n_output_bins)  # shape = (10, 38)
    
    # (3) Normalize by the maximum of *this* spectrogram
    peak = np.max(spec_log)
    if peak > 0:
        spec_log = spec_log / peak
    
    # (4) Add channel dimension → (10, 38, 1)
    spec_final = spec_log[..., np.newaxis]
    
    return spec_final

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
    
    return bin_boundaries, bin_ranges

###############################################################################
# Load a single old‐style .bin file (non‐log‐scaled).
###############################################################################
def load_stft_file(filename: str) -> dict:
    """
    Reads a .bin file written by the OLD spectrogram acquisition C code.
    File format:
      - 7 uint32 metadata values:
          meta[0] = SAMPLES_20MS
          meta[1] = nperseg
          meta[2] = noverlap
          meta[3] = num_subwindows   (should be 38)
          meta[4] = fft_out_size     (should be 129)
          meta[5] = effective_sr
          meta[6] = time_offset_ms
      - SAMPLES_20MS float32 values: raw time‐domain data (unused here)
      - (num_subwindows * fft_out_size) float32 values: STFT power array
            in row‐major order, shape = (num_subwindows, fft_out_size) = (38, 129)

    Returns:
      {
        'stft': 2D np.ndarray of shape (38, 129)
      }
    """
    meta = np.fromfile(filename, dtype=np.uint32, count=7)
    if meta.size < 7:
        raise ValueError(f"File {filename} is too short to read metadata.")

    samples_20ms   = int(meta[0])
    nperseg        = int(meta[1])
    noverlap       = int(meta[2])
    num_subwindows = int(meta[3])  # expected 38
    fft_out_size   = int(meta[4])  # expected 129

    # Skip raw time‐domain data:
    meta_bytes     = 7 * 4                 # 28 bytes
    raw_time_bytes = samples_20ms * 4      # each float32 is 4 bytes
    stft_offset    = meta_bytes + raw_time_bytes

    stft_data = np.fromfile(
        filename,
        dtype=np.float32,
        count=num_subwindows * fft_out_size,
        offset=stft_offset
    )
    if stft_data.size < num_subwindows * fft_out_size:
        raise ValueError(f"File {filename} seems too short for full STFT data.")

    # Reshape into (num_subwindows, fft_out_size)
    stft_data = stft_data.reshape((num_subwindows, fft_out_size))
    return {'stft': stft_data}

###############################################################################
# Predict on one .bin file.
###############################################################################
def predict_on_bin(filename: str, model, threshold: float = 0.8):
    """
    1) Loads old‐style .bin → (38,129)
    2) Preprocess → (10,38,1)
    3) Batch dim → (1,10,38,1)
    4) model.predict → [prob_bg, prob_bubble]
    Prints and returns (predicted_class, [prob_bg, prob_bubble]).
    """
    data = load_stft_file(filename)
    raw_spec = data['stft']                 # (38,129)
    proc_spec = preprocess_spectrogram(raw_spec)  # (10,38,1)

    # Batch dimension → (1,10,38,1)
    input_data = np.expand_dims(proc_spec, axis=0)
    pred_prob = model.predict(input_data, verbose=0)[0]  # shape = (2,)

    print("Predicted probabilities [background, bubble]:", pred_prob)
    predicted_class = 1 if (pred_prob[1] > threshold) else 0
    print(f"Predicted class (threshold={threshold}): {predicted_class}")
    return predicted_class, pred_prob

###############################################################################
# Predict across a directory subset to compute metrics & plot ROC/PR.
###############################################################################
def predict_on_directory(
    directory: str,
    model,
    bubble_files: list,
    threshold: float = 0.8,
    sample_fraction: float = 0.4
):
    """
    Loops over:
      - all files in `bubble_files` ∩ (directory/*.bin), plus
      - a random `sample_fraction` subset of the non‐bubble files.
    Returns (y_true, y_scores, y_pred_labels), where:
      y_true[i] ∈ {0,1}, y_scores[i] = model's P(bubble), y_pred_labels = thresholded.
    """
    bin_files = sorted([f for f in os.listdir(directory) if f.endswith('.bin')])
    bubble_set = set(bubble_files)

    bubble_in_dir    = [f for f in bin_files if f in bubble_set]
    non_bubble_files = [f for f in bin_files if f not in bubble_set]

    sample_size = int(len(non_bubble_files) * sample_fraction)
    sampled_non_bubble = random.sample(non_bubble_files, sample_size) if sample_size > 0 else []

    final_files = sorted(set(bubble_in_dir + sampled_non_bubble))

    y_true   = []
    y_scores = []

    for fname in final_files:
        full_path = os.path.join(directory, fname)
        data = load_stft_file(full_path)
        raw_spec = data['stft']                     # (38,129)
        proc_spec = preprocess_spectrogram(raw_spec) # (10,38,1)

        input_data = np.expand_dims(proc_spec, axis=0)    # (1,10,38,1)
        prob_bubble = model.predict(input_data, verbose=0)[0, 1]
        y_scores.append(prob_bubble)
        y_true.append(1 if (fname in bubble_set) else 0)

    y_true   = np.array(y_true, dtype=np.int32)
    y_scores = np.array(y_scores, dtype=np.float32)
    y_pred   = (y_scores > threshold).astype(np.int32)

    return y_true, y_scores, y_pred

###############################################################################
# Main entry‐point.
###############################################################################
if __name__ == "__main__":
    # Test the new binning strategy first
    print("=== Testing New Binning Strategy ===")
    visualize_binning_strategy(129, 10)
    
    # Test with dummy data to verify shapes
    print("\n=== Testing Data Pipeline ===")
    dummy_spectrogram = np.random.rand(38, 129)  # Your input format
    processed = preprocess_spectrogram(dummy_spectrogram, n_output_bins=10)
    print(f"Input shape: {dummyhttps://github.com/catr1na/RedPitaya/tree/cnn-rebinning/Examples/C/Daniel_thesis_spectrogram.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Output min/max: {processed.min():.4f} / {processed.max():.4f}")
    
    # IMPORTANT: You need to retrain your model with input_shape=(10,38,1) instead of (513,38,1)
    # Once you retrain your model, uncomment the code below:
    
    """
    # (1) Load the retrained model with input_shape=(10,38,1)
    model = load_model('bubble_detector_model_10bins.h5')  # New model name

    # (2) Single‐file example
    single_bin = '/Users/Catrina/Desktop/CombinedTrainingData2/stft_436851.bin'
    print("=== Single File Prediction ===")
    predict_on_bin(single_bin, model, threshold=0.8)

    # (3) Evaluate on a directory subset
    eval_dir = '/Users/Catrina/Desktop/CombinedTrainingData2/'
    bubble_files = [
        "stft_436851.bin",
        "stft_472453.bin",
        "stft_480039.bin",
        "stft_489712.bin",
        "stft_509922.bin",
        "stft_539593.bin",
        "stft_553732.bin",
        "stft_553733.bin",
        "stft_565593.bin",
        "stft_581307.bin",
        "stft_606454.bin",
        "stft_636685.bin",
        "stft_646469.bin",
        "stft_671133.bin",
        "stft_689650.bin",
        "stft_698391.bin",
        "stft_734491.bin",
        "stft_734492.bin",
        "stft_743329.bin",
        "stft_776322.bin",
        "stft_838205.bin",
        "stft_850833.bin",
        "stft_859606.bin",
        "stft_869352.bin",
        "stft_869353.bin",
        "stft_879223.bin",
        "stft_888666.bin",
        "stft_888667.bin",
        "stft_910538.bin",
        "stft_910539.bin",
        "stft_1044807.bin",
        "stft_1044808.bin",
        "stft_1072303.bin",
        "stft_1086451.bin",
        "stft_1109856.bin",
        "stft_1109857.bin",
        "stft_1216826.bin",
        "stft_1216827.bin",
        "stft_1242322.bin",
        "stft_1259305.bin",
        "stft_1259306.bin",
        "stft_1272396.bin",
        "stft_1272397.bin",
        "stft_1289728.bin",
        "stft_1289729.bin",
        "stft_1313179.bin",
        "stft_1313180.bin",
        "stft_1348371.bin",
        "stft_1377342.bin",
        "stft_1377343.bin",
        "stft_1422123.bin",
        "stft_1422124.bin",
        "stft_1435531.bin",
        "stft_1435532.bin",
        "stft_1463687.bin",
        "stft_1463688.bin",
        "stft_1496675.bin",
        "stft_1517557.bin",
        "stft_1517558.bin",
        "stft_1525196.bin",
        "stft_1540053.bin",
        "stft_1548748.bin",
        "stft_1564438.bin",
        "stft_1590565.bin",
        "stft_1601066.bin",
        "stft_1601067.bin",
        "stft_1613430.bin",
        "stft_1613431.bin",
        "stft_1621451.bin",
        "stft_1621452.bin",
        "stft_1706510.bin",
        "stft_1706511.bin",
        "stft_1727662.bin",
        "stft_1742283.bin",
        "stft_1742284.bin",
        "stft_1763324.bin",
        "stft_1804621.bin",
        "stft_1804622.bin",
        "stft_1828362.bin",
        "stft_1828363.bin",
        "stft_1839805.bin",
        "stft_1839806.bin",
        "stft_1855476.bin",
        "stft_1855477.bin",
        "stft_1877592.bin",
        "stft_1877593.bin",
        "stft_1890728.bin",
        "stft_1890729.bin",
        "stft_1941875.bin",
        "stft_1941876.bin",
        "stft_1963675.bin",
        "stft_1971793.bin",
        "stft_1971794.bin",
        "stft_2026020.bin",
        "stft_2035599.bin",
        "stft_2050312.bin",
        "stft_2050313.bin",
        "stft_2080193.bin",
        "stft_2091291.bin"
    ]

    print("\n=== Evaluating on Directory Subset for ROC & Metrics ===")
    y_true, y_scores, y_pred = predict_on_directory(
        eval_dir,
        model,
        bubble_files,
        threshold=0.9997,
        sample_fraction=0.4
    )

    # Compute metrics
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred)

    print(f"\nMetrics (threshold=0.9997):")
    print(f"  Precision:           {precision:.4f}")
    print(f"  Recall (Trigger Eff): {recall:.4f}")
    print(f"  False Positive Rate: {false_positive_rate:.5f}")
    print(f"  Accuracy:            {accuracy:.4f}")
    print(f"  F1 Score:            {f1:.4f}")

    # Plot ROC curve (log‐scale on x‐axis)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    fpr_nonzero = np.clip(fpr, 1e-6, 1.0)
    plt.plot(fpr_nonzero, tpr, label=f'ROC (AUC={roc_auc:.4f})')
    plt.xscale('log')
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('Trigger Efficiency')
    plt.title('Receiver Operating Characteristic')
    diag = np.logspace(-6, 0, 1000)
    plt.plot(diag, diag, 'k--', label='Random')
    plt.ylim([0.0, 1.01])
    plt.xlim([1e-6, 1.0])
    plt.grid(True, which='both', ls='--')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve_10bins.png')
    plt.close()
    print("Saved: roc_curve_10bins.png")

    # Plot Precision‐Recall curve
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.plot(recalls, precisions, label=f'PR (AP={pr_auc:.3f})')
    plt.xlabel('Recall (Trigger Eff)')
    plt.ylabel('Precision')
    plt.title('Precision‐Recall Curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('precision_recall_curve_10bins.png')
    plt.close()
    print("Saved: precision_recall_curve_10bins.png")
    """
    
    print("\n=== Next Steps ===")
    print("1. Retrain your model with input_shape=(10, 38, 1)")
    print("2. Save the new model with a different name (e.g., 'bubble_detector_model_10bins.h5')")
    print("3. Uncomment the prediction code above")
    print("4. Test the new model with the 10-bin preprocessing")
