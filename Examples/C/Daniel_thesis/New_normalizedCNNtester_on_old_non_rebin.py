#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for Bubble vs. Background Classification (using old, non‐log‐scaled .bin files).

This script:
  1) Reads each old‐style .bin file (metadata → raw time → STFT power array of shape (38,129)).
  2) Applies a log‐frequency resampling to 513 bins → (513,38).
  3) Normalizes each spectrogram by its own peak.
  4) Adds a channel dimension so that the final input is (513,38,1).
  5) Calls `model.predict()` on shape (1,513,38,1).
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
# Helper: Log‐frequency resampling along the “frequency” axis.
# Input: 2D array of shape (orig_freq_bins, num_time_steps)
# Output: 2D array of shape (new_num_freq_bins, num_time_steps)
###############################################################################
def log_scale_spectrogram(spectrogram: np.ndarray, new_num_freq_bins: int = 513) -> np.ndarray:
    """
    spectrogram: shape = (orig_freq_bins, num_time_steps)
    new_num_freq_bins: e.g. 513
    Returns: shape = (new_num_freq_bins, num_time_steps)
    """
    orig_num_freq_bins, num_time_steps = spectrogram.shape
    x_old = np.arange(orig_num_freq_bins, dtype=np.float32)
    # Create logarithmically spaced “positions” in [0, orig_num_freq_bins−1]
    x_new = np.logspace(0, np.log10(orig_num_freq_bins - 1), new_num_freq_bins, dtype=np.float32)
    new_spec = np.zeros((new_num_freq_bins, num_time_steps), dtype=spectrogram.dtype)

    for t in range(num_time_steps):
        new_spec[:, t] = np.interp(x_new, x_old, spectrogram[:, t])
    return new_spec

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
# Preprocess one STFT (38×129) → (513×38×1):
#   1) Transpose to (129×38)
#   2) Log‐scale freq‐axis to 513 → (513×38)
#   3) Normalize by its own max
#   4) Add channel dim → (513, 38, 1)
###############################################################################
def preprocess_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    Input: spectrogram shape = (num_subwindows=38, fft_out_size=129)
    Steps:
      (1) Transpose → (129, 38)
      (2) Log‐scale frequency axis → (513, 38)
      (3) Divide by peak value → still (513, 38)
      (4) Expand dims → (513, 38, 1)
    Returns: (513, 38, 1)
    """
    # (1) Transpose (38×129 → 129×38)
    spec_T = spectrogram.T  # shape = (129, 38)

    # (2) Log‐resample from 129 bins → 513 bins along freq axis
    spec_log = log_scale_spectrogram(spec_T, new_num_freq_bins=513)  # shape = (513, 38)

    # (3) Normalize by the maximum of *this* spectrogram
    peak = np.max(spec_log)
    if peak > 0:
        spec_log = spec_log / peak

    # (4) Add channel dimension → (513, 38, 1)
    spec_final = spec_log[..., np.newaxis]

    return spec_final

###############################################################################
# Predict on one .bin file.
###############################################################################
def predict_on_bin(filename: str, model, threshold: float = 0.8):
    """
    1) Loads old‐style .bin → (38,129)
    2) Preprocess → (513,38,1)
    3) Batch dim → (1,513,38,1)
    4) model.predict → [prob_bg, prob_bubble]
    Prints and returns (predicted_class, [prob_bg, prob_bubble]).
    """
    data = load_stft_file(filename)
    raw_spec = data['stft']                 # (38,129)
    proc_spec = preprocess_spectrogram(raw_spec)  # (513,38,1)

    # Batch dimension → (1,513,38,1)
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
      y_true[i] ∈ {0,1}, y_scores[i] = model’s P(bubble), y_pred_labels = thresholded.
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
        proc_spec = preprocess_spectrogram(raw_spec) # (513,38,1)

        input_data = np.expand_dims(proc_spec, axis=0)    # (1,513,38,1)
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
    # (1) Load the Keras model saved by your “fixed trainer” (with input_shape=(513,38,1))
    model = load_model('bubble_detector_modelNEW1.h5')

    # (2) Single‐file example
    single_bin = '/Users/danielcampos/Desktop/CombinedTrainingData/stft_436851.bin'
    print("=== Single File Prediction ===")
    predict_on_bin(single_bin, model, threshold=0.8)

    # (3) Evaluate on a directory subset
    eval_dir = '/Users/danielcampos/Desktop/CombinedTrainingData/'
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

    print(f"\nMetrics (threshold=0.95):")
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
    plt.savefig('roc_curve_logNEW1_6.png')
    plt.close()
    print("Saved: roc_curve_logNEW1_6.png")

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
    plt.savefig('precision_recall_curveNEW1_6.png')
    plt.close()
    print("Saved: precision_recall_curvenNEW1_6.png")
