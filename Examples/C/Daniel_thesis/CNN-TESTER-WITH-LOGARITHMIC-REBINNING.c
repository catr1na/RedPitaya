/*
 * Inference script for Bubble vs. Background Classification (using old, non-log-scaled .bin files).
 * 
 * This script:
 *   1) Reads each old-style .bin file (metadata → raw time → STFT power array of shape (38,129)).
 *   2) Applies a log-frequency rebinning with integration to 17 bins → (17,38).
 *   3) Normalizes each spectrogram by its own peak.
 *   4) Adds a channel dimension so that the final input is (17,38,1).
 *   5) Calls model prediction on shape (1,17,38,1).
 *   6) Can loop over a directory subset (all bubble files + 40% of the non-bubble files),
 *      compute precision/recall/FPR/accuracy/F1, and plot ROC/PR curves.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>

#define MAX_FILENAME 512
#define MAX_FILES 10000
#define ORIG_FREQ_BINS 129
#define N_OUTPUT_BINS 17
#define NUM_TIME_STEPS 38
#define EXPECTED_SUBWINDOWS 38
#define EXPECTED_FFT_SIZE 129

// Structure to hold STFT data
typedef struct {
    float stft[EXPECTED_SUBWINDOWS][EXPECTED_FFT_SIZE];
} stft_data_t;

// Structure for bin ranges
typedef struct {
    int start_idx;
    int end_idx;
} bin_range_t;

// Structure for evaluation results
typedef struct {
    float *y_true;
    float *y_scores;
    int *y_pred;
    int count;
} eval_results_t;

// Structure for metrics
typedef struct {
    float precision;
    float recall;
    float false_positive_rate;
    float accuracy;
    float f1_score;
    int tp, tn, fp, fn;
} metrics_t;

/*
 * Log-frequency rebinning with integration (NEW APPROACH)
 */

void create_log_binning_for_setup(int orig_freq_bins, int n_output_bins, 
                                  int *bin_boundaries, bin_range_t *bin_ranges) {
    /*
     * Create logarithmic binning strategy for specific setup:
     * - Original: 129 frequency bins from FFT
     * - Target: 17 logarithmic bins
     */
    
    // Skip DC bin (index 0), work with bins 1 to 128
    // Create logarithmically spaced positions in the range [1, 128]
    double log_start = log10(1.0);
    double log_end = log10((double)(orig_freq_bins - 1));
    double log_step = (log_end - log_start) / n_output_bins;
    
    // Generate log positions and convert to integer indices
    for (int i = 0; i <= n_output_bins; i++) {
        double log_pos = log_start + i * log_step;
        double linear_pos = pow(10.0, log_pos);
        bin_boundaries[i] = (int)round(linear_pos);
        
        // Ensure we don't exceed bounds
        if (bin_boundaries[i] < 1) bin_boundaries[i] = 1;
        if (bin_boundaries[i] >= orig_freq_bins) bin_boundaries[i] = orig_freq_bins - 1;
    }
    
    // Remove duplicates while preserving order
    int unique_count = 1;
    for (int i = 1; i <= n_output_bins; i++) {
        if (bin_boundaries[i] > bin_boundaries[unique_count - 1]) {
            bin_boundaries[unique_count] = bin_boundaries[i];
            unique_count++;
        }
    }
    
    // If we lost boundaries due to duplicates, spread them out
    if (unique_count < n_output_bins + 1) {
        int remaining_bins = n_output_bins + 1 - unique_count;
        int start_idx = bin_boundaries[unique_count - 1] + 1;
        int end_idx = orig_freq_bins - 1;
        
        if (start_idx <= end_idx && remaining_bins > 0) {
            for (int i = 0; i < remaining_bins; i++) {
                double frac = (double)(i + 1) / (remaining_bins + 1);
                bin_boundaries[unique_count + i] = start_idx + (int)round(frac * (end_idx - start_idx));
            }
            unique_count += remaining_bins;
        }
    }
    
    // Create ranges for integration
    for (int i = 0; i < n_output_bins && i < unique_count - 1; i++) {
        bin_ranges[i].start_idx = bin_boundaries[i];
        bin_ranges[i].end_idx = bin_boundaries[i + 1] - 1;  // Make ranges non-overlapping
    }
}

void new_rebinning_function(float spectrogram[ORIG_FREQ_BINS][NUM_TIME_STEPS],
                           float output[N_OUTPUT_BINS][NUM_TIME_STEPS]) {
    /*
     * Integration-based logarithmic binning
     * Input: spectrogram shape = (orig_freq_bins=129, num_time_steps=38)
     * Output: shape = (n_output_bins=17, num_time_steps=38)
     */
    
    bin_range_t bin_ranges[N_OUTPUT_BINS];
    int bin_boundaries[N_OUTPUT_BINS + 1];
    
    // Get the logarithmic bin ranges
    create_log_binning_for_setup(ORIG_FREQ_BINS, N_OUTPUT_BINS, bin_boundaries, bin_ranges);
    
    // Initialize output
    memset(output, 0, N_OUTPUT_BINS * NUM_TIME_STEPS * sizeof(float));
    
    // Integrate power in each logarithmic bin
    for (int i = 0; i < N_OUTPUT_BINS; i++) {
        int start_idx = bin_ranges[i].start_idx;
        int end_idx = bin_ranges[i].end_idx;
        
        if (start_idx <= end_idx) {
            for (int t = 0; t < NUM_TIME_STEPS; t++) {
                for (int f = start_idx; f <= end_idx; f++) {
                    output[i][t] += spectrogram[f][t];
                }
            }
        }
    }
}

void preprocess_spectrogram(float input_spec[EXPECTED_SUBWINDOWS][EXPECTED_FFT_SIZE],
                           float output[N_OUTPUT_BINS][NUM_TIME_STEPS]) {
    /*
     * Preprocess spectrogram for model input
     * Input: spectrogram shape = (num_subwindows=38, fft_out_size=129)
     * Steps:
     *   (1) Transpose → (129, 38)
     *   (2) Log-bin with integration from 129 → 17 bins → (17, 38)
     *   (3) Divide by peak value → still (17, 38)
     *   (4) Channel dimension handled separately
     * Returns: (17, 38)
     */
    
    // (1) Transpose (38×129 → 129×38)
    float spec_T[ORIG_FREQ_BINS][NUM_TIME_STEPS];
    for (int i = 0; i < EXPECTED_SUBWINDOWS; i++) {
        for (int j = 0; j < EXPECTED_FFT_SIZE; j++) {
            spec_T[j][i] = input_spec[i][j];
        }
    }
    
    // (2) Integration-based log binning
    float spec_log[N_OUTPUT_BINS][NUM_TIME_STEPS];
    new_rebinning_function(spec_T, spec_log);
    
    // (3) Normalize by the maximum of *this* spectrogram
    float peak = 0.0f;
    for (int i = 0; i < N_OUTPUT_BINS; i++) {
        for (int j = 0; j < NUM_TIME_STEPS; j++) {
            if (spec_log[i][j] > peak) {
                peak = spec_log[i][j];
            }
        }
    }
    
    if (peak > 0.0f) {
        for (int i = 0; i < N_OUTPUT_BINS; i++) {
            for (int j = 0; j < NUM_TIME_STEPS; j++) {
                output[i][j] = spec_log[i][j] / peak;
            }
        }
    } else {
        memcpy(output, spec_log, N_OUTPUT_BINS * NUM_TIME_STEPS * sizeof(float));
    }
}

void visualize_binning_strategy(int orig_freq_bins, int n_output_bins) {
    /*
     * Show how the 129 original bins map to 17 logarithmic bins
     */
    
    bin_range_t bin_ranges[n_output_bins];
    int bin_boundaries[n_output_bins + 1];
    
    create_log_binning_for_setup(orig_freq_bins, n_output_bins, bin_boundaries, bin_ranges);
    
    printf("Binning Strategy: %d bins → %d bins\n", orig_freq_bins, n_output_bins);
    printf("Bin boundaries (indices): ");
    for (int i = 0; i <= n_output_bins; i++) {
        printf("%d ", bin_boundaries[i]);
    }
    printf("\n\nBin ranges:\n");
    
    int total_bins_used = 0;
    for (int i = 0; i < n_output_bins; i++) {
        int width = bin_ranges[i].end_idx - bin_ranges[i].start_idx + 1;
        total_bins_used += width;
        printf("Output bin %d: indices %d-%d (width: %d bins)\n", 
               i + 1, bin_ranges[i].start_idx, bin_ranges[i].end_idx, width);
    }
    
    printf("\nTotal input bins used: %d out of %d (excluding DC bin)\n", 
           total_bins_used, orig_freq_bins - 1);
}

/*
 * Load a single old-style .bin file (non-log-scaled).
 */
int load_stft_file(const char *filename, stft_data_t *data) {
    /*
     * Reads a .bin file written by the OLD spectrogram acquisition C code.
     * File format:
     *   - 7 uint32 metadata values:
     *       meta[0] = SAMPLES_20MS
     *       meta[1] = nperseg
     *       meta[2] = noverlap
     *       meta[3] = num_subwindows   (should be 38)
     *       meta[4] = fft_out_size     (should be 129)
     *       meta[5] = effective_sr
     *       meta[6] = time_offset_ms
     *   - SAMPLES_20MS float32 values: raw time-domain data (unused here)
     *   - (num_subwindows * fft_out_size) float32 values: STFT power array
     *         in row-major order, shape = (num_subwindows, fft_out_size) = (38, 129)
     */
    
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    // Read metadata
    uint32_t meta[7];
    if (fread(meta, sizeof(uint32_t), 7, file) != 7) {
        fprintf(stderr, "Error: File %s is too short to read metadata.\n", filename);
        fclose(file);
        return -1;
    }
    
    uint32_t samples_20ms = meta[0];
    uint32_t num_subwindows = meta[3];
    uint32_t fft_out_size = meta[4];
    
    // Skip raw time-domain data
    if (fseek(file, samples_20ms * sizeof(float), SEEK_CUR) != 0) {
        fprintf(stderr, "Error: Cannot seek past raw time data in %s\n", filename);
        fclose(file);
        return -1;
    }
    
    // Read STFT data
    size_t expected_count = num_subwindows * fft_out_size;
    float *stft_buffer = (float*)malloc(expected_count * sizeof(float));
    if (!stft_buffer) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return -1;
    }
    
    if (fread(stft_buffer, sizeof(float), expected_count, file) != expected_count) {
        fprintf(stderr, "Error: File %s seems too short for full STFT data.\n", filename);
        free(stft_buffer);
        fclose(file);
        return -1;
    }
    
    // Reshape into (num_subwindows, fft_out_size)
    for (int i = 0; i < num_subwindows && i < EXPECTED_SUBWINDOWS; i++) {
        for (int j = 0; j < fft_out_size && j < EXPECTED_FFT_SIZE; j++) {
            data->stft[i][j] = stft_buffer[i * fft_out_size + j];
        }
    }
    
    free(stft_buffer);
    fclose(file);
    return 0;
}

/*
 * Mock model prediction function
 * In a real implementation, this would interface with TensorFlow C API or similar
 */
void mock_model_predict(float input[N_OUTPUT_BINS][NUM_TIME_STEPS], float *prob_bg, float *prob_bubble) {
    /*
     * Mock prediction function that simulates model behavior
     * In practice, this would call TensorFlow C API or similar ML framework
     */
    
    // Simple heuristic for demonstration: higher energy in higher frequency bins suggests bubble
    float high_freq_energy = 0.0f;
    float total_energy = 0.0f;
    
    for (int i = 0; i < N_OUTPUT_BINS; i++) {
        for (int j = 0; j < NUM_TIME_STEPS; j++) {
            total_energy += input[i][j];
            if (i > N_OUTPUT_BINS / 2) {  // Higher frequency bins
                high_freq_energy += input[i][j];
            }
        }
    }
    
    float ratio = (total_energy > 0) ? high_freq_energy / total_energy : 0.0f;
    
    // Mock probabilities based on energy distribution
    *prob_bubble = fminf(0.95f, fmaxf(0.05f, ratio * 2.0f));
    *prob_bg = 1.0f - *prob_bubble;
}

/*
 * Predict on one .bin file.
 */
int predict_on_bin(const char *filename, float threshold, int *predicted_class, float prob[2]) {
    /*
     * 1) Loads old-style .bin → (38,129)
     * 2) Preprocess → (17,38)
     * 3) Mock model prediction → [prob_bg, prob_bubble]
     * Prints and returns (predicted_class, [prob_bg, prob_bubble]).
     */
    
    stft_data_t data;
    if (load_stft_file(filename, &data) != 0) {
        return -1;
    }
    
    float processed[N_OUTPUT_BINS][NUM_TIME_STEPS];
    preprocess_spectrogram(data.stft, processed);
    
    mock_model_predict(processed, &prob[0], &prob[1]);
    
    printf("Predicted probabilities [background, bubble]: [%.4f, %.4f]\n", prob[0], prob[1]);
    *predicted_class = (prob[1] > threshold) ? 1 : 0;
    printf("Predicted class (threshold=%.4f): %d\n", threshold, *predicted_class);
    
    return 0;
}

/*
 * String utilities
 */
int string_ends_with(const char *str, const char *suffix) {
    if (!str || !suffix) return 0;
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > str_len) return 0;
    return strcmp(str + str_len - suffix_len, suffix) == 0;
}

int is_in_bubble_list(const char *filename, const char *bubble_files[], int bubble_count) {
    for (int i = 0; i < bubble_count; i++) {
        if (strcmp(filename, bubble_files[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

/*
 * Fisher-Yates shuffle algorithm for random sampling
 */
void shuffle_array(int *array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

/*
 * Predict across a directory subset to compute metrics.
 */
int predict_on_directory(const char *directory, const char *bubble_files[], int bubble_count,
                        float threshold, float sample_fraction, eval_results_t *results) {
    /*
     * Loops over:
     *   - all files in bubble_files ∩ (directory/*.bin), plus
     *   - a random sample_fraction subset of the non-bubble files.
     * Returns y_true, y_scores, y_pred_labels
     */
    
    DIR *dir = opendir(directory);
    if (!dir) {
        fprintf(stderr, "Error: Cannot open directory %s\n", directory);
        return -1;
    }
    
    // Collect all .bin files
    char bin_files[MAX_FILES][MAX_FILENAME];
    int bin_count = 0;
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && bin_count < MAX_FILES) {
        if (string_ends_with(entry->d_name, ".bin")) {
            strncpy(bin_files[bin_count], entry->d_name, MAX_FILENAME - 1);
            bin_files[bin_count][MAX_FILENAME - 1] = '\0';
            bin_count++;
        }
    }
    closedir(dir);
    
    // Separate bubble and non-bubble files
    char bubble_in_dir[MAX_FILES][MAX_FILENAME];
    int bubble_in_dir_count = 0;
    int non_bubble_indices[MAX_FILES];
    int non_bubble_count = 0;
    
    for (int i = 0; i < bin_count; i++) {
        if (is_in_bubble_list(bin_files[i], bubble_files, bubble_count)) {
            strncpy(bubble_in_dir[bubble_in_dir_count], bin_files[i], MAX_FILENAME);
            bubble_in_dir_count++;
        } else {
            non_bubble_indices[non_bubble_count] = i;
            non_bubble_count++;
        }
    }
    
    // Sample non-bubble files
    int sample_size = (int)(non_bubble_count * sample_fraction);
    shuffle_array(non_bubble_indices, non_bubble_count);
    
    // Prepare final file list
    int total_files = bubble_in_dir_count + sample_size;
    
    results->y_true = (float*)malloc(total_files * sizeof(float));
    results->y_scores = (float*)malloc(total_files * sizeof(float));
    results->y_pred = (int*)malloc(total_files * sizeof(int));
    results->count = total_files;
    
    if (!results->y_true || !results->y_scores || !results->y_pred) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return -1;
    }
    
    int file_idx = 0;
    
    // Process bubble files
    for (int i = 0; i < bubble_in_dir_count; i++) {
        char full_path[MAX_FILENAME * 2];
        snprintf(full_path, sizeof(full_path), "%s/%s", directory, bubble_in_dir[i]);
        
        stft_data_t data;
        if (load_stft_file(full_path, &data) == 0) {
            float processed[N_OUTPUT_BINS][NUM_TIME_STEPS];
            preprocess_spectrogram(data.stft, processed);
            
            float prob[2];
            mock_model_predict(processed, &prob[0], &prob[1]);
            
            results->y_scores[file_idx] = prob[1];  // Probability of bubble
            results->y_true[file_idx] = 1.0f;      // True label: bubble
            results->y_pred[file_idx] = (prob[1] > threshold) ? 1 : 0;
            file_idx++;
        }
    }
    
    // Process sampled non-bubble files
    for (int i = 0; i < sample_size; i++) {
        char full_path[MAX_FILENAME * 2];
        snprintf(full_path, sizeof(full_path), "%s/%s", directory, 
                bin_files[non_bubble_indices[i]]);
        
        stft_data_t data;
        if (load_stft_file(full_path, &data) == 0) {
            float processed[N_OUTPUT_BINS][NUM_TIME_STEPS];
            preprocess_spectrogram(data.stft, processed);
            
            float prob[2];
            mock_model_predict(processed, &prob[0], &prob[1]);
            
            results->y_scores[file_idx] = prob[1];  // Probability of bubble
            results->y_true[file_idx] = 0.0f;      // True label: background
            results->y_pred[file_idx] = (prob[1] > threshold) ? 1 : 0;
            file_idx++;
        }
    }
    
    results->count = file_idx;
    return 0;
}

/*
 * Calculate metrics
 */
void calculate_metrics(eval_results_t *results, metrics_t *metrics) {
    metrics->tp = metrics->tn = metrics->fp = metrics->fn = 0;
    
    for (int i = 0; i < results->count; i++) {
        int true_label = (int)results->y_true[i];
        int pred_label = results->y_pred[i];
        
        if (true_label == 1 && pred_label == 1) metrics->tp++;
        else if (true_label == 0 && pred_label == 0) metrics->tn++;
        else if (true_label == 0 && pred_label == 1) metrics->fp++;
        else if (true_label == 1 && pred_label == 0) metrics->fn++;
    }
    
    metrics->precision = (metrics->tp + metrics->fp > 0) ? 
                        (float)metrics->tp / (metrics->tp + metrics->fp) : 0.0f;
    metrics->recall = (metrics->tp + metrics->fn > 0) ? 
                     (float)metrics->tp / (metrics->tp + metrics->fn) : 0.0f;
    metrics->false_positive_rate = (metrics->fp + metrics->tn > 0) ? 
                                  (float)metrics->fp / (metrics->fp + metrics->tn) : 0.0f;
    metrics->accuracy = (float)(metrics->tp + metrics->tn) / results->count;
    metrics->f1_score = (metrics->precision + metrics->recall > 0) ? 
                       2.0f * metrics->precision * metrics->recall / 
                       (metrics->precision + metrics->recall) : 0.0f;
}

/*
 * Main entry-point.
 */
int main(int argc, char *argv[]) {
    // Seed random number generator
    srand((unsigned int)time(NULL));
    
    // Test the new binning strategy first
    printf("=== Testing New Binning Strategy ===\n");
    visualize_binning_strategy(129, 10);
    
    // Test with dummy data to verify shapes
    printf("\n=== Testing Data Pipeline ===\n");
    float dummy_spectrogram[EXPECTED_SUBWINDOWS][EXPECTED_FFT_SIZE];
    float processed[N_OUTPUT_BINS][NUM_TIME_STEPS];
    
    // Initialize dummy data
    for (int i = 0; i < EXPECTED_SUBWINDOWS; i++) {
        for (int j = 0; j < EXPECTED_FFT_SIZE; j++) {
            dummy_spectrogram[i][j] = (float)rand() / RAND_MAX;
        }
    }
    
    preprocess_spectrogram(dummy_spectrogram, processed);
    
    printf("Input shape: (%d, %d)\n", EXPECTED_SUBWINDOWS, EXPECTED_FFT_SIZE);
    printf("Output shape: (%d, %d)\n", N_OUTPUT_BINS, NUM_TIME_STEPS);
    
    // Find min/max
    float min_val = processed[0][0], max_val = processed[0][0];
    for (int i = 0; i < N_OUTPUT_BINS; i++) {
        for (int j = 0; j < NUM_TIME_STEPS; j++) {
            if (processed[i][j] < min_val) min_val = processed[i][j];
            if (processed[i][j] > max_val) max_val = processed[i][j];
        }
    }
    printf("Output min/max: %.4f / %.4f\n", min_val, max_val);
    
    printf("\n=== Next Steps ===\n");
    printf("1. Replace mock_model_predict() with actual TensorFlow C API calls\n");
    printf("2. Train your model with input_shape=(10, 38, 1)\n");
    printf("3. Save the new model and load it using TensorFlow C API\n");
    printf("4. Test the new model with the 10-bin preprocessing\n");
    printf("5. Implement ROC/PR curve plotting using a graphics library\n");
    
    /* 
     * Example usage (uncomment when model is ready):
     * 
     * // Single file prediction
     * const char *single_bin = "/Users/Catrina/Desktop/CombinedTrainingData2/stft_436851.bin";
     * int predicted_class;
     * float prob[2];
     * if (predict_on_bin(single_bin, 0.8f, &predicted_class, prob) == 0) {
     *     printf("Prediction successful\n");
     * }
     * 
     * // Directory evaluation
     * const char *bubble_files[] = {
     *     "stft_436851.bin", "stft_472453.bin", "stft_480039.bin", // ... add all bubble files
     * };
     * int bubble_count = sizeof(bubble_files) / sizeof(bubble_files[0]);
     * 
     * eval_results_t results;
     * if (predict_on_directory("/Users/Catrina/Desktop/CombinedTrainingData2/", 
     *                         bubble_files, bubble_count, 0.9997f, 0.4f, &results) == 0) {
     *     metrics_t metrics;
     *     calculate_metrics(&results, &metrics);
     *     
     *     printf("\nMetrics (threshold=0.9997):\n");
     *     printf("  Precision:           %.4f\n", metrics.precision);
     *     printf("  Recall (Trigger Eff): %.4f\n", metrics.recall);
     *     printf("  False Positive Rate: %.5f\n", metrics.false_positive_rate);
     *     printf("  Accuracy:            %.4f\n", metrics.accuracy);
     *     printf("  F1 Score:            %.4f\n", metrics.f1_score);
     *     
     *     free(results.y_true);
     *     free(results.y_scores);
     *     free(results.y_pred);
     * }
     */
    
    return 0;
}
