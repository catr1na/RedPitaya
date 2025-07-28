#ifndef BUBBLE_DETECTOR_H
#define BUBBLE_DETECTOR_H

#include <stdbool.h>
#include <stdint.h>

void save_spectrogram_csv(float* log_power_array, int new_freq_bins, int num_subwindows, uint32_t frame_num);

// CNN Architecture Parameters
// used to be CONV1_FILTERS: 64, CONV2_FILTERS: 128, CONV3_FILTERS & DENSE1_UNITS: 256
#define CONV1_FILTERS 32
#define CONV2_FILTERS 64
#define CONV3_FILTERS 128
#define DENSE1_UNITS 128
#define CONV_KERNEL_SIZE 3
#define POOL_SIZE 2

// Input dimensions
#define INPUT_HEIGHT 17  // Number of frequency bins (nperseg=1024)
#define INPUT_WIDTH 38    // Number of subwindows per chunk (num_subwindows=38)

// Detection Results
typedef enum {
    DETECTION_BACKGROUND = 0,
    DETECTION_BUBBLE = 1
} DetectionResult;

// Initialize detector with pre-trained weights directory
bool detector_init(const char* weights_dir);

// Process a single spectrogram frame
// 'spectrogram' should be an array of floats with shape [INPUT_HEIGHT * INPUT_WIDTH]
DetectionResult detector_process_frame(float* spectrogram);

// Clean up detector resources
void detector_cleanup(void);

#endif // BUBBLE_DETECTOR_H
