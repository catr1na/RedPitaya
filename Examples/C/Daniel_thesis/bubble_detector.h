#ifndef BUBBLE_DETECTOR_H
#define BUBBLE_DETECTOR_H

#include <stdbool.h>
#include <stdint.h>

#define CONV1_FILTERS 32
#define CONV2_FILTERS 64
#define DENSE1_UNITS 128
#define CONV_KERNEL_SIZE 3
#define POOL_SIZE 2

typedef enum {
    DETECTION_BACKGROUND = 0,
    DETECTION_BUBBLE = 1
} DetectionResult;

// Initialize detector with weights
bool detector_init(const char* weights_dir);

// Process single spectrogram frame
DetectionResult detector_process_frame(float* spectrogram, int size);

// Clean up detector resources
void detector_cleanup(void);

// Get the required input dimensions for the model
void detector_get_input_dims(int* height, int* width);

#endif // BUBBLE_DETECTOR_H