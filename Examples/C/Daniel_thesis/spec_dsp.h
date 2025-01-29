#ifndef SPEC_DSP_H
#define SPEC_DSP_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    RECTANGULAR = 0,
    HANNING = 1,
    HAMMING = 2,
    BLACKMAN_HARRIS = 3
} window_mode_t;

// Initialize spectrogram processing
int spec_init(int fft_size, window_mode_t window_type);

// Process one frame of data and compute its spectrogram
int spec_process_frame(const float* input, float* output, int size);

// Clean up resources
void spec_cleanup(void);

// Get output size (FFT_SIZE/2 + 1)
int spec_get_output_size(void);

// Get current window type
window_mode_t spec_get_window_type(void);

// Change window type
int spec_change_window(window_mode_t new_type);

#endif // SPEC_DSP_H
