#include "spec_dsp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <stdbool.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

static struct {
    int fft_size;
    float* window;
    float* fft_in;
    fftwf_complex* fft_out;
    fftwf_plan plan;
    bool initialized;
    window_mode_t current_window;
} spec = {0};

static void create_window(window_mode_t type, int size) {
    if (spec.window) {
        free(spec.window);
    }
    
    spec.window = (float*)malloc(size * sizeof(float));
    spec.current_window = type;
    
    for (int i = 0; i < size; i++) {
        float phase = 2.0 * M_PI * i / (size - 1);
        switch (type) {
            case HANNING:
                spec.window[i] = 0.5 * (1.0 - cos(phase));
                break;
            case HAMMING:
                spec.window[i] = 0.54 - 0.46 * cos(phase);
                break;
            case BLACKMAN_HARRIS:
                spec.window[i] = 0.35875 - 
                                0.48829 * cos(phase) +
                                0.14128 * cos(2.0 * phase) -
                                0.01168 * cos(3.0 * phase);
                break;
            case RECTANGULAR:
            default:
                spec.window[i] = 1.0;
                break;
        }
    }

    // Normalize window
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += spec.window[i];
    }
    float norm = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        spec.window[i] *= norm;
    }
}

int spec_init(int fft_size, window_mode_t window_type) {
    if (spec.initialized) {
        spec_cleanup();
    }

    spec.fft_size = fft_size;
    
    // Create window function
    create_window(window_type, fft_size);
    
    // Initialize FFT
    spec.fft_in = (float*)fftwf_malloc(fft_size * sizeof(float));
    spec.fft_out = (fftwf_complex*)fftwf_malloc((fft_size/2 + 1) * sizeof(fftwf_complex));
    
    if (!spec.fft_in || !spec.fft_out || !spec.window) {
        spec_cleanup();
        return -1;
    }
    
    spec.plan = fftwf_plan_dft_r2c_1d(fft_size, spec.fft_in, spec.fft_out, FFTW_MEASURE);
    if (!spec.plan) {
        spec_cleanup();
        return -1;
    }
    
    spec.initialized = true;
    return 0;
}

int spec_process_frame(const float* input, float* output, int size) {
    if (!spec.initialized || !input || !output || size != spec.fft_size) {
        return -1;
    }
    
    // Apply window and copy to FFT input buffer
    for (int i = 0; i < spec.fft_size; i++) {
        spec.fft_in[i] = input[i] * spec.window[i];
    }
    
    // Compute FFT
    fftwf_execute(spec.plan);
    
    // Compute magnitude spectrum
    for (int i = 0; i < spec.fft_size/2 + 1; i++) {
        float re = spec.fft_out[i][0];
        float im = spec.fft_out[i][1];
        output[i] = sqrtf(re * re + im * im);
    }
    
    return 0;
}

void spec_cleanup(void) {
    if (spec.window) {
        free(spec.window);
        spec.window = NULL;
    }
    if (spec.plan) {
        fftwf_destroy_plan(spec.plan);
        spec.plan = NULL;
    }
    if (spec.fft_in) {
        fftwf_free(spec.fft_in);
        spec.fft_in = NULL;
    }
    if (spec.fft_out) {
        fftwf_free(spec.fft_out);
        spec.fft_out = NULL;
    }
    spec.initialized = false;
}

int spec_get_output_size(void) {
    if (!spec.initialized) {
        return -1;
    }
    return spec.fft_size/2 + 1;
}

window_mode_t spec_get_window_type(void) {
    return spec.current_window;
}

int spec_change_window(window_mode_t new_type) {
    if (!spec.initialized) {
        return -1;
    }
    if (new_type != spec.current_window) {
        create_window(new_type, spec.fft_size);
    }
    return 0;
}
