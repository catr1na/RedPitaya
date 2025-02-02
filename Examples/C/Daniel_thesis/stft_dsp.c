#include "stft_dsp.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Function to create a Hann window
static void create_hann_window(float *w, int size) {
    for (int i = 0; i < size; i++) {
        w[i] = 0.5f - 0.5f * cosf((2.0f * M_PI * i) / (size - 1));
    }
}

STFT_Handle* stft_init(int nperseg, int noverlap) {
    if (noverlap >= nperseg) {
        fprintf(stderr, "stft_init: noverlap (%d) >= nperseg (%d)!\n", noverlap, nperseg);
        return NULL;
    }

    STFT_Handle* h = (STFT_Handle*)calloc(1, sizeof(STFT_Handle));
    if (!h) {
        fprintf(stderr, "stft_init: Memory allocation failed for STFT_Handle\n");
        return NULL;
    }

    h->nperseg = nperseg;
    h->noverlap = noverlap;
    h->hop = nperseg - noverlap;
    h->fft_out_size = (nperseg / 2) + 1;

    // Allocate memory
    h->window = (float*)malloc(nperseg * sizeof(float));
    if (!h->window) {
        fprintf(stderr, "stft_init: Memory allocation failed for window\n");
        stft_cleanup(h);
        return NULL;
    }

    h->fft_in = (float*)fftwf_malloc(nperseg * sizeof(float));
    if (!h->fft_in) {
        fprintf(stderr, "stft_init: FFTW malloc failed for fft_in\n");
        stft_cleanup(h);
        return NULL;
    }

    h->fft_out = (fftwf_complex*)fftwf_malloc(h->fft_out_size * sizeof(fftwf_complex));
    if (!h->fft_out) {
        fprintf(stderr, "stft_init: FFTW malloc failed for fft_out\n");
        stft_cleanup(h);
        return NULL;
    }

    // Create Hann window
    create_hann_window(h->window, nperseg);

    // Create FFT plan
    h->plan = fftwf_plan_dft_r2c_1d(nperseg, h->fft_in, h->fft_out, FFTW_ESTIMATE);
    if (!h->plan) {
        fprintf(stderr, "stft_init: FFTW plan creation failed\n");
        stft_cleanup(h);
        return NULL;
    }

    return h;
}

void stft_cleanup(STFT_Handle* h) {
    if (!h) return;
    if (h->plan) fftwf_destroy_plan(h->plan);
    if (h->fft_out) fftwf_free(h->fft_out);
    if (h->fft_in) fftwf_free(h->fft_in);
    if (h->window) free(h->window);
    free(h);
}

int stft_compute(STFT_Handle* h, const float* data, int num_samples, float* out_power_dB) {
    if (!h || !data || !out_power_dB) return -1;

    int nperseg = h->nperseg;
    int hop = h->hop;
    int fft_out_size = h->fft_out_size;

    // Calculate number of sub-windows
    int num_subwindows = (num_samples - nperseg) / hop + 1;
    if (num_subwindows < 1) {
        fprintf(stderr, "stft_compute: Not enough samples for even one sub-window\n");
        return -1;
    }

    for (int sw = 0; sw < num_subwindows; sw++) {
        int start_idx = sw * hop;

        // Apply window and copy to fft_in
        for (int i = 0; i < nperseg; i++) {
            h->fft_in[i] = data[start_idx + i] * h->window[i];
        }

        // Execute FFT
        fftwf_execute(h->plan);

        // Compute power in dB
        for (int k = 0; k < fft_out_size; k++) {
            float re = h->fft_out[k][0];
            float im = h->fft_out[k][1];
            float mag = sqrtf(re * re + im * im);
            // Avoid log(0)
            if (mag < 1e-12f) mag = 1e-12f;
            float db = 20.0f * log10f(mag);
            out_power_dB[sw * fft_out_size + k] = db;
        }
    }

    return num_subwindows;
}
