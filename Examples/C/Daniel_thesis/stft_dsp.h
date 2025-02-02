#ifndef STFT_DSP_H
#define STFT_DSP_H

#include <fftw3.h>

typedef struct {
    int nperseg;               // Sub-window size (samples per FFT)
    int noverlap;              // Overlap between sub-windows (samples)
    int hop;                   // Hop size = nperseg - noverlap
    int fft_out_size;          // Number of frequency bins = nperseg/2 + 1
    float *window;             // Window function array
    float *fft_in;             // FFT input buffer
    fftwf_complex *fft_out;    // FFT output buffer
    fftwf_plan plan;           // FFTW plan
} STFT_Handle;

/**
 * Initialize STFT for repeated sub-window FFTs.
 * nperseg: samples per sub-window (e.g., 1024)
 * noverlap: overlap in samples (set to 0 for no overlap)
 * Returns: pointer to allocated STFT_Handle or NULL on failure.
 */
STFT_Handle* stft_init(int nperseg, int noverlap);

/**
 * Free all STFT resources.
 */
void stft_cleanup(STFT_Handle* h);

/**
 * Perform STFT on 'data' (length num_samples).
 * Output goes into 'out_power_dB', which must have size >= num_subwindows * (nperseg/2 + 1).
 * Returns: number of sub-windows actually processed, or -1 on error.
 *
 * num_subwindows is computed as:
 *   (num_samples - nperseg) / hop + 1
 * where hop = (nperseg - noverlap).
 */
int stft_compute(STFT_Handle* h,
                const float* data,
                int num_samples,
                float* out_power_dB);

#endif // STFT_DSP_H
