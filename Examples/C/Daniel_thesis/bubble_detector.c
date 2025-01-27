#include "bubble_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static struct {
    float* conv1_weights;
    float* conv1_bias;
    float* conv2_weights;
    float* conv2_bias;
    float* dense1_weights;
    float* dense1_bias;
    float* dense2_weights;
    float* dense2_bias;
    int input_height;
    int input_width;
    bool initialized;
} model = {0};

// Helper functions
static float relu(float x) {
    return x > 0 ? x : 0;
}

static void softmax(float* input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float val = expf(input[i] - max_val);
        input[i] = val;
        sum += val;
    }

    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

static void conv2d(const float* input, int in_h, int in_w,
                  const float* weights, const float* bias,
                  int kernel_size, int num_filters,
                  float* output, int out_h, int out_w) {
    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            for (int f = 0; f < num_filters; f++) {
                float sum = 0.0f;
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_h_idx = h + kh;
                        int in_w_idx = w + kw;
                        if (in_h_idx < in_h && in_w_idx < in_w) {
                            sum += input[in_h_idx * in_w + in_w_idx] *
                                  weights[(f * kernel_size * kernel_size) + (kh * kernel_size) + kw];
                        }
                    }
                }
                output[(h * out_w * num_filters) + (w * num_filters) + f] = relu(sum + bias[f]);
            }
        }
    }
}

bool detector_init(const char* weights_dir) {
    char filepath[256];
    FILE* fp;

    // Example for loading conv1 weights:
    snprintf(filepath, sizeof(filepath), "%s/conv1_weights.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to load weights: %s\n", filepath);
        return false;
    }

    // Allocate and read weights
    model.conv1_weights = malloc(CONV1_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * sizeof(float));
    if (!model.conv1_weights) {
        fclose(fp);
        return false;
    }
    
    if (fread(model.conv1_weights, sizeof(float), 
              CONV1_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE, fp) != 
              CONV1_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE) {
        fprintf(stderr, "Failed to read weights completely\n");
        fclose(fp);
        return false;
    }
    fclose(fp);

    // Similarly load other weights...
    model.initialized = true;
    return true;
}

DetectionResult detector_process_frame(float* spectrogram, int size) {
    if (!model.initialized) {
        return DETECTION_BACKGROUND;
    }

    // Normalize input
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        if (spectrogram[i] > max_val) max_val = spectrogram[i];
    }

    float* input = malloc(size * sizeof(float));
    if (!input) {
        return DETECTION_BACKGROUND;
    }

    for (int i = 0; i < size; i++) {
        input[i] = (max_val > 0) ? spectrogram[i] / max_val : spectrogram[i];
    }

    // Allocate memory for intermediate results
    int conv1_out_size = (model.input_height - CONV_KERNEL_SIZE + 1) * 
                        (model.input_width - CONV_KERNEL_SIZE + 1) * CONV1_FILTERS;
    float* conv1_output = malloc(conv1_out_size * sizeof(float));
    if (!conv1_output) {
        free(input);
        return DETECTION_BACKGROUND;
    }

    // Forward pass processing
    float output[2] = {0.0f, 0.0f};  // Initialize the output array

    // First convolution
    conv2d(input, model.input_height, model.input_width,
           model.conv1_weights, model.conv1_bias,
           CONV_KERNEL_SIZE, CONV1_FILTERS,
           conv1_output, model.input_height - CONV_KERNEL_SIZE + 1,
           model.input_width - CONV_KERNEL_SIZE + 1);

    // ... Rest of network processing ...

    // Apply softmax to get final probabilities
    softmax(output, 2);

    // Cleanup
    free(input);
    free(conv1_output);

    return (output[1] > 0.5) ? DETECTION_BUBBLE : DETECTION_BACKGROUND;
}

void detector_cleanup(void) {
    if (model.initialized) {
        free(model.conv1_weights);
        free(model.conv1_bias);
        free(model.conv2_weights);
        free(model.conv2_bias);
        free(model.dense1_weights);
        free(model.dense1_bias);
        free(model.dense2_weights);
        free(model.dense2_bias);
        model.initialized = false;
    }
}

void detector_get_input_dims(int* height, int* width) {
    if (height) *height = model.input_height;
    if (width) *width = model.input_width;
}