#ifdef CNN_ENABLED  // Compile this file only if CNN_ENABLED is defined

#include "bubble_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

//---------------------------------------------------------------------
// Structure to hold CNN weights and biases
//---------------------------------------------------------------------
typedef struct {
    // Convolutional Layer 1
    float* conv1_weights; // [CONV1_FILTERS, CONV_KERNEL_SIZE, 
CONV_KERNEL_SIZE, 1]
    float* conv1_bias;    // [CONV1_FILTERS]

    // Convolutional Layer 2
    float* conv2_weights; // [CONV2_FILTERS, CONV_KERNEL_SIZE, 
CONV_KERNEL_SIZE, CONV1_FILTERS]
    float* conv2_bias;    // [CONV2_FILTERS]

    // Convolutional Layer 3
    float* conv3_weights; // [CONV3_FILTERS, CONV_KERNEL_SIZE, 
CONV_KERNEL_SIZE, CONV2_FILTERS]
    float* conv3_bias;    // [CONV3_FILTERS]

    // Dense Layer 1
    float* dense1_weights; // [DENSE1_UNITS, <flattened_size>]
    float* dense1_bias;    // [DENSE1_UNITS]

    // Dense Layer 2 (Output Layer)
    float* dense2_weights; // [2, DENSE1_UNITS]
    float* dense2_bias;    // [2]

    // Output buffers for intermediate layers
    float* conv2d_output_1;
    float* pool_output_1;
    float* conv2d_output_2;
    float* pool_output_2;
    float* conv2d_output_3;
    float* pool_output_3;
    float* dense_output_1;
    float* dense_output_2;
} CNNModel;

static CNNModel model = {0};
static int is_initialized = 0;

//---------------------------------------------------------------------
// Activation functions
//---------------------------------------------------------------------
static float relu(float x) {
    return x > 0 ? x : 0;
}

static void softmax(float* input, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }
    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}

//---------------------------------------------------------------------
// Helper: Load binary weights from a file
//---------------------------------------------------------------------
static float* load_weights(const char* filepath, size_t count) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open weight file: %s\n", filepath);
        return NULL;
    }
    float* weights = (float*)malloc(sizeof(float) * count);
    if (!weights) {
        fprintf(stderr, "Failed to allocate memory for weights from: 
%s\n", filepath);
        fclose(fp);
        return NULL;
    }
    size_t read_count = fread(weights, sizeof(float), count, fp);
    if (read_count != count) {
        fprintf(stderr, "Failed to read weights from: %s (expected %zu, 
got %zu)\n", filepath, count, read_count);
        free(weights);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    return weights;
}

//---------------------------------------------------------------------
// Helper: Build a full file path from the directory and filename
//---------------------------------------------------------------------
static char* build_filepath(const char* weights_dir, const char* filename) 
{
    size_t len = strlen(weights_dir) + 1 + strlen(filename) + 1;
    char* full_path = (char*)malloc(len);
    if (!full_path) {
        fprintf(stderr, "Failed to allocate memory for file path.\n");
        return NULL;
    }
    snprintf(full_path, len, "%s/%s", weights_dir, filename);
    return full_path;
}

//---------------------------------------------------------------------
// Detector initialization: load all weights from files.
//---------------------------------------------------------------------
bool detector_init(const char* weights_dir) {
    if (is_initialized) {
        fprintf(stderr, "Detector is already initialized.\n");
        return false;
    }

    // Load Conv1 weights and bias
    char* filepath = build_filepath(weights_dir, "conv1_weights.bin");
    if (!filepath) return false;
    model.conv1_weights = load_weights(filepath, CONV1_FILTERS * 
CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * 1);
    free(filepath);
    if (!model.conv1_weights) return false;

    filepath = build_filepath(weights_dir, "conv1_bias.bin");
    if (!filepath) return false;
    model.conv1_bias = load_weights(filepath, CONV1_FILTERS);
    free(filepath);
    if (!model.conv1_bias) return false;

    // Load Conv2 weights and bias
    filepath = build_filepath(weights_dir, "conv2_weights.bin");
    if (!filepath) return false;
    model.conv2_weights = load_weights(filepath, CONV2_FILTERS * 
CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV1_FILTERS);
    free(filepath);
    if (!model.conv2_weights) return false;

    filepath = build_filepath(weights_dir, "conv2_bias.bin");
    if (!filepath) return false;
    model.conv2_bias = load_weights(filepath, CONV2_FILTERS);
    free(filepath);
    if (!model.conv2_bias) return false;

    // Load Conv3 weights and bias
    filepath = build_filepath(weights_dir, "conv3_weights.bin");
    if (!filepath) return false;
    model.conv3_weights = load_weights(filepath, CONV3_FILTERS * 
CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV2_FILTERS);
    free(filepath);
    if (!model.conv3_weights) return false;

    filepath = build_filepath(weights_dir, "conv3_bias.bin");
    if (!filepath) return false;
    model.conv3_bias = load_weights(filepath, CONV3_FILTERS);
    free(filepath);
    if (!model.conv3_bias) return false;

    // Calculate dimensions after each conv/pool stage
    // Stage 1: Conv + Pool
    int h1 = INPUT_HEIGHT - CONV_KERNEL_SIZE + 1;
    int w1 = INPUT_WIDTH - CONV_KERNEL_SIZE + 1;
    int c1 = CONV1_FILTERS;
    int ph1 = h1 / POOL_SIZE, pw1 = w1 / POOL_SIZE;

    // Stage 2: Conv + Pool
    int h2 = ph1 - CONV_KERNEL_SIZE + 1;
    int w2 = pw1 - CONV_KERNEL_SIZE + 1;
    int c2 = CONV2_FILTERS;
    int ph2 = h2 / POOL_SIZE, pw2 = w2 / POOL_SIZE;

    // Stage 3: Conv + Pool
    int h3 = ph2 - CONV_KERNEL_SIZE + 1;
    int w3 = pw2 - CONV_KERNEL_SIZE + 1;
    int c3 = CONV3_FILTERS;
    int ph3 = h3 / POOL_SIZE, pw3 = w3 / POOL_SIZE;

    int flattened_size = ph3 * pw3 * c3;

    // Load Dense1 weights and bias
    filepath = build_filepath(weights_dir, "dense1_weights.bin");
    if (!filepath) return false;
    model.dense1_weights = load_weights(filepath, DENSE1_UNITS * 
flattened_size);
    free(filepath);
    if (!model.dense1_weights) return false;

    filepath = build_filepath(weights_dir, "dense1_bias.bin");
    if (!filepath) return false;
    model.dense1_bias = load_weights(filepath, DENSE1_UNITS);
    free(filepath);
    if (!model.dense1_bias) return false;

    // Load Dense2 weights and bias
    filepath = build_filepath(weights_dir, "dense2_weights.bin");
    if (!filepath) return false;
    model.dense2_weights = load_weights(filepath, 2 * DENSE1_UNITS);
    free(filepath);
    if (!model.dense2_weights) return false;

    filepath = build_filepath(weights_dir, "dense2_bias.bin");
    if (!filepath) return false;
    model.dense2_bias = load_weights(filepath, 2);
    free(filepath);
    if (!model.dense2_bias) return false;

    // Allocate output buffers
    model.conv2d_output_1 = malloc(sizeof(float) * h1 * w1 * c1);
    model.pool_output_1 = malloc(sizeof(float) * ph1 * pw1 * c1);
    model.conv2d_output_2 = malloc(sizeof(float) * h2 * w2 * c2);
    model.pool_output_2 = malloc(sizeof(float) * ph2 * pw2 * c2);
    model.conv2d_output_3 = malloc(sizeof(float) * h3 * w3 * c3);
    model.pool_output_3 = malloc(sizeof(float) * ph3 * pw3 * c3);
    model.dense_output_1 = malloc(sizeof(float) * DENSE1_UNITS);
    model.dense_output_2 = malloc(sizeof(float) * 2);

    // Check for allocation failure
    if (!model.conv2d_output_1 || !model.pool_output_1 ||
        !model.conv2d_output_2 || !model.pool_output_2 ||
        !model.conv2d_output_3 || !model.pool_output_3 ||
        !model.dense_output_1 || !model.dense_output_2) {
        fprintf(stderr, "Failed to allocate CNN buffers\n");
        detector_cleanup();
        return false;
    }

    is_initialized = 1;
    return true;
}

//---------------------------------------------------------------------
// CNN forward-pass helper functions
//---------------------------------------------------------------------
static void conv2d_forward(
    float *output,
    const float *input,
    int in_h,
    int in_w,
    int in_c,
    const float *weights,
    const float* bias,
    int kernel_size,
    int num_filters
) {
    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;
    
    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float sum = 0.0f;
                for (int c = 0; c < in_c; c++) {
                    for (int ki = 0; ki < kernel_size; ki++) {
                        for (int kj = 0; kj < kernel_size; kj++) {
                            int in_idx = (i + ki) * in_w * in_c + (j + kj) 
* in_c + c;
                            int w_idx = f * (kernel_size * kernel_size * 
in_c) +
                                       ki * (kernel_size * in_c) +
                                       kj * in_c + c;
                            sum += input[in_idx] * weights[w_idx];
                        }
                    }
                }
                sum += bias[f];
                output[i * out_w * num_filters + j * num_filters + f] = 
relu(sum);
            }
        }
    }
}

static void max_pool2d_forward(const float* input, int in_h, int in_w, int 
in_c,
                              int pool_size, float* output) {
    int out_h = in_h / pool_size;
    int out_w = in_w / pool_size;
    
    for (int c = 0; c < in_c; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float max_val = -INFINITY;
                for (int pi = 0; pi < pool_size; pi++) {
                    for (int pj = 0; pj < pool_size; pj++) {
                        int in_i = i * pool_size + pi;
                        int in_j = j * pool_size + pj;
                        int in_idx = (in_i * in_w + in_j) * in_c + c;
                        if (input[in_idx] > max_val) {
                            max_val = input[in_idx];
                        }
                    }
                }
                int out_idx = (i * out_w + j) * in_c + c;
                output[out_idx] = max_val;
            }
        }
    }
}

static void dense_forward(
    const float* input, 
    int input_size,
    const float* weights,
    const float* bias,
    int output_size,
    bool apply_relu,
    float* output
) {
    for (int o = 0; o < output_size; o++) {
        float sum = bias[o];
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[o * input_size + i];
        }
        output[o] = apply_relu ? relu(sum) : sum;
    }
}

//---------------------------------------------------------------------
// Forward pass through the CNN
//---------------------------------------------------------------------
static float* forward_pass(float* spectrogram) {
    int h = INPUT_HEIGHT, w = INPUT_WIDTH, c = 1;

    // Layer 1: Conv2D + ReLU + MaxPool
    conv2d_forward(model.conv2d_output_1, spectrogram, h, w, c,
                   model.conv1_weights, model.conv1_bias,
                   CONV_KERNEL_SIZE, CONV1_FILTERS);
    h = h - CONV_KERNEL_SIZE + 1;
    w = w - CONV_KERNEL_SIZE + 1;
    c = CONV1_FILTERS;

    max_pool2d_forward(model.conv2d_output_1, h, w, c, POOL_SIZE, 
model.pool_output_1);
    h /= POOL_SIZE;
    w /= POOL_SIZE;

    // Layer 2: Conv2D + ReLU + MaxPool
    conv2d_forward(model.conv2d_output_2, model.pool_output_1, h, w, c,
                   model.conv2_weights, model.conv2_bias,
                   CONV_KERNEL_SIZE, CONV2_FILTERS);
    h = h - CONV_KERNEL_SIZE + 1;
    w = w - CONV_KERNEL_SIZE + 1;
    c = CONV2_FILTERS;

    max_pool2d_forward(model.conv2d_output_2, h, w, c, POOL_SIZE, 
model.pool_output_2);
    h /= POOL_SIZE;
    w /= POOL_SIZE;

    // Layer 3: Conv2D + ReLU + MaxPool
    conv2d_forward(model.conv2d_output_3, model.pool_output_2, h, w, c,
                   model.conv3_weights, model.conv3_bias,
                   CONV_KERNEL_SIZE, CONV3_FILTERS);
    h = h - CONV_KERNEL_SIZE + 1;
    w = w - CONV_KERNEL_SIZE + 1;
    c = CONV3_FILTERS;

    max_pool2d_forward(model.conv2d_output_3, h, w, c, POOL_SIZE, 
model.pool_output_3);
    h /= POOL_SIZE;
    w /= POOL_SIZE;

    // Dense layers
    int flattened_size = h * w * c;
    dense_forward(model.pool_output_3, flattened_size,
                  model.dense1_weights, model.dense1_bias,
                  DENSE1_UNITS, true, model.dense_output_1);

    dense_forward(model.dense_output_1, DENSE1_UNITS,
                  model.dense2_weights, model.dense2_bias,
                  2, false, model.dense_output_2);

    // Apply softmax
    softmax(model.dense_output_2, 2);
    return model.dense_output_2;
}

//---------------------------------------------------------------------
// Process a single spectrogram frame using the CNN
//---------------------------------------------------------------------
DetectionResult detector_process_frame(float* spectrogram) {
    if (!is_initialized) {
        fprintf(stderr, "Detector is not initialized. Call detector_init 
first.\n");
        return DETECTION_BACKGROUND;
    }
    
    float* probabilities = forward_pass(spectrogram);
    if (!probabilities) {
        fprintf(stderr, "Forward pass failed.\n");
        return DETECTION_BACKGROUND;
    }
    
    DetectionResult result = DETECTION_BACKGROUND;
    if (probabilities[1] > 0.5f) {
        result = DETECTION_BUBBLE;
    }
    return result;
}

//---------------------------------------------------------------------
// Cleanup: free all allocated CNN resources
//---------------------------------------------------------------------
void detector_cleanup(void) {
    if (!is_initialized) return;
    
    // Free weights and biases
    free(model.conv1_weights); model.conv1_weights = NULL;
    free(model.conv1_bias);    model.conv1_bias = NULL;
    free(model.conv2_weights); model.conv2_weights = NULL;
    free(model.conv2_bias);    model.conv2_bias = NULL;
    free(model.conv3_weights); model.conv3_weights = NULL;
    free(model.conv3_bias);    model.conv3_bias = NULL;
    free(model.dense1_weights); model.dense1_weights = NULL;
    free(model.dense1_bias);   model.dense1_bias = NULL;
    free(model.dense2_weights); model.dense2_weights = NULL;
    free(model.dense2_bias);   model.dense2_bias = NULL;
    
    // Free output buffers
    free(model.conv2d_output_1); model.conv2d_output_1 = NULL;
    free(model.pool_output_1);   model.pool_output_1 = NULL;
    free(model.conv2d_output_2); model.conv2d_output_2 = NULL;
    free(model.pool_output_2);   model.pool_output_2 = NULL;
    free(model.conv2d_output_3); model.conv2d_output_3 = NULL;
    free(model.pool_output_3);   model.pool_output_3 = NULL;
    free(model.dense_output_1);  model.dense_output_1 = NULL;
    free(model.dense_output_2);  model.dense_output_2 = NULL;

    memset(&model, 0, sizeof(CNNModel));
    is_initialized = 0;
}

#endif // CNN_ENABLED
