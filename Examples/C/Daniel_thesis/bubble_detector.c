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
    float* conv1_weights; // [CONV1_FILTERS, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, 1]
    float* conv1_bias;    // [CONV1_FILTERS]

    // Convolutional Layer 2
    float* conv2_weights; // [CONV2_FILTERS, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV1_FILTERS]
    float* conv2_bias;    // [CONV2_FILTERS]

    // Convolutional Layer 3
    float* conv3_weights; // [CONV3_FILTERS, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, CONV2_FILTERS]
    float* conv3_bias;    // [CONV3_FILTERS]

    // Dense Layer 1
    float* dense1_weights; // [DENSE1_UNITS, <flattened_size>]
    float* dense1_bias;    // [DENSE1_UNITS]

    // Dense Layer 2 (Output Layer)
    float* dense2_weights; // [2, DENSE1_UNITS]
    float* dense2_bias;    // [2]
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
    float max = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max) max = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        input[i] = expf(input[i] - max);
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
        fprintf(stderr, "Failed to allocate memory for weights from: %s\n", filepath);
        fclose(fp);
        return NULL;
    }
    size_t read = fread(weights, sizeof(float), count, fp);
    if (read != count) {
        fprintf(stderr, "Failed to read weights from: %s (expected %zu, got %zu)\n", filepath, count, read);
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
static char* build_filepath(const char* weights_dir, const char* filename) {
    size_t len = strlen(weights_dir) + 1 + strlen(filename) + 1; // weights_dir + "/" + filename + '\0'
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
// If mode==0 (CNN detection) this function is called with the weights directory.
// In mode==1 (save file mode) this function is never called.
//---------------------------------------------------------------------
bool detector_init(const char* weights_dir) {
    if (is_initialized) {
        fprintf(stderr, "Detector is already initialized.\n");
        return false;
    }

    // Load Conv1 weights and bias
    char* filepath = build_filepath(weights_dir, "conv1_weights.bin");
    if (!filepath) return false;
    model.conv1_weights = load_weights(filepath, CONV1_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * 1);
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
    model.conv2_weights = load_weights(filepath, CONV2_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV1_FILTERS);
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
    model.conv3_weights = load_weights(filepath, CONV3_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV2_FILTERS);
    free(filepath);
    if (!model.conv3_weights) return false;

    filepath = build_filepath(weights_dir, "conv3_bias.bin");
    if (!filepath) return false;
    model.conv3_bias = load_weights(filepath, CONV3_FILTERS);
    free(filepath);
    if (!model.conv3_bias) return false;

    // Calculate the flattened size after 3 poolings.
    // Each pooling layer reduces dimensions by a factor of POOL_SIZE.
    int pooled_height = INPUT_HEIGHT;
    int pooled_width  = INPUT_WIDTH;
    for (int i = 0; i < 3; i++) {
        pooled_height /= POOL_SIZE;
        pooled_width  /= POOL_SIZE;
    }
    int flattened_size = CONV3_FILTERS * pooled_height * pooled_width;

    // Load Dense1 weights and bias
    filepath = build_filepath(weights_dir, "dense1_weights.bin");
    if (!filepath) return false;
    model.dense1_weights = load_weights(filepath, DENSE1_UNITS * flattened_size);
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

    is_initialized = 1;
    return true;
}

//---------------------------------------------------------------------
// CNN forward-pass helper functions
//---------------------------------------------------------------------
static float* conv2d_forward(const float* input, int in_h, int in_w, int in_c,
                              const float* weights, const float* bias,
                              int kernel_size, int num_filters) {
    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;
    float* output = (float*)calloc(out_h * out_w * num_filters, sizeof(float));
    if (!output) {
        fprintf(stderr, "Failed to allocate memory for conv2d output.\n");
        return NULL;
    }
    for (int f = 0; f < num_filters; f++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float sum = 0.0f;
                for (int ki = 0; ki < kernel_size; ki++) {
                    for (int kj = 0; kj < kernel_size; kj++) {
                        for (int c = 0; c < in_c; c++) {
                            int in_idx = (i + ki) * in_w * in_c + (j + kj) * in_c + c;
                            int w_idx = f * kernel_size * kernel_size * in_c + ki * kernel_size * in_c + kj * in_c + c;
                            sum += input[in_idx] * weights[w_idx];
                        }
                    }
                }
                sum += bias[f];
                if (sum < 0) sum = 0; // ReLU activation
                int out_idx = i * out_w * num_filters + j * num_filters + f;
                output[out_idx] = sum;
            }
        }
    }
    return output;
}

static float* max_pool2d_forward(const float* input, int in_h, int in_w, int in_c,
                                 int pool_size) {
    int out_h = in_h / pool_size;
    int out_w = in_w / pool_size;
    float* output = (float*)calloc(out_h * out_w * in_c, sizeof(float));
    if (!output) {
        fprintf(stderr, "Failed to allocate memory for max_pool2d output.\n");
        return NULL;
    }
    for (int c = 0; c < in_c; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float max_val = -INFINITY;
                for (int pi = 0; pi < pool_size; pi++) {
                    for (int pj = 0; pj < pool_size; pj++) {
                        int in_i = i * pool_size + pi;
                        int in_j = j * pool_size + pj;
                        int in_idx = in_i * in_w * in_c + in_j * in_c + c;
                        if (input[in_idx] > max_val)
                            max_val = input[in_idx];
                    }
                }
                int out_idx = i * out_w * in_c + j * in_c + c;
                output[out_idx] = max_val;
            }
        }
    }
    return output;
}

static float* dense_forward(const float* input, int input_size,
                            const float* weights, const float* bias,
                            int output_size) {
    float* output = (float*)calloc(output_size, sizeof(float));
    if (!output) {
        fprintf(stderr, "Failed to allocate memory for dense_forward output.\n");
        return NULL;
    }
    for (int o = 0; o < output_size; o++) {
        float sum = bias[o];
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[o * input_size + i];
        }
        if (sum < 0) sum = 0; // ReLU activation
        output[o] = sum;
    }
    return output;
}

//---------------------------------------------------------------------
// Forward pass through the CNN
//---------------------------------------------------------------------
static float* forward_pass(const float* spectrogram) {
    // Assume spectrogram is a flattened array with shape [INPUT_HEIGHT * INPUT_WIDTH]
    // Step 1: Convolution Layer 1
    float* conv1 = conv2d_forward(spectrogram, INPUT_HEIGHT, INPUT_WIDTH, 1,
                                  model.conv1_weights, model.conv1_bias,
                                  CONV_KERNEL_SIZE, CONV1_FILTERS);
    if (!conv1) return NULL;
    // Step 2: Max Pooling Layer 1
    int conv1_out_h = INPUT_HEIGHT - CONV_KERNEL_SIZE + 1;
    int conv1_out_w = INPUT_WIDTH - CONV_KERNEL_SIZE + 1;
    float* pool1 = max_pool2d_forward(conv1, conv1_out_h, conv1_out_w, CONV1_FILTERS, POOL_SIZE);
    free(conv1);
    if (!pool1) return NULL;

    // Step 3: Convolution Layer 2
    int pool1_h = conv1_out_h / POOL_SIZE;
    int pool1_w = conv1_out_w / POOL_SIZE;
    float* conv2 = conv2d_forward(pool1, pool1_h, pool1_w, CONV1_FILTERS,
                                  model.conv2_weights, model.conv2_bias,
                                  CONV_KERNEL_SIZE, CONV2_FILTERS);
    free(pool1);
    if (!conv2) return NULL;

    // Step 4: Max Pooling Layer 2
    int conv2_out_h = pool1_h - CONV_KERNEL_SIZE + 1;
    int conv2_out_w = pool1_w - CONV_KERNEL_SIZE + 1;
    float* pool2 = max_pool2d_forward(conv2, conv2_out_h, conv2_out_w, CONV2_FILTERS, POOL_SIZE);
    free(conv2);
    if (!pool2) return NULL;

    // Step 5: Convolution Layer 3
    int pool2_h = conv2_out_h / POOL_SIZE;
    int pool2_w = conv2_out_w / POOL_SIZE;
    float* conv3 = conv2d_forward(pool2, pool2_h, pool2_w, CONV2_FILTERS,
                                  model.conv3_weights, model.conv3_bias,
                                  CONV_KERNEL_SIZE, CONV3_FILTERS);
    free(pool2);
    if (!conv3) return NULL;

    // Step 6: Max Pooling Layer 3
    int conv3_out_h = (pool2_h - CONV_KERNEL_SIZE + 1);
    int conv3_out_w = (pool2_w - CONV_KERNEL_SIZE + 1);
    float* pool3 = max_pool2d_forward(conv3, conv3_out_h, conv3_out_w, CONV3_FILTERS, POOL_SIZE);
    free(conv3);
    if (!pool3) return NULL;

    // Step 7: Flatten the output for the Dense layer
    int flattened_size = (conv3_out_h / POOL_SIZE) * (conv3_out_w / POOL_SIZE) * CONV3_FILTERS;
    float* flattened = (float*)malloc(sizeof(float) * flattened_size);
    if (!flattened) {
        fprintf(stderr, "Failed to allocate memory for flatten layer.\n");
        free(pool3);
        return NULL;
    }
    for (int i = 0; i < flattened_size; i++) {
        flattened[i] = pool3[i];
    }
    free(pool3);

    // Step 8: Dense Layer 1
    float* dense1 = dense_forward(flattened, flattened_size,
                                  model.dense1_weights, model.dense1_bias,
                                  DENSE1_UNITS);
    free(flattened);
    if (!dense1) return NULL;

    // Step 9: Dense Layer 2 (Output Layer)
    float* dense2 = dense_forward(dense1, DENSE1_UNITS,
                                  model.dense2_weights, model.dense2_bias,
                                  2);
    free(dense1);
    if (!dense2) return NULL;

    // Step 10: Apply Softmax to obtain probabilities
    softmax(dense2, 2);
    return dense2;  // Contains probabilities for [Background, Bubble]
}

//---------------------------------------------------------------------
// Process a single spectrogram frame using the CNN
//---------------------------------------------------------------------
DetectionResult detector_process_frame(float* spectrogram) {
    if (!is_initialized) {
        fprintf(stderr, "Detector is not initialized. Call detector_init first.\n");
        return DETECTION_BACKGROUND;
    }
    float* probabilities = forward_pass(spectrogram);
    if (!probabilities) {
        fprintf(stderr, "Forward pass failed.\n");
        return DETECTION_BACKGROUND;
    }
    DetectionResult result = DETECTION_BACKGROUND;
    if (probabilities[1] > 0.5f) { // Assuming index 1 corresponds to "Bubble"
        result = DETECTION_BUBBLE;
    }
    free(probabilities);
    return result;
}

//---------------------------------------------------------------------
// Cleanup: free all allocated CNN resources
//---------------------------------------------------------------------
void detector_cleanup(void) {
    if (!is_initialized) return;
    free(model.conv1_weights);
    free(model.conv1_bias);
    free(model.conv2_weights);
    free(model.conv2_bias);
    free(model.conv3_weights);
    free(model.conv3_bias);
    free(model.dense1_weights);
    free(model.dense1_bias);
    free(model.dense2_weights);
    free(model.dense2_bias);
    memset(&model, 0, sizeof(CNNModel));
    is_initialized = 0;
}

#endif // CNN_ENABLED
