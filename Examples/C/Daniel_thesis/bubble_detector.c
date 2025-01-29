#ifdef CNN_ENABLED  // Only compile CNN code if this flag is set

#include "bubble_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_HEIGHT 257  // FFT_SIZE/2 + 1
#define INPUT_WIDTH 1     // Single time frame
#define POOL1_HEIGHT (INPUT_HEIGHT/POOL_SIZE)
#define POOL1_WIDTH (INPUT_WIDTH/POOL_SIZE)
#define POOL2_HEIGHT (POOL1_HEIGHT/POOL_SIZE)
#define POOL2_WIDTH (POOL1_WIDTH/POOL_SIZE)
#define POOL3_HEIGHT (POOL2_HEIGHT/POOL_SIZE)
#define POOL3_WIDTH (POOL2_WIDTH/POOL_SIZE)
#define CONV1_SIZE ((INPUT_HEIGHT - CONV_KERNEL_SIZE + 1) * (INPUT_WIDTH - CONV_KERNEL_SIZE + 1) * CONV1_FILTERS)
#define POOL1_SIZE (CONV1_SIZE / (POOL_SIZE * POOL_SIZE))
#define CONV2_SIZE ((POOL1_SIZE - CONV_KERNEL_SIZE + 1) * (1 - CONV_KERNEL_SIZE + 1) * CONV2_FILTERS)
#define POOL2_SIZE (CONV2_SIZE / (POOL_SIZE * POOL_SIZE))
#define CONV3_SIZE ((POOL2_HEIGHT - CONV_KERNEL_SIZE + 1) * (POOL2_WIDTH - CONV_KERNEL_SIZE + 1) * CONV3_FILTERS)
#define POOL3_SIZE ((POOL3_HEIGHT) * (POOL3_WIDTH))
#ifndef CNN_MODE
#define CNN_MODE 1  // Set to 0 to disable CNN during compilation
#endif

static struct {
    float* conv1_weights;
    float* conv1_bias;
    float* conv2_weights;
    float* conv2_bias;
    float* conv3_weights;
    float* conv3_bias;
    float* dense1_weights;
    float* dense1_bias;
    float* dense2_weights;
    float* dense2_bias;
    bool initialized;
} model = {0};

static float relu(float x) {
    return x > 0 ? x : 0;
}

static void softmax(float* input, int size) {
    float max_val = input[0];
    for(int i = 1; i < size; i++) {
        if(input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0.0f;
    for(int i = 0; i < size; i++) {
        float val = expf(input[i] - max_val);
        input[i] = val;
        sum += val;
    }
    
    for(int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

static void conv2d(const float* input, int in_h, int in_w, int in_c,
                  const float* weights, const float* bias,
                  int kernel_size, int num_filters,
                  float* output) {
    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;
    
    for(int h = 0; h < out_h; h++) {
        for(int w = 0; w < out_w; w++) {
            for(int f = 0; f < num_filters; f++) {
                float sum = 0.0f;
                for(int kh = 0; kh < kernel_size; kh++) {
                    for(int kw = 0; kw < kernel_size; kw++) {
                        for(int c = 0; c < in_c; c++) {
                            int in_idx = ((h+kh) * in_w + (w+kw)) * in_c + c;
                            int w_idx = (f * kernel_size * kernel_size * in_c) + 
                                      (kh * kernel_size * in_c) + 
                                      (kw * in_c) + c;
                            sum += input[in_idx] * weights[w_idx];
                        }
                    }
                }
                int out_idx = (h * out_w + w) * num_filters + f;
                output[out_idx] = relu(sum + bias[f]);
            }
        }
    }
}

static void max_pool_2d(const float* input, int in_h, int in_w, int channels,
                       float* output) {
    int out_h = in_h / POOL_SIZE;
    int out_w = in_w / POOL_SIZE;
    
    for(int h = 0; h < out_h; h++) {
        for(int w = 0; w < out_w; w++) {
            for(int c = 0; c < channels; c++) {
                float max_val = -INFINITY;
                for(int ph = 0; ph < POOL_SIZE; ph++) {
                    for(int pw = 0; pw < POOL_SIZE; pw++) {
                        int in_idx = ((h*POOL_SIZE + ph) * in_w * channels) +
                                   ((w*POOL_SIZE + pw) * channels) + c;
                        if(input[in_idx] > max_val) {
                            max_val = input[in_idx];
                        }
                    }
                }
                output[(h * out_w * channels) + (w * channels) + c] = max_val;
            }
        }
    }
}

static void dense_layer(const float* input, int input_size,
                       const float* weights, const float* bias,
                       int output_size, float* output) {
    for(int i = 0; i < output_size; i++) {
        float sum = bias[i];
        for(int j = 0; j < input_size; j++) {
            sum += input[j] * weights[i * input_size + j];
        }
        output[i] = relu(sum);
    }
}

bool detector_init(const char* weights_dir) {
    char filepath[256];
    FILE* fp;
    
    // Load conv1 weights
    snprintf(filepath, sizeof(filepath), "%s/conv1_weights.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if(!fp) {
        fprintf(stderr, "Failed to load weights: %s\n", filepath);
        return false;
    }
    
    model.conv1_weights = malloc(CONV1_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * sizeof(float));
    if(!model.conv1_weights) {
        fclose(fp);
        return false;
    }
    
    if(fread(model.conv1_weights, sizeof(float), 
             CONV1_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE, fp) != 
             CONV1_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE) {
        fclose(fp);
        return false;
    }
    fclose(fp);
    
    // Load conv1 bias
    snprintf(filepath, sizeof(filepath), "%s/conv1_bias.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if(!fp) {
        detector_cleanup();
        return false;
    }
    
    model.conv1_bias = malloc(CONV1_FILTERS * sizeof(float));
    if(!model.conv1_bias) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    
    if(fread(model.conv1_bias, sizeof(float), CONV1_FILTERS, fp) != CONV1_FILTERS) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    fclose(fp);

    // Load conv2 weights
    snprintf(filepath, sizeof(filepath), "%s/conv2_weights.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if(!fp) {
        detector_cleanup();
        return false;
    }
    
    model.conv2_weights = malloc(CONV2_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV1_FILTERS * sizeof(float));
    if(!model.conv2_weights) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    
    if(fread(model.conv2_weights, sizeof(float), 
             CONV2_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV1_FILTERS, fp) != 
             CONV2_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV1_FILTERS) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    fclose(fp);

    // Load conv2 bias
    snprintf(filepath, sizeof(filepath), "%s/conv2_bias.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if(!fp) {
        detector_cleanup();
        return false;
    }
    
    model.conv2_bias = malloc(CONV2_FILTERS * sizeof(float));
    if(!model.conv2_bias) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    
    if(fread(model.conv2_bias, sizeof(float), CONV2_FILTERS, fp) != CONV2_FILTERS) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    fclose(fp);

    // Load conv3 weights
    snprintf(filepath, sizeof(filepath), "%s/conv3_weights.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if(!fp) {
        detector_cleanup();
        return false;
    }
    
    model.conv3_weights = malloc(CONV3_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV2_FILTERS * sizeof(float));
    if(!model.conv3_weights) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    
    if(fread(model.conv3_weights, sizeof(float), 
             CONV3_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV2_FILTERS, fp) != 
             CONV3_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV2_FILTERS) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    fclose(fp);

    // Load conv3 bias
    snprintf(filepath, sizeof(filepath), "%s/conv3_bias.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if(!fp) {
        detector_cleanup();
        return false;
    }
    
    model.conv3_bias = malloc(CONV3_FILTERS * sizeof(float));
    if(!model.conv3_bias) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    
    if(fread(model.conv3_bias, sizeof(float), CONV3_FILTERS, fp) != CONV3_FILTERS) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    fclose(fp);

    // Load dense1 weights
    snprintf(filepath, sizeof(filepath), "%s/dense1_weights.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if(!fp) {
        detector_cleanup();
        return false;
    }
    
    int flattened_size = POOL3_SIZE * CONV3_FILTERS;
    model.dense1_weights = malloc(DENSE1_UNITS * flattened_size * sizeof(float));
    if(!model.dense1_weights) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    
    if(fread(model.dense1_weights, sizeof(float), DENSE1_UNITS * flattened_size, fp) != 
             DENSE1_UNITS * flattened_size) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    fclose(fp);

    // Load dense1 bias
    snprintf(filepath, sizeof(filepath), "%s/dense1_bias.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if(!fp) {
        detector_cleanup();
        return false;
    }
    
    model.dense1_bias = malloc(DENSE1_UNITS * sizeof(float));
    if(!model.dense1_bias) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    
    if(fread(model.dense1_bias, sizeof(float), DENSE1_UNITS, fp) != DENSE1_UNITS) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    fclose(fp);

    // Load dense2 weights
    snprintf(filepath, sizeof(filepath), "%s/dense2_weights.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if(!fp) {
        detector_cleanup();
        return false;
    }
    
    model.dense2_weights = malloc(2 * DENSE1_UNITS * sizeof(float));
    if(!model.dense2_weights) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    
    if(fread(model.dense2_weights, sizeof(float), 2 * DENSE1_UNITS, fp) != 2 * DENSE1_UNITS) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    fclose(fp);

    // Load dense2 bias
    snprintf(filepath, sizeof(filepath), "%s/dense2_bias.bin", weights_dir);
    fp = fopen(filepath, "rb");
    if(!fp) {
        detector_cleanup();
        return false;
    }
    
    model.dense2_bias = malloc(2 * sizeof(float));
    if(!model.dense2_bias) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    
    if(fread(model.dense2_bias, sizeof(float), 2, fp) != 2) {
        fclose(fp);
        detector_cleanup();
        return false;
    }
    fclose(fp);
    
    model.initialized = true;
    return true;
}

DetectionResult detector_process_frame(float* spectrogram, int size) {
    if(!model.initialized || size != INPUT_HEIGHT) {
        return DETECTION_BACKGROUND;
    }
    
    float* conv1_output = malloc(CONV1_SIZE * sizeof(float));
    float* pool1_output = malloc(POOL1_SIZE * sizeof(float));
    float* conv2_output = malloc(CONV2_SIZE * sizeof(float));
    float* pool2_output = malloc(POOL2_SIZE * sizeof(float));
    float* conv3_output = malloc(CONV3_SIZE * sizeof(float));
    float* pool3_output = malloc(POOL3_SIZE * sizeof(float));
    float* dense1_output = malloc(DENSE1_UNITS * sizeof(float));
    float* final_output = malloc(2 * sizeof(float));
    
    if(!conv1_output || !pool1_output || !conv2_output || !pool2_output ||
       !conv3_output || !pool3_output || !dense1_output || !final_output) {
        free(conv1_output);
        free(pool1_output);
        free(conv2_output);
        free(pool2_output);
        free(conv3_output);
        free(pool3_output);
        free(dense1_output);
        free(final_output);
        return DETECTION_BACKGROUND;
    }
    
    // Normalize input
    float max_val = 0.0f;
    for(int i = 0; i < size; i++) {
        if(spectrogram[i] > max_val) max_val = spectrogram[i];
    }
    if(max_val > 0) {
        for(int i = 0; i < size; i++) {
            spectrogram[i] /= max_val;
        }
    }
    
    // Forward pass through network
    // Conv1 + Pool1
    conv2d(spectrogram, INPUT_HEIGHT, INPUT_WIDTH, 1,
           model.conv1_weights, model.conv1_bias,
           CONV_KERNEL_SIZE, CONV1_FILTERS, conv1_output);
    
    max_pool_2d(conv1_output, 
                INPUT_HEIGHT - CONV_KERNEL_SIZE + 1,
                INPUT_WIDTH - CONV_KERNEL_SIZE + 1,
                CONV1_FILTERS, pool1_output);
    
    // Conv2 + Pool2
    conv2d(pool1_output, POOL1_HEIGHT, POOL1_WIDTH, CONV1_FILTERS,
           model.conv2_weights, model.conv2_bias,
           CONV_KERNEL_SIZE, CONV2_FILTERS, conv2_output);
    
    max_pool_2d(conv2_output,
                POOL1_HEIGHT - CONV_KERNEL_SIZE + 1,
                POOL1_WIDTH - CONV_KERNEL_SIZE + 1,
                CONV2_FILTERS, pool2_output);
    
    // Conv3 + Pool3
    conv2d(pool2_output, POOL2_HEIGHT, POOL2_WIDTH, CONV2_FILTERS,
           model.conv3_weights, model.conv3_bias,
           CONV_KERNEL_SIZE, CONV3_FILTERS, conv3_output);
    
    max_pool_2d(conv3_output,
                POOL2_HEIGHT - CONV_KERNEL_SIZE + 1,
                POOL2_WIDTH - CONV_KERNEL_SIZE + 1,
                CONV3_FILTERS, pool3_output);
    
    // Dense1 (fully connected layer)
    int flattened_size = POOL3_SIZE * CONV3_FILTERS;
    dense_layer(pool3_output, flattened_size,
                model.dense1_weights, model.dense1_bias,
                DENSE1_UNITS, dense1_output);
    
    // Dense2 (output layer)
    dense_layer(dense1_output, DENSE1_UNITS,
                model.dense2_weights, model.dense2_bias,
                2, final_output);
    
    // Apply softmax to get probabilities
    softmax(final_output, 2);
    
    // Get prediction
    DetectionResult result = (final_output[1] > 0.5f) ? DETECTION_BUBBLE : DETECTION_BACKGROUND;
    
    // Cleanup
    free(conv1_output);
    free(pool1_output);
    free(conv2_output);
    free(pool2_output);
    free(conv3_output);
    free(pool3_output);
    free(dense1_output);
    free(final_output);
    
    return result;
}

void detector_cleanup(void) {
    if(model.initialized) {
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
        model.initialized = false;
    }
}

#endif  // End of CNN_ENABLED
