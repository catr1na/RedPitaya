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

    // output from conv2d calls
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
    float max_val = input[0]; // Corrected variable name from 'max' to 'max_val' to avoid conflict if math.h defines max as a macro
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
        fprintf(stderr, "Failed to allocate memory for weights from: %s\n", filepath);
        fclose(fp);
        return NULL;
    }
    size_t read_count = fread(weights, sizeof(float), count, fp); // Corrected variable name from 'read' to 'read_count'
    if (read_count != count) {
        fprintf(stderr, "Failed to read weights from: %s (expected %zu, got %zu)\n", filepath, count, read_count);
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
//---------------------------------------------------------------------
bool detector_init(const char* weights_dir) {
    if (is_initialized) {
        fprintf(stderr, "Detector is already initialized.\n");
        return false; // Return false, not true, if already initialized and not re-initializing
    }

    // Load Conv1 weights and bias
    char* filepath = build_filepath(weights_dir, "conv1_weights.bin");
    if (!filepath) return false;
    // Weights for Conv1: CONV1_FILTERS output channels, 1 input channel, KxK kernel
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
    // Weights for Conv2: CONV2_FILTERS output channels, CONV1_FILTERS input channels, KxK kernel
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
    // Weights for Conv3: CONV3_FILTERS output channels, CONV2_FILTERS input channels, KxK kernel
    model.conv3_weights = load_weights(filepath, CONV3_FILTERS * CONV_KERNEL_SIZE * CONV_KERNEL_SIZE * CONV2_FILTERS);
    free(filepath);
    if (!model.conv3_weights) return false;

    filepath = build_filepath(weights_dir, "conv3_bias.bin");
    if (!filepath) return false;
    model.conv3_bias = load_weights(filepath, CONV3_FILTERS);
    free(filepath);
    if (!model.conv3_bias) return false;

    // ******************************************************************
    // ** CORRECTED FLATTENED_SIZE CALCULATION **
    // Calculate the flattened size after 3 conv/pool stages.
    // This matches the Keras model output: (None, 62, 3, 256) -> flattened 47616
    // if CONV3_FILTERS is 256.
    // ******************************************************************
    // BELOW: DANIEL'S H W AND C CALCULATIONS
    //int current_h = INPUT_HEIGHT;
    //int current_w = INPUT_WIDTH;
    //int k_size = CONV_KERNEL_SIZE; // Kernel size
    //int p_size = POOL_SIZE;    // Pool size

    // Stage 1
    //current_h = current_h - k_size + 1; // After Conv1
    //current_w = current_w - k_size + 1;
    //current_h /= p_size;                // After Pool1
    //current_w /= p_size;

    // Stage 2
    //current_h = current_h - k_size + 1; // After Conv2
    //current_w = current_w - k_size + 1;
    //current_h /= p_size;                // After Pool2
    //current_w /= p_size;

    // Stage 3
    //current_h = current_h - k_size + 1; // After Conv3
    //current_w = current_w - k_size + 1;
    //current_h /= p_size;                // After Pool3
    //current_w /= p_size;
    // Now current_h = 62, current_w = 3 with INPUT_HEIGHT=513, INPUT_WIDTH=38, K=3, P=2

    //int flattened_size = current_h * current_w * CONV3_FILTERS;
    // If CONV3_FILTERS is 256 (as per corrected .h file):
    // flattened_size = 62 * 3 * 256 = 47616. This matches model_info.txt.

    // *********  BELOW: MY OWN CODE TO COMPUTE INTERMEDIATE H W AND C //
    //stage 1
    int h1 = INPUT_HEIGHT - CONV_KERNEL_SIZE + 1;
    int w1 = INPUT_WIDTH - CONV_KERNEL_SIZE + 1;
    int c1 = CONV1_FILTERS;
    int ph1 = h1 / POOL_SIZE, pw1 = w1 / POOL_SIZE;

    // stage 2
    int h2 = ph1 - CONV_KERNEL_SIZE + 1;
    int w2 = pw1 - CONV_KERNEL_SIZE + 1;
    int c2 = CONV2_FILTERS;
    int ph2 = h2 / POOL_SIZE, pw2 = w2 / POOL_SIZE;

    // stage 3
    int h3 = ph2 - CONV_KERNEL_SIZE + 1;
    int w3 = pw2 - CONV_KERNEL_SIZE + 1;
    int c3 = CONV3_FILTERS;
    int ph3 = h3 / POOL_SIZE, pw3 = w3 / POOL_SIZE;

    int flattened_size = ph3 * pw3 * c3;
        


    
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
    model.dense2_weights = load_weights(filepath, 2 * DENSE1_UNITS); // Output units = 2
    free(filepath);
    if (!model.dense2_weights) return false;

    filepath = build_filepath(weights_dir, "dense2_bias.bin");
    if (!filepath) return false;
    model.dense2_bias = load_weights(filepath, 2); // Output units = 2
    free(filepath);
    if (!model.dense2_bias) return false;

    model.conv2d_output_1 = malloc(sizeof(float) * h1 * w1 * c1);
    model.pool_output_1  = malloc(sizeof(float) * ph1 * pw1 * c1);

    model.conv2d_output_2 = malloc(sizeof(float) * h2 * w2 * c2);
    model.pool_output_2  = malloc(sizeof(float) * ph2 * pw2 * c2);

    model.conv2d_output_3 = malloc(sizeof(float) * h3 * w3 * c3);
    model.pool_output_3  = malloc(sizeof(float) * ph3 * pw3 * c3);

    model.dense_output_1 = malloc(sizeof(float) * DENSE1_UNITS);
    model.dense_output_2 = malloc(sizeof(float) * 2);

      // 3) Check for allocation failure
    if (!model.conv2d_output_1 || !model.pool_output_1  ||
        !model.conv2d_output_2 || !model.pool_output_2  ||
        !model.conv2d_output_3 || !model.pool_output_3  ||
        !model.dense_output_1 || !model.dense_output_2) {
        fprintf(stderr, "Failed to alloc CNN buffers\n");
        // free any non-NULL pointers here before returning…
        return false;
    }
    
    is_initialized = 1;
    return true;
}

//---------------------------------------------------------------------
// CNN forward-pass helper functions
//---------------------------------------------------------------------
static void conv2d_forward(
    float *output
    const float *input,
    int in_h,
    int in_w,
    int in_c,
    const float *weights,
    const float* bias,
    int kernel_size,
    int num_filters,
)
{
    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;
    // DC: float* output = (float*)malloc(out_h * out_w * num_filters, sizeof(float));
   // DC: if (!output) {
       //  DC: fprintf(stderr, "Failed to allocate memory for conv2d output.\n");
        // DC: return NULL;
   // DC: }
    // Weight layout: [filter_out, kernel_h, kernel_w, filter_in]
    // Input/Output layout: HWC (Height, Width, Channels)
    for (int f = 0; f < num_filters; f++) {        // Output filter
        for (int i = 0; i < out_h; i++) {          // Output height
            for (int j = 0; j < out_w; j++) {      // Output width
                float sum = 0.0f;
                for (int c = 0; c < in_c; c++) {   // Input channel
                    for (int ki = 0; ki < kernel_size; ki++) { // Kernel height
                        for (int kj = 0; kj < kernel_size; kj++) { // Kernel width
                            int in_idx = (i + ki) * in_w * in_c + (j + kj) * in_c + c;
                            // Corrected weight indexing to match typical [f_out][kH][kW][f_in]
                            int w_idx = f * (kernel_size * kernel_size * in_c) + // Offset for current output filter
                                        ki * (kernel_size * in_c) +            // Offset for kernel row
                                        kj * in_c + c;                          // Offset for kernel + input??                                      // Offset for input channel
                            sum += input[in_idx] * weights[w_idx];
                        }
                    }
                }
                sum += bias[f];
                output[i * out_w * num_filters + j * num_filters + f] = relu(sum); // Apply ReLU
            }
        }
    } //return output?
}

static void max_pool2d_forward(const float* input, int in_h, int in_w, int in_c,
                                 int pool_size, float*output)
{
    // Assuming stride = pool_size for non-overlapping pooling
    int out_h = in_h / pool_size;
    int out_w = in_w / pool_size;
    //DC: float* output = (float*)malloc(out_h * out_w * in_c, sizeof(float));
    //DC: if (!output) {
        //DC: fprintf(stderr, "Failed to allocate memory for max_pool2d output.\n");
        //return NULL;
    //}
    for (int c = 0; c < in_c; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float max_val = -INFINITY;
                for (int pi = 0; pi < pool_size; pi++) {
                    for (int pj = 0; pj < pool_size; pj++) {
                    int in_i = i * pool_size + pi;
                    int in_j = j * pool_size + pj;

                    // compute input index (assuming channel-last)
                    int in_idx = (in_i * in_w + in_j) * in_c + c;
                    if (input[in_idx] > max_val) {
                        max_val = input[in_idx];
                    }
                }
            }

            // compute output index (channel-last)
            int out_idx = (i * out_w + j) * in_c + c;
            output[out_idx] = max_val;
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
 // Added apply_relu flag
    //float* output = (float*)malloc(output_size, sizeof(float));
    //if (!output) {
       // fprintf(stderr, "Failed to allocate memory for dense_forward output.\n");
        //return NULL;
    //}
    // Weight layout: [output_unit, input_unit]
    for (int o = 0; o < output_size; o++) {
        float sum = bias[o];
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[o * input_size + i];
        }
       // if (apply_relu) {
        output[o] = apply_relu ? relu(sum) : sum;
        } //else {
            //output[o] = sum;
    }
}

//---------------------------------------------------------------------
// Forward pass through the CNN
//---------------------------------------------------------------------
static float* forward_pass(float* spectrogram) {
    int h = INPUT_HEIGHT, w = INPUT_WIDTH, c = 1;
    // Assume spectrogram is a flattened array with shape [INPUT_HEIGHT * INPUT_WIDTH], representing 1 channel input
    // Layer dimensions will be taken from bubble_detector.h constants

    // Reshape/Treat spectrogram as [INPUT_HEIGHT, INPUT_WIDTH, 1]
    //const float* current_input = spectrogram;
   // int h = INPUT_HEIGHT;
    //int w = INPUT_WIDTH;
    //int c = 1; // Initial input channels

    // Layer 1: Conv2D + ReLU
    //float* conv1_out = model.conv2d_output_1;
    conv2d_forward(spectrogram, h, w, c,
                   model.conv1_weights, model.conv1_bias,
                   CONV_KERNEL_SIZE, CONV1_FILTERS, model.conv2d_output_1);
    //if (!conv1_out) return NULL;
    h = h - CONV_KERNEL_SIZE + 1; // Update height
    w = w - CONV_KERNEL_SIZE + 1; // Update width
    c = CONV1_FILTERS;          // Update channels

    max_pool2d_forward(model.conv2d_output_1, h, w, c, POOL_SIZE, model.pool_output_1);

    // free(conv1_out);
    //if (!pool1_out) return NULL;
    h /= POOL_SIZE; // Update height
    w /= POOL_SIZE; // Update width
    // 'c' (channels) remains CONV1_FILTERS

    // Layer 2:
    // Layer 2: conv → pool
    conv2d_forward(
        model.pool_output_1,      // 2️⃣ input buffer
        h,                        // 3️⃣ in_h
        w,                        // 4️⃣ in_w
        c,                        // 5️⃣ in_c
        model.conv2_weights,      // 6️⃣ weights
        model.conv2_bias,         // 7️⃣ bias
        CONV_KERNEL_SIZE,         // 8️⃣ kernel_size
        CONV2_FILTERS             // 9️⃣ num_filters
        model.conv2d_output_2,    // 1️⃣ output buffer

    );

// update spatial dims & channels
    h = h - CONV_KERNEL_SIZE + 1;
    w = w - CONV_KERNEL_SIZE + 1;
    c = CONV2_FILTERS;

// now max‐pool
    max_pool2d_forward(
        model.conv2d_output_2,    // 2️⃣ input buffer
        h,                        // 3️⃣ in_h
        w,                        // 4️⃣ in_w
        c,                        // 5️⃣ in_c
        POOL_SIZE,                // 6️⃣ pool_size
        model.pool_output_2       // 7️⃣ output ptr
    );
    h /= POOL_SIZE;
    w /= POOL_SIZE;
 //LAYER 3:
    
    conv2d_forward(
        model.conv2d_output_3,
        model.pool_output_2,
        h, w, c,
        model.conv3_weights,
        model.conv3_bias,
        CONV_KERNEL_SIZE,
        CONV3_FILTERS,
    );
    h = h - CONV_KERNEL_SIZE + 1;
    w = w - CONV_KERNEL_SIZE + 1;
    c = CONV3_FILTERS;

    max_pool2d_forward(
        model.conv2d_output_3,
        h, w, c,
        POOL_SIZE,
        model.pool_output_3
    );
    h /= POOL_SIZE;
    w /= POOL_SIZE;

    // Layer 4: MaxPooling2D
    // 'c' (channels) remains CONV3_FILTERS
    max_pool2d_forward(
        model.conv2d_output_4,
        h, w, c,
        POOL_SIZE,
        model.pool_output_3
    );
    h /= POOL_SIZE;
    w /= POOL_SIZE;

    // Layer 5: Conv2D + ReLU
   // float* conv3_out = model.conv2d_output_3;
    conv2d_forward(
        model.conv2d_output_5,
        model.pool_outtput_4,
        h, w, c,
        model.conv5_weights, model.conv5_bias,
        CONV_KERNEL_SIZE, CONV3_FILTERS,
        model.conv2d_output_3
    );
    h = h - CONV_KERNEL_SIZE + 1;
    w = w - CONV_KERNEL_SIZE + 1;
    c = CONV5_FILTERS;
    
    //conv3_out, pool2_out, h, w, c, // 'c' is CONV2_FILTERS
                 //  model.conv3_weights, model.conv3_bias,
                   //CONV_KERNEL_SIZE, CONV3_FILTERS);
    //free(pool2_out);
    //if (!conv3_out) return NULL;

    // Layer 6: MaxPooling2D
    //float* pool3_out = max_pool2d_forward(conv3_out, h, w, c, POOL_SIZE);
    // free(conv3_out);
    //if (!pool3_out) return NULL;
    //h /= POOL_SIZE; // h should now be 62
    //w /= POOL_SIZE; // w should now be 3
    // 'c' (channels) remains CONV3_FILTERS
    max_pool2d_forward(
        model.conv2d_output_6,
        h, w, c,
        POOL_SIZE,
        model.pool_output_5
    );
    h /= POOL_SIZE;
    w /= POOL_SIZE;

    // Layer 7: Flatten (pool3_out is already HWC, effectively flattened by treating as 1D array)
    int flattened_size = h * w * c; // This should be 62 * 3 * CONV3_FILTERS
                                    // e.g., 62 * 3 * 256 = 47616

    // Layer 8: Dense + ReLU
     dense_forward(model.pool_output_3, h*w*c,
         model.dense1_weights, model.dense1_bias,
         DENSE1_UNITS, true,
         model.dense_output_1);
    
                                      //model.dense1_weights, model.dense1_bias,
                                      //DENSE1_UNITS, true); // Apply ReLU
  //  free(pool3_out); // pool3_out was used as the flattened input
   // if (!dense1_out) return NULL;

    // Layer 9: Dense (Output Layer, no ReLU here, Softmax applied later)
    dense_forward(model.dense_output_1, DENSE1_UNITS, model.dense2_weights, 
        model.dense2_bias, 2, false, model.dense_output_2); // No ReLU for output layer before Softmax
//    free(dense1_out);
  //  if (!dense2_out) return NULL;

    // Layer 10: Softmax
    softmax(model.dense_output_2, 2);
    return model.dense_output_2; // Contains probabilities for [Background, Bubble]
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
    // Assuming index 1 corresponds to "Bubble" and index 0 to "Background"
    // A threshold of 0.5 is common after softmax for binary classification,
    // but model_info.txt used 0.9f. Sticking to a common default unless specified.
    if (probabilities[1] > 0.5f) {
        result = DETECTION_BUBBLE;
    }
    return result;
}

//---------------------------------------------------------------------
// Cleanup: free all allocated CNN resources
//---------------------------------------------------------------------
//-----------
/* void detector_cleanup(void) {
    if (!is_initialized) return;
    free(model.conv1_weights); model.conv1_weights = NULL;
    free(model.conv1_bias);    model.conv1_bias = NULL;
    free(model.conv2_weights); model.conv2_weights = NULL;
    free(model.conv2_bias);    model.conv2_bias = NULL;
    free(model.conv3_weights); model.conv3_weights = NULL;
    free(model.conv3_bias);    model.conv3_bias = NULL;
    free(model.dense1_weights);model.dense1_weights = NULL;
    free(model.dense1_bias);   model.dense1_bias = NULL;
    free(model.dense2_weights);model.dense2_weights = NULL;
    free(model.dense2_bias);   model.dense2_bias = NULL;
    free(model.conv2d_output_1); model.conv2d_output_1 = NULL;
    free(model.conv2d_output_2); model.conv2d_output_2 = NULL;
    free(model.conv2d_output_3); model.conv2d_output_3 = NULL;
    memset(&model, 0, sizeof(CNNModel));
    is_initialized = 0; 
*/

    free(model.conv2d_output_1); model.conv2d_output_1 = NULL;
    free(model.pool_output_1);   model.pool_output_1   = NULL;
    free(model.conv2d_output_2); model.conv2d_output_2 = NULL;
    free(model.pool_output_2);   model.pool_output_2   = NULL;
    free(model.conv2d_output_3); model.conv2d_output_3 = NULL;
    free(model.pool_output_3);   model.pool_output_3   = NULL;
    free(model.dense_output_1);  model.dense_output_1  = NULL;
    free(model.dense_output_2);  model.dense_output_2  = NULL;

    memset(&model, 0, sizeof(CNNModel));
    is_initialized = 0; 

}

#endif // CNN_ENABLED
