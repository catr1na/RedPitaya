// Compile this file only if CNN_ENABLED is defined

#include "bubble_detector.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC 1000000
#endif
#ifdef CNN_ENABLED

typedef struct {
        double conv1_time, conv2_time, pool1_time, pool2_time, conv3_time, pool3_time;
        double dense1_time, dense2_time, softmax_time, total_time;
} timing_results_t;

void print_timing_results(timing_results_t* timings) {
    if (timings) {
         printf("\n=== CNN TIMING BREAKDOWN ===\n");
         printf("Conv1:    %.4f seconds (%.1f%%)\n", timings->conv1_time, (timings->conv1_time/timings->total_time)*100);
         printf("Pool1:    %.4f seconds (%.1f%%)\n", timings->pool1_time, (timings->pool1_time/timings->total_time)*100);
         printf("Conv2:    %.4f seconds (%.1f%%)\n", timings->conv2_time, (timings->conv2_time/timings->total_time)*100);
         printf("Pool2:    %.4f seconds (%.1f%%)\n", timings->pool2_time, (timings->pool2_time/timings->total_time)*100);
         printf("Conv3:    %.4f seconds (%.1f%%)\n", timings->conv3_time, (timings->conv3_time/timings->total_time)*100);
         printf("Pool3:    %.4f seconds (%.1f%%)\n", timings->pool3_time, (timings->pool3_time/timings->total_time)*100);
         printf("Dense1:   %.4f seconds (%.1f%%)\n", timings->dense1_time, (timings->dense1_time/timings->total_time)*100);
         printf("Dense2:   %.4f seconds (%.1f%%)\n", timings->dense2_time, (timings->dense2_time/timings->total_time)*100);
         printf("Softmax:  %.4f seconds (%.1f%%)\n", timings->softmax_time, (timings->softmax_time/timings->total_time)*100);
         printf("TOTAL:    %.4f seconds\n", timings->total_time);
         printf("============================\n");
    }
}

//---------------------------------------------------------------------
// Structure to hold CNN weights and biases
//---------------------------------------------------------------------
typedef struct {
    // Convolutional Layer 1
    float* conv1_weights; // [CONV1_FILTERS, CONV_KERNEL_SIZE, 
//CONV_KERNEL_SIZE, 1]
    float* conv1_bias;    // [CONV1_FILTERS]

    // Convolutional Layer 2
    float* conv2_weights; // [CONV2_FILTERS, CONV_KERNEL_SIZE, 
//CONV_KERNEL_SIZE, CONV1_FILTERS]
    float* conv2_bias;    // [CONV2_FILTERS]

    // Convolutional Layer 3
    float* conv3_weights; // [CONV3_FILTERS, CONV_KERNEL_SIZE, 
//CONV_KERNEL_SIZE, CONV2_FILTERS]
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
        fprintf(stderr, "Failed to allocate memory for weights from: %s\n", filepath);
        fclose(fp);
        return NULL;
    }
    size_t read_count = fread(weights, sizeof(float), count, fp);
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
    int in_h, int in_w, int in_c,
    const float *weights,
    const float* bias,
    int kernel_size,
    int num_filters
) {
    int out_h = in_h - kernel_size + 1;
    int out_w = in_w - kernel_size + 1;
    int kernel_area = kernel_size * kernel_size;

//Precalculate strides for better cache performance
    int input_stride_h = in_w * in_c;
    int input_stride_w = in_c;
    int weight_stride_filter = kernel_area * in_c;
    int weight_stride_h = kernel_size * in_c;

    for (int f = 0; f < num_filters; f++) {
        const float *weight_base = weights + f * weight_stride_filter;
	float bias_val = bias[f];

         for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float sum = bias_val;  // Start with bias
 
		//Pointer to input window
		const float *input_window = input + i * input_stride_h + j * input_stride_w;
		const float *weight_ptr = weight_base;

		//Optimized kernel Convolution
                for (int ki = 0; ki < kernel_size; ki++) {
			const float *input_row = input_window + ki * input_stride_h;
                        for (int kj = 0; kj < kernel_size; kj++) {
                        	 const float *input_pixel = input_row + kj * input_stride_w;

				// Vectorizable innner loop over channels
				for (int c = 0; c < in_c; c++) {
					sum += input_pixel[c] * weight_ptr[c];
				}
				weight_ptr += in_c;
			}
		}

		//Apply ReLu and store
		int out_idx = (i * out_w + j) * num_filters + f;
		output[out_idx] = fmaxf(0.0f, sum);
            }
        }
    }
}

//Optimized max pooling
static void max_pool2d_forward
	(const float* input,
	int in_h, int in_w, int in_c,
        int pool_size, float* output
) {
    int out_h = in_h / pool_size;
    int out_w = in_w / pool_size;
    
    for (int c = 0; c < in_c; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float max_val = -INFINITY;

		//Pool window base position
		int base_i = i * pool_size;
		int base_j = j * pool_size;

		//Find max in pool window
                for (int pi = 0; pi < pool_size; pi++) {
                    for (int pj = 0; pj < pool_size; pj++) {
                        int in_idx = ((base_i + pi) * in_w + (base_j + pj)) * in_c + c;
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


static float* forward_pass_with_timing(float* spectrogram, timing_results_t* timings) {
    printf("DEBUG: About to call forward_pass_with_timing\n");
    clock_t start, total_start = clock();
    int h = INPUT_HEIGHT, w = INPUT_WIDTH, c = 1;

    // Layer 1: Conv2D + ReLU + MaxPool
    start = clock();
    conv2d_forward(model.conv2d_output_1, spectrogram, h, w, c,
                   model.conv1_weights, model.conv1_bias,
                   CONV_KERNEL_SIZE, CONV1_FILTERS);
    timings->conv1_time = ((double)(clock() - start)) /  CLOCKS_PER_SEC;
    h = h - CONV_KERNEL_SIZE + 1;
    w = w - CONV_KERNEL_SIZE + 1;
    c = CONV1_FILTERS;

    start = clock();
    max_pool2d_forward(model.conv2d_output_1, h, w, c, POOL_SIZE, 
model.pool_output_1);
    h /= POOL_SIZE;
    w /= POOL_SIZE;

    // Layer 2: Conv2D + ReLU + MaxPool
    start = clock();
    conv2d_forward(model.conv2d_output_2, model.pool_output_1, h, w, c,
                   model.conv2_weights, model.conv2_bias,
                   CONV_KERNEL_SIZE, CONV2_FILTERS);
    timings->conv2_time = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    h = h - CONV_KERNEL_SIZE + 1;
    w = w - CONV_KERNEL_SIZE + 1;
    c = CONV2_FILTERS;

    start = clock();
    max_pool2d_forward(model.conv2d_output_2, h, w, c, POOL_SIZE, 
model.pool_output_2);
    timings->pool2_time = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    h /= POOL_SIZE;
    w /= POOL_SIZE;

    // Layer 3: Conv2D + ReLU + MaxPool
    start = clock();
    conv2d_forward(model.conv2d_output_3, model.pool_output_2, h, w, c,
                   model.conv3_weights, model.conv3_bias,
                   CONV_KERNEL_SIZE, CONV3_FILTERS);
    timings->conv3_time = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    h = h - CONV_KERNEL_SIZE + 1;
    w = w - CONV_KERNEL_SIZE + 1;
    c = CONV3_FILTERS;

    start = clock();
    max_pool2d_forward(model.conv2d_output_3, h, w, c, POOL_SIZE, 
model.pool_output_3);
    timings->pool3_time = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    h /= POOL_SIZE;
    w /= POOL_SIZE;

    // Dense layers
    int flattened_size = h * w * c;

    start = clock();
    dense_forward(model.pool_output_3, flattened_size,
                  model.dense1_weights, model.dense1_bias,
                  DENSE1_UNITS, true, model.dense_output_1);
    timings->dense1_time = ((double)(clock() - start)) / CLOCKS_PER_SEC;

    start = clock();
    dense_forward(model.dense_output_1, DENSE1_UNITS,
                  model.dense2_weights, model.dense2_bias,
                  2, false, model.dense_output_2);

    timings->dense2_time = ((double)(clock() - start)) / CLOCKS_PER_SEC;

    // Apply softmax
    start = clock();
    softmax(model.dense_output_2, 2);
    timings->softmax_time = ((double)(clock() - start)) / CLOCKS_PER_SEC;

    timings->total_time = ((double)(clock() - total_start)) / CLOCKS_PER_SEC;
    
    printf("DEBUG: Forward pass completed\n");
    {
    return model.dense_output_2;
    }

//---------------------------------------------------------------------
// Process a single spectrogram frame using the CNN
//---------------------------------------------------------------------


DetectionResult detector_process_frame(float* spectrogram) {
    printf("DEBUG: Starting detector_process_frame\n");
    timing_results_t timings;
    {
	printf("DEBUG: About to call forward_pass_with_timing\n");
    }
    float* probabilities = forward_pass_with_timing(spectrogram, &timings);

    {
	printf("DEBUG: Forward pass completed\n");
    }

    print_timing_results(&timings);
    if (!is_initialized) {
        fprintf(stderr, "Detector is not initialized. Call detector_init first.\n");
        return DETECTION_BACKGROUND;
    }
    //return DETECTION_BUBBLE;
    
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
//static float* create_test_spectrogram() {
//    float* test_data = malloc(sizeof(float) * INPUT_HEIGHT * INPUT_WIDTH);
  //  if (!test_data) {
    //    fprintf(stderr, "Failed to allocate test data\n");
      //  return NULL;
   // }
   //
    // Fill with some test pattern (you can modify this)
   // for (int i = 0; i < INPUT_HEIGHT * INPUT_WIDTH; i++) {
     //   test_data[i] = (float)rand() / RAND_MAX; // Random values between 0 and 1
//	}
 //   }
    
   // return test_data;
}

#else
//---------------------------------------------------------------------
int main(int argc, char* argv[]) {
    printf("Bubble Detector CNN Test Program\n");
    printf("================================\n");
    
    // Check if weights directory is provided
    const char* weights_dir = "./weights";  // default directory
    if (argc > 1) {
        weights_dir = argv[1];
    }
    
    printf("Using weights directory: %s\n", weights_dir);
    
    // Initialize the detector
    printf("Initializing detector...\n");
    if (!detector_init(weights_dir)) {
        fprintf(stderr, "Failed to initialize detector. Make sure weight files exist in: %s\n", weights_dir);
        printf("Expected files:\n");
        printf("  - conv1_weights.bin, conv1_bias.bin\n");
        printf("  - conv2_weights.bin, conv2_bias.bin\n");
        printf("  - conv3_weights.bin, conv3_bias.bin\n");
        printf("  - dense1_weights.bin, dense1_bias.bin\n");
        printf("  - dense2_weights.bin, dense2_bias.bin\n");
        return 1;
    }
    
    printf("Detector initialized successfully!\n");
    
    // Create test spectrogram data
    printf("Creating test spectrogram data...\n");
    float* test_spectrogram = create_test_spectrogram();
    if (!test_spectrogram) {
        detector_cleanup();
        return 1;
    }
    
    // Process the test frame
    printf("Processing test frame...\n");
    DetectionResult result = detector_process_frame(test_spectrogram);
    
    // Display results
    printf("Detection result: %s\n", 
           result == DETECTION_BUBBLE ? "BUBBLE DETECTED" : "BACKGROUND");
    
    // Clean up
    free(test_spectrogram);
    detector_cleanup();
    
    printf("Test completed successfully!\n");
    return 0;
}

//*void print_timing_results(timing_results_t* timings) {
    if (timings) {
   	 printf("\n=== CNN TIMING BREAKDOWN ===\n");
    	 printf("Conv1:    %.4f seconds (%.1f%%)\n", timings->conv1_time, (timings->conv1_time/timings->total_time)*100);
   	 printf("Pool1:    %.4f seconds (%.1f%%)\n", timings->pool1_time, (timings->pool1_time/timings->total_time)*100);
   	 printf("Conv2:    %.4f seconds (%.1f%%)\n", timings->conv2_time, (timings->conv2_time/timings->total_time)*100);
   	 printf("Pool2:    %.4f seconds (%.1f%%)\n", timings->pool2_time, (timings->pool2_time/timings->total_time)*100);
   	 printf("Conv3:    %.4f seconds (%.1f%%)\n", timings->conv3_time, (timings->conv3_time/timings->total_time)*100);
   	 printf("Pool3:    %.4f seconds (%.1f%%)\n", timings->pool3_time, (timings->pool3_time/timings->total_time)*100);
   	 printf("Dense1:   %.4f seconds (%.1f%%)\n", timings->dense1_time, (timings->dense1_time/timings->total_time)*100);
   	 printf("Dense2:   %.4f seconds (%.1f%%)\n", timings->dense2_time, (timings->dense2_time/timings->total_time)*100);
   	 printf("Softmax:  %.4f seconds (%.1f%%)\n", timings->softmax_time, (timings->softmax_time/timings->total_time)*100);
   	 printf("TOTAL:    %.4f seconds\n", timings->total_time);
   	 printf("============================\n");
    }
}*//

//#else
//#include <stdio.h>
//
//int main() {
//    printf("Error: CNN_ENABLED is not defined.\n");
//    printf("Compile with: gcc -DCNN_ENABLED -lm newbubble_detector.c -o bubble_detector\n");
//    return 1;
//}
#endif

