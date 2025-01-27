#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <stdbool.h>
#include <pthread.h>
#include "rp.h"
#include "spec_dsp.h"
#include "bubble_detector.h"

#define SAMPLE_RATE 125000000  // RedPitaya's sampling rate
#define DECIMATION RP_DEC_64   // Decimation factor
#define SAMPLES_20MS (SAMPLE_RATE / 64 * 0.02)  // Samples for 20ms at given decimation
#define OVERLAP_SAMPLES (SAMPLES_20MS / 2)  // 10ms overlap
#define BUFFER_COUNT 32        // Circular buffer size
#define FFT_SIZE SAMPLES_20MS  // Size of FFT

typedef struct {
    float timestamp;      // Time in seconds
    float spectrum[FFT_SIZE / 2 + 1];  // Spectrogram data
} SpectrogramFrame;

typedef struct {
    float* data;
    bool ready;
    bool processed;
} Buffer;

typedef struct {
    Buffer buffers[BUFFER_COUNT];
    int write_index;
    int read_index;
    bool running;
    bool save_to_file;
    pthread_mutex_t mutex;
    pthread_cond_t data_ready;
} CircularBuffer;

static CircularBuffer cbuf = {0};
static char* output_directory = NULL;
static volatile bool running = true;
static uint32_t frame_counter = 0;

void signal_handler(int signum) {
    running = false;
    cbuf.running = false;
    pthread_cond_broadcast(&cbuf.data_ready);
}

void init_circular_buffer() {
    cbuf.write_index = 0;
    cbuf.read_index = 0;
    cbuf.running = true;
    pthread_mutex_init(&cbuf.mutex, NULL);
    pthread_cond_init(&cbuf.data_ready, NULL);

    for (int i = 0; i < BUFFER_COUNT; i++) {
        cbuf.buffers[i].data = (float*)malloc(SAMPLES_20MS * sizeof(float));
        if (!cbuf.buffers[i].data) {
            fprintf(stderr, "Failed to allocate buffer %d\n", i);
            exit(1);
        }
        cbuf.buffers[i].ready = false;
        cbuf.buffers[i].processed = true;
    }
}

void cleanup_system() {
    // Clean up buffers
    for (int i = 0; i < BUFFER_COUNT; i++) {
        free(cbuf.buffers[i].data);
    }
    pthread_mutex_destroy(&cbuf.mutex);
    pthread_cond_destroy(&cbuf.data_ready);

    // Clean up DSP resources
    spec_cleanup();
    detector_cleanup();

    // Release RedPitaya
    rp_Release();
}

void* acquisition_thread(void* arg) {
    uint32_t buff_size = SAMPLES_20MS;
    float* temp_buffer = (float*)malloc(SAMPLES_20MS * sizeof(float));
    if (!temp_buffer) {
        fprintf(stderr, "Failed to allocate temporary buffer\n");
        return NULL;
    }

    while (cbuf.running) {
        // Wait for the required amount of data
        rp_AcqGetLatestDataV(RP_CH_1, &buff_size, temp_buffer);

        pthread_mutex_lock(&cbuf.mutex);

        // Check if buffer is available
        if (!cbuf.buffers[cbuf.write_index].processed) {
            pthread_mutex_unlock(&cbuf.mutex);
            usleep(1000);  // Wait 1ms and try again
            continue;
        }

        // Copy data to circular buffer
        memcpy(cbuf.buffers[cbuf.write_index].data, 
               temp_buffer, 
               SAMPLES_20MS * sizeof(float));

        cbuf.buffers[cbuf.write_index].ready = true;
        cbuf.buffers[cbuf.write_index].processed = false;

        cbuf.write_index = (cbuf.write_index + 1) % BUFFER_COUNT;

        pthread_cond_signal(&cbuf.data_ready);
        pthread_mutex_unlock(&cbuf.mutex);

        // Wait for overlap period
        usleep(10000);  // 10ms delay
    }

    free(temp_buffer);
    return NULL;
}

void process_buffer(int index) {
    float* spectrum = malloc((FFT_SIZE / 2 + 1) * sizeof(float));
    if (!spectrum) {
        fprintf(stderr, "Failed to allocate spectrum buffer\n");
        return;
    }

    // Compute spectrogram
    if (spec_process_frame(cbuf.buffers[index].data, spectrum, FFT_SIZE) != 0) {
        fprintf(stderr, "Failed to process frame %d\n", frame_counter);
        free(spectrum);
        return;
    }

    if (cbuf.save_to_file) {
        // Save mode: write spectrogram with timestamp to file
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/spectrogram_%06u.bin", 
                 output_directory, frame_counter);
        
        SpectrogramFrame frame;
        frame.timestamp = frame_counter * 0.01;  // Assuming 10 ms overlap
        memcpy(frame.spectrum, spectrum, (FFT_SIZE / 2 + 1) * sizeof(float));

        FILE* fp = fopen(filename, "wb");
        if (fp) {
            fwrite(&frame, sizeof(SpectrogramFrame), 1, fp);
            fclose(fp);
            printf("Saved frame %u with timestamp %.2f seconds\n", frame_counter, frame.timestamp);
        } else {
            fprintf(stderr, "Failed to save frame %u\n", frame_counter);
        }
    } else {
        // Process mode: detect bubbles
        DetectionResult result = detector_process_frame(spectrum, FFT_SIZE / 2 + 1);
        if (result == DETECTION_BUBBLE) {
            printf("Bubble detected in frame %u\n", frame_counter);
        }
    }

    frame_counter++;
    free(spectrum);
}

void* processing_thread(void* arg) {
    while (cbuf.running) {
        pthread_mutex_lock(&cbuf.mutex);

        while (cbuf.buffers[cbuf.read_index].processed && cbuf.running) {
            pthread_cond_wait(&cbuf.data_ready, &cbuf.mutex);
        }

        if (!cbuf.running) {
            pthread_mutex_unlock(&cbuf.mutex);
            break;
        }

        if (cbuf.buffers[cbuf.read_index].ready) {
            process_buffer(cbuf.read_index);
            cbuf.buffers[cbuf.read_index].processed = true;
            cbuf.read_index = (cbuf.read_index + 1) % BUFFER_COUNT;
        }

        pthread_mutex_unlock(&cbuf.mutex);
    }

    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <mode> <path>\n", argv[0]);
        printf("mode: 0 for processing (CNN detection), 1 for saving files\n");
        printf("path: model path for mode 0, output directory for mode 1\n");
        return 1;
    }

    // Parse command line arguments
    cbuf.save_to_file = atoi(argv[1]) != 0;
    output_directory = argv[2];

    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize RedPitaya
    if (rp_Init() != RP_OK) {
        fprintf(stderr, "Failed to initialize RedPitaya\n");
        return 1;
    }

    // Configure acquisition
    rp_AcqReset();
    rp_AcqSetDecimation(DECIMATION);
    rp_AcqSetTriggerLevel(RP_CH_1, 0.5);
    rp_AcqSetTriggerDelay(0);

    // Initialize DSP
    if (spec_init(FFT_SIZE, HANNING) != 0) {
        fprintf(stderr, "Failed to initialize DSP\n");
        rp_Release();
        return 1;
    }

    // Initialize circular buffer
    init_circular_buffer();

    if (!cbuf.save_to_file) {
        // Processing mode: initialize detector
        if (!detector_init(output_directory)) {
            fprintf(stderr, "Failed to initialize detector\n");
            cleanup_system();
            return 1;
        }
    }

    // Start acquisition
    rp_AcqStart();
    rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW);

    // Create processing threads
    pthread_t acq_thread, proc_thread;
    if (pthread_create(&acq_thread, NULL, acquisition_thread, NULL) != 0) {
        fprintf(stderr, "Failed to create acquisition thread\n");
        cleanup_system();
        return 1;
    }
    if (pthread_create(&proc_thread, NULL, processing_thread, NULL) != 0) {
        fprintf(stderr, "Failed to create processing thread\n");
        cbuf.running = false;
        pthread_join(acq_thread, NULL);
        cleanup_system();
        return 1;
    }

    printf("Running. Press Ctrl+C to stop...\n");

    // Wait for threads to complete (will be interrupted by Ctrl+C)
    pthread_join(acq_thread, NULL);
    pthread_join(proc_thread, NULL);

    // Cleanup
    cleanup_system();

    printf("\nAcquisition complete. Processed %u frames.\n", frame_counter);
    return 0;
}