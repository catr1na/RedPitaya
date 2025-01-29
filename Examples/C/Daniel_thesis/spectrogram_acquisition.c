#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <stdbool.h>
#include <pthread.h>
#include <math.h>
#include "rp.h"
#include "spec_dsp.h"
#include "bubble_detector.h"

#define SAMPLE_RATE 125000000  // Red Pitaya's sampling rate
#define DECIMATION RP_DEC_64   // Decimation factor
#define SAMPLES_20MS (SAMPLE_RATE / 64 * 0.02)  // Samples for 20ms at given decimation
#define OVERLAP_SAMPLES (SAMPLES_20MS / 2)      // 10ms overlap
#define BUFFER_COUNT 32        // Circular buffer size
#define FFT_SIZE SAMPLES_20MS  // Size of FFT

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
    for (int i = 0; i < BUFFER_COUNT; i++) {
        free(cbuf.buffers[i].data);
    }
    pthread_mutex_destroy(&cbuf.mutex);
    pthread_cond_destroy(&cbuf.data_ready);

    spec_cleanup();

#ifdef CNN_ENABLED
    detector_cleanup();
#endif

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
        // Acquire the latest data
        rp_AcqGetLatestDataV(RP_CH_1, &buff_size, temp_buffer);

        pthread_mutex_lock(&cbuf.mutex);

        // Wait if the next buffer hasn't been processed yet
        if (!cbuf.buffers[cbuf.write_index].processed) {
            pthread_mutex_unlock(&cbuf.mutex);
            usleep(1000);
            continue;
        }

        // Copy data into the circular buffer
        memcpy(cbuf.buffers[cbuf.write_index].data, temp_buffer, SAMPLES_20MS * sizeof(float));
        cbuf.buffers[cbuf.write_index].ready = true;
        cbuf.buffers[cbuf.write_index].processed = false;
        cbuf.write_index = (cbuf.write_index + 1) % BUFFER_COUNT;

        pthread_cond_signal(&cbuf.data_ready);
        pthread_mutex_unlock(&cbuf.mutex);

        // Sleep 10 ms (simulating 50% overlap)
        usleep(10000);
    }

    free(temp_buffer);
    return NULL;
}

void process_buffer(int index) {
    float* time_data = cbuf.buffers[index].data;
    float* spectrum = malloc((FFT_SIZE/2 + 1) * sizeof(float));
    float* power_db = malloc((FFT_SIZE/2 + 1) * sizeof(float));

    if (!spectrum || !power_db) {
        fprintf(stderr, "Failed to allocate buffers for spectrum\n");
        free(spectrum);
        free(power_db);
        return;
    }

    // Compute FFT magnitude
    if (spec_process_frame(time_data, spectrum, FFT_SIZE) != 0) {
        fprintf(stderr, "Failed to process frame %d\n", frame_counter);
        free(spectrum);
        free(power_db);
        return;
    }

    // Convert magnitude to dB
    for(int i = 0; i < FFT_SIZE/2 + 1; i++) {
        if(spectrum[i] > 0) {
            power_db[i] = 20 * log10(spectrum[i]);
        } else {
            power_db[i] = -200;
        }
    }

    // -------- SAVE TO FILE MODE -----------
    if (cbuf.save_to_file) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/spectrogram_%06u.bin",
                 output_directory, frame_counter);

        FILE* fp = fopen(filename, "wb");
        if (fp) {
            // metadata: [0] = SAMPLES_20MS, [1] = FFT_SIZE/2+1, [2] = (SAMPLE_RATE/DECIMATION), [3] = time offset in ms
            uint32_t metadata[4] = {
                SAMPLES_20MS,
                FFT_SIZE/2 + 1,
                (uint32_t)(SAMPLE_RATE / 64),
                frame_counter * 10
            };
            fwrite(metadata, sizeof(uint32_t), 4, fp);
            fwrite(time_data, sizeof(float), SAMPLES_20MS, fp);
            fwrite(spectrum, sizeof(float), FFT_SIZE/2 + 1, fp);
            fwrite(power_db, sizeof(float), FFT_SIZE/2 + 1, fp);
            fclose(fp);

            printf("Saved frame %u (time offset: %u ms)\n",
                   frame_counter, frame_counter * 10);
        } else {
            fprintf(stderr, "Could not open file %s for writing\n", filename);
        }
    }

#ifdef CNN_ENABLED
    // -------- CNN MODE -----------
    else {
        // Run bubble detection (if compiled with CNN)
        DetectionResult result = detector_process_frame(power_db, FFT_SIZE/2 + 1);
        if (result == DETECTION_BUBBLE) {
            printf("Bubble detected at frame %u\n", frame_counter);
        }
    }
#endif

    free(spectrum);
    free(power_db);
    frame_counter++;
}

void* processing_thread(void* arg) {
    while (cbuf.running) {
        pthread_mutex_lock(&cbuf.mutex);

        // Wait if the read buffer has already been processed or no data yet
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
        printf("  mode: 0 for processing (CNN detection), 1 for saving files\n");
        printf("  path: model path for mode=0, output directory for mode=1\n");
        return 1;
    }

    int mode = atoi(argv[1]);
    output_directory = argv[2];

#ifdef CNN_ENABLED
    // If compiled with CNN
    if (mode == 0) {
        // CNN-based detection
        if (!detector_init(output_directory)) {
            fprintf(stderr, "Failed to initialize detector\n");
            return 1;
        }
        cbuf.save_to_file = false;
    } else if (mode == 1) {
        // Save-to-file mode
        cbuf.save_to_file = true;
    } else {
        fprintf(stderr, "Invalid mode: %d\n", mode);
        return 1;
    }
#else
    // If compiled WITHOUT CNN
    // We only support saving mode in this compilation
    if (mode != 1) {
        fprintf(stderr, "CNN is disabled, so only mode=1 (save) is valid.\n");
        return 1;
    }
    printf("Running in save mode. CNN will be disabled.\n");
    cbuf.save_to_file = true;
#endif

    // Handle Ctrl+C / kill signals
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize Red Pitaya
    if (rp_Init() != RP_OK) {
        fprintf(stderr, "Failed to initialize RedPitaya\n");
        return 1;
    }

    // Configure acquisition
    rp_AcqReset();
    rp_AcqSetDecimation(DECIMATION);
    rp_AcqSetTriggerLevel(RP_CH_1, 0.5);
    rp_AcqSetTriggerDelay(0);

    // Initialize DSP for FFT
    if (spec_init(FFT_SIZE, HANNING) != 0) {
        fprintf(stderr, "Failed to initialize DSP\n");
        rp_Release();
        return 1;
    }

    // Initialize and start data acquisition
    init_circular_buffer();
    rp_AcqStart();
    rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW);

    // Create threads: acquisition + processing
    pthread_t acq_thread, proc_thread;
    pthread_create(&acq_thread, NULL, acquisition_thread, NULL);
    pthread_create(&proc_thread, NULL, processing_thread, NULL);

    // Wait for threads to finish
    pthread_join(acq_thread, NULL);
    pthread_join(proc_thread, NULL);

    // Cleanup
    cleanup_system();
    printf("\nAcquisition complete. Processed %u frames.\n", frame_counter);

    return 0;
}
