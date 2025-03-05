/****************************************************************************
 * spectrogram_acquisition.c
 * Using STFT with 38 FFTs per 20ms chunk (nperseg=256, no overlap).
 *
 * Modes:
 *   Mode 0: CNN detection (if CNN_ENABLED is defined), no trigger
 *   Mode 1: Background Save (no trigger)
 *   Mode 2: Trigger Save (hardware trigger on IN1 at 15 mV)
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <stdbool.h>
#include <pthread.h>
#include <math.h>
#include "rp.h"
#include "stft_dsp.h"          // STFT code
#include "bubble_detector.h"   // CNN (if enabled)

// Red Pitaya sample rate and decimation
#define SAMPLE_RATE      125000000
#define DECIMATION       RP_DEC_256
#define EFFECTIVE_SR     (SAMPLE_RATE / (float)DECIMATION) // ~488281.25

// 20 ms chunk => ~9765 samples at decimation=256
#define CHUNK_DURATION_S 0.02
#define SAMPLES_20MS     ((uint32_t)(EFFECTIVE_SR * CHUNK_DURATION_S + 0.5f))

// Number of ring buffer slots
#define BUFFER_COUNT     32

// Trigger threshold: 15 mV = 0.015 V
#ifndef TRIGGER_THRESHOLD
#define TRIGGER_THRESHOLD 0.015f
#endif


// Data buffer structure for one 20ms chunk
typedef struct {
    float* data;  // Time-domain samples for one 20ms chunk
    bool   ready;
    bool   processed;
} Buffer;

// Circular buffer structure
typedef struct {
    Buffer buffers[BUFFER_COUNT];
    int  write_index;
    int  read_index;
    bool running;
    bool save_to_file;
    pthread_mutex_t mutex;
    pthread_cond_t  data_ready;
} CircularBuffer;

static CircularBuffer cbuf = {0};
static char* output_directory = NULL;
static volatile bool running_flag = true;
static uint32_t frame_counter = 0;

// STFT parameters
static STFT_Handle* stft_handle = NULL;
static int nperseg = 256; // Sub-window size => expecting 38 sub-windows per chunk
static int noverlap = 0;  // No overlap => hop=256

// Global flag for trigger-saving logic in process_buffer
bool trigger_mode_enabled = false;

//
// Signal handler for clean exit (Ctrl+C, SIGTERM)
//
void signal_handler(int signum) {
    running_flag = false;
    cbuf.running = false;
    pthread_cond_broadcast(&cbuf.data_ready);
}

//
// Initialize the circular buffer and its buffers
//
void init_circular_buffer() {
    cbuf.write_index = 0;
    cbuf.read_index  = 0;
    cbuf.running     = true;
    pthread_mutex_init(&cbuf.mutex, NULL);
    pthread_cond_init(&cbuf.data_ready, NULL);

    for (int i = 0; i < BUFFER_COUNT; i++) {
        cbuf.buffers[i].data = (float*)malloc(SAMPLES_20MS * sizeof(float));
        if (!cbuf.buffers[i].data) {
            fprintf(stderr, "Failed to allocate buffer %d\n", i);
            exit(1);
        }
        cbuf.buffers[i].ready     = false;
        cbuf.buffers[i].processed = true;
    }
}

//
// Cleanup system resources (buffers, STFT, CNN, RP resources)
//
void cleanup_system() {
    for (int i = 0; i < BUFFER_COUNT; i++) {
        free(cbuf.buffers[i].data);
    }
    pthread_mutex_destroy(&cbuf.mutex);
    pthread_cond_destroy(&cbuf.data_ready);

    stft_cleanup(stft_handle);

#ifdef CNN_ENABLED
    detector_cleanup();
#endif

    rp_Release();
}

//
// Acquisition thread: continuously acquires 20ms chunks from IN1 (Channel A).
//
void* acquisition_thread(void* arg) {
    uint32_t buff_size = SAMPLES_20MS;
    float* temp_buffer = (float*)malloc(SAMPLES_20MS * sizeof(float));
    if (!temp_buffer) {
        fprintf(stderr, "Failed to allocate temp_buffer\n");
        return NULL;
    }

    while (cbuf.running) {
        // Acquire ~20ms of data from Channel A
        rp_AcqGetLatestDataV(RP_CH_1, &buff_size, temp_buffer);

        pthread_mutex_lock(&cbuf.mutex);
        if (!cbuf.buffers[cbuf.write_index].processed) {
            pthread_mutex_unlock(&cbuf.mutex);
            usleep(1000); // wait for buffer to be processed
            continue;
        }

        // Copy the acquired data into the ring buffer
        memcpy(cbuf.buffers[cbuf.write_index].data, temp_buffer, SAMPLES_20MS * sizeof(float));
        cbuf.buffers[cbuf.write_index].ready     = true;
        cbuf.buffers[cbuf.write_index].processed = false;
        cbuf.write_index = (cbuf.write_index + 1) % BUFFER_COUNT;

        pthread_cond_signal(&cbuf.data_ready);
        pthread_mutex_unlock(&cbuf.mutex);

        // Sleep 10ms => consecutive ~20ms chunks overlap by ~10ms
        usleep(10000);
    }

    free(temp_buffer);
    return NULL;
}

#ifdef CNN_ENABLED
//
// In CNN mode, process the 2D STFT array and feed it to the detector
//
static void run_cnn_detection(float* stft_power_db, int num_subwindows, int fft_out_size) {
    // Example: average across time sub-windows => single frequency vector
    float* freq_vector = (float*)malloc(fft_out_size * sizeof(float));
    if (!freq_vector) return;

    for (int f = 0; f < fft_out_size; f++) {
        double sum = 0.0;
        for (int sw = 0; sw < num_subwindows; sw++) {
            sum += stft_power_db[sw * fft_out_size + f];
        }
        freq_vector[f] = (float)(sum / num_subwindows);
    }

    // Pass to CNN
    DetectionResult result = detector_process_frame(freq_vector);
    if (result == DETECTION_BUBBLE) {
        printf("Bubble detected at frame %u\n", frame_counter);
    }

    free(freq_vector);
}
#endif

//
// Process one buffer from the circular buffer
// If trigger_mode_enabled == true, we first verify the chunk has a sample >= TRIGGER_THRESHOLD.
// Then compute STFT. If saving, write to .bin. Otherwise pass to CNN (mode 0).
//
void process_buffer(int index) {
    float* time_data = cbuf.buffers[index].data;

    // If trigger mode is enabled (Mode 2), check for a sample above threshold
    if (trigger_mode_enabled) {
        bool triggered = false;
        for (int i = 0; i < SAMPLES_20MS; i++) {
            if (fabsf(time_data[i]) >= TRIGGER_THRESHOLD) {
                triggered = true;
                break;
            }
        }
        if (!triggered) {
            printf("Frame %u not triggered (max < %.3f V). Skipping.\n",
                   frame_counter, TRIGGER_THRESHOLD);
            frame_counter++;
            return;
        }
    }

    // 1) Compute STFT
    //    We want 38 subwindows => subwindow size=256 => no overlap => chunk ~9765
    int hop = nperseg - noverlap;  // 256
    int num_subwindows = (SAMPLES_20MS - nperseg) / hop + 1;
    if (num_subwindows != 38) {
        fprintf(stderr,
                "process_buffer: Expected 38 subwindows, got %d (SAMPLES_20MS=%u)\n",
                num_subwindows, SAMPLES_20MS);
    }

    int fft_out_size = (nperseg / 2) + 1; // e.g. 129 for nperseg=256
    float* power_db_array = (float*)malloc(num_subwindows * fft_out_size * sizeof(float));
    if (!power_db_array) {
        fprintf(stderr, "process_buffer: Failed to allocate power_db_array\n");
        return;
    }

    int ret = stft_compute(stft_handle, time_data, SAMPLES_20MS, power_db_array);
    if (ret != num_subwindows) {
        fprintf(stderr, "process_buffer: stft_compute returned %d subwindows, expected %d\n",
                ret, num_subwindows);
    }

    // 2) If in saving mode => write .bin
    if (cbuf.save_to_file) {
        char filename[256];
        uint32_t time_offset_ms = frame_counter * 10;
        snprintf(filename, sizeof(filename), "%s/stft_%06u.bin", output_directory, frame_counter);

        FILE* fp = fopen(filename, "wb");
        if (!fp) {
            fprintf(stderr, "process_buffer: Failed to open %s\n", filename);
            free(power_db_array);
            return;
        }

        // Write metadata: [SAMPLES_20MS, nperseg, noverlap, num_subwindows,
        //                 fft_out_size, (uint32_t)(EFFECTIVE_SR), time_offset_ms]
        uint32_t meta[7];
        meta[0] = SAMPLES_20MS;
        meta[1] = nperseg;
        meta[2] = noverlap;
        meta[3] = num_subwindows;
        meta[4] = fft_out_size;
        meta[5] = (uint32_t)(EFFECTIVE_SR);
        meta[6] = time_offset_ms;
        fwrite(meta, sizeof(uint32_t), 7, fp);

        // Optionally store raw time-domain data
        fwrite(time_data, sizeof(float), SAMPLES_20MS, fp);

        // Write STFT power array
        fwrite(power_db_array, sizeof(float), num_subwindows * fft_out_size, fp);

        fclose(fp);
        printf("Saved STFT chunk %u => %s\n", frame_counter, filename);
    }
#ifdef CNN_ENABLED
    else {
        // Otherwise => CNN detection (mode 0)
        run_cnn_detection(power_db_array, num_subwindows, fft_out_size);
    }
#endif

    free(power_db_array);
    frame_counter++;
}

//
// Processing thread: waits for new data in the ring buffer, processes it
//
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

//
// Main: parse arguments, configure modes, init Red Pitaya & STFT, start threads
//
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <mode> <output_dir_or_weights>\n", argv[0]);
        fprintf(stderr, "  mode=0 => CNN detection (no trigger)\n");
        fprintf(stderr, "  mode=1 => Background Save (no trigger)\n");
        fprintf(stderr, "  mode=2 => Trigger Save (trigger on IN1 at 15 mV)\n");
        return 1;
    }

    int mode = atoi(argv[1]);
    output_directory = argv[2];

#ifdef CNN_ENABLED
    if (mode == 0) {
        // Mode 0: CNN detection, free-running
        cbuf.save_to_file     = false;
        trigger_mode_enabled  = false;
        if (!detector_init(output_directory)) {
            fprintf(stderr, "detector_init failed!\n");
            return 1;
        }
    }
    else if (mode == 1) {
        cbuf.save_to_file     = true;
        trigger_mode_enabled  = false;
    }
    else if (mode == 2) {
        cbuf.save_to_file     = true;
        trigger_mode_enabled  = true;
    }
    else {
        fprintf(stderr, "Invalid mode %d\n", mode);
        return 1;
    }
#else
    // CNN not enabled => only mode=1 or mode=2
    if (mode == 0) {
        fprintf(stderr, "CNN disabled.  Use mode=1 or mode=2.\n");
        return 1;
    }
    else if (mode == 1) {
        cbuf.save_to_file     = true;
        trigger_mode_enabled  = false;
    }
    else if (mode == 2) {
        cbuf.save_to_file     = true;
        trigger_mode_enabled  = true;
    }
    else {
        fprintf(stderr, "Invalid mode %d\n", mode);
        return 1;
    }
#endif

    // Handle Ctrl+C
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize Red Pitaya
    if (rp_Init() != RP_OK) {
        fprintf(stderr, "rp_Init failed\n");
        return 1;
    }
    rp_AcqReset();

    // Use decimation=256 => ~488 kS/s
    rp_AcqSetDecimation(DECIMATION);

    // Trigger config
    if (trigger_mode_enabled) {
        // Mode 2 => hardware trigger on Channel A positive edge
        rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);
        rp_AcqSetTriggerLevel(RP_T_CH_1, TRIGGER_THRESHOLD);

        // Pre-trigger: half chunk => ~10 ms
        rp_AcqSetTriggerDelay(SAMPLES_20MS / 2);
    } else {
        // Mode 0 or 1 => free running
        rp_AcqSetTriggerSrc(RP_TRIG_SRC_DISABLED);
    }

    // Initialize STFT
    stft_handle = stft_init(nperseg, noverlap);
    if (!stft_handle) {
        fprintf(stderr, "Failed to initialize STFT\n");
        rp_Release();
        return 1;
    }

    init_circular_buffer();
    rp_AcqStart();

    // Start threads
    pthread_t acq_thread, proc_thread;
    pthread_create(&acq_thread, NULL, acquisition_thread, NULL);
    pthread_create(&proc_thread, NULL, processing_thread, NULL);

    // Wait for them to finish
    pthread_join(acq_thread, NULL);
    pthread_join(proc_thread, NULL);

    cleanup_system();
    printf("Done. Processed %u frames.\n", frame_counter);
    return 0;
}
