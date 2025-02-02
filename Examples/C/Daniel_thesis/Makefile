# Compiler and flags
CC = gcc
CFLAGS = -g -std=gnu11 -Wall -O2
CFLAGS += -I/opt/redpitaya/include
CFLAGS += -I/usr/local/include  # FFTW3 headers

# Library paths and libraries
LDFLAGS = -L/opt/redpitaya/lib
LDFLAGS += -L/usr/local/lib
LDLIBS = -lm -lpthread -lrp -lfftw3f

# Determine if CNN should be included based on environment variable
ifeq ($(MODEL_PATH),)
    # No CNN
    SRCS = spectrogram_acquisition.c stft_dsp.c
else
    # Include CNN
    SRCS = spectrogram_acquisition.c stft_dsp.c bubble_detector.c
    CFLAGS += -DCNN_ENABLED  # Define CNN_ENABLED for conditional compilation
endif

OBJS = $(SRCS:.c=.o)
TARGET = spectrogram_acquisition

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

# Dependencies
spectrogram_acquisition.o: spectrogram_acquisition.c stft_dsp.h bubble_detector.h
stft_dsp.o: stft_dsp.c stft_dsp.h
bubble_detector.o: bubble_detector.c bubble_detector.h
