# Compiler and flags
CC = gcc
CFLAGS = -g -std=gnu11 -Wall -O2
CFLAGS += -I/opt/redpitaya/include
CFLAGS += -I/usr/local/include              # FFTW3 headers

# Library paths and libraries
LDFLAGS = -L/opt/redpitaya/lib
LDFLAGS += -L/usr/local/lib
LDLIBS = -lm -lpthread -lrp -lfftw3f

# Source files
SRCS = spectrogram_acquisition.c spec_dsp.c bubble_detector.c
OBJS = $(SRCS:.c=.o)
TARGET = spectrogram_acquisition

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

# Dependencies
spectrogram_acquisition.o: spectrogram_acquisition.c spec_dsp.h bubble_detector.h
spec_dsp.o: spec_dsp.c spec_dsp.h
bubble_detector.o: bubble_detector.c bubble_detector.h