
CC = gcc

MODEL ?= Z10

CFLAGS  = -g -std=gnu11 -Wall -Werror -v
CFLAGS += -I/root/RedPitaya/rp-api/api_copy/api/src
CFLAGS += -I/opt/redpitaya/include
#LDFLAGS = -L/opt/redpitaya/lib
#LDLIBS =   -static -lrp

ifeq ($(MODEL),Z20_250_12)
INCLUDE += -I/opt/redpitaya/include/api250-12
LDLIBS += -lrp-gpio -lrp-i2c 
endif

LDLIBS +=  -lm -lstdc++ -lpthread

LDFLAGS = -L/root/RedPitaya/rp_api/api_copy/api/output

LIBRARY_PATH = /root/RedPitaya/rp-api/api_copy/api/output:$(LIBRARY_PATH)
LDLIBS += -lrp 

SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=)

#all: $(OBJS)
all: prog positive negative background

%.o: %.c
	$(CC) -c -V  $(CFLAGS) $(LDFLAGS) spec_dsp.c  $< -o $@ -lrp

clean:
	$(RM) *.o
	$(RM) $(OBJS)
	$(RM) positive negative background mask template trigger
	
prog:
	$(CC) acquire_on_trigger.c spec_dsp.c kiss_fft/kiss_fft.c\
	 kiss_fft/kiss_fftr.c -o acquire_on_trigger -lrp -lfftw3f -lm


positive:
	$(CC) acquire_positive.c acquire.c spec_dsp.c kiss_fft/kiss_fft.c\
	 kiss_fft/kiss_fftr.c -o positive -lrp -lfftw3f -lm

negative:
	$(CC) acquire_negative.c acquire.c spec_dsp.c kiss_fft/kiss_fft.c\
		 kiss_fft/kiss_fftr.c -o negative -lrp -lfftw3f -lm

background:
	$(CC) acquire_background.c acquire.c spec_dsp.c kiss_fft/kiss_fft.c\
		 kiss_fft/kiss_fftr.c -o background -lrp -lfftw3f -lm

mask:
	$(CC) acquire_mask.c acquire.c spec_dsp.c kiss_fft/kiss_fft.c\
			 kiss_fft/kiss_fftr.c -o mask -lrp -lfftw3f -lm

template:
	$(CC) template_creator.c -o template

trigger:
	$(CC) final.c acquire.c spec_dsp.c kiss_fft/kiss_fft.c\
			kiss_fft/kiss_fftr.c -o trigger -lrp -lfftw3f -lm -lpthread
