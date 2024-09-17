#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fftw3.h>
#include <time.h>
#include <pthread.h>
#include "rp.h"
#include "acquire.h"

void* acq_func(void* arg){
	printf("step 1\n");
	while(true){
		acquire_read_and_transform_select();
	}
}

int trigger_counter;

void* trig_func(void* arg) {
	while(true){
		if(select_buffer_max()>3000 && check_trigger()) printf("TRIGGERED\n");
	}
}

int main() {
	acquire_init(32,512,256,0.2);
	sleep(1);
	int count = 0;
	rp_acq_trig_state_t state;
	while(true){
		rp_AcqGetTriggerState(&state);
		acquire_read_and_transform_select();
		count++;
		if(count %16==0) {
			if(check_trigger()) printf("TRIGGERED\n");
		}
	}
	acquire_clean();
	return 0;
}
