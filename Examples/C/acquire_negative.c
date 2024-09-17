#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fftw3.h>
#include <time.h>
#include "rp.h"
#include "acquire.h"
#define DELAY 25
#define COUNT 200


int main() {
	acquire_init(32,512,256,0.2);
	int count = COUNT;
	int delay;
	rp_acq_trig_state_t state;
	char filename[] = "data/selected/more_training/negative_trigger/negative_trigger_0000.bin";
	long clocks_per_micro = CLOCKS_PER_SEC/1000000;

	while(count){
		delay = rand()%32;
		rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);
		clock_t begin = clock();
		while(delay) {
			if(clock()-begin>=200*clocks_per_micro) {
				rp_AcqGetTriggerState(&state);
				if(state==RP_TRIG_STATE_TRIGGERED) {
					delay--;
					//printf("here\n");
				}
				acquire_read_and_transform_select();
				//acquire_read_and_transform();
				begin = begin + 200*clocks_per_micro;
			}
		}
		filename[65] = (COUNT-count)%10+'0';
		filename[64] = (COUNT-count)%100/10 + '0';
		filename[63] = (COUNT-count)/100+'0';
		filename[62] = (COUNT-count)/1000+'0';
		acquire_write_out_select(filename);
		//acquire_write_out(filename);
		printf("Trigger number %d\n", COUNT -count);
		count--;
		usleep(1000000);
	}
	acquire_clean();
	return 0;
}
