#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fftw3.h>
#include "rp.h"
#include "acquire.h"
#define DELAY 30
#define COUNT 200


int main() {
	acquire_init(32,512,256,0.2);
	int count = COUNT;
	int delay;
	rp_acq_trig_state_t state;
	char filename[] = "data/masks/positive_000.bin";
	
	while(count){
		delay = DELAY;
		rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);
		while(delay) {
			rp_AcqGetTriggerState(&state);
			if(state==RP_TRIG_STATE_TRIGGERED) {
				delay--;
			}
			acquire_read_and_transform();
		}
		filename[22] = (COUNT-count)%10+'0';
		filename[21] = (COUNT-count)%100/10+'0';
		filename[20] = (COUNT-count)/100+'0';
		acquire_write_out(filename);
		printf("Trigger number %d\n", COUNT -count);
		count--;
		usleep(500000);
	}
	acquire_clean();
	return 0;
}
