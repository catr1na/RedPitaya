#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fftw3.h>
#include <time.h>
#include "rp.h"
#include "acquire.h"
#define DELAY 40
#define COUNT 4000


int main() {
	acquire_init(32,512,256,0);
	int count = COUNT;
	int delay;
	rp_acq_trig_state_t state;
	char filename[] = "data/selected/more_training/background/background_trigger_0000.bin";
	sleep(1);
	//long clocks_per_micro  = CLOCKS_PER_SEC/1000000;
	int offset=0;
	
	while(count){
		delay = DELAY;
		rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW);
		//clock_t begin = clock();
		while(delay) {
			//if(clock()-begin>=200*clocks_per_micro){
				rp_AcqGetTriggerState(&state);
				if(state==RP_TRIG_STATE_TRIGGERED) {
					delay--;
				}
				acquire_read_and_transform_select();
				//acquire_read_and_transform();
				//begin = begin + 200*clocks_per_micro;
				//}
		}
		filename[61] = (COUNT-count+offset)%10+'0';
		filename[60] = (COUNT-count+offset)%100/10+'0';
		filename[59] = (COUNT-count+offset)%1000/100+'0';
		filename[58] = (COUNT-count+offset)/1000+'0';
		acquire_write_out_select(filename);
		//acquire_write_out(filename);
		printf("Trigger number %d\n", COUNT -count);
		count--;
		usleep(1000000);
	}
	acquire_clean();
	return 0;
}
