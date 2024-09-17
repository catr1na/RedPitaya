#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fftw3.h>
#include "rp.h"
#include "spec_dsp.h"
#define NUM_SELECTED 6

int num_segments, in_signal_length, out_signal_length, decimation, size;
float* signal_in;
fftwf_complex* signal_out;
float* fft_buff;
uint32_t buff_size;
int offset;
int curr=0;
FILE* write_ptr;
float* template;
float* template_buff;
static int cols[8] = {28,37,43,51,54,110};
float* select_buff;
fftwf_plan p;

int acquire_init(int ns, int sl, int d, float trigger_level) {
	num_segments = ns;
	in_signal_length = sl;
	out_signal_length = in_signal_length/2+1;
	decimation = d;
	size = out_signal_length*num_segments;

	signal_in = (float*)malloc(in_signal_length*sizeof(float));
	signal_out = (fftwf_complex*)malloc(out_signal_length*sizeof(fftwf_complex));
	fft_buff = (float*)malloc(out_signal_length*num_segments*sizeof(float));
	template = (float*)malloc(out_signal_length*num_segments*sizeof(float));
	FILE* fp;
	fp = fopen("data/masks/trigger_mask.bin","rb");
	fread(template,out_signal_length*num_segments*sizeof(float),1,fp);
	fclose(fp);
	template_buff = (float*)malloc(num_segments*sizeof(float));
	p = fftwf_plan_dft_r2c_1d(in_signal_length, signal_in, signal_out, FFTW_MEASURE);
	select_buff = (float*)malloc(NUM_SELECTED*num_segments*sizeof(float));
	if(rp_Init() != RP_OK) {
		fprintf(stderr, "Rp api init failed!\n");
	}
	buff_size = in_signal_length;
	rp_Reset();
	rp_set_spectr_signal_length(in_signal_length);
	rp_spectr_window_init(HANNING);

	rp_AcqReset();
	rp_AcqSetDecimation(decimation);
	rp_AcqSetArmKeep(true);
	rp_AcqStart();
	rp_AcqSetTriggerLevel(RP_T_CH_1,trigger_level);
	buff_size = in_signal_length;
	rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);
	return 0;
}



void acquire_read_and_transform() {
	rp_AcqGetLatestDataV(RP_CH_1,&buff_size,signal_in);
	one_window_filter(signal_in, &signal_in);
	offset = curr * out_signal_length;
	fftwf_execute(p);
	for(int j=0; j<out_signal_length; j++) {
		fft_buff[offset+j] = signal_out[j][0]*signal_out[j][0]\
						   + signal_out[j][1]*signal_out[j][1];
	}
	curr = (curr+1)%num_segments;
}

void acquire_read_and_transform_select() {
	rp_AcqGetLatestDataV(RP_CH_1,&buff_size,signal_in);
	one_window_filter(signal_in,&signal_in);
	offset = curr*NUM_SELECTED;
	fftwf_execute(p);
	int temp;
	for(int i=0; i<NUM_SELECTED; i++){
		temp = cols[i];
		select_buff[offset+i] = signal_out[temp][0]*signal_out[temp][0]\
						   + signal_out[temp][1]*signal_out[temp][1];
	}
	curr = (curr+1)%num_segments;
}

void acquire_apply_template() {
	template_buff[curr] = 0;
	int l = curr*out_signal_length;
	int c = 0;
	for(int i=l; i<size; i++) {
		template_buff[curr]+=template[c]*fft_buff[i];
		c+=4;
	}
	for(int i=0; i<l; i++) {
		template_buff[curr]+=template[c]*fft_buff[i];
		c+=4;
	}
}


int acquire_write_out(char* filename) {
	rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);
	rp_AcqStart();
	write_ptr = fopen(filename, "wb");
	fwrite(fft_buff+(curr*out_signal_length),sizeof(float)*out_signal_length*(num_segments-curr),\
	       1,write_ptr);
	fclose(write_ptr);
	write_ptr = fopen(filename, "ab");
	fwrite(fft_buff, sizeof(float)*out_signal_length*curr,1,write_ptr);
	fclose(write_ptr);
	return 0;
}

int acquire_write_out_template_buff(char* filename) {
	rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);
	rp_AcqStart();
	write_ptr = fopen(filename, "wb");
	fwrite(template_buff+curr,sizeof(float)*(num_segments-curr),\
	       1,write_ptr);
	fclose(write_ptr);
	write_ptr = fopen(filename, "ab");
	fwrite(fft_buff, sizeof(float)*curr,1,write_ptr);
	fclose(write_ptr);
	return 0;
}

int acquire_write_out_select(char* filename) {
	rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);
	rp_AcqStart();
	write_ptr = fopen(filename, "wb");
	fwrite(select_buff+curr*NUM_SELECTED,sizeof(float)*NUM_SELECTED*(num_segments-curr),\
			1,write_ptr);
	fclose(write_ptr);
	write_ptr = fopen(filename, "ab");
	fwrite(select_buff,sizeof(float)*curr*NUM_SELECTED,1,write_ptr);
	fclose(write_ptr);
	return 0;	
}

int acquire_clean() {
	free(signal_in);
	free(fft_buff);
	free(template);
	free(select_buff);
	rp_spectr_window_clean();
	rp_Release();

	fftwf_destroy_plan(p);
	fftwf_free(signal_out);
	return 0;
}

float get_select_buffer_at(int i) {
	if(i+NUM_SELECTED*curr<192) return select_buff[i+NUM_SELECTED*curr];
	return select_buff[i+NUM_SELECTED*curr-192];
}

float select_buffer_max() {
	float max = 0;
	for(int i=0; i<192; i++) {
		if(select_buff[i]>max) max = select_buff[i];
	}
	return max;
}

int check_trigger() {
	if (get_select_buffer_at(96) <= 0.0019433139823377132) {
		if (get_select_buffer_at(90) <= 0.012699482962489128) {
			return 0;
		}
		else {
			return 1;
		}
	}
	else {
		if (get_select_buffer_at(187) <= 0.001474678865633905) {
			if (get_select_buffer_at(2) <= 0.0018088282085955143) {
				if (get_select_buffer_at(144) <= 0.00036885861482005566) {
					if (get_select_buffer_at(179) <= 4.705363608081825e-05) {
						return 1;
					}
					else {
						return 0;
					}
				}
				else {
					if (get_select_buffer_at(140) <= 0.0022403853945434093) {
						if (get_select_buffer_at(176) <= 0.0020718638552352786) {
							return 1;
						}
						else {
							return 0;
						}
					}
					else {
						return 0;
					}
				}
			}
			else {
				return 0;
			}
		}
		else {
			return 0;
		}
	}
}
