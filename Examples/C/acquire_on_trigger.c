#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <fftw3.h>
#define FIXED_POINT 16
#define CHECK_OVERFLOW TRUE
#include "rp.h"
#include "kiss_fft/_kiss_fft_guts.h"
#include "spec_dsp.h"
#include "kiss_fft/kiss_fftr.h"


double get_seconds() {
  struct timeval t;
  gettimeofday(&t,0);
  return (double)t.tv_sec+(double)t.tv_usec/1e6;
}


int main(int argc, char **argv){
  int num_segments = 32;
  int segment_length = ADC_BUFFER_SIZE/num_segments;
  float *fft_buff=(float*)malloc(rp_get_spectr_out_signal_length()*num_segments*sizeof(float));
  int curr = 0;  

  if(rp_Init() != RP_OK){
    fprintf(stderr, "Rp api init failed!\n");
  }
  rp_Reset();

  rp_set_spectr_signal_length(segment_length);
  rp_spectr_window_init(HANNING);
  rp_spectr_fft_init();

  rp_AcqReset();
  rp_AcqSetDecimation(RP_DEC_256);
	
  rp_AcqSetArmKeep(true);
  rp_AcqStart();
  uint32_t buff_size = rp_get_spectr_signal_length();
  float *buff = (float *)malloc(buff_size * sizeof(float));
 
  sleep(1);

  //rp_AcqSetTriggerSrc(RP_TRIG_SRC_NOW);
  rp_AcqSetTriggerLevel(RP_T_CH_1,0.2);
  rp_AcqSetTriggerLevel(RP_T_CH_2,0.2);
  float level;
  rp_AcqGetTriggerLevel(RP_T_CH_1, &level);
  printf("trigger set: %f\n",level);

  sleep(1);

  float *cha_in = (float *)malloc(rp_get_spectr_signal_length() * sizeof(float));
  float *chb_in = (float *)malloc(rp_get_spectr_signal_length() * sizeof(float));
  float *cha_fft = (float *)malloc(rp_get_spectr_out_signal_length() * sizeof(float));
  float *chb_fft = (float *)malloc(rp_get_spectr_out_signal_length() * sizeof(float));
  float *f_temp = (float *)malloc(rp_get_spectr_out_signal_length() * sizeof(float));
  double *freq_vector = (double *)malloc(rp_get_spectr_out_signal_length() * sizeof(double));
  if(!cha_in || !chb_in || !cha_fft || !chb_fft) return -1;

  rp_spectr_prepare_freq_vector(&f_temp, 1,1);
  for(int i = 0; i<rp_get_spectr_out_signal_length(); i++) {
    freq_vector[i] = (double) f_temp[i]; 
  }

  sleep(1);
  //int sig_len = rp_get_spectr_out_signal_length();
  double start = get_seconds();



  fftwf_complex *in, *out;
  fftwf_plan p;
  in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * segment_length);
  out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * segment_length);
  p = fftwf_plan_dft_r2c_1d(segment_length, chb_in, out,  FFTW_MEASURE);
  //getting data
  /*
  for(int i=0; i<1000; i++){
    rp_AcqGetLatestDataV(RP_CH_1, &buff_size,cha_in);
    //rp_spectr_int_fft(cha_in,&cha_fft);
    //rp_AcqGetLatestDataV(RP_CH_2, &buff_size,chb_in);
    //uint32_t trig_pos;
    //rp_AcqGetWritePointerAtTrig(&trig_pos);
    //rp_AcqGetDataV2D(trig_pos, &buff_size, cha_in, chb_in);
    //rp_spectr_window_filter(cha_in, chb_in, &cha_in, &chb_in);

    //double *row = fft_buff+curr*sig_len;
    //rp_spectr_fft(cha_in, chb_in, &row, &chb_fft);
    //curr = (curr + 1)%num_segments;
    int offset = curr*segment_length/2;
    fftwf_execute(p);
    for(int j=0; j<segment_length/2; j++) {
    	fft_buff[offset+j] = out[j][0]*out[j][0]+ out[j][1]*out[j][1];
    	//printf("%f\n",cha_fft[j]);
    }
    curr=(curr+1)%(2*num_segments);
  }
  */
  int count = 0;
  char file_name[] = "data/positive_trigger/positive_trigger_00.bin";
  int delay=50;
  while(count<5){
  delay = 50;
  rp_AcqSetTriggerSrc(RP_TRIG_SRC_CHA_PE);
  rp_acq_trig_state_t state = RP_TRIG_STATE_WAITING;
  while(delay) {
  	rp_AcqGetTriggerState(&state);
  	if(state==RP_TRIG_STATE_TRIGGERED) {
  		delay--;
  	}
  	rp_AcqGetLatestDataV(RP_CH_2, &buff_size, chb_in);
  	one_window_filter(chb_in, &chb_in);
  	int offset = curr*segment_length/2;
  	fftwf_execute(p);
  	for(int j=0; j<segment_length/2; j++) {
  		fft_buff[offset+j] = out[j][0]*out[j][0]+out[j][1]*out[j][1];
  	}
  	curr =(curr+1)%(2*num_segments);
  }


  //For testing that it is deterministic
  //for(int i=0; i<segment_length; i++) {
  //	cha_in[i] =( 1<<15)-1;
  //}
  //rp_spectr_int_fft(cha_in,&cha_fft);
  //curr=atoi(argv[1]);
  file_name[40] = count + '0';
  FILE *write_ptr;
  write_ptr = fopen(file_name,"wb");
  fwrite(fft_buff+(curr*segment_length/2),sizeof(float)*rp_get_spectr_out_signal_length()\
  		 *(2*num_segments-curr),1,write_ptr);
  fclose(write_ptr);
  write_ptr = fopen(file_name,"ab");
  fwrite(fft_buff, sizeof(float)*rp_get_spectr_out_signal_length()*curr,1,write_ptr);
  //fwrite(cha_fft, sizeof(float)*rp_get_spectr_out_signal_length(),1,write_ptr);
  fclose(write_ptr); 
  printf("Trigger number %d\n", count);
  count++;
  rp_AcqStart();
  sleep(1);
  }

  
  FILE *f_ptr;
  f_ptr = fopen("freqs.bin","wb");
  fwrite(freq_vector, sizeof(double) * rp_get_spectr_out_signal_length(),1,f_ptr);
  fclose(f_ptr);
  
  double time = get_seconds()-start;
  printf("Time taken: %f seconds\n", time);

  int i;
  for(i=0; i<100; i++){
    printf("%f\n",cha_fft[i]);
  }
  free(cha_in);
  free(chb_in);
  free(cha_fft);
  free(chb_fft);
  free(freq_vector);
  free(fft_buff);
  
  double test[20];
  for(i=0; i<20; i++){
    test[i] = 20;
  }
  kiss_fft_scalar *test2 = (kiss_fft_scalar *)test;
  for(i=0; i<0; i++){
    printf("%d\n",test2[i]);
  }

  rp_spectr_fft_clean();
  rp_spectr_window_clean();
  free(buff);
  rp_Release();

  fftwf_destroy_plan(p);
  fftwf_free(in); fftwf_free(out);
  return 0;
}
    	
