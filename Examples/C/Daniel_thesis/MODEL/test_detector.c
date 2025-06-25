// test_detector.c
#include "bubble_detector.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  if (!detector_init("path/to/weights")) {
    fprintf(stderr,"init failed\n");
    return 1;
  }
  float *spect = calloc(INPUT_HEIGHT*INPUT_WIDTH,sizeof *spect);
  for(int i=0;i<10;i++){
    DetectionResult r = detector_process_frame(spect);
    printf("frame %2d → %s\n", i,
      r==DETECTION_BUBBLE?"bubble":"background");
  }
  free(spect);
  detector_cleanup();
  return 0;
}
