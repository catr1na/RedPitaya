#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define COUNT 200

int main() {
	int size = 32*256;
	char filename[] = "data/masks/positive_000.bin";
	int count = COUNT;
	float template[size];
	for(int i=0; i<size;i++) template[i]=0;
	float temp[size];
	float background[size];
	while(count) {
		filename[22] = (COUNT-count)%10+'0';
		filename[21] = (COUNT-count)%100/10+'0';
		filename[20] = (COUNT-count)/100+'0';
		FILE* fp;
		fp = fopen(filename,"rb");
		fread(temp,size*sizeof(float),1,fp);
		for(int i=0; i<size; i++) {
			template[i] += temp[i];
		}
		fclose(fp);
		//if((COUNT-count)%10==0) printf("iteration %d\n", COUNT-count);
		count--;
	}
	

	count = 1000;
	char filename2[] = "data/training/negative_trigger/background_trigger_000.bin";
	while(count) {
		filename2[52] = (1000-count)%10+'0';
		filename2[51] = (1000-count)%100/10+'0';
		filename2[50] = (1000-count)/100+'0';
		FILE* fp;
		fp = fopen(filename2,"rb");
		fread(temp,size*sizeof(float),1,fp);
		for(int i=0; i<size; i++) {
			background[i] += temp[i];
		}
		fclose(fp);
		if((COUNT-count)%10==0) printf("iteration %d\n", COUNT-count);
		count--;
	}

	for(int i=0; i<size; i++) {
		background[i]=background[i]/200;
	}

	for(int i=0; i<size; i++) {
		template[i]=template[i]/200-background[i];
	}

	FILE* fp;
	fp = fopen("data/masks/trigger_mask.bin","wb");
	fwrite(template, size*sizeof(float), 1, fp);
	fclose(fp);
}
