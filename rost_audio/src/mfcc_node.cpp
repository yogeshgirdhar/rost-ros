#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <ros/ros.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <deque>
#include "rost_common/WordObservation.h"
#include "rost_audio/AudioRaw.h"

#ifdef __cplusplus
extern "C"{
#endif
#include "libmfcc.h"
#ifdef __cplusplus
}
#endif

using namespace std;

#define FFT_BUF_SIZE 4096 // 4096~92 ms, also 2^12 samples
#define HOP_SIZE (FFT_BUF_SIZE/2)
#define WORD_SIZE 13

double hamming[FFT_BUF_SIZE];

double ** vocab;
int vocab_size;
deque<double> iQ;

ros::Publisher words_pub;

void initWindow(void){
  for (int i = 0; i < FFT_BUF_SIZE; i++){
    hamming[i] = 0.54 - 0.46*cos((2*M_PI*i)/(FFT_BUF_SIZE - 1));
  }
}

double euDist(double * v1, double * v2, int vec_length){
  float dist = 0.0;
  for (int i = 0; i < vec_length; i++){
    dist += (v2[i] - v1[i])*(v2[i] - v1[i]);
  }
  dist = sqrt(dist);
  return dist;
}

int applyVocab(double * mfcc){
  int i;
  int best_d = -1;
  int best_label = -1;
  for (i = 0; i < vocab_size; i++){
    int d = euDist(mfcc, vocab[i], WORD_SIZE);
    if (best_d < 0 or d < best_d){
      best_d = d;
      best_label = i;
    }
  }
  return best_label;
}

vector<int> getMFCCs(int num_samples, int samplerate){
  vector<int> words_out;
  // Need at least FFT_BUF_SIZE elements
  if (num_samples > FFT_BUF_SIZE && num_samples <= iQ.size()){
    // Set up the fft
    fftw_complex *out;
    fftw_plan plan;
    double spectrum[FFT_BUF_SIZE];
    double * in = (double*) fftw_malloc(sizeof(double)*FFT_BUF_SIZE);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(FFT_BUF_SIZE/2+1));
    plan = fftw_plan_dft_r2c_1d(FFT_BUF_SIZE, in, out, FFTW_MEASURE);
    
    int pos = 0;
    while (iQ.size() > FFT_BUF_SIZE && pos < num_samples){
      int i;
      // Take each time slot and get a windowed copy of it
      for (i = 0; i < FFT_BUF_SIZE; i++){
	in[i] = iQ[i]*hamming[i];
      }
      for (i = 0; i < HOP_SIZE; i++){
	iQ.pop_front();
      }
      pos += HOP_SIZE;
     
      // Run the fft
      fftw_execute(plan);
      for (i = 0; i < FFT_BUF_SIZE/2+1; i++){
	spectrum[i] = out[i][0]/FFT_BUF_SIZE;
      }
      // Compute the MFCC filterbank
      double mfcc_raw[WORD_SIZE];
      int coeff;
      for(coeff = 0; coeff < WORD_SIZE; coeff++) {
	double curCoeff = GetCoefficient(spectrum, samplerate, 48, 128, coeff);
	mfcc_raw[coeff] = curCoeff;
      }
      
      // Find the closest label in the vocab
      words_out.push_back(applyVocab(mfcc_raw));
    }
  }
  return words_out;
}

void audioCallback(const rost_audio::AudioRawConstPtr &msg){
  // Can do this better!
  for (int i = 0; i < msg->data.size(); i++){
    iQ.push_back(msg->data[i]);
  }
  vector <int> words = getMFCCs(msg->samplerate, msg->samplerate);
  vector <int> pose;
  for (int i = 0; i < words.size(); i++){
    pose.push_back(i);
  }
  vector <int> scale(words.size(), 1);
  if (words.size() > 0){
    rost_common::WordObservation words_msg;
    words_msg.words = words;
    words_msg.word_pose = pose;
    words_msg.word_scale = scale;
    words_msg.source = "audio";
    words_msg.vocabulary_begin = 0;
    words_msg.vocabulary_size = vocab_size;
    words_pub.publish(words_msg);
  }
}

int main(int argc, char *argv[]){
  ros::init(argc, argv, "audio_words");
  ros::NodeHandle nh("");
  
  // Load the vocab
  FILE * vocab_f;
  vocab_f = fopen(argv[1], "r");
  fscanf(vocab_f, "%d", &vocab_size);
  //printf("vocab size: %d\n", vocab_size);
  vocab = (double **) malloc(sizeof(double)*vocab_size*WORD_SIZE);
  for (int i = 0; i < vocab_size; i++){  
   vocab[i] = (double*) malloc(WORD_SIZE*sizeof(double));
  }
  int i = 0;
  int label;
  while( !feof(vocab_f) && i < vocab_size){
    //fscanf(vocab_f, "%d:", &label);
    for (int j = 0; j < WORD_SIZE; j++){
	float curCoeff;
	fscanf(vocab_f, "%f", &curCoeff);
	vocab[i][j] = (double) curCoeff;
	//printf("%f,",vocab[i][j]);
    }
    //printf("\n");
    i++;
  }
  fclose(vocab_f);
  
  // Initialize the hamming window (applied before doing the fft)  
  initWindow();
  words_pub = nh.advertise<rost_common::WordObservation>("words", 1);
  ros::Subscriber audio_sub = nh.subscribe("audio", 1, audioCallback);
  ros::spin();
}
