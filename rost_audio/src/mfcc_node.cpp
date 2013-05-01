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
#include <math.h>
#ifdef __cplusplus
}
#endif

using namespace std;

#define WORD_SIZE 13


double ** vocab;
int vocab_size;
int fft_buf_size;
int fft_hop_size;
int seq;
double *hamming;
double MAX = pow(2, 16);

deque<double> iQ;

ros::Publisher words_pub;

void initWindow(void){
  for (int i = 0; i < fft_buf_size; i++){
    hamming[i] = 0.54 - 0.46*cos((2*M_PI*i)/(fft_buf_size - 1));
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
  double best_d = -1;
  int best_label = -1;
  for (i = 0; i < vocab_size; i++){
    double d = euDist(mfcc, vocab[i], WORD_SIZE);
    if (best_d < 0 or d <= best_d){
      best_d = d;
      best_label = i;
    }
  }
  return best_label;
}

vector<int> getMFCCs(int num_samples, int samplerate){
  vector<int> words_out;
  // Need at least fft_buf_size elements
  if (num_samples > fft_buf_size && num_samples <= iQ.size()){
    // Set up the fft
    fftw_complex *out;
    fftw_plan plan;
    double spectrum[fft_buf_size/2+1];
    double * in = (double*) fftw_malloc(sizeof(double)*fft_buf_size);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(fft_buf_size/2+1));
    plan = fftw_plan_dft_r2c_1d(fft_buf_size, in, out, FFTW_MEASURE);
    
    int pos = 0;
    while (iQ.size() > fft_buf_size && pos < num_samples){
      int i;
      // Take each time slot and get a windowed copy of it
      for (i = 0; i < fft_buf_size; i++){
	in[i] = iQ[i]*hamming[i];
      }
      for (i = 0; i < fft_hop_size; i++){
	iQ.pop_front();
      }
      pos += fft_hop_size;
     
      // Run the fft
      fftw_execute(plan);
      for (i = 0; i < fft_buf_size/2+1; i++){
	spectrum[i] = out[i][0]/fft_buf_size;
      }
      // Compute the MFCC filterbank
      double mfcc_raw[WORD_SIZE];
      int coeff;
      for(coeff = 0; coeff < WORD_SIZE; coeff++) {
	double curCoeff = GetCoefficient(spectrum, samplerate, 48, 128, coeff);
	mfcc_raw[coeff] = curCoeff;
	cout << curCoeff << ",";
      }
      cout << endl;
      
      // Find the closest label in the vocab
      words_out.push_back(applyVocab(mfcc_raw));
    }
  }
  return words_out;
}

void audioCallback(const rost_audio::AudioRawConstPtr &msg){
  for (int i = 0; i < msg->data.size(); i++){
    iQ.push_back(((double) msg->data[i]*2.0/MAX) - 1.0);
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
    seq++;
    words_msg.seq = seq;
    words_msg.header.seq = seq;
    words_msg.observation_pose.push_back(seq);
    words_pub.publish(words_msg);
  }
}

int main(int argc, char *argv[]){
  if (argv[1] != NULL && strcmp(argv[1], "--help") == 0){
    cout << endl;
    cout << "Usage: rosrun rost_audio" << endl;
    cout << "Parameters:" << endl;
    cout << "    _vocab (string), default MontrealSounds2k.txt" << endl;
    cout << "    _fft_buf_size (int), default 4096" << endl;
    cout << "    _overlap (double), default 0.5, must be < 1, starts lagging when < 0.15" << endl;
    cout << endl;
    return -1;
  }
  ros::init(argc, argv, "audio_words");
  ros::NodeHandle nh("");
  ros::NodeHandle nhp("~");
  
  string vocab_name;
  nhp.param<string>("vocab", vocab_name, "MontrealSounds2k.txt");
  cout << vocab_name << ": name" << endl;
  
  // By Default ~ 92 ms or 2^12 samples
  nhp.param<int>("fft_buf_size",fft_buf_size, 4096);
  cout << fft_buf_size << ": buffer size" << endl;
  
  double overlap;
  nhp.param<double>("overlap",overlap, 0.5);
  fft_hop_size = overlap * fft_buf_size;
  cout << fft_hop_size << ": hop size" << endl;
  
  // Load the vocab
  FILE * vocab_f;
  vocab_f = fopen(vocab_name.c_str(), "r");
  if (vocab_f == NULL){
    cout << vocab_name << " was not a valid vocabulary" << endl;
    return -1;
  }
  fscanf(vocab_f, "%d", &vocab_size);
  //printf("vocab size: %d\n", vocab_size);
  vocab = (double **) malloc(sizeof(double)*vocab_size*WORD_SIZE);
  for (int i = 0; i < vocab_size; i++){  
   vocab[i] = (double*) malloc(WORD_SIZE*sizeof(double));
  }
  int i = 0;
  int label;
  while( !feof(vocab_f) && i < vocab_size){
    fscanf(vocab_f, "%d:", &label);
    for (int j = 0; j < WORD_SIZE; j++){
	float curCoeff;
	fscanf(vocab_f, "%f", &curCoeff);
	vocab[i][j] = (double) curCoeff;
	//printf("%f,",vocab[i][j]);
    }
    //printf("\n");
    i++;
  }
  cout << "Read " << i << " words from the vocabulary" << endl;
  fclose(vocab_f);
  
  // Initialize the hamming window (applied before doing the fft)  
  hamming  = (double *) malloc(fft_buf_size * sizeof(double));
  initWindow();
  seq = 0;
  words_pub = nh.advertise<rost_common::WordObservation>("words", 1);
  ros::Subscriber audio_sub = nh.subscribe("audio", 1, audioCallback);
  ros::spin();
}
