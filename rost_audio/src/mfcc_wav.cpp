#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include <string.h>
#include <deque>
#include <ros/ros.h>
#include <sndfile.h>

#include "rost_common/WordObservation.h"

#ifdef __cplusplus
extern "C"{
#endif
#include "libmfcc.h"
#ifdef __cplusplus
}
#endif

using namespace std;

#define WORD_SIZE 13

struct CalcMFCC {
  // Parameters
  int msg_size;
  int fft_buf_size;
  int fft_hop_size;
  int samplerate;
  
  // Structs for computing the fft
  double *hamming;
  fftw_complex *fft_out;
  double *fft_in;
  fftw_plan p;
  
  // Data in
  int cur_pos;
  int max_length;
  int seq;
  double *wav_in;
  
  // Data out
  int cur_label;
};

ros::Publisher words_pub;


double ** vocab;
int vocab_size;

void initWindow(double *hamming, int size){
  for (int i = 0; i < size; i++){
    hamming[i] = 0.54 - 0.46*cos((2*M_PI*i)/(size - 1));
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

void publishWord(vector<int> labels, vector<int> poses, int seq){
  rost_common::WordObservation words_msg;
  
  words_msg.words = labels;
  words_msg.word_pose = poses;
  
  for (uint i = 0; i < labels.size(); i++) {
    words_msg.word_scale.push_back(1);
  }
  
  words_msg.source = "audio-offline";
  words_msg.vocabulary_begin = 0;
  words_msg.vocabulary_size = vocab_size;

  words_msg.seq = seq;
  words_msg.observation_pose.push_back(poses[0]);
  words_pub.publish(words_msg);
}

void calcMFCC(CalcMFCC * cm) {
  double spectrum[cm->fft_buf_size];
  double curCoeff;

  cm->fft_in = (double*) fftw_malloc(sizeof(double)*cm->fft_buf_size);
  cm->fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(cm->fft_buf_size/2+1));
  cm->p = fftw_plan_dft_r2c_1d(cm->fft_buf_size, cm->fft_in, cm->fft_out, FFTW_MEASURE);

  vector<int> labels;
  vector<int> poses;
  
  for (int word = 0; word < cm->msg_size; word++) {
    
    if (cm->cur_pos >= cm->max_length) {
      break;
    }
    
    int local_pos = 0;
    while (cm->cur_pos < cm->max_length && local_pos < cm->fft_buf_size) {

      cm->fft_in[local_pos] = cm->wav_in[cm->cur_pos]*cm->hamming[local_pos];
      
      local_pos++;
      
      if (local_pos < cm->fft_hop_size) {
	cm->cur_pos++;
      }
    }
      
    fftw_execute(cm->p);
    
    int i;
    for (i = 0; i < cm->fft_buf_size/2+1; i++){
      spectrum[i] = cm->fft_out[i][0]/cm->fft_buf_size;
    }

    int coeff;
    double curMFCC[WORD_SIZE];
    for(coeff = 0; coeff < WORD_SIZE; coeff++) {
	curCoeff = GetCoefficient(spectrum, cm->samplerate, 48, 128, coeff);
	curMFCC[coeff] = curCoeff;
    }
    
    cm->cur_label = applyVocab(curMFCC);
    labels.push_back(cm->cur_label);
    
    poses.push_back((int) ((double) cm->cur_pos*1000/(double) cm->samplerate));
  }
  
  publishWord(labels, poses, cm->seq);  
  cm->seq++;
}

int loadVocab(const char *fname){
  // Load the vocab
  FILE * vocab_f;
  vocab_f = fopen(fname, "r");
  if (vocab_f == NULL){
    cout << fname << " was not a valid vocabulary" << endl;
    return -1;
  }  
    
  // allocate it 
  fscanf(vocab_f, "%d", &vocab_size);
  vocab = (double **) malloc(sizeof(double)*vocab_size*WORD_SIZE);
  for (int i = 0; i < vocab_size; i++){  
   vocab[i] = (double*) malloc(WORD_SIZE*sizeof(double));
  }
  
  // and parse it
  int i = 0;
  int label;
  while( !feof(vocab_f) && i < vocab_size){
    fscanf(vocab_f, "%d:", &label);
    for (int j = 0; j < WORD_SIZE; j++){
	float curCoeff;
	fscanf(vocab_f, "%f", &curCoeff);
	vocab[i][j] = (double) curCoeff;
    }
    i++;
  }
  fclose(vocab_f);
  return 0;
}

int main(int argc, char *argv[]){
  CalcMFCC calc_mfcc;
  calc_mfcc.cur_pos = 0;
  calc_mfcc.seq = 0;
  
  ros::init(argc, argv, "audio_words");
  ros::NodeHandle nh("");
  ros::NodeHandle nhp("~");
  
  string vocab_name;
  nhp.param<string>("vocab", vocab_name, "MontrealSounds2k.txt");
  
  // By Default ~ 92 ms or 2^12 samples
  nhp.param<int>("fft_buf_size", calc_mfcc.fft_buf_size, 4096);
  
  double overlap;
  nhp.param<double>("overlap", overlap, 0.5);
  calc_mfcc.fft_hop_size = (1 - overlap) * calc_mfcc.fft_buf_size;
  
  nhp.param<int>("msg_size", calc_mfcc.msg_size, 1);
  
  if (argv[1] != NULL && strcmp(argv[1], "--help") == 0 || overlap < 0 || overlap > 1){
    cout << endl;
    cout << "Usage: rosrun rost_audio mfcc_wav" << endl;
    cout << "Parameters:" << endl;
    cout << "    _file (string), must be specified" << endl;
    cout << "    _vocab (string), default MontrealSounds2k.txt" << endl;
    cout << "    _fft_buf_size (int), default 4096" << endl;
    cout << "    _overlap (double), default 0.5, must be < 1, starts lagging when < 0.15" << endl;
    cout << "    _msg_size (int), number of words to be published at once. default 1. " << endl;
    cout << endl;
    return -1;
  }
  
  
  if (loadVocab(vocab_name.c_str()) < 0){
    return -1;
  }
  // Initialize the hamming window (applied before doing the fft)  
  calc_mfcc.hamming  = (double *) malloc(calc_mfcc.fft_buf_size * sizeof(double));
  initWindow(calc_mfcc.hamming, calc_mfcc.fft_buf_size);
  
  // Load the wav file
  string file;
  nhp.param<string>("file", file, "");
  cout << "Opening " << file << endl;
  SF_INFO info;
  SNDFILE *sf;
  
  sf = sf_open(file.c_str(), SFM_READ, &info);
  if (sf == NULL) {
      cout << "Failed to open the file." << endl;
      return(-1);
  }
  if(info.channels != 1){
      cout << "Only tested on single channel audio." << endl;
      return(-1);
  }
  
  calc_mfcc.wav_in = (double *) malloc(info.frames*sizeof(double));
  calc_mfcc.max_length = sf_read_double(sf, calc_mfcc.wav_in, info.frames);
  sf_close(sf);
  
  cout << "Read " << calc_mfcc.max_length << " samples from " << file << endl;
  
  words_pub = nh.advertise<rost_common::WordObservation>("words", 1);
  
  calc_mfcc.samplerate = info.samplerate;
  
  ros::Rate r((double) calc_mfcc.samplerate/((double) calc_mfcc.msg_size*calc_mfcc.fft_hop_size));

  while (calc_mfcc.cur_pos < calc_mfcc.max_length) {
    calcMFCC(&calc_mfcc); 
    ros::spinOnce();
    r.sleep();
  }
  
}
