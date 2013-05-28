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

ros::Publisher words_pub;
double ** vocab;
int vocab_size;
int fft_buf_size;
int fft_hop_size;
double *hamming;
int seq;
deque<int> words_out;

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

void publishWord(const ros::TimerEvent&){
  if ( words_out.empty() ) {
    return;
  }
  rost_common::WordObservation words_msg;
  words_msg.words.push_back(words_out.front());
  words_out.pop_front();
  words_msg.word_pose.push_back(0);
  words_msg.word_scale.push_back(1);
  words_msg.source = "audio-offline";
  words_msg.vocabulary_begin = 0;
  words_msg.vocabulary_size = vocab_size;
  seq++;

  words_msg.seq = seq;
  words_msg.header.seq = seq;
  words_msg.observation_pose.push_back(seq);
  words_pub.publish(words_msg);
}

void calcMFCC(double ** wav_p, int blocks_read, int mfcc_order, int sr=44100){
    cout << "Generating MFCCs for " << blocks_read << " blocks." << endl;
    fftw_complex *out;
    double *in;
    double *wav = *wav_p;
    fftw_plan p;

    double spectrum[fft_buf_size];
    double curCoeff;

    in = (double*) fftw_malloc(sizeof(double)*fft_buf_size);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(fft_buf_size/2+1));
    p = fftw_plan_dft_r2c_1d(fft_buf_size, in, out, FFTW_MEASURE);

    int pos;
    int win_num = 0;

    for (pos = 0; pos < blocks_read - fft_buf_size; pos += fft_hop_size){
        win_num++;
        int i;
        for (i = 0; i < fft_buf_size; i++){
            in[i] = wav[pos+i]*hamming[i];
        }
        fftw_execute(p);

        for (i = 0; i < fft_buf_size/2+1; i++){
            spectrum[i] = out[i][0]/fft_buf_size;
        }

        int coeff;
        double curMFCC[WORD_SIZE];
        for(coeff = 0; coeff < mfcc_order; coeff++) {
            curCoeff = GetCoefficient(spectrum, sr, 48, 128, coeff);
            curMFCC[coeff] = curCoeff;
        }
        int label = applyVocab(curMFCC);
       
        words_out.push_back(label);
 
    }
    cout << endl;
}

int loadVocab(const char *fname){
  // Load the vocab
  FILE * vocab_f;
  vocab_f = fopen(fname, "r");
  if (vocab_f == NULL){
    cout << fname << " was not a valid vocabulary" << endl;
    return -d1;
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
	//printf("%f,",vocab[i][j]);
    }
    //printf("\n");
    i++;
  }
  fclose(vocab_f);
  return 0;
}

int main(int argc, char *argv[]){
  if (argv[1] != NULL && strcmp(argv[1], "--help") == 0){
    cout << endl;
    cout << "Usage: rosrun rost_audio mfcc_wav" << endl;
    cout << "Parameters:" << endl;
    cout << "    _vocab (string), default MontrealSounds2k.txt" << endl;
    cout << "    _fft_buf_size (int), default 4096" << endl;
    cout << "    _overlap (double), default 0.5, must be < 1, starts lagging when < 0.15" << endl;
    cout << "    _file (string), must be specified" << endl;
    cout << endl;
    return -1;
  }
  ros::init(argc, argv, "audio_words");
  ros::NodeHandle nh("");
  ros::NodeHandle nhp("~");
  
  string vocab_name;
  nhp.param<string>("vocab", vocab_name, "MontrealSounds2k.txt");
  
  // By Default ~ 92 ms or 2^12 samples
  nhp.param<int>("fft_buf_size",fft_buf_size, 4096);
  cout << fft_buf_size << ": buffer size" << endl;
  
  double overlap;
  nhp.param<double>("overlap",overlap, 0.5);
  fft_hop_size = overlap * fft_buf_size;
  cout << fft_hop_size << ": hop size" << endl;  
  
  if (loadVocab(vocab_name.c_str()) < 0){
    return -1;
  }
  // Initialize the hamming window (applied before doing the fft)  
  hamming  = (double *) malloc(fft_buf_size * sizeof(double));
  initWindow();
  
  // Load the wav file
  string file;
  nhp.param<string>("file", file, "");
  cout << "Opening " << file << endl;
  SF_INFO info;
  SNDFILE *sf;
  int blocks_read;
  double *wav;
  sf = sf_open(file.c_str(), SFM_READ, &info);
  if (sf == NULL) {
      cout << "Failed to open the file." << endl;
      return(-1);
  }
  if(info.channels != 1){
      cout << "Only tested on single channel audio." << endl;
      return(-1);
  }
  wav = (double *) malloc(info.frames*sizeof(double));
  blocks_read = sf_read_double(sf, wav, info.frames);
  sf_close(sf);
  cout << "Read " << blocks_read << " blocks from " << file << endl;
  
  words_pub = nh.advertise<rost_common::WordObservation>("words", 1);
  seq = 0;
  
  ros::Timer pub_timer = nh.createTimer(ros::Duration((double) fft_hop_size/info.samplerate), publishWord, false);

  calcMFCC(&wav, info.frames, WORD_SIZE, info.samplerate);

  ros::spin();
}
