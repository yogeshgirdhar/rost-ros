#include <iostream>
#include <fstream>
#include <sndfile.h>
#include <fftw3.h>
#include <ros/ros.h>
#include <boost/program_options.hpp>

#ifdef __cplusplus
extern "C"{
#endif
#include "libmfcc.h"
#ifdef __cplusplus
}
#endif

using namespace std;

namespace po = boost::program_options;

#define WORD_SIZE 13

using namespace std;

#define WORD_SIZE 13

double ** vocab;
int vocab_size;

void initWindow(double *hamming, int size){
  for (int i = 0; i < size; i++){
    hamming[i] = 0.54 - 0.46*cos((2*M_PI*i)/(size - 1));
  }
}

double euDist(vector<double> v1, double * v2){
  float dist = 0.0;
  for (uint i = 0; i < v1.size(); i++){
    dist += (v2[i] - v1[i])*(v2[i] - v1[i]);
  }
  dist = sqrt(dist);
  return dist;
}

int applyVocab(vector<double> mfcc){
  int i;
  double best_d = -1;
  int best_label = -1;
  for (i = 0; i < vocab_size; i++){
    double d = euDist(mfcc, vocab[i]);
    if (best_d < 0 or d <= best_d){
      best_d = d;
      best_label = i;
    }
  }
  return best_label;
}

vector<vector<double> > calcMFCC(double * wav, double * hamming, int blocks_read, int fft_buf_size, int fft_hop_size, int sr=44100) {
  fftw_complex *out;
  double *in;
  vector<vector<double> > mfcc_result;
  fftw_plan p;

  double spectrum[fft_buf_size];
  double curCoeff;

  in = (double*) fftw_malloc(sizeof(double)*fft_buf_size);
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(fft_buf_size/2+1));
  p = fftw_plan_dft_r2c_1d(fft_buf_size, in, out, FFTW_MEASURE);

  int pos;
  for (pos = 0; pos < blocks_read - fft_buf_size; pos += fft_hop_size) {
      vector<double> curMFCC;
      int i;
      for (i = 0; i < fft_buf_size; i++) {
	  in[i] = wav[pos+i]*hamming[i];
      }
      
      fftw_execute(p);
      
      for (i = 0; i < fft_buf_size/2+1; i++) {
	  spectrum[i] = out[i][0]/fft_buf_size;
      }
      
      int coeff;
      int fft_size = fft_buf_size/2+1;
      if (fft_size > 512) {
          fft_size = 512;
      }
      for(coeff = 0; coeff < WORD_SIZE; coeff++) {
	  curCoeff = GetCoefficient(spectrum, sr, 48, fft_size/2, coeff);
	  curMFCC.push_back(curCoeff);
      }
      
      mfcc_result.push_back(curMFCC);
      int cur_pct = (int) ( (float)pos/blocks_read*100 );
      cout << "\r" << "Computing MFCCs...(" << cur_pct << "%)    ";
      
  }
  
  return mfcc_result;
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
  vector<vector<double> > mfcc_result;
  
  string vocab_name;
  int fft_buf_size;
  double overlap;
  string wav_name;
  string wordfile_name;
  string mfccfile_name;
  
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("wav_name", po::value<string>(&wav_name), "The wav file to be processed. Must be specified.")
    ("vocab_name", po::value<string>(&vocab_name), "The name of an mfcc vocabulary file. Must be specified")
    ("fft_buf_size", po::value<int>(&fft_buf_size)->default_value(4096), "Number of samples taken into account when calculating the fft and mfcc.")
    ("overlap", po::value<double>(&overlap)->default_value(0.5), "Amount of overlap between successive mfccs. Must be < 1.")
    ("wordfile_name", po::value<string>(&wordfile_name), "The name of the file where the output labels will be saved.")
    ("mfccfile_name", po::value<string>(&mfccfile_name)->default_value(""), "The name of the file where the raw mfcc output will be saved.")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  
  if ( vm.count("help") || overlap > 1 || overlap < 0 ) { 
    cout << desc << endl; 
    return -1; 
  }
  
  int fft_hop_size = (1 - overlap) * fft_buf_size;
    
  
  if (loadVocab(vocab_name.c_str()) < 0){
    return -1;
  }
  // Initialize the hamming window (applied before doing the fft)  
  double *hamming  = (double *) malloc(fft_buf_size * sizeof(double));
  initWindow(hamming, fft_buf_size);
  

  // Read the file
  cout << "Opening " << wav_name << endl;
  SF_INFO info;
  SNDFILE *sf;
  int blocks_read;
  double *wav;
  sf = sf_open( wav_name.c_str(), SFM_READ, &info );
  
  // Ensure that it is in the right format
  if (sf == NULL) {
      cout << "Failed to open the file." << endl;
      return(-1);
  }
  
  if(info.channels != 1) {
      cout << "Only tested on single channel audio." << endl;
      return(-1);
  }
  
  if (info.samplerate != 44100) {
    cout << "Warning: Only tested on 44.1 kHz samplerate audio. Proceed at your own risk!" << endl;
  }
  
  wav = (double *) malloc(info.frames*sizeof(double));
  blocks_read = sf_read_double(sf, wav, info.frames);
  
  sf_close(sf);
  cout << "Read " << blocks_read << " samples from " << wav_name << endl;

  mfcc_result = calcMFCC(wav, hamming, info.frames, fft_buf_size, fft_hop_size, info.samplerate);
  cout << "\nComputed " << mfcc_result.size() << " mfccs @ " << fft_hop_size << " samples/mfcc = " << mfcc_result.size()*fft_hop_size << " samples" << endl;
  
  free(wav); 
  
  if (mfccfile_name != "") {
    ofstream mfcc_file;
    mfcc_file.open(mfccfile_name.c_str());
  
    cout << "Writing raw results to " << mfccfile_name << endl;
  
    for (uint i = 0; i < mfcc_result.size(); i++) {
      mfcc_file << ((float) (i*fft_hop_size))/info.samplerate << ",";
      for (int j = 0; j < WORD_SIZE-1; j++) {
	mfcc_file << mfcc_result[i][j] << ",";
      }
      mfcc_file << mfcc_result[i][ WORD_SIZE-1 ] << endl;
    }

    mfcc_file.close();
  }
  
  if (wordfile_name != "") {
    ofstream word_file;
    word_file.open(wordfile_name.c_str());
    
    cout << "Writing word labels to " << wordfile_name;
    for (uint i = 0; i < mfcc_result.size(); i++) {
      word_file << ((float) (i*fft_hop_size))/info.samplerate << ",";
      word_file << applyVocab(mfcc_result[i]) << endl;
    }
    
    word_file.close();
    
  }
  
  return 0;
}
