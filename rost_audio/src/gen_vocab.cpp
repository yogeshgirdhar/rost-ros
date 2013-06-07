#include <iostream>
#include <fstream>
#include <sndfile.h>
#include <fftw3.h>
#include <ros/ros.h>
#include <boost/program_options.hpp>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

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

double *hamming;

vector<vector<double> > kMeansVocab(vector<vector<vector<double> > > vocab_raw, int vocab_size, int totalWords) {
  int num_files = vocab_raw.size();
  cv::Mat vocab_raw_flat(totalWords, WORD_SIZE, CV_32FC1);
  cv::Mat labels;
  cv::Mat centers(vocab_size, WORD_SIZE, CV_32FC1);
  
  int wordsAssigned = 0;
  for (int i = 0; i < num_files; i++) {
    for (uint j = 0; j < vocab_raw[i].size(); j++) {
      for (int x = 0; x < WORD_SIZE; x++) {
	vocab_raw_flat.at<float>(wordsAssigned, x) = (float)vocab_raw[i][j][x];
      }
    wordsAssigned++;
    }
  }
  
  cv::kmeans(vocab_raw_flat, vocab_size, labels, cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, FLT_EPSILON ), 3, cv::KMEANS_PP_CENTERS, centers);

  int i;
  vector<vector<double> > centers_out;
  for (i = 0; i < centers.rows; i++) {
    vector<double> center;
    for (int j = 0; j < WORD_SIZE; j++) {
      center.push_back( (double) centers.at<float>(i, j) );
    }
    centers_out.push_back(center);
  }
  
  return centers_out;
}

void initWindow(int size) {
  for (int i = 0; i < size; i++) {
    hamming[i] = 0.54 - 0.46*cos((2*M_PI*i)/(size - 1));
  }
}

vector<vector<double> > calcMFCC(double ** wav_p, int blocks_read, int fft_buf_size, int fft_hop_size, int sr=44100) {
  fftw_complex *out;
  double *in;
  double *wav = *wav_p;
  vector<vector<double> > mfccResult;
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
      for(coeff = 0; coeff < WORD_SIZE; coeff++) {
	  curCoeff = GetCoefficient(spectrum, sr, 48, fft_buf_size/2+1, coeff);
	  curMFCC.push_back(curCoeff);
      }
      
      mfccResult.push_back(curMFCC);
      cout << "\r" << "Computing MFCCs...(" << (int) ( (float)pos/blocks_read*100 ) << "%)    ";
  }
  
  return mfccResult;
}

void printVocab(vector<vector<double> > vocab, string fname) {
  ofstream vocab_file;
  vocab_file.open(fname.c_str());
  
  cout << "Writing results to " << fname << endl;
  
  vocab_file << vocab.size() << endl;
  for (uint i = 0; i < vocab.size(); i++) {
    vocab_file << i << ":";
    for (int j = 0; j < WORD_SIZE; j++) {
      vocab_file << vocab[i][j] << " ";
    }
    vocab_file << "\n";
  }
    
  vocab_file.close();
}


int main(int argc, char * argv[]) {
  
  vector<vector<double> > mfccResult;
  vector<vector<vector<double> > > allMfccResult;
  
  string vocab_name = "./newVocab";
  int vocab_size;
  int fft_buf_size;
  double overlap;
  vector<string> wav_names;
  
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("wav_names", po::value<vector<string> >(&wav_names)->multitoken(), "A list of the wav files to use as source audio for the vocabulary. Must be specified.")
    ("vocab_name", po::value<string>(&vocab_name), "The text file to save the vocab to.")
    ("vocab_size", po::value<int>(&vocab_size)->default_value(2000), "Number of distinct words in the output vocabulary.")
    ("fft_buf_size", po::value<int>(&fft_buf_size)->default_value(4096), "Number of samples taken into account when calculating the fft and mfcc.")
    ("overlap", po::value<double>(&overlap)->default_value(0.5), "Amount of overlap between successive mfccs. Must be < 1.")
  ;
  
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  
  if ( vm.count("help") || overlap > 1 || overlap < 0 ) { 
    cout << desc << endl; 
    return -1; 
  }
  
  int fft_hop_size = fft_buf_size*(1 - overlap);
  
  cout << "vocab_name: " << vocab_name << endl;
  cout << "wav_names: ";
  for (uint i = 0; i < wav_names.size(); i++) {
    cout << wav_names[i] << " ";
  }
  cout << endl;
  
  // Initialize the hamming window (applied before doing the fft)  
  hamming  = (double *) malloc(fft_buf_size * sizeof(double));
  initWindow(fft_buf_size);
  
  // MAIN LOOP
  // For each specified wav file
  for (int i = 0; i < wav_names.size(); ++i) {
    cout << "Opening " << wav_names[i] << endl;
    SF_INFO info;
    SNDFILE *sf;
    int blocks_read;
    double *wav;
    sf = sf_open( wav_names[i].c_str(), SFM_READ, &info );
    
    if (sf == NULL) {
	cout << "Failed to open the file." << endl;
	return(-1);
    }
    
    if(info.channels != 1) {
	cout << "Only tested on single channel audio." << endl;
	return(-1);
    }
    
    wav = (double *) malloc(info.frames*sizeof(double));
    blocks_read = sf_read_double(sf, wav, info.frames);
    
    sf_close(sf);
    cout << "Read " << blocks_read << " blocks from " << wav_names[i] << endl;

    mfccResult = calcMFCC(&wav, info.frames, fft_buf_size, fft_hop_size, info.samplerate);
    allMfccResult.push_back(mfccResult);
    cout << "\nComputed " << mfccResult.size() << " samples" << endl;
    
    free(wav);
    
  }
  
  int totalWords = 0;
  for (uint i = 0; i < allMfccResult.size(); i++) {
      totalWords += allMfccResult[i].size();
  }
    
  vector<vector<double> > vocab;
  if ( totalWords > vocab_size ) {
    cout << "Quantizing MFCCs from " << totalWords << " samples to " << vocab_size << " centroids. " << endl;
    printVocab( kMeansVocab(allMfccResult, vocab_size, totalWords), vocab_name ); 
  }

  return 0;
}
