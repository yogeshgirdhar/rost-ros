#include <stdio.h>
#include <stdlib.h>
#include <sndfile.h>
#include <fftw3.h>
#include <iostream>
#include <fstream>

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

#define FFT_BUF_SIZE 4096 // 4096~92 ms, also 2^12 samples
#define HOP_SIZE (FFT_BUF_SIZE/2)
#define WORD_SIZE 13

double hamming[FFT_BUF_SIZE];

void initWindow(void){
    for (int i = 0; i < FFT_BUF_SIZE; i++){
        hamming[i] = 0.54 - 0.46*cos((2*M_PI*i)/(FFT_BUF_SIZE - 1));
    }
}

vector< vector<double> > calcMFCC(double ** wav_p, int blocks_read, int mfcc_order, int sr=44100){
    cout << "Generating MFCCs for " << blocks_read << " blocks." << endl;
    //COMPUTE THE FFTs
    fftw_complex *out;
    double *in;
    double *wav = *wav_p;
    vector< vector<double> > mfccResult;
    fftw_plan p;

    double spectrum[FFT_BUF_SIZE];
    double curCoeff;

    in = (double*) fftw_malloc(sizeof(double)*FFT_BUF_SIZE);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(FFT_BUF_SIZE/2+1));
    p = fftw_plan_dft_r2c_1d(FFT_BUF_SIZE, in, out, FFTW_MEASURE);

    int pos;
    int win_num = 0;

    for (pos = 0; pos < blocks_read - FFT_BUF_SIZE; pos += HOP_SIZE){
        win_num++;
        int i;
        for (i = 0; i < FFT_BUF_SIZE; i++){
            in[i] = wav[pos+i]*hamming[i];
        }
        fftw_execute(p);

        for (i = 0; i < FFT_BUF_SIZE/2+1; i++){
            spectrum[i] = out[i][0]/FFT_BUF_SIZE;
        }

        int coeff;
        vector<double> curMFCC(WORD_SIZE, 0.0);
        for(coeff = 0; coeff < mfcc_order; coeff++) {
            curCoeff = GetCoefficient(spectrum, sr, 48, 128, coeff);
            curMFCC[coeff] = curCoeff;
        }

        mfccResult.push_back(curMFCC);
        //if (pos % printInterval == 0){
          cout << "\rCompleted " << pos << " blocks. ( " << 100*((double)pos/(double)blocks_read) << " %)";
        //}
    }
    cout << endl;
    return mfccResult;
}

vector< vector<double> > kMeansVocab(vector< vector<double> > in, int k){
    int word_size = in[0].size();
    int num_words = in.size();

    cv::Mat inFlat(num_words, word_size, CV_32FC1);
    cv::Mat labels;
    cv::Mat centers(k, word_size, CV_32FC1);

    for (int i = 0; i < in.size(); i++){
        for (int j = 0; j < word_size; j++){
            inFlat.at<float>(i, j) = (float)in[i][j];
        }
    }

    cv::kmeans(inFlat, k, labels, cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, FLT_EPSILON ), 3, cv::KMEANS_PP_CENTERS, centers);

    //find all the centroids
    vector<double> empty(word_size, 0.0);
    vector<vector<double> > centroids(k, empty);
    vector <int> numsAssigned(k, 1);
    for (int i = 0; i < num_words; i++){
        int j = labels.at<int>(i, 1);
        for (int x = 0; x < word_size; x++){
            centroids[j][x] += (double) inFlat.at<float>(i, x);
        }
        numsAssigned[j]++;
    }
    for (int i = 0; i < k; i++){
        for (int x = 0; x < word_size; x++){
            centroids[i][x] /= (double) numsAssigned[i];
        }
    }

    return centroids;
}

int main(int argc, char * argv[]){
    //For reading the wav file
    initWindow();
    vector< vector<double> > mfccResult;
    int samplerate;
    int numWords = atoi(argv[2]);
    for (int i = 3; i < argc; i++){
        char *fname = argv[i];
        cout << "Opening " << fname << endl;
        SF_INFO info;
        SNDFILE *sf;
        int blocks_read;
        double *wav;
        sf = sf_open(fname, SFM_READ, &info);
        if (sf == NULL) {
            printf("Failed to open the file.\n");
            return(-1);
        }
        if(info.channels != 1){
            printf("Only tested on single channel audio.\n");
            return(-1);
        }
        samplerate = info.samplerate;
        wav = (double *) malloc(info.frames*sizeof(double));
        blocks_read = sf_read_double(sf, wav, info.frames);
        sf_close(sf);
        cout << "Read " << blocks_read << " blocks from " << fname << endl;
        vector< vector<double> > this_result = calcMFCC(&wav, info.frames, WORD_SIZE, info.samplerate);
        mfccResult.insert(mfccResult.end(), this_result.begin(), this_result.end());
        free(wav);
    }

    char rawFName[128];
    strcpy(rawFName, argv[1]);
    strcat(rawFName, "_raw.txt");
    ofstream outfileraw;
    outfileraw.open(rawFName);
    cout << "Writing Raw Output to " << rawFName << endl;
    double time = 0;
    double interval = (double)HOP_SIZE/samplerate;
    for (int i = 0; i < mfccResult.size(); i++){
        outfileraw << time << ":";
        for (int j = 0; j < WORD_SIZE; j++){
            outfileraw << mfccResult[i][j] << " ";
        }
        outfileraw << endl;
        time += interval;
    }
    outfileraw.close();

    if (mfccResult.size() > numWords){
        cout << "Clustering ( k=" << numWords << " )" << endl;
        vector< vector<double> > allClusteredResults = kMeansVocab(mfccResult, numWords);
        char vocabFName[128];
        strcpy(vocabFName, argv[1]);
        strcat(vocabFName, ".txt");
        cout << "Writing FinalVocab to " << vocabFName << endl;
        ofstream outfile;
        outfile.open(vocabFName);
	outfile << allClusteredResults.size() << endl;
        for (int i = 0; i < allClusteredResults.size(); i++){
            outfile << i << ":";
            for (int j = 0; j < WORD_SIZE; j++){
                outfile << mfccResult[i][j] << " ";
            }
            outfile << endl;
        }
        outfile.close();
    }

    return 0;
}

