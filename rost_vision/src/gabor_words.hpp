#ifndef ROST_GABOR_BOW_HPP
#define ROST_GABOR_BOW_HPP
#include "bow.hpp"


namespace rost{
  using namespace rost_common;
  using namespace cv;

  struct GaborBOW:public BOW{
    int size_cols;
    double img_scale;
    int hvocab0, ivocab0;
    vector<double> thetas, lambdas;
    vector<Mat> sin_gabor, cos_gabor;
    vector<Mat_<float> >filter_response;

    GaborBOW(int vocabulary_begin_=0, int size_cols_=64, double img_scale_=1.0);

    vector<unsigned int> make_words(vector<Mat_<float> >& response);

    WordObservation::Ptr operator()(cv::Mat& imgs, unsigned image_seq, const vector<int>& pose);
  };
}
#endif
