#ifndef ROST_BOW_HPP
#define ROST_BOW_HPP
#include "rost_common/WordObservation.h"
namespace rost{
  using namespace rost_common;
  struct BOW{    
    string name;
    int vocabulary_begin;
    int vocabulary_size;
    BOW(const string& name_, int vb, int vs=0):name(name_), vocabulary_begin(vb), vocabulary_size(vs){}
    virtual WordObservation::Ptr operator()(cv::Mat& img, unsigned image_seq, const vector<int>& pose)=0;   
  };
}
#endif
