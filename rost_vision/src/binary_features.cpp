#include "binary_features.hpp"
cv::Mat binary_to_float(cv::Mat features){
  cv::Mat points(features.rows, features.cols*8, CV_32F);
  for(int r = 0; r < features.rows; ++r){
    int j=0;
    for(int c = 0; c < features.cols; ++c){
      unsigned char byte = features.at<unsigned char>(r,c);
      for(int b = 0; b < 8; ++b){
	points.at<float>(r,j++) = byte & 1<<b ? 1.0 : 0.0;
      }
    } 
  }
  return points;
}

cv::Mat float_to_binary(cv::Mat features){
  cv::Mat points(features.rows, features.cols/8, CV_8U);
  for(int r = 0; r < features.rows; ++r){
    for(int c = 0; c < features.cols; c+=8){
      unsigned char byte=0;

      for(int i=0;i<8;++i){
	float f=features.at<float>(r,c+i);
	assert(f>=0.0 && f<=1.0);
	unsigned char bit=std::floor(f+0.5); 
	assert(bit ==0 || bit ==1);
	byte |= (bit << i);
      }
      points.at<unsigned char>(r,c/8) = byte;
    } 
  }
  return points;
}
