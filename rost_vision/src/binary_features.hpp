#ifndef ROST_BINARY_FEATURES_HPP
#define ROST_BINARY_FEATURES_HPP
#include <cv.h>

cv::Mat binary_to_float(cv::Mat features);
cv::Mat float_to_binary(cv::Mat features);
#endif
