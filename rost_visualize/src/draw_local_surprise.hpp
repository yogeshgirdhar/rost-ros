#ifndef ROST_DRAW_LOCAL_SURPRISE
#define ROST_DRAW_LOCAL_SURPRISE
#include <algorithm>
#include <numeric>

using namespace std;

cv::Mat draw_local_surprise(const rost_common::LocalSurprise::ConstPtr&  z, cv::Mat img){
  //float total = accumulate(z->surprise.begin(), z->surprise.end(),0);
  size_t max_i= max_element(z->surprise.begin(), z->surprise.end()) - z->surprise.begin();
  float max_surprise = z->surprise[max_i];
  cv::Mat img_surp(img.rows,img.cols,CV_8UC3);

  for(size_t i=0;i<z->surprise.size();++i){
    int x = z->centers[2*i];
    int y = z->centers[2*i+1];
    int r = z->radii[i];
    float s = z->surprise[i]/max_surprise;
    if(i == max_i)
      cv::circle(img_surp,cv::Point(x,y),r,cv::Scalar(0,0,255),-1);
    else
      cv::circle(img_surp,cv::Point(x,y),r,cv::Scalar(255*s,0*s,0*s),-1);
  }


  cv::addWeighted(img,0.5,img_surp,0.5,0,img_surp);
  return img_surp;
}

#endif

