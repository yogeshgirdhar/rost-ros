#ifndef ROST_FEATURE_DETECTOR
#define ROST_FEATURE_DETECTOR
#include "opencv2/features2d/features2d.hpp"
#include <vector>
using namespace std;

namespace cv{
struct orient_keypoints{
  float IC_Angle(const Mat& image, const int half_k, Point2f pt,
		 const vector<int> & u_max)
  {
    int m_01 = 0, m_10 = 0;
    
    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));
    // Treat the center line differently, v=0                                                                                                               
    for (int u = -half_k; u <= half_k; ++u)
      m_10 += u * center[u];
    
    // Go line by line in the circular patch                                                                                                                
    int step = (int)image.step1();
    for (int v = 1; v <= half_k; ++v)
      {
	// Proceed over the two lines                                                                                                                       
	int v_sum = 0;
	int d = u_max[v];
	for (int u = -d; u <= d; ++u)
	  {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
	  }
        m_01 += v * v_sum;
      }
    
    return fastAtan2((float)m_01, (float)m_10);
  }
  void operator()(const Mat& image, vector<KeyPoint>& keypoints,  int patchSize)
  {
    int halfPatchSize = patchSize / 2;
    vector<int> umax(halfPatchSize + 1);
    
    int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v)
      umax[v] = cvRound(sqrt((double)halfPatchSize * halfPatchSize - v * v));

    // Make sure we are symmetric      
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
       while (umax[v0] == umax[v0 + 1])
	 ++v0;
       umax[v] = v0;
       ++v0;
    }
    
    // Process each keypoint 
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
	   keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
      {
        keypoint->angle = IC_Angle(image, halfPatchSize, keypoint->pt, umax);
      }
  }
  
};
}


static cv::Ptr<cv::FeatureDetector> get_feature_detector(const string& name, int num_features){
  cv::Ptr<cv::FeatureDetector> fd;
  if(name=="Grid2ORB"){
    cv::Ptr<cv::FeatureDetector> orb(new cv::OrbFeatureDetector(num_features/4));
    fd = cv::Ptr<cv::FeatureDetector>(new cv::GridAdaptedFeatureDetector(orb,num_features,2,2));;
  }
  else if(name=="Grid3ORB"){
    cv::Ptr<cv::FeatureDetector> orb(new cv::OrbFeatureDetector(num_features/9));
    fd = cv::Ptr<cv::FeatureDetector>(new cv::GridAdaptedFeatureDetector(orb,num_features,3,3));;
  }
  else if(name=="Grid4ORB"){
    cv::Ptr<cv::FeatureDetector> orb(new cv::OrbFeatureDetector(num_features/16));
    fd = cv::Ptr<cv::FeatureDetector>(new cv::GridAdaptedFeatureDetector(orb,num_features,4,4));;
  }
  else if(name=="Dense"){
    fd = cv::Ptr<cv::FeatureDetector>(new cv::DenseFeatureDetector(32, 1, 0.1, 16, 32));
  }
  else 
    fd = cv::FeatureDetector::create(name);
  return fd;
}


static void get_keypoints(cv::Mat& imgs, vector<string>& feature_detector_names, vector<cv::Ptr<cv::FeatureDetector> >& feature_detectors, vector<cv::KeyPoint>& keypoints)
{
  static cv::orient_keypoints keypoint_orienter;
  keypoints.clear();
  for(size_t i=0;i<feature_detectors.size(); ++i){
    std::vector<cv::KeyPoint> keypoints_i;
    feature_detectors[i]->detect(imgs, keypoints_i);
    if(feature_detector_names[i]=="Dense"){
      cv::Mat img_gray;
      cvtColor(imgs,img_gray,CV_RGB2GRAY);
      keypoint_orienter(img_gray, keypoints_i, 31);
    }
    keypoints.insert(keypoints.end(), keypoints_i.begin(), keypoints_i.end());
  }  
}

#endif
