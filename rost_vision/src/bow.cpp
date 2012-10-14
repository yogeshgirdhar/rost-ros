#include <ros/ros.h>
#include <ros/topic.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include "rost_common/WordObservation.h"
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include "feature_detector.hpp"
using namespace std;

namespace rost{
  using namespace rost_common;

  //sensor_msgs::CvBridge bridge;


  struct BOW{    
    string name;
    int vocabulary_begin;
    int vocabulary_size;
    BOW(const string& name_, int vb, int vs=0):name(name_), vocabulary_begin(vb), vocabulary_size(vs){}
    virtual WordObservation::Ptr operator()(cv::Mat& img, unsigned image_seq, const vector<int>& pose)=0;   
  };

  vector<cv::Ptr<BOW> > word_extractors;

  struct ColorBOW:public BOW{
    int size_cols;
    double img_scale;
    bool use_hue, use_intensity;
    int hvocab0, ivocab0;
    ColorBOW(int vocabulary_begin_=0, int size_cols_=32, double img_scale_=1.0, bool use_hue_=true, bool use_intensity_=true):
      BOW("Color",vocabulary_begin_,0),
      size_cols(size_cols_),
      img_scale(img_scale_),
      use_hue(use_hue_), use_intensity(use_intensity_)
    {
      assert(use_hue || use_intensity);
      if(use_hue){
	cerr<<"Initializing Hue Words"<<endl;
	hvocab0=vocabulary_begin+vocabulary_size;
	BOW::vocabulary_size+=180;
      }
      if(use_intensity){
	cerr<<"Initializing Intensity Words"<<endl;
	ivocab0=vocabulary_begin+vocabulary_size;
	BOW::vocabulary_size+=256;
      }
    }

    WordObservation::Ptr operator()(cv::Mat& imgs, unsigned image_seq, const vector<int>& pose){


      cv::Mat thumb;
      int size_rows = size_cols*static_cast<float>(imgs.rows)/imgs.cols;
      cv::resize(imgs,thumb,cv::Size(size_cols,size_rows));
      cv::Mat hsv (size_rows, size_cols, CV_8UC3);
      cv::Mat_<uchar> hue (hsv.rows, hsv.cols);
      cv::Mat_<uchar> saturation (hsv.rows, hsv.cols);
      cv::Mat_<uchar> value (hsv.rows, hsv.cols);
      cv::cvtColor(thumb,hsv,CV_BGR2HSV);
      cv::Mat splitchannels[]={hue,saturation,value};
      cv::split(hsv,splitchannels);


      WordObservation::Ptr z(new rost_common::WordObservation);
      z->source=name;
      z->seq = image_seq;
      z->observation_pose=pose;
      z->vocabulary_begin=vocabulary_begin;
      z->vocabulary_size=vocabulary_size;

      //width of each pixel in the original image
      float word_scale = static_cast<float>(imgs.cols)/size_cols/img_scale;

      vector<int> word_pose(2,0);
      for(int i=0;i<thumb.rows; ++i) // y
	for(int j=0;j<thumb.cols; ++j){ //x
	  if(use_intensity){
	    z->words.push_back(ivocab0 + value(i,j));
	    z->word_pose.push_back(j*word_scale + word_scale/2);
	    z->word_pose.push_back(i*word_scale + word_scale/2);
	    z->word_scale.push_back(word_scale/2);
	  }
	  if(use_hue){
	    z->words.push_back( hvocab0 + hue(i,j)); //hue is from 0..180
	    z->word_pose.push_back(j*word_scale + word_scale/2);
	    z->word_pose.push_back(i*word_scale + word_scale/2);
	    z->word_scale.push_back(word_scale/2);
	  }
	}
      cerr<<"#color-words: "<<z->words.size();
      return z;
    }
  };

  struct FeatureBOW:public BOW{
    //    cv::Ptr<cv::FeatureDetector> feature_detector;
    vector<cv::Ptr<cv::FeatureDetector> >feature_detectors;
    vector<string >feature_detector_names;
    cv::Ptr<cv::DescriptorExtractor> desc_extractor;
    cv::Ptr<cv::DescriptorMatcher> desc_matcher;
    cv::Mat vocabulary;
    
    double img_scale;


    FeatureBOW(int vocabulary_begin_, 
	       const string& vocabulary_filename, 
	       const vector<string>& feature_detector_names_, 
	       const vector<int>& feature_sizes_, 
	       const string& feature_descriptor_name="SURF", 
	       double img_scale_=1.0):
      BOW(feature_descriptor_name, vocabulary_begin_),
      feature_detector_names(feature_detector_names_),
      img_scale(img_scale_)
    {
      cerr<<"Initializing Feature BOW"<<endl;
      for(size_t i=0;i<feature_detector_names.size(); ++i){
	feature_detectors.push_back(get_feature_detector(feature_detector_names[i],feature_sizes_[i]));
      }

      desc_extractor = cv::DescriptorExtractor::create(feature_descriptor_name);
      if(feature_descriptor_name=="SURF"){
	cerr<<"Using SURF descriptor"<<endl;
	desc_matcher = cv::DescriptorMatcher::create("FlannBased");
      }
      else{
	cerr<<"Using ORB descriptor"<<endl;
	desc_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
      }

      ROS_INFO("Opening vocabulary file: %s",vocabulary_filename.c_str());
      cv::FileStorage fs(vocabulary_filename, cv::FileStorage::READ);
      if(fs.isOpened()){
	fs["vocabulary"]>>vocabulary;
	fs.release();
	cerr<<"Read vocabulary: "<<vocabulary.rows<<" "<<vocabulary.cols<<endl;
	vocabulary_size=static_cast<int>(vocabulary.rows);
      }
      else{
	ROS_ERROR("ERROR opening file: %s\n",vocabulary_filename.c_str());
	exit(0);
      }
    }
  
    WordObservation::Ptr operator()(cv::Mat& img, unsigned image_seq, const vector<int>& pose){
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      WordObservation::Ptr z(new rost_common::WordObservation);
      z->source=name;
      z->seq = image_seq;
      z->observation_pose=pose;
      z->vocabulary_begin=vocabulary_begin;
      z->vocabulary_size=vocabulary_size;

      get_keypoints(img, feature_detector_names, feature_detectors, keypoints);
      //      feature_detector->detect(img, keypoints); 
      if(keypoints.size()>0){
	vector<cv::DMatch> matches;
	desc_extractor->compute(img,keypoints,descriptors);
	desc_matcher->match(descriptors,vocabulary,matches);	
	assert(matches.size()==(size_t)descriptors.rows);
	z->words.resize(matches.size());
	for(size_t i=0;i<matches.size(); ++i){
	  z->words[matches[i].queryIdx] = matches[i].trainIdx;
	}
      }

      z->word_pose.resize(keypoints.size()*2);//x,y
      z->word_scale.resize(keypoints.size());//x,y
      rost_common::WordObservation::_word_pose_type::iterator ci = z->word_pose.begin();
      rost_common::WordObservation::_word_scale_type::iterator si = z->word_scale.begin();
      for(size_t ki=0;ki<keypoints.size();ki++){
	*ci++ = static_cast<int>(keypoints[ki].pt.x/img_scale);
	*ci++ = static_cast<int>(keypoints[ki].pt.y/img_scale);
	*si++ = static_cast<int>(keypoints[ki].size/img_scale);
      }
      cerr<<"#feature-words: "<<z->words.size()<<endl;
      return z;
    }
  };

  ros::Publisher words_pub; //words publisher
  double img_scale;

  void imageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat img = cv_ptr->image;
    cv::Mat imgs; //scaled color image;

    if(img_scale >1.01 || img_scale < 0.99){
      cv::resize(img,imgs,cv::Size(),img_scale,img_scale);
    }
    else{
      imgs=img;
    }
    vector<int> pose(3);
    pose[0]=msg->header.seq;
    pose[1]=0;
    pose[2]=0;
    //    int nw=3;
    //    for(int x=0; x<nw; ++x){
    //      for(int y=0; y<nw; ++y){
    //	cv::Mat imgwindow = imgs(cv::Range(x*imgs.rows/nw, (x+1)*imgs.rows/nw), cv::Range(y*imgs.cols/nw, (y+1)*imgs.cols/nw))
	for(size_t i=0;i<word_extractors.size(); ++i){
	  words_pub.publish((*word_extractors[i])(imgs, msg->header.seq, pose));
	}    
	//      }
	//    }
  }
}

int main(int argc, char**argv){

  
  ros::init(argc, argv, "bow");
  ros::NodeHandle nhp("~");
  //ros::NodeHandle nh;
  std::string vocabulary_filename, image_topic_name, feature_descriptor_name;
  int num_surf, num_orb, num_grid_orb;
  bool use_surf, use_hue, use_intensity, use_orb, use_grid_orb;
  cerr<<"namespace:"<<nhp.getNamespace()<<endl;

  double rate; //looping rate

  nhp.param<bool>("use_surf",use_surf, false);
  nhp.param<int>("num_surf",num_surf, 1000);

  nhp.param<bool>("use_orb",use_orb, false);
  nhp.param<int>("num_orb",num_orb, 1000);

  nhp.param<bool>("use_grid_orb",use_grid_orb, true);
  nhp.param<int>("num_grid_orb",num_grid_orb, 1000);

  nhp.param<bool>("use_hue",use_hue, true);
  nhp.param<bool>("use_intensity",use_intensity, false);

  nhp.param<double>("scale",rost::img_scale, 1.0);
  nhp.param<string>("image",image_topic_name, "/image");
  nhp.param<double>("rate",rate, 10);

  nhp.param<string>("feature_descriptor",feature_descriptor_name, "ORB");

  cerr<<"Image scaling: "<<rost::img_scale<<endl;

  cv::initModule_nonfree();

  ros::topic::waitForMessage<sensor_msgs::Image>(image_topic_name);

  int v_begin=0;
  vector<string> feature_detector_names;
  vector<int> feature_sizes;
  if(use_surf || use_orb || use_grid_orb){
    if(!nhp.getParam("vocabulary",vocabulary_filename)){
      ROS_ERROR("Must specify a vocabulary!");
      return 0;
    }
  }

  if(use_surf){
    feature_detector_names.push_back("SURF");
    feature_sizes.push_back(num_surf);
  }
  if(use_orb){
    feature_detector_names.push_back("ORB");
    feature_sizes.push_back(num_orb);
  }
  if(use_grid_orb){
    feature_detector_names.push_back("Grid3ORB");
    feature_sizes.push_back(num_grid_orb);
    feature_detector_names.push_back("Grid2ORB");
    feature_sizes.push_back(num_grid_orb);
  }

  if(use_hue || use_orb){
    rost::word_extractors.push_back(cv::Ptr<rost::BOW>(new rost::FeatureBOW(v_begin,
									    vocabulary_filename, 
									    feature_detector_names,
									    feature_sizes, 
									    feature_descriptor_name,
									    rost::img_scale)));
    v_begin+=rost::word_extractors.back()->vocabulary_size;
  }

  if(use_hue || use_intensity){
    rost::word_extractors.push_back(cv::Ptr<rost::BOW>(new rost::ColorBOW(v_begin, 32, rost::img_scale, use_hue, use_intensity)));
    v_begin+=rost::word_extractors.back()->vocabulary_size;
  }
  

  image_transport::ImageTransport it(nhp);
  image_transport::Subscriber sub = it.subscribe(image_topic_name, 1, rost::imageCallback);
  rost::words_pub = nhp.advertise<rost_common::WordObservation>("/words", 1);


  ros::Rate loop_rate(rate);
  while (ros::ok()){
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
