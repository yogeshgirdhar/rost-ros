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
using namespace std;

namespace rost{
  using namespace rost_common;

  //sensor_msgs::CvBridge bridge;


  struct BOW{    
    int vocabulary_begin;
    int vocabulary_size;
    string name;
    BOW(const string& name_, int vb, int vs=0):name(name_), vocabulary_begin(vb), vocabulary_size(vs){}
    virtual WordObservation::Ptr operator()(cv::Mat& img, unsigned image_seq, const vector<int>& pose)=0;   
  };

  vector<cv::Ptr<BOW> > word_extractors;

  struct ColorBOW:public BOW{
    int size_cols;
    double img_scale;
    ColorBOW(int vocabulary_begin_, int size_cols_, double img_scale_=1.0):
      BOW("Hue",vocabulary_begin_,180),
      size_cols(size_cols_),
      img_scale(img_scale_)
    {}
    WordObservation::Ptr operator()(cv::Mat& img, unsigned image_seq, const vector<int>& pose){


      cv::Mat thumb;
      int size_rows = size_cols*static_cast<float>(imgs.rows)/imgs.cols;
      cv::resize(img,thumb,cv::Size(size_cols,size_rows));
      cv::Mat hsv (size_rows, size_cols, CV_8UC3);
      cv::Mat hue (hsv.rows, hsv.cols, CV_8UC1);
      cv::Mat saturation (hsv.rows, hsv.cols, CV_8UC1);
      cv::Mat value (hsv.rows, hsv.cols, CV_8UC1);
      cv::cvtColor(thumb,hsv,CV_BGR2HSV);
      cv::Mat splitchannels[]={hue,saturation,value};
      cv::split(hsv,splitchannels);

      vector<int> word_pose(2,0);
      for(int i=0;i<thumb.rows; ++i) // y
	for(int j=0;j<thumb.cols; ++j){ //x
	  if(use_intensity_words){
	    z->words.push_back( ivocab0 + iwords(i,j)*ivocab_size/256);
	    z->coordinates.push_back(j*iword_scale + iword_scale/2);
	    z->coordinates.push_back(i*iword_scale + iword_scale/2);
	    z->word_scales.push_back(iword_scale/2);
	  }
	  if(use_hue_words){
	    vw_msg->words.push_back( hvocab0 + hue_words(i,j)*hvocab_size/180); //hue is from 0..180
	    vw_msg->coordinates.push_back(j*iword_scale + iword_scale/2);
	    vw_msg->coordinates.push_back(i*iword_scale + iword_scale/2);
	    vw_msg->scales.push_back(iword_scale/2);
	  }
	}      
    }
  };

  struct FeatureBOW:public BOW{
    cv::Ptr<cv::FeatureDetector> feature_detector;
    cv::Ptr<cv::DescriptorExtractor> desc_extractor;
    cv::Ptr<cv::DescriptorMatcher> desc_matcher;
    cv::Ptr<cv::BOWImgDescriptorExtractor>  bow_extractor;
    double img_scale;
    FeatureBOW(int vocabulary_begin_, const string& vocabulary_filename, const string& feature_detector_name="SURF", const string& feature_descriptor_name="SURF", double img_scale_=1.0):
      BOW(feature_detector_name+"+"+feature_descriptor_name, vocabulary_begin_),
      img_scale(img_scale_)
    {
      if(feature_detector_name=="Dense"){
	feature_detector = new cv::DenseFeatureDetector();
      
      }
      else{
	feature_detector = cv::FeatureDetector::create(feature_detector_name);
      }
      desc_extractor = cv::DescriptorExtractor::create(feature_descriptor_name);
      desc_matcher = cv::DescriptorMatcher::create("FlannBased");
      bow_extractor = cv::Ptr<cv::BOWImgDescriptorExtractor>(new cv::BOWImgDescriptorExtractor(desc_extractor, desc_matcher));
      cv::FileStorage fs(vocabulary_filename, cv::FileStorage::READ);
      cv::Mat vocabulary;
      fs["vocabulary"]>>vocabulary;
      fs.release();
      cerr<<"Read vocabulary: "<<vocabulary.rows<<" "<<vocabulary.cols<<endl;
      bow_extractor->setVocabulary(vocabulary);
      vocabulary_size=static_cast<int>(vocabulary.rows);
    }
  
    WordObservation::Ptr operator()(cv::Mat& img, unsigned image_seq, const vector<int>& pose){
      std::vector<std::vector<int> > word_map;  
      std::vector<int> inverse_word_map; 
      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;
      cv::Mat_<float>bow_descriptors;

      WordObservation::Ptr z(new rost_common::WordObservation);
      z->source=name;
      z->seq = image_seq;
      z->observation_pose=pose;
      z->vocabulary_begin=vocabulary_begin;
      z->vocabulary_size=vocabulary_size;
      feature_detector->detect(img, keypoints);     
      bow_extractor->compute(img,keypoints,bow_descriptors, &word_map);   
      z->words.resize(keypoints.size());

      for(size_t vi=0;vi<word_map.size(); ++vi){
	for(size_t wi=0;wi<word_map[vi].size(); ++wi){
	  z->words[word_map[vi][wi]]=vi;
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
      cerr<<"#surf-words: "<<z->words.size()<<endl;
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
    for(size_t i=0;i<word_extractors.size(); ++i){
      words_pub.publish((*word_extractors[i])(imgs, msg->header.seq, pose));
    }    
  }
}

int main(int argc, char**argv){

  
  ros::init(argc, argv, "bow");
  ros::NodeHandle nhp("~");
  //ros::NodeHandle nh;
  std::string surf_vocabulary_filename, image_topic_name;
  cerr<<"namespace:"<<nhp.getNamespace()<<endl;

  double rate; //looping rate
  bool use_surf, use_hue, use_intensity;
  nhp.param<bool>("use_surf",use_surf, true);
  nhp.param<bool>("use_hue",use_hue, true);
  nhp.param<bool>("use_intensity",use_intensity, true);

  nhp.param<double>("scale",rost::img_scale, 1.0);
  nhp.param<string>("image",image_topic_name, "/image");
  nhp.param<double>("rate",rate, 10);
  //  nhp.param<bool>("intensity_words",use_intensity_words, true);
  //  nhp.param<bool>("hue_words",use_hue_words, true);
  //  nhp.param<bool>("surf_words",use_surf_words, true);

  cerr<<"Image scaling: "<<rost::img_scale<<endl;
  //      <<"Feature detector: "<<feature_detector_name<<endl
  //      <<"Feature descriptor: "<<feature_descriptor_name<<endl;

  //  ROS_DEBUG("Using vocabulary: %s",vocabulary_filename.c_str());
  //  cerr<<"Using vocabulary: "<<vocabulary_filename<<endl;
  cv::initModule_nonfree();

  ros::topic::waitForMessage<sensor_msgs::Image>(image_topic_name);

  int v_begin=0;
  if(use_surf){
    if(!nhp.getParam("surf_vocabulary",surf_vocabulary_filename)){
      ROS_ERROR("Must specify a vocabulary!");
      return 0;
    }
    rost::word_extractors.push_back(cv::Ptr<rost::BOW>(new rost::FeatureBOW(v_begin,surf_vocabulary_filename, "SURF","SURF", rost::img_scale)));
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
