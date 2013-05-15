#include <ros/ros.h>
#include <ros/topic.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/flann/flann.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include "rost_common/WordObservation.h"
#include "rost_common/Pause.h"
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include "feature_detector.hpp"
#include "bow.hpp"
#include "gabor_words.hpp"

//#include "binary_feature_matcher.hpp"
#include "flann_matcher.hpp"
using namespace std;

namespace rost{
  using namespace rost_common;

  vector<cv::Ptr<BOW> > word_extractors;


  struct LBPBOW:public BOW{
    cv::RNG rng;
    cv::Mat_<unsigned char> gray;
    cv::Mat_<unsigned char> tmp;
    int num_words;
    LBPBOW(int vocabulary_begin_=0, double img_scale_=1.0, int num_words_=1000):
      BOW("LBP",vocabulary_begin_,256),
      num_words(num_words_)
    {
      ROS_INFO(">>>>>>>>>>>>>>>>>>>>>>>>>>>>Inilizing LBP words: %d",num_words);
    }
    WordObservation::Ptr operator()(cv::Mat& imgs, unsigned image_seq, const vector<int>& pose){
      ROS_INFO("LBP: %d",image_seq);

      WordObservation::Ptr z(new rost_common::WordObservation);
      z->source="LBP";
      z->seq = image_seq;
      z->observation_pose=pose;
      z->vocabulary_begin=vocabulary_begin;
      z->vocabulary_size=256;

      int nwords=num_words;
      gray.create(imgs.rows,imgs.cols);
      cvtColor(imgs,tmp,CV_BGR2GRAY);

      float local_scale=0.25;
      cv::resize(tmp,gray,cv::Size(),local_scale,local_scale);
      vector<int> word_hist(256);
      for(int i=0;i<nwords;++i){
	int scale = 1;
	int x = rng.next()%(gray.cols -8) +4;
	int y = rng.next()%(gray.rows -8) +4;
	int word =0;
	//ROS_INFO("LBP x:%d, y:%d  word:%d",x,y,word);
	fill(word_hist.begin(), word_hist.end(), 0);
	int r=0;
	for(int rx=x-r; rx<=x+r; ++rx)	for(int ry=y-r; ry<=y+r; ++ry){
	    word=0;
	    for(int gx=rx-1; gx <=rx+1; gx++) for(int gy=ry-1; gy <=ry+1; gy++){
	      //ROS_INFO("LBP gx:%d, gy:%d  word:%d",gx,gy,word);
	      if(gx != rx || gy !=ry)
		{
		  word = word << 1;
		  word = word | (gray.at<unsigned char>(gy,gx) > gray.at<unsigned char>(ry,rx)); 
		  //cerr<<word<<endl;
		  //assert(word<256);
		}
	      }
	    word_hist[word]++;
	  }
	vector<int>::iterator it = max_element(word_hist.begin(), word_hist.end());
	//copy(word_hist.begin(), word_hist.end(), ostream_iterator<int>(cerr," "));cerr<<endl;
	word = it - word_hist.begin();
	//cerr<<"word= "<<word<<endl;
	z->words.push_back(vocabulary_begin + word); 
	z->word_pose.push_back(x/local_scale);
	z->word_pose.push_back(y/local_scale);
	z->word_scale.push_back(scale*2/local_scale);	
      }        
      return z;
    }
  };

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
      //cerr<<"#color-words: "<<z->words.size();
      return z;
    }
  };

  struct FeatureBOW:public BOW{
    //    cv::Ptr<cv::FeatureDetector> feature_detector;
    vector<cv::Ptr<cv::FeatureDetector> >feature_detectors;
    vector<string >feature_detector_names;
    string feature_descriptor_name;
    cv::Ptr<cv::DescriptorExtractor> desc_extractor;
    //    cv::Ptr<cv::DescriptorMatcher> desc_matcher;
    cv::Ptr<cv::DescriptorMatcher>  desc_matcher;
    cv::Ptr<FlannMatcher> flann_matcher;
    cv::Mat vocabulary;
    
    double img_scale;


    FeatureBOW(int vocabulary_begin_, 
	       const string& vocabulary_filename, 
	       const vector<string>& feature_detector_names_, 
	       const vector<int>& feature_sizes_, 
	       const string& feature_descriptor_name_="SURF", 
	       double img_scale_=1.0):
      BOW(feature_descriptor_name_, vocabulary_begin_),
      feature_detector_names(feature_detector_names_),
      feature_descriptor_name(feature_descriptor_name_),
      img_scale(img_scale_)
    {
      cerr<<"Initializing Feature BOW with detectors:";
      for(size_t i=0;i<feature_detector_names.size(); ++i){
	cerr<<feature_detector_names[i]<<endl;
	feature_detectors.push_back(get_feature_detector(feature_detector_names[i],feature_sizes_[i]));
      }

      desc_extractor = cv::DescriptorExtractor::create(feature_descriptor_name);
      if(feature_descriptor_name=="SURF"){
	cerr<<"Using SURF descriptor"<<endl;
	desc_matcher = cv::DescriptorMatcher::create("FlannBased");
	//	desc_matcher = cv::Ptr<cv::FlannBasedMatcher>(new cv::FlannBasedMatcher());
	flann_matcher = cv::Ptr<FlannMatcher>(new L2FlannMatcher<float>());
      }
      else{
	cerr<<"Using ORB descriptor"<<endl;
	desc_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	flann_matcher = cv::Ptr<FlannMatcher>(new BinaryFlannMatcher());
	//desc_matcher = cv::Ptr<cv::FlannBasedMatcher>(new cv::BinaryFlannBasedMatcher());
	//desc_matcher = new cv::BinaryFlannBasedMatcher();

	//desc_matcher = cv::Ptr<cv::DescriptorMatcher>(new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(12,20,2)));
	//desc_matcher = new FlannBasedMatcher(new flann::LshIndexParams(20,10,2));

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
      vector<cv::Mat> vocab_vector;
      for(int i=0;i<vocabulary.rows; ++i){
	vocab_vector.push_back(vocabulary.row(i));
      }
      desc_matcher->add(vocab_vector);

      //      if(feature_descriptor_name=="ORB"){
      flann_matcher->set_vocabulary(vocabulary);
	//      }

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
      if(keypoints.size()>0) desc_extractor->compute(img,keypoints,descriptors);
      if(keypoints.size()>0) flann_matcher->get_words(descriptors, z->words);

      z->word_pose.resize(keypoints.size()*2);//x,y
      z->word_scale.resize(keypoints.size());//x,y
      rost_common::WordObservation::_word_pose_type::iterator ci = z->word_pose.begin();
      rost_common::WordObservation::_word_scale_type::iterator si = z->word_scale.begin();
      for(size_t ki=0;ki<keypoints.size();ki++){
	*ci++ = static_cast<int>(keypoints[ki].pt.x/img_scale);
	*ci++ = static_cast<int>(keypoints[ki].pt.y/img_scale);
	*si++ = static_cast<int>(keypoints[ki].size/img_scale);
      }
      //cerr<<"#feature-words: "<<z->words.size()<<endl;
      return z;
    }
  };

  ros::Publisher words_pub; //words publisher
  double img_scale;
  bool pause_bow;
  bool pause(rost_common::Pause::Request& request, rost_common::Pause::Response& response){
    pause_bow=request.pause;
    return true;
  }

  void imageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    if(words_pub.getNumSubscribers() == 0 || pause_bow)
      return;

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
  ros::NodeHandle nh("");
  //ros::NodeHandle nh;
  std::string vocabulary_filename, image_topic_name, feature_descriptor_name;
  int num_surf, num_orb, num_aqua_orb, num_grid_orb, num_lbp, num_dense, color_cell_size, gabor_cell_size;
  bool use_surf, use_hue, use_intensity, use_orb, use_aqua_orb, use_grid_orb, use_lbp, use_dense, use_gabor;
  cerr<<"namespace:"<<nhp.getNamespace()<<endl;

  double rate; //looping rate

  nhp.param<bool>("use_surf",use_surf, false);
  nhp.param<int>("num_surf",num_surf, 1000);

  nhp.param<bool>("use_orb",use_orb, true);
  nhp.param<int>("num_orb",num_orb, 1000);

  nhp.param<bool>("use_aqua_orb",use_aqua_orb, false);
  nhp.param<int>("num_aqua_orb",num_aqua_orb, 1000);

  nhp.param<bool>("use_grid_orb",use_grid_orb, false);
  nhp.param<int>("num_grid_orb",num_grid_orb, 1000);


  nhp.param<bool>("use_hue",use_hue, true);
  nhp.param<bool>("use_intensity",use_intensity, false);
  nhp.param<int>("color_cell_size",color_cell_size, 32);

  nhp.param<bool>("use_gabor",use_gabor, true);
  nhp.param<int>("gabor_cell_size",gabor_cell_size, 64);

  nhp.param<bool>("use_lbp",use_lbp, false);
  nhp.param<int>("num_lbp",num_lbp, 1000);


  nhp.param<bool>("use_dense",use_dense, false);
  nhp.param<int>("num_dense",num_dense, 1000);

  nhp.param<double>("scale",rost::img_scale, 1.0);
  nhp.param<string>("image",image_topic_name, "/image");
  nhp.param<double>("rate",rate, 0);

  nhp.param<string>("feature_descriptor",feature_descriptor_name, "ORB");

  nhp.param<bool>("paused",rost::pause_bow, false);

  cerr<<"Image scaling: "<<rost::img_scale<<endl;

  cv::initModule_nonfree();


  //ros::topic::waitForMessage<sensor_msgs::Image>(image_topic_name);

  int v_begin=0;
  vector<string> feature_detector_names;
  vector<int> feature_sizes;
  if(use_surf || use_orb || use_grid_orb || use_dense){
    if(!nhp.getParam("vocabulary",vocabulary_filename)){
      ROS_ERROR("Must specify a vocabulary!");
      return 0;
    }
    ROS_INFO("BOW using vocabulary: %s", vocabulary_filename.c_str());
  }

  if(use_surf){
    feature_detector_names.push_back("SURF");
    feature_sizes.push_back(num_surf);
  }
  if(use_orb || use_aqua_orb){
    if(use_aqua_orb){ //aqua optimized orb features
      feature_detector_names.push_back("AquaORB");
      feature_sizes.push_back(num_aqua_orb);
    }
    else{
      feature_detector_names.push_back("ORB");
      feature_sizes.push_back(num_orb);
    }
  }
  if(use_grid_orb){
    feature_detector_names.push_back("Grid3ORB");
    feature_sizes.push_back(num_grid_orb);
    feature_detector_names.push_back("Grid2ORB");
    feature_sizes.push_back(num_grid_orb);
  }
  if(use_dense){
    feature_detector_names.push_back("Dense");
    feature_sizes.push_back(num_dense);
  }

  if(use_surf||use_grid_orb || use_orb || use_dense){
    rost::word_extractors.push_back(cv::Ptr<rost::BOW>(new rost::FeatureBOW(v_begin,
									    vocabulary_filename, 
									    feature_detector_names,
									    feature_sizes, 
									    feature_descriptor_name,
									    rost::img_scale)));
    v_begin+=rost::word_extractors.back()->vocabulary_size;
  }

  if(use_hue || use_intensity){
    rost::word_extractors.push_back(cv::Ptr<rost::BOW>(new rost::ColorBOW(v_begin, color_cell_size, rost::img_scale, use_hue, use_intensity)));
    v_begin+=rost::word_extractors.back()->vocabulary_size;
  }

  if(use_gabor){
    rost::word_extractors.push_back(cv::Ptr<rost::BOW>(new rost::GaborBOW(v_begin, gabor_cell_size, rost::img_scale)));
    v_begin+=rost::word_extractors.back()->vocabulary_size;
  }

  if(use_lbp){
    rost::word_extractors.push_back(cv::Ptr<rost::BOW>(new rost::LBPBOW(v_begin, rost::img_scale, num_lbp)));
    v_begin+=rost::word_extractors.back()->vocabulary_size;
  }
  

  image_transport::ImageTransport it(nhp);
  image_transport::Subscriber sub = it.subscribe(image_topic_name, 1, rost::imageCallback);
  rost::words_pub = nh.advertise<rost_common::WordObservation>("words", 1);

  ros::ServiceServer pause_service = nhp.advertiseService("pause", rost::pause);

  if(rate==0)
    ros::spin();
  else{
    ros::Rate loop_rate(rate);
    while (ros::ok()){
      ros::spinOnce();
      loop_rate.sleep();
    }
  }

  return 0;
}
