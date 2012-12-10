#include <ros/ros.h>
#include <rost_common/WordObservation.h>
#include <rost_common/LocalSurprise.h>
#include <rost_common/Perplexity.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "draw_keypoints.hpp"
#include "draw_local_surprise.hpp"
#include "draw_plot.hpp"
using namespace std;
namespace enc = sensor_msgs::image_encodings;
map<unsigned, sensor_msgs::ImageConstPtr> image_cache;

ScorePlot perplexity_plot;
void perplexity_callback(const rost_common::Perplexity::Ptr& msg){
  cv::Mat img = perplexity_plot.push(msg->perplexity);
  cv::imshow("Perplexity", img);
  cv::waitKey(5);  
}

void image_callback(const sensor_msgs::ImageConstPtr& msg){
  image_cache[msg->header.seq]=msg;
  if(image_cache.size()>100)
    image_cache.erase(image_cache.begin());
}

void words_callback(const rost_common::WordObservation::ConstPtr&  z){
  sensor_msgs::ImageConstPtr img_msg = image_cache[z->seq];
  if(!img_msg) return;
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img_msg, enc::BGR8);
  cv::Mat out_img = draw_keypoints(z, cv_ptr->image.clone());
  cv::imshow(z->source, out_img);
  cv::waitKey(5);  
}

void local_surprise_callback(const rost_common::LocalSurprise::ConstPtr&  msg){
  sensor_msgs::ImageConstPtr img_msg = image_cache[msg->seq];
  if(!img_msg) return;
  cv_bridge::CvImagePtr cv_ptr;
  try{
    cv_ptr = cv_bridge::toCvCopy(img_msg, enc::BGR8);
  }
  catch (cv_bridge::Exception& e){
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
  cv::Mat img = cv_ptr->image.clone();
  img = draw_local_surprise(msg,img);
  cv::imshow("Look!", img);
  cv::waitKey(5);  
}


int main(int argc, char**argv){
  ros::init(argc, argv, "viewer");
  ros::NodeHandle *nh = new ros::NodeHandle("~");

  bool show_topics, show_local_surprise, show_perplexity;
  string image_topic_name;
  nh->param<bool>("topics", show_topics, true);
  nh->param<bool>("local_surprise", show_local_surprise, true);
  nh->param<string>("image", image_topic_name, "/image");
  nh->param<bool>("perplexity", show_perplexity, true);

  ROS_INFO("reading images from: %s", image_topic_name.c_str());
  image_transport::ImageTransport it(*nh);
  image_transport::Subscriber image_sub = it.subscribe(image_topic_name, 1, image_callback);
  ros::Subscriber word_sub = nh->subscribe("/topics", 1, words_callback);
  ros::Subscriber local_surprise_sub = nh->subscribe("/local_surprise", 1, local_surprise_callback);
  ros::Subscriber perplexity_sub;

  if(show_topics)
    word_sub = nh->subscribe("/topics", 1, words_callback);
  if(show_local_surprise)
    local_surprise_sub = nh->subscribe("/local_surprise", 1, local_surprise_callback);
  if(show_perplexity)
    perplexity_sub = nh->subscribe("/perplexity", 1, perplexity_callback);

  ros::spin();

  return 0;
}
