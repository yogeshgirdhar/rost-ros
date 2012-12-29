#include <ros/ros.h>
#include <rost_common/WordObservation.h>
#include <rost_common/LocalSurprise.h>
#include <rost_common/Perplexity.h>
#include <rost_common/TopicWeights.h>

#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "draw_keypoints.hpp"
#include "draw_local_surprise.hpp"
#include "draw_plot.hpp"
#include "draw_topic_hist.hpp"
using namespace std;
namespace enc = sensor_msgs::image_encodings;
map<unsigned, cv::Mat> image_cache;

ScorePlot perplexity_plot;
void perplexity_callback(const rost_common::Perplexity::Ptr& msg){
  cv::Mat img = perplexity_plot.push(msg->perplexity);
  cv::imshow("Perplexity", img);
  cv::waitKey(5);  
}

void image_callback(const sensor_msgs::ImageConstPtr& msg){

  cv_bridge::CvImagePtr cv_ptr;
  try{
    cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
  }
  catch (cv_bridge::Exception& e){
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  image_cache[msg->header.seq]=cv_ptr->image.clone();
  if(image_cache.size()>100)
    image_cache.erase(image_cache.begin());
}

void words_callback(const rost_common::WordObservation::ConstPtr&  z){
  cv::Mat img = image_cache[z->seq];
  if(img.empty()) return;
  cv::Mat out_img = draw_keypoints(z, img);
  cv::imshow(z->source, out_img);
  cv::waitKey(5);  
}

void local_surprise_callback(const rost_common::LocalSurprise::ConstPtr&  msg){
  cv::Mat img = image_cache[msg->seq];
  if(img.empty()) return;
  cv::Mat out_img = draw_local_surprise(msg,img);
  cv::imshow("Look!", out_img);
  cv::waitKey(5);  
}

void cell_ppx_callback(const rost_common::LocalSurprise::ConstPtr&  msg){
  cv::Mat img = image_cache[msg->seq];
  if(img.empty()) return;
  cv::Mat out_img = draw_local_surprise(msg,img);
  cv::imshow("cell perplexity", out_img);
  cv::waitKey(5);  
}

void topic_weight_callback(const rost_common::TopicWeights::ConstPtr&  msg){
  cv::Mat out_img = draw_log_barchart(msg->weight,640,240,cv::Scalar(255,255,255));
  cv::imshow("topic weights", out_img);
  cv::waitKey(5);  
}


int main(int argc, char**argv){
  ros::init(argc, argv, "viewer");
  ros::NodeHandle *nhp = new ros::NodeHandle("~");
  ros::NodeHandle *nh = new ros::NodeHandle("");

  bool show_topics, show_local_surprise, show_perplexity;
  string image_topic_name;
  nhp->param<bool>("topics", show_topics, true);
  nhp->param<bool>("local_surprise", show_local_surprise, true);
  nhp->param<string>("image", image_topic_name, "/image");
  nhp->param<bool>("perplexity", show_perplexity, true);

  ROS_INFO("reading images from: %s", image_topic_name.c_str());
  image_transport::ImageTransport it(*nh);
  image_transport::Subscriber image_sub = it.subscribe(image_topic_name, 1, image_callback);
  ros::Subscriber word_sub = nh->subscribe("topics", 1, words_callback);
  ros::Subscriber local_surprise_sub = nh->subscribe("local_surprise", 1, local_surprise_callback);
  ros::Subscriber cell_ppx_sub = nh->subscribe("cell_perplexity", 1, cell_ppx_callback);
  ros::Subscriber topic_weight_sub = nh->subscribe("topic_weight", 1, topic_weight_callback);
  ros::Subscriber perplexity_sub;

  if(show_topics)
    word_sub = nh->subscribe("topics", 1, words_callback);
  if(show_local_surprise)
    local_surprise_sub = nh->subscribe("local_surprise", 1, local_surprise_callback);
  if(show_perplexity)
    perplexity_sub = nh->subscribe("perplexity", 1, perplexity_callback);

  ros::spin();

  return 0;
}
