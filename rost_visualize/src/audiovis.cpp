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
#include "video_writer.hpp"

using namespace std;


ScorePlot perplexity_plot;
void perplexity_callback(const rost_common::Perplexity::Ptr& msg){
  cv::Mat img = perplexity_plot.push(msg->perplexity);
  cv::imshow("Perplexity", img);
  cv::waitKey(5);  
}

void topics_callback(const rost_common::WordObservation::ConstPtr&  z){
  vector<float> hist = calc_hist(z->words, z->vocabulary_size);
  cv::Mat out_img = draw_barchart(hist, 600, 300, cv::Scalar(255,255,255));

  cv::imshow("topic histogram", out_img);
  cv::waitKey(5); 
}

void words_callback(const rost_common::WordObservation::ConstPtr&  z){
  vector<float> hist = calc_hist(z->words, z->vocabulary_size, 0);
  cv::Mat out_img = draw_barchart_sparse(hist, 600, 300, cv::Scalar(255,255,255), cv::Scalar(255,128,128) );
  cv::imshow("word histogram", out_img);
  cv::waitKey(5); 
}


void topic_weight_callback(const rost_common::TopicWeights::ConstPtr&  msg){
  cv::Mat out_img = draw_barchart(msg->weight,640,240,cv::Scalar(255,255,255));
  cv::imshow("topic weights", out_img);
  cv::waitKey(5);  
}



int main(int argc, char**argv){
  ros::init(argc, argv, "viewer");
  ros::NodeHandle *nhp = new ros::NodeHandle("~");
  ros::NodeHandle *nh = new ros::NodeHandle("");

  bool show_topics, show_words, show_perplexity;
  nhp->param<bool>("topics", show_topics, true);
  nhp->param<bool>("words", show_words, false);
  nhp->param<bool>("perplexity", show_perplexity, true);

  ros::Subscriber word_sub = nh->subscribe("words", 1, words_callback);
  ros::Subscriber topic_sub = nh->subscribe("topics", 1, topics_callback);
  ros::Subscriber topic_weight_sub = nh->subscribe("topic_weight", 1, topic_weight_callback);
  ros::Subscriber perplexity_sub;

  if(show_perplexity)
    perplexity_sub = nh->subscribe("perplexity", 1, perplexity_callback);

  ros::spin();

  return 0;
}
