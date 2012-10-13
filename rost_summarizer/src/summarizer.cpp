#include <ros/ros.h>
#include "rost_common/WordObservation.h"
#include "rost_common/Summary.h"
#include "summarizer.hpp"
#include "rost/markov.hpp"
//#include "random.hpp"
#include <map>
#include <iostream>
#include <vector>
#include <limits>
using namespace std;
//namespace h2 = entropy2;


////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
typedef Summary<> SummaryT;
SummaryT *summary;
ros::Publisher summary_pub; 
int S;
double alpha;

void words_callback(const rost_common::WordObservation::ConstPtr&  msg){
  vector<float> observation = normalize(histogram(msg->words, msg->vocabulary_size),alpha); 

  float surprise;
  int closest_idx;
  tie(surprise,closest_idx)=summary->surprise(observation);
  //cerr<<"Surprise: "<<surprise<<"  Threshold:"<<summary->threshold<<endl;
  if(surprise >= summary->threshold){
    //cerr<<"ADD"<<endl;
    summary->add(observation, msg->image_seq);
    summary->update_threshold();
  }
  rost_common::Summary::Ptr summary_msg(new rost_common::Summary);
  summary_msg->seq = msg->seq;
  summary_msg->surprise = surprise;
  summary_msg->threshold = summary->threshold;
  summary_msg->closest_seq = closest_seq;
  summary_msg->summary.insert(summary_msg->summary.begin(),summary->uids.begin(), summary->uids.end());
  summary_pub.publish(summary_msg);
}


int main(int argc, char**argv){
  ros::init(argc, argv, "summarizer");
  ros::NodeHandle nh("~");
  string thresholding;
  nh.param<int>("S", S, 9); //size of the summary
  nh.param<double>("alpha", alpha,1.0); //histogram smoothness
  nh.param<string>("threshold", thresholding,"auto"); //2min, mean, doubling

  summary = new Summary<>(S,thresholding);

  ros::Subscriber sub = nh.subscribe("/topics", 1, words_callback);
  summary_pub = nh.advertise<summarizer::Summary>("/summary", 1);
  ros::spin();
  return 0;
}



