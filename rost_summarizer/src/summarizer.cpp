#include <ros/ros.h>
#include <rost_common/WordObservation.h>
#include <rost_common/SummaryObservations.h>
#include <rost_common/Summary.h>
#include <rost_common/GetTopicsForTime.h>
#include "summarizer.hpp"
#include "markov.hpp"
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
ros::Publisher summary_pub, summary_observations_pub; 
int S;
double alpha;
bool observations_are_topics;
ros::ServiceClient topics_client;
map<int, rost_common::WordObservation::Ptr> summary_observations;

//update the topics for summary. 
void update_summary_topics(){
  cerr<<"Updating summary...";
  vector<size_t> summary_uid(summary->uids.begin(), summary->uids.end());
  if(summary_uid.size()==0) return;
  static size_t last = 0;
  if(last >= summary_uid.size()) last = 0;
  size_t id = summary_uid[last++];
  cerr<<" id ="<<id<<endl;
  rost_common::GetTopicsForTime srv;
  srv.request.seq=id;
  if(! topics_client.call(srv)){
    ROS_ERROR("Failed to call get_topics_for_time service");
  }
  else{
    summary->remove(id);
    summary->add( normalize(histogram(srv.response.topics, srv.response.K),alpha), id); 
    summary_observations[id]->words=srv.response.topics;
  }
}

void publish_summary_observations(){
  rost_common::SummaryObservations::Ptr summary_observations_msg(new rost_common::SummaryObservations);
  vector<size_t> summary_uid(summary->uids.begin(), summary->uids.end());
  map<int, rost_common::WordObservation::Ptr> summary_observations_new;
  for(size_t i=0; i< summary_uid.size(); ++i){
    summary_observations_new[summary_uid[i]] = summary_observations[summary_uid[i]];
    summary_observations_msg->summary.push_back(*summary_observations[summary_uid[i]]);
  }
  summary_observations = summary_observations_new;
  summary_observations_pub.publish(summary_observations_msg);
}

void words_callback(rost_common::WordObservation::Ptr  msg){
  if(observations_are_topics){
    update_summary_topics();
  }
  vector<float> observation = normalize(histogram(msg->words, msg->vocabulary_size),alpha); 
  float surprise;
  int closest_seq;
  tie(surprise,closest_seq)=summary->surprise(observation);
  if(surprise >= summary->threshold){
    summary->add(observation, msg->seq);
    summary->update_threshold();
    summary_observations[msg->seq]=msg;
  }
  rost_common::Summary::Ptr summary_msg(new rost_common::Summary);
  summary_msg->seq = msg->seq;
  summary_msg->surprise = surprise;
  summary_msg->threshold = summary->threshold;
  summary_msg->closest_seq = closest_seq;
  summary_msg->summary.insert(summary_msg->summary.begin(),summary->uids.begin(), summary->uids.end());
  summary_pub.publish(summary_msg);
  publish_summary_observations();
}


int main(int argc, char**argv){
  ros::init(argc, argv, "summarizer");
  ros::NodeHandle nh("~");
  string thresholding;
  nh.param<int>("S", S, 9); //size of the summary
  nh.param<double>("alpha", alpha,1.0); //histogram smoothness
  nh.param<string>("threshold", thresholding,"auto"); //2min, mean, doubling
  nh.param<bool>("topics", observations_are_topics,false); //auto update topic labels

  summary = new Summary<>(S,thresholding);

  ros::Subscriber sub = nh.subscribe("/topics", 10, words_callback);
  summary_pub = nh.advertise<rost_common::Summary>("/summary", 10);
  summary_observations_pub = nh.advertise<rost_common::SummaryObservations>("/summary_observations", 10);

  if(observations_are_topics){
    topics_client =  nh.serviceClient<rost_common::GetTopicsForTime>("/rost/get_topics_for_time", true);
    topics_client.waitForExistence();
  }

  ros::spin();
  return 0;
}



