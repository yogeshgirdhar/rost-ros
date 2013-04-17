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
ros::NodeHandle *nhp, *nh;
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
  //cerr<<"Updating summary...";
  vector<size_t> summary_uid(summary->uids.begin(), summary->uids.end());
  if(summary_uid.size()==0) return;
  static size_t last = 0;
  if(last >= summary_uid.size()) last = 0;
  size_t id = summary_uid[last++];
  //cerr<<" id ="<<id<<endl;
  rost_common::GetTopicsForTime srv;
  srv.request.seq=id;
  if(! topics_client.call(srv)){
    ROS_ERROR("Failed to call get_topics_for_time service.");
    topics_client =  nh->serviceClient<rost_common::GetTopicsForTime>("get_topics_for_time", true);
    ROS_INFO("Waiting to reistablish connection.");
    topics_client.waitForExistence();
    ROS_INFO("Conneciton reistablished.");
  }
  else{
    summary->remove(id);
    summary->add( normalize(histogram(srv.response.topics, srv.response.K),alpha), id); 
    if(summary_observations.find(id)!=summary_observations.end()){
      summary_observations[id]->words=srv.response.topics;
    }		
    else{
      ROS_WARN("WARNING: %d  not found in summary_observations[]",id);
    }
  }
}

void publish_summary_observations(){
  rost_common::SummaryObservations::Ptr summary_observations_msg(new rost_common::SummaryObservations);
  vector<size_t> summary_uid(summary->uids.begin(), summary->uids.end());
  map<int, rost_common::WordObservation::Ptr> summary_observations_new;
  for(size_t i=0; i< summary_uid.size(); ++i){
    rost_common::WordObservation::Ptr z = summary_observations[summary_uid[i]];
    if(z){
      assert(z);
      summary_observations_new[summary_uid[i]] = z;
      summary_observations_msg->summary.push_back(*z);
    }
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
    cerr<<"Summarizer Adding : "<<msg->seq<<endl;
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
  nhp = new ros::NodeHandle ("~");
  nh = new ros::NodeHandle ("");

  string thresholding;
  nhp->param<int>("S", S, 9); //size of the summary
  nhp->param<double>("alpha", alpha,1.0); //histogram smoothness
  nhp->param<string>("threshold", thresholding,"auto"); //2min, mean, doubling
  nhp->param<bool>("topics", observations_are_topics,false); //auto update topic labels

  summary = new Summary<>(S,thresholding);

  ros::Subscriber sub = nh->subscribe("topics", 100, words_callback);
  summary_pub = nh->advertise<rost_common::Summary>("summary", 100);
  summary_observations_pub = nh->advertise<rost_common::SummaryObservations>("summary_observations", 100);

  if(observations_are_topics){
    topics_client =  nh->serviceClient<rost_common::GetTopicsForTime>("get_topics_for_time", true);
    topics_client.waitForExistence();
  }

  ros::spin();
  return 0;
}



