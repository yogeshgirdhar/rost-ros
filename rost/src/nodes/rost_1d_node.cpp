#include <ros/ros.h>
#include <rost_common/WordObservation.h>
#include "rost.hpp"

using namespace std;
typedef int pose_t;
typedef ROST<pose_t,neighbors<pose_t>, hash<pose_t> > ROST_t;
ROST_t * rost=NULL;
pose_t last_pose;
size_t last_refine_count;
vector<int> last_word_pose, last_word_scale;
int K, V; //number of topic types, number of word types
ros::Publisher topics_pub; 


void broadcast_topics(pose_t pose, vector<int>& word_pose, vector<int>& word_scale){
  rost_common::WordObservation::Ptr z(new rost_common::WordObservation);
  z->source="topics";
  z->seq = pose;
  z->observation_pose.push_back(pose);
  z->word_pose        = word_pose;
  z->word_pose        = word_scale;
  z->vocabulary_begin = 0;
  z->vocabulary_size  = K;  
  cerr<<"Requesting topics for pose: "<<pose<<endl;
  z->words            = rost->get_topics_for_pose(pose);
  cerr<<"Pushing topics "<<pose<<": "<<z->words.size()<<endl;
  topics_pub.publish(z);
}

void words_callback(const rost_common::WordObservation::ConstPtr&  words){
  cerr<<"Got words: "<<words->words.size()<<endl;
  if(last_pose>=0){
    broadcast_topics(last_pose, last_word_pose, last_word_scale);
  }
  size_t refine_count = rost->get_refine_count();
  rost->add_observation(words->observation_pose[0], words->words);
  cerr<<"#cells_refine: "<<refine_count - last_refine_count<<endl;
  last_pose = words->observation_pose[0];
  last_word_pose = words->word_pose;
  last_word_scale = words->word_scale;
  last_refine_count = refine_count;
}


int main(int argc, char**argv){
  ros::init(argc, argv, "rost_1d");
  ros::NodeHandle *nh = new ros::NodeHandle("~");

  double alpha, beta, gamma, tau;//, G_width_space;
  int G_width_time, G_max, num_threads;
  nh->param<int>("K", K,16); //number of topics
  nh->setParam("K",K);
  nh->param<int>("G_max", G_max,0); //max neighborhood size 0=no limit
  nh->param<int>("V", V,1500); //vocabulary size
  //  nh->param<int>("max_refines_per_iter", max_refines_per_iter,0); //vocabulary size 1000 + 16 + 18
  nh->param<double>("alpha", alpha,0.1);
  nh->param<double>("beta", beta,0.1);
  nh->param<double>("gamma", gamma,0.0);
  nh->param<double>("tau", tau,2.0);  //beta(1,tau) is used to pick cells for refinement
  nh->param<int>("num_threads", num_threads,2);  //beta(1,tau) is used to pick cells for refinement
  nh->param<int>("G_width_time", G_width_time,8);


  ROS_INFO("Starting online topic modeling: K=%d, alpha=%f, beta=%f, gamma=%f",K,alpha,beta,gamma);


  topics_pub = nh->advertise<rost_common::WordObservation>("/topics", 1);
  ros::Subscriber sub = nh->subscribe("/words", 1, words_callback);

  //  ros::ServiceServer write_topics_service = nh->advertiseService("write_topics", service_write_topics_callback);
  //  ros::ServiceServer write_state_service = nh->advertiseService("write_state", service_write_state_callback);
  //  ros::ServiceServer refine_service = nh->advertiseService("refine_global", refine_topics_callback);
  //  ros::ServiceServer refine_online_service = nh->advertiseService("refine_online", refine_online_topics_callback);

  rost = new ROST<int,neighbors<int>, hash<int> > (V, K, alpha, beta, G_width_time, hash<int>());
  last_pose = -1;
  cerr<<"Processing words online."<<endl;
  atomic<bool> stop;   stop.store(false);
  auto workers =  parallel_refine_online(rost, tau, num_threads, &stop);

  cerr<<"Spinning..."<<endl;
  ros::spin();
  stop.store(true);  //signal workers to stop
  for(auto t:workers){  //wait for them to stop
    t->join();
  }
  delete rost;
  return 0;
}
