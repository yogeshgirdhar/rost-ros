#include <ros/ros.h>
#include <rost_common/WordObservation.h>
#include "rost.hpp"

using namespace std;

typedef array<int,3> pose_t;
typedef ROST<pose_t,neighbors<pose_t>, hash_container<pose_t> > ROST_t;
ROST_t * rost=NULL;
int last_time;
size_t last_refine_count;
//vector<int> last_observation_pose;
//vector<int> last_word_pose, last_word_scale;
//set<pose_t> last_observation_poses;
//map<int, vector<int>> word_for_pose; 
map<pose_t, vector<int>> worddata_for_pose;  //stores [pose]-> {x_i,y_i,scale_i,.....}. 

int K, V; //number of topic types, number of word types
ros::Publisher topics_pub; 

template<typename W>
void broadcast_topics(int time, const W& worddata_for_poses){
  cerr<<"Requesting topics for time: "<<time<<endl;
  rost_common::WordObservation::Ptr z(new rost_common::WordObservation);
  z->source="topics";
  z->vocabulary_begin = 0;
  z->vocabulary_size  = K;  
  z->seq =time;
  z->observation_pose.push_back(time);
  for(auto& pose_data: worddata_for_pose){
    vector<int> topics= rost->get_topics_for_pose(pose_data.first);
    z->words.insert(z->words.end(), topics.begin(), topics.end()); 
    vector<int>& word_data = pose_data.second;
    assert(topics.size()*3 == word_data.size()); //x,y,scale
    auto wi = word_data.begin();
    for(size_t i=0;i<topics.size(); ++i){
      z->word_pose.push_back(*wi++);  z->word_pose.push_back(*wi++);  //x,y
      z->word_scale.push_back(*wi++);  //scale
    }
  }
  topics_pub.publish(z);
  //cerr<<"Publish topics "<<pose<<": "<<z->words.size()<<endl;
}

void words_callback(const rost_common::WordObservation::ConstPtr&  words){
  cerr<<"Got words: "<<words->source<<"  #"<<words->words.size()<<endl;
  int observation_time = words->observation_pose[0];
  if(last_time>=0 && (last_time != observation_time)){
    broadcast_topics(last_time, worddata_for_pose);
    size_t refine_count = rost->get_refine_count();
    cerr<<"#cells_refine: "<<refine_count - last_refine_count<<endl;  
    last_time = observation_time;
    last_refine_count = refine_count;
    worddata_for_pose.clear();
  }
  vector<int> word_pose_cell(words->word_pose.size());
  int cell_width=32;

  //split the words into different windows, each with its own pose (t,x,y)
  map<pose_t, vector<int>> words_for_pose;
  for(size_t i=0;i<words->words.size(); ++i){
    pose_t pose {{observation_time, words->word_pose[2*i]/cell_width, words->word_pose[2*i+1]/cell_width}};
    words_for_pose[pose].push_back(words->words[i]);
    auto&v = worddata_for_pose[pose];
    v.push_back(words->word_pose[2*i]); v.push_back(words->word_pose[2*i+1]);
    v.push_back(words->word_scale[i]);
  }

  for(auto & p: words_for_pose){
    rost->add_observation(p.first, p.second);
  }
}


int main(int argc, char**argv){
  ros::init(argc, argv, "rost_1d");
  ros::NodeHandle *nh = new ros::NodeHandle("~");

  double alpha, beta, gamma, tau;//, G_width_space;
  int G_time, G_space, num_threads;
  nh->param<int>("K", K, 16); //number of topics
  nh->param<int>("V", V,1500); //vocabulary size
  //  nh->param<int>("max_refines_per_iter", max_refines_per_iter,0); //vocabulary size 1000 + 16 + 18
  nh->param<double>("alpha", alpha,0.1);
  nh->param<double>("beta", beta,0.1);
  nh->param<double>("gamma", gamma,0.0);
  nh->param<double>("tau", tau,2.0);  //beta(1,tau) is used to pick cells for refinement
  nh->param<int>("num_threads", num_threads,2);  //beta(1,tau) is used to pick cells for refinement
  nh->param<int>("G_time", G_time,4);
  nh->param<int>("G_space", G_space,1);


  ROS_INFO("Starting online topic modeling: K=%d, alpha=%f, beta=%f, gamma=%f",K,alpha,beta,gamma);


  topics_pub = nh->advertise<rost_common::WordObservation>("/topics", 1);
  ros::Subscriber sub = nh->subscribe("/words", 10, words_callback);

  //  ros::ServiceServer write_topics_service = nh->advertiseService("write_topics", service_write_topics_callback);
  //  ros::ServiceServer write_state_service = nh->advertiseService("write_state", service_write_state_callback);
  //  ros::ServiceServer refine_service = nh->advertiseService("refine_global", refine_topics_callback);
  //  ros::ServiceServer refine_online_service = nh->advertiseService("refine_online", refine_online_topics_callback);
  pose_t G{{G_time, G_space, G_space}};
  rost = new ROST_t (V, K, alpha, beta, G);
  last_time = -1;
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
