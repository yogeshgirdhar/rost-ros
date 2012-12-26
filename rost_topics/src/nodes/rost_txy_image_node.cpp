#include <ros/ros.h>
#include <rost_common/WordObservation.h>
#include <rost_common/GetTopicsForTime.h>
#include <rost_common/RefineTopics.h>
#include <rost_common/Perplexity.h>
#include <rost_common/GetModelPerplexity.h>
#include <rost_common/GetTopicModel.h>
#include <rost_common/TopicWeights.h>
#include <rost_common/Pause.h>
#include <std_srvs/Empty.h>
#include "rost.hpp"

using namespace std;

typedef array<int,3> pose_t;
typedef ROST<pose_t,neighbors<pose_t>, hash_container<pose_t> > ROST_t;
ROST_t * rost=NULL;
int last_time;
size_t last_refine_count;

map<int, set<pose_t>> cellposes_for_time;  //list of all poses observed at a given time
map<pose_t, vector<int>> worddata_for_pose;  //stores [pose]-> {x_i,y_i,scale_i,.....} for the current time

int K, V, cell_width; //number of topic types, number of word types
double k_alpha, k_beta, k_gamma, k_tau, p_refine_last_observation;
int G_time, G_space, num_threads, observation_size;
ros::Publisher topics_pub, perplexity_pub, topic_weights_pub; 


//service callback:
//refine topics for given number of cells
bool get_model_perplexity(rost_common::GetModelPerplexity::Request& request, rost_common::GetModelPerplexity::Response& response){
  ROS_INFO("Computing model perplexity");
  for(auto& time_poses : cellposes_for_time){
    double seq_ppx=0;
    int n_words=0;
    for(auto& pose : time_poses.second){
      vector<int> topics; double ppx;
      tie(topics,ppx) = rost->get_topics_and_ppx_for_pose(pose);
      n_words += topics.size();
      seq_ppx+=ppx;
    }
    response.perplexity.push_back(exp(-seq_ppx/n_words));
  }
  return true;
}

//service callback:
//refine topics for given number of cells
bool refine_topics(rost_common::RefineTopics::Request& request, rost_common::RefineTopics::Response& response){
  ROS_INFO("Refining %u cells, with tau=%f",request.iterations, k_tau);
  parallel_refine_tau(rost, num_threads, k_tau, request.iterations);
  return true;
}

//service callback:
//refine topics for given number of cells
bool reshuffle_topics(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response){
  ROS_INFO("Reshuffling topics");
  rost->shuffle_topics();
  return true;
}


//service callback:
//returns max likelihood topic label estimates for given pose, along with its perplexity
bool get_topics_for_time(rost_common::GetTopicsForTime::Request& request, rost_common::GetTopicsForTime::Response& response){
  
  set<pose_t>& poses = cellposes_for_time[request.seq];
  response.perplexity=0;
  for(auto& pose : poses){
    vector<int> topics; double ppx;
    tie(topics,ppx) = rost->get_topics_and_ppx_for_pose(pose);
    response.topics.insert(response.topics.end(), topics.begin(), topics.end());
    response.perplexity+=ppx;
  }
  response.K = rost->K;
  response.perplexity=exp(-response.perplexity/response.topics.size());
  return true;
}


//service callback:
//returns the current topic model, a flattened KxV matrix
bool get_topic_model(rost_common::GetTopicModel::Request& request, rost_common::GetTopicModel::Response& response){
  response.topic_model.resize(K*V);
  response.K=K;
  response.V=V;
  auto topic_model = rost->get_topic_model(); //returns a [K][V] matrix
  assert(topic_model.size()==static_cast<size_t>(K));
  assert(topic_model[0].size()==static_cast<size_t>(V));
  auto it = response.topic_model.begin();
  for(auto& topic : topic_model){
    it = copy(topic.begin(), topic.end(), it);
  }
  return true;
}

//service callback:
//returns the current topic model, a flattened KxV matrix
bool pause(rost_common::Pause::Request& request, rost_common::Pause::Response& response){
  ROS_INFO("pause service called");
  rost->pause(request.pause);
  return true;
}

//service callback:
//returns the current topic model, a flattened KxV matrix
/*bool set_topic_model(rost_common::GetTopicModel::Request& request, rost_common::GetTopicModel::Response& response){
  response.topic_model.resize(K*V);
  response.topic_model.K=K;
  response.topic_model.V=V;
  auto topic_model = rost->get_topic_model(); //returns a [K][V] matrix
  assert(topic_model.size()==static_cast<size_t>(K));
  assert(topic_model[0].size()==static_cast<size_t>(V));
  auto it = response.topic_model.begin();
  for(auto& topic : topic_model){
    it = copy(topic.begin(), topic.end(), it);
  }
  return true;
}
*/
template<typename W>
void broadcast_topics(int time, const W& worddata_for_poses){
  //  cerr<<"Requesting topics for time: "<<time<<endl;
  rost_common::WordObservation::Ptr z(new rost_common::WordObservation);
  rost_common::Perplexity::Ptr msg_ppx(new rost_common::Perplexity);
  rost_common::TopicWeights::Ptr msg_topic_weights(new rost_common::TopicWeights);
  z->source="topics";
  z->vocabulary_begin = 0;
  z->vocabulary_size  = K;  
  z->seq =time;
  z->observation_pose.push_back(time);
  msg_ppx->perplexity=0;
  msg_ppx->seq = time;
  msg_topic_weights->seq=time;
  msg_topic_weights->weight=rost->get_topic_weights();

  int n_words=0; double sum_log_p_word=0;
  for(auto& pose_data: worddata_for_pose){
    vector<int> topics; double ppx;
    tie(topics,ppx)=rost->get_topics_and_ppx_for_pose(pose_data.first);
    z->words.insert(z->words.end(), topics.begin(), topics.end()); 
    vector<int>& word_data = pose_data.second;
    assert(topics.size()*3 == word_data.size()); //x,y,scale
    auto wi = word_data.begin();
    for(size_t i=0;i<topics.size(); ++i){
      z->word_pose.push_back(*wi++);  z->word_pose.push_back(*wi++);  //x,y
      z->word_scale.push_back(*wi++);  //scale
    }
    n_words+=topics.size();
    sum_log_p_word+=ppx;
  }
  msg_ppx->perplexity= exp(-sum_log_p_word/n_words);
  topics_pub.publish(z);
  perplexity_pub.publish(msg_ppx);
  topic_weights_pub.publish(msg_topic_weights);
  //cerr<<"Publish topics for seq"<<z->seq<<": "<<z->words.size()<<endl;
}

void words_callback(const rost_common::WordObservation::ConstPtr&  words){
  //cerr<<"Got words: "<<words->source<<"  #"<<words->words.size()<<endl;
  int observation_time = words->observation_pose[0];
  if(last_time>=0 && (last_time != observation_time)){
    broadcast_topics(last_time, worddata_for_pose);
    size_t refine_count = rost->get_refine_count();
    ROS_INFO("#cells_refine: %u",static_cast<unsigned>(refine_count - last_refine_count));
    //    cerr<<"#cells_refine: "<<refine_count - last_refine_count<<endl;  
    last_refine_count = refine_count;
    worddata_for_pose.clear();
  }
  vector<int> word_pose_cell(words->word_pose.size());

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
    cellposes_for_time[words->seq].insert(p.first);
  }
  last_time = observation_time;
}


int main(int argc, char**argv){
  ros::init(argc, argv, "rost");
  ros::NodeHandle *nh = new ros::NodeHandle("~");

  bool polled_refine;
  nh->param<int>("K", K, 64); //number of topics
  nh->param<int>("V", V,1500); //vocabulary size
  //  nh->param<int>("max_refines_per_iter", max_refines_per_iter,0); //vocabulary size 1000 + 16 + 18
  nh->param<double>("alpha", k_alpha,0.1);
  nh->param<double>("beta", k_beta,0.1);
  nh->param<double>("gamma", k_gamma,0.0);
  nh->param<double>("tau", k_tau,2.0);  //beta(1,tau) is used to pick cells for global refinement
  nh->param<int>("observation_size", observation_size, 64);  //number of cells in an observation
  nh->param<double>("p_refine_last_observation", p_refine_last_observation, 0.5);  //probability of refining last observation
  nh->param<int>("num_threads", num_threads,2);  //beta(1,tau) is used to pick cells for refinement
  nh->param<int>("cell_width", cell_width, 64);
  nh->param<int>("G_time", G_time,4);
  nh->param<int>("G_space", G_space,1);
  nh->param<bool>("polled_refine", polled_refine,false);


  ROS_INFO("Starting online topic modeling: K=%d, alpha=%f, beta=%f, gamma=%f tau=%f",K,k_alpha,k_beta,k_gamma,k_tau);


  topics_pub = nh->advertise<rost_common::WordObservation>("/topics", 1);
  perplexity_pub = nh->advertise<rost_common::Perplexity>("/perplexity", 1);
  topic_weights_pub = nh->advertise<rost_common::TopicWeights>("/topic_weight", 1);
  ros::Subscriber sub = nh->subscribe("/words", 100, words_callback);
  ros::ServiceServer get_topics_for_time_service = nh->advertiseService("get_topics_for_time", get_topics_for_time);
  ros::ServiceServer refine_service = nh->advertiseService("refine", refine_topics);
  ros::ServiceServer get_model_perplexity_service = nh->advertiseService("get_model_perplexity", get_model_perplexity);
  ros::ServiceServer reshuffle_topics_service = nh->advertiseService("reshuffle_topics", reshuffle_topics);
  ros::ServiceServer get_topic_model_service = nh->advertiseService("get_topic_model", get_topic_model);
  //ros::ServiceServer get_topic_model_service = nh->advertiseService("get_topic_model", get_topic_model);
  ros::ServiceServer pause_service = nh->advertiseService("pause", pause);


  pose_t G{{G_time, G_space, G_space}};
  rost = new ROST_t (V, K, k_alpha, k_beta, G);
  last_time = -1;


  if(polled_refine){ //refine when requested    
    ROS_INFO("Topics will be refined on request.");
    ros::spin();
  }
  else{ //refine automatically
    ROS_INFO("Topics will be refined online.");
    atomic<bool> stop;   stop.store(false);
    //    auto workers =  parallel_refine_online(rost, k_tau, num_threads, &stop);
    auto workers =  parallel_refine_online2(rost, k_tau,  p_refine_last_observation, observation_size, num_threads, &stop);

    ros::MultiThreadedSpinner spinner(2);
    cerr<<"Spinning..."<<endl;
    //ros::spin();
    spinner.spin();
    stop.store(true);  //signal workers to stop
    for(auto t:workers){  //wait for them to stop
      t->join();
    }
  }
  delete rost;
  return 0;
}
