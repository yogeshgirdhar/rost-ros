#include <signal.h> //for on_shutdown()
#include <ros/ros.h>
#include <rost_common/WordObservation.h>
#include <rost_common/GetTopicsForTime.h>
#include <rost_common/RefineTopics.h>
#include <rost_common/Perplexity.h>
#include <rost_common/GetModelPerplexity.h>
#include <rost_common/LocalSurprise.h>
#include <rost_common/GetTopicModel.h>
#include <rost_common/SetTopicModel.h>
#include <rost_common/LoadFile.h>
#include <rost_common/SaveObservationModel.h>
#include <rost_common/TopicWeights.h>
#include <rost_common/Pause.h>
#include <std_srvs/Empty.h>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include "rost.hpp"

using namespace std;

typedef array<int,3> pose_t;
typedef ROST<pose_t,neighbors<pose_t>, hash_container<pose_t> > ROST_t;
ROST_t * rost=NULL;
int last_time;
size_t last_refine_count;
bool paused;

map<int, set<pose_t>> cellposes_for_time;  //list of all poses observed at a given time
map<pose_t, vector<int>> worddata_for_pose;  //stores [pose]-> {x_i,y_i,scale_i,.....} for the current time
vector<int> observation_times; //list of all time seq ids observed thus far.

int K, V, cell_width; //number of topic types, number of word types
double k_alpha, k_beta, k_gamma, k_tau, p_refine_last_observation;
int G_time, G_space, num_threads, observation_size;
ros::Publisher topics_pub, perplexity_pub, cell_perplexity_pub, topic_weights_pub; 
ros::Subscriber word_sub;
ros::NodeHandle *nh, *nhp;


//pause the topic model
void pause(bool p);
bool pause(rost_common::Pause::Request& request, rost_common::Pause::Response& response);

//process incoming observation
void words_callback(const rost_common::WordObservation::ConstPtr&  words);

//publish topic labels for given time
template<typename W>
void broadcast_topics(int time, const W& worddata_for_poses);

//returns the current observed data model, a flattened TxK matrix
bool save_observation_model(rost_common::SaveObservationModel::Request& request, rost_common::SaveObservationModel::Response& response);

//returns the current topic model, a flattened KxV matrix
bool set_topic_model(rost_common::SetTopicModel::Request& request, rost_common::SetTopicModel::Response& response);

//returns the current topic model, a flattened KxV matrix
bool get_topic_model(rost_common::GetTopicModel::Request& request, rost_common::GetTopicModel::Response& response);


//returns max likelihood topic label estimates for given pose, along with its perplexity
bool get_topics_for_time(rost_common::GetTopicsForTime::Request& request, rost_common::GetTopicsForTime::Response& response);


//reset topic labels to random labels
bool reshuffle_topics(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);

//refine topics for given number of cells
bool refine_topics(rost_common::RefineTopics::Request& request, rost_common::RefineTopics::Response& response);

bool get_model_perplexity(rost_common::GetModelPerplexity::Request& request, rost_common::GetModelPerplexity::Response& response);









//service callback:
bool get_model_perplexity(rost_common::GetModelPerplexity::Request& request, rost_common::GetModelPerplexity::Response& response){
  ROS_INFO("Computing model perplexity");
  for(auto& time_poses : cellposes_for_time){
    double seq_ppx=0;
    int n_words=0;
    for(auto& pose : time_poses.second){
      vector<int> topics; double ppx;
      tie(topics,ppx) = rost->get_ml_topics_and_ppx_for_pose(pose);
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
//reset topic labels to random labels
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
    tie(topics,ppx) = rost->get_ml_topics_and_ppx_for_pose(pose);
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

  response.topic_model.K=K;
  response.topic_model.V=V;
  response.topic_model.alpha=k_alpha;
  response.topic_model.beta=k_beta;

  //populate the flattened phi matrix (topic-word distributions)
  response.topic_model.phi.resize(K*V);
  auto topic_model = rost->get_topic_model(); //returns a [K][V] matrix
  assert(topic_model.size()==static_cast<size_t>(K));
  assert(topic_model[0].size()==static_cast<size_t>(V));
  auto it = response.topic_model.phi.begin();
  for(auto& topic : topic_model){
    it = copy(topic.begin(), topic.end(), it);
  }

  //populate topic weights
  response.topic_model.topic_weights = rost->get_topic_weights();
  return true;
}

//service callback:
//returns the current topic model, a flattened KxV matrix
bool set_topic_model(rost_common::SetTopicModel::Request& request, rost_common::SetTopicModel::Response& response){

  pause(true);
  K = request.topic_model.K;
  V = request.topic_model.V;
  k_alpha = request.topic_model.alpha;
  k_beta = request.topic_model.beta;

  rost->set_topic_model(request.topic_model.phi, request.topic_model.topic_weights);

  pause(false);
  return true;
}

//service callback:
//returns the current topic model, a flattened KxV matrix
bool load_topic_model(rost_common::LoadFile::Request& request, rost_common::LoadFile::Response& response){

  pause(true);
  ifstream fin(request.filename.c_str());
  YAML::Parser parser(fin);
  YAML::Node doc;
  parser.GetNextDocument(doc);
  const YAML::Node& topic_model_node = doc["topic_model"];

  int new_K=0, new_V=0;
  topic_model_node["K"] >> new_K;
  topic_model_node["V"] >> new_V;
  if(new_K != K || new_V !=V){
    ROS_ERROR("Attempting to load topic model of wrong size");
    return false;
  }
  
  
  const YAML::Node &topic_weights_node = topic_model_node["topic_weights"];
  if(topic_weights_node.size() != K){
    return false;
  }
  vector<int> new_topic_weights(K,0);
  for(int i=0;i<K;++i){
    topic_weights_node[i] >> new_topic_weights[i];
  }

  ROS_INFO("Reading phi..");
  vector<int> new_phi(K*V,0);
  const YAML::Node &phi_node = topic_model_node["phi"];
  if(phi_node.size() !=K*V){
    return false;
  }
  for(int i=0;i<K*V;++i){
    phi_node[i] >> new_phi[i];
  }
  ROS_INFO("Done reading phi..");

  ROS_INFO("Setting topic model");
  //  copy(new_topic_weights.begin(), new_topic_weights.end(), ostream_iterator<int>(cerr," "));
  rost->set_topic_model(new_phi, new_topic_weights);
  ROS_INFO("Done setting topic model");
  pause(false);
  return true;
}

//service callback:
//returns the current observed data model, a flattened TxK matrix
bool save_observation_model(rost_common::SaveObservationModel::Request& request, rost_common::SaveObservationModel::Response& response){

  ROS_INFO("SaveObservationModel: Computing maximum likelihood topic estimate for all %d observations.", static_cast<int>(observation_times.size()));
  YAML::Emitter out;
  out << YAML::BeginMap
      << YAML::Key << "K" << YAML::Value << K
      << YAML::Key << "V" << YAML::Value << V
      << YAML::Key << "T" << YAML::Value << observation_times.size()
      << YAML::Key << "alpha" << YAML::Value << k_alpha
      << YAML::Key << "beta"  << YAML::Value << k_beta
      << YAML::Key << "gamma"  << YAML::Value << k_gamma
      << YAML::Key << "tau"  << YAML::Value << k_tau
      << YAML::Key << "G_time" << YAML::Value << G_time
      << YAML::Key << "G_space" << YAML::Value << G_space
      << YAML::Key << "p_refine_last_observation" << YAML::Value << p_refine_last_observation
      << YAML::Key << "observation_size" << YAML::Value << observation_size
      << YAML::Key << "refine_count" << YAML::Value<< rost->get_refine_count()
      << YAML::Key << "observations" << YAML::Value;
  out<<YAML::BeginSeq; //begin observations
  for(size_t i=0;i<observation_times.size(); ++i){
    out<<YAML::BeginMap //begin observation
       <<YAML::Key << "seq" << YAML::Value << observation_times[i]
       <<YAML::Key << "topics" <<YAML::Value;

    set<pose_t>& poses = cellposes_for_time[observation_times[i]];
    double perplexity=0;
    vector<int> topics;
    for(auto& pose : poses){
      vector<int> pose_topics; double ppx=0;
      tie(pose_topics,ppx) = rost->get_topics_and_ppx_for_pose(pose);
      topics.insert(topics.end(), pose_topics.begin(), pose_topics.end());
      perplexity+=ppx;
    }
    perplexity=exp(-perplexity/topics.size());    
    out<<YAML::Flow<<topics
       <<YAML::Key << "perplexity" << YAML::Value << perplexity;
    
    out<<YAML::EndMap;//end observation
  }  
  
  out<<YAML::EndSeq; //end observations
  out<<YAML::EndMap;//end

  ROS_INFO("SaveObservationModel: writing to model to %s",request.filename.c_str());
  ofstream outf(request.filename.c_str());
  outf<<out.c_str();
  outf.close();
  
  return true;
}



template<typename W>
void broadcast_topics(int time, const W& worddata_for_poses){

  //if nobody is listening, then why speak?
  if(topics_pub.getNumSubscribers()          == 0 &&
     perplexity_pub.getNumSubscribers()      == 0 &&
     topic_weights_pub.getNumSubscribers()   == 0 &&
     cell_perplexity_pub.getNumSubscribers() == 0    ){  
  return;
  }

  //  cerr<<"Requesting topics for time: "<<time<<endl;
  rost_common::WordObservation::Ptr z(new rost_common::WordObservation);
  rost_common::Perplexity::Ptr msg_ppx(new rost_common::Perplexity);
  rost_common::TopicWeights::Ptr msg_topic_weights(new rost_common::TopicWeights);
  rost_common::LocalSurprise::Ptr cell_perplexity(new rost_common::LocalSurprise);
  z->source="topics";
  z->vocabulary_begin = 0;
  z->vocabulary_size  = K;  
  z->seq =time;
  z->observation_pose.push_back(time);
  msg_ppx->perplexity=0;
  msg_ppx->seq = time;
  msg_topic_weights->seq=time;
  msg_topic_weights->weight=rost->get_topic_weights();
  cell_perplexity->seq= time;
  cell_perplexity->cell_width=cell_width;


  int n_words=0; double sum_log_p_word=0;

  int max_x=0, max_y=0;
  for(auto& pose_data: worddata_for_pose){
    const pose_t & pose = pose_data.first;
    max_x = max(pose[1],max_x);
    max_y = max(pose[2],max_y);
  }
  cell_perplexity->surprise.resize((max_x+1)*(max_y+1),0);
  cell_perplexity->width=max_x+1;
  cell_perplexity->height=max_y+1;

  for(auto& pose_data: worddata_for_pose){

    const pose_t & pose = pose_data.first;
    vector<int> topics;  //topic labels for each word in the cell
    double log_likelihood; //cell's sum_w log(p(w | model) = log p(cell | model)
    tie(topics,log_likelihood)=rost->get_ml_topics_and_ppx_for_pose(pose_data.first);

    //populate the topic label message
    z->words.insert(z->words.end(), topics.begin(), topics.end()); 
    vector<int>& word_data = pose_data.second;
    assert(topics.size()*3 == word_data.size()); //x,y,scale
    auto wi = word_data.begin();
    for(size_t i=0;i<topics.size(); ++i){
      z->word_pose.push_back(*wi++);  //x
      z->word_pose.push_back(*wi++);  //y
      z->word_scale.push_back(*wi++);  //scale      
    }
    n_words+=topics.size();
    sum_log_p_word+=log_likelihood;

    //populate the cell_perplexity message
    //cell_perplexity->centers.insert(cell_perplexity->centers.end(), 
    //				    pose.begin()+1, pose.end()); // x,y only. no t
  //cell_perplexity->radii.push_back(cell_width/2);
    int idx = pose[2]*(max_x+1) + pose[1];
    cell_perplexity->surprise[idx]=exp(-log_likelihood/topics.size());
  }

  //  transform(cell_perplexity->centers.begin(), 
  //	    cell_perplexity->centers.end(), 
  //	    cell_perplexity->centers.begin(), 
  //	    [](int x){return x*cell_width + cell_width/2;});
  msg_ppx->perplexity= exp(-sum_log_p_word/n_words);
  topics_pub.publish(z);
  perplexity_pub.publish(msg_ppx);
  topic_weights_pub.publish(msg_topic_weights);
  cell_perplexity_pub.publish(cell_perplexity);
  //cerr<<"Publish topics for seq"<<z->seq<<": "<<z->words.size()<<endl;
}

void words_callback(const rost_common::WordObservation::ConstPtr&  words){
  //cerr<<"Got words: "<<words->source<<"  #"<<words->words.size()<<endl;
  int observation_time = words->observation_pose[0];
  //update the  list of observed time step ids
  if(observation_times.empty() || observation_times.back() < observation_time){
    observation_times.push_back(observation_time);
  }

  //if we are receiving observations from the next time step, then spit out
  //topics for the current time step.
  if(last_time>=0 && (last_time != observation_time)){
    broadcast_topics(last_time, worddata_for_pose);
    size_t refine_count = rost->get_refine_count();
    ROS_INFO("#cells_refined: %u",static_cast<unsigned>(refine_count - last_refine_count));
    last_refine_count = refine_count;
    worddata_for_pose.clear();
  }
  vector<int> word_pose_cell(words->word_pose.size());

  //split the words into different cells, each with its own pose (t,x,y)
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

//service callback:
//pause the topic model
void pause(bool p){
  if(p){
    ROS_INFO("stopped listening to words");
    word_sub.shutdown();
  }
  else{
    ROS_INFO("started listening to words");
    word_sub = nh->subscribe("words", 10, words_callback);
  }
  rost->pause(p);
  paused=p;
}
bool pause(rost_common::Pause::Request& request, rost_common::Pause::Response& response){
  ROS_INFO("pause service called");
  pause(request.pause);
  return true;
}

void on_shutdown(int sig){
  pause(false);
  ros::shutdown();
}

int main(int argc, char**argv){
  ros::init(argc, argv, "rost");
  nhp = new ros::NodeHandle("~");
  nh = new ros::NodeHandle("");

  bool polled_refine;
  nhp->param<int>("K", K, 64); //number of topics
  nhp->param<int>("V", V,1500); //vocabulary size
  //  nh->param<int>("max_refines_per_iter", max_refines_per_iter,0); //vocabulary size 1000 + 16 + 18
  nhp->param<double>("alpha", k_alpha,0.1);
  nhp->param<double>("beta", k_beta,0.1);
  nhp->param<double>("gamma", k_gamma,0.0);
  nhp->param<double>("tau", k_tau,2.0);  //beta(1,tau) is used to pick cells for global refinement
  nhp->param<int>("observation_size", observation_size, 64);  //number of cells in an observation
  nhp->param<double>("p_refine_last_observation", p_refine_last_observation, 0.5);  //probability of refining last observation
  nhp->param<int>("num_threads", num_threads,2);  //beta(1,tau) is used to pick cells for refinement
  nhp->param<int>("cell_width", cell_width, 64);
  nhp->param<int>("G_time", G_time,4);
  nhp->param<int>("G_space", G_space,1);
  nhp->param<bool>("polled_refine", polled_refine,false);
  nhp->param<bool>("paused", paused,false);



  ROS_INFO("Starting online topic modeling: K=%d, alpha=%f, beta=%f, gamma=%f tau=%f",K,k_alpha,k_beta,k_gamma,k_tau);


  topics_pub = nh->advertise<rost_common::WordObservation>("topics", 10);
  perplexity_pub = nh->advertise<rost_common::Perplexity>("perplexity", 10);
  cell_perplexity_pub = nh->advertise<rost_common::LocalSurprise>("cell_perplexity", 10);
  topic_weights_pub = nh->advertise<rost_common::TopicWeights>("topic_weight", 10);
  word_sub = nh->subscribe("words", 10, words_callback);
  ros::ServiceServer get_topics_for_time_service = nh->advertiseService("get_topics_for_time", get_topics_for_time);
  ros::ServiceServer refine_service = nh->advertiseService("refine", refine_topics);
  ros::ServiceServer get_model_perplexity_service = nh->advertiseService("get_model_perplexity", get_model_perplexity);
  ros::ServiceServer reshuffle_topics_service = nh->advertiseService("reshuffle_topics", reshuffle_topics);
  ros::ServiceServer get_topic_model_service = nh->advertiseService("get_topic_model", get_topic_model);
  ros::ServiceServer set_topic_model_service = nh->advertiseService("set_topic_model", set_topic_model);
  ros::ServiceServer load_topic_model_service = nh->advertiseService("load_topic_model", load_topic_model);
  ros::ServiceServer save_observation_model_service = nh->advertiseService("save_observation_model", save_observation_model);
  //ros::ServiceServer get_topic_model_service = nh->advertiseService("get_topic_model", get_topic_model);
  ros::ServiceServer pause_service = nhp->advertiseService("pause", pause);

  signal(SIGINT, on_shutdown);

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
    pause(paused);
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
