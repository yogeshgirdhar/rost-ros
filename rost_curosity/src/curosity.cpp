#include <ros/ros.h>
#include <rost_common/WordObservation.h>
#include <rost_common/TopicWeights.h>
#include <rost_common/LocalSurprise.h>
#include <rost_common/LocalSurprise1D.h>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;
////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
ros::Publisher local_surprise_pub, local_surprise1d_pub;

vector<int> topic_weights;
int total_topic_weights;
vector<float> surprise_grid;
vector<int> surprise_grid_centers;
vector<int> surprise_grid_radii;
int img_width, img_height, cell_size, s_width, s_height;

void topic_weights_callback(rost_common::TopicWeights::Ptr  msg){
  topic_weights.resize(msg->weight.size());
  copy(msg->weight.begin(), msg->weight.end(), topic_weights.begin());
  total_topic_weights = accumulate(topic_weights.begin(), topic_weights.end(),0);
}

void topics_callback(rost_common::WordObservation::Ptr  msg){
  if(topic_weights.empty()) return;

  assert(msg->word_pose.size() == 2* msg->words.size());
  //decay the surprise score
  //  transform(surprise_grid.begin(), surprise_grid.end(), surprise_grid.begin(), 
  //	    [](float s){
  //	      return s*0.5;
  //	    });

  vector<float> current_sg(surprise_grid.size(),0);
  vector<int> sg_count(surprise_grid.size(),0);

  float log_P_topic_prior = log(1.0/msg->vocabulary_size);
  for(size_t i=0;i<msg->words.size(); ++i){
    int x = msg->word_pose[i*2]/cell_size;
    int y = msg->word_pose[i*2+1]/cell_size;
    int z = msg->words[i];
    //    float P_topic_given_data = 1.0/(static_cast<float>(topic_weights[z]));
    float P_topic_given_data = static_cast<float>(topic_weights[z])/total_topic_weights;
    current_sg[static_cast<size_t>(y*s_width + x)]+= log(P_topic_given_data);
    sg_count[static_cast<size_t>(y*s_width + x)]++;
  }

  for(size_t i=0;i<surprise_grid.size(); ++i){
    surprise_grid[i]= exp(-current_sg[i]/sg_count[i]);
  }

  rost_common::LocalSurprise::Ptr local_surprise_msg(new rost_common::LocalSurprise);

  local_surprise_msg->seq = msg->seq;
  //  local_surprise_msg->centers = surprise_grid_centers;
  //  local_surprise_msg->radii = surprise_grid_radii;
  local_surprise_msg->surprise = surprise_grid;
  local_surprise_msg->width = s_width;
  local_surprise_msg->height = s_height;
  local_surprise_msg->cell_width = cell_size;
  local_surprise_pub.publish(local_surprise_msg);

  rost_common::LocalSurprise1D::Ptr local_surprise1d_msg(new rost_common::LocalSurprise1D);
  local_surprise1d_msg->seq = msg->seq;
  local_surprise1d_msg->surprise.resize(s_width,0);
  size_t iter=0;
  //add up the vertical axis surprise to get x axis surprise
  for(int j=0; j<s_height; ++j){
    for(int i=0;i<s_width; ++i){
      //      local_surprise1d_msg->surprise[i]+=surprise_grid[iter++];
      local_surprise1d_msg->surprise[i] = max(local_surprise1d_msg->surprise[i],surprise_grid[iter++]);
    }
  }
  local_surprise1d_pub.publish(local_surprise1d_msg);
}


int main(int argc, char**argv){
  ros::init(argc, argv, "curosity");
  ros::NodeHandle nh("");
  ros::NodeHandle nhp("~");
  string thresholding;
  nhp.param<int>("img_width", img_width, 640);
  nhp.param<int>("img_height", img_height, 480);
  nhp.param<int>("cell_size", cell_size, 64);
  s_width=ceil(static_cast<float>(img_width)/cell_size);
  s_height=ceil(static_cast<float>(img_height)/cell_size);
  surprise_grid.resize(s_width*s_height,0);
  surprise_grid_centers.resize(surprise_grid.size()*2,0);
  size_t iter=0;
  for(int j=0; j<s_height; ++j){
    for(int i=0;i<s_width; ++i){
      surprise_grid_centers[iter++]=i*cell_size + 0.5*cell_size;
      surprise_grid_centers[iter++]=j*cell_size + 0.5*cell_size;
    }
  }
  surprise_grid_radii.resize(surprise_grid.size(),cell_size/2);

  ros::Subscriber sub_topics = nh.subscribe("topics", 1, topics_callback);
  ros::Subscriber sub_topics_weights = nh.subscribe("topic_weight", 1, topic_weights_callback);
  local_surprise_pub = nh.advertise<rost_common::LocalSurprise>("local_surprise", 1);
  local_surprise1d_pub = nh.advertise<rost_common::LocalSurprise1D>("local_surprise_yaw", 1);
  ros::spin();
  return 0;
}



