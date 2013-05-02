#include <fstream>
#include <string>
#include <vector>
#include <ros/ros.h>
#include "rost_common/WordObservation.h"


using namespace std;

int seq, num_words_in_msg, vocab_size;
double window_size_in_seconds, rate;
ros::Publisher words_pub;
ifstream word_in;
double last_time=0;
void publish_words(const ros::TimerEvent&){

  double time; char delim; int word;
  vector <int> words;
  word_in >> time >> delim >> word;
  while(word_in){
    words.push_back(word);
    if(time > last_time + window_size_in_seconds)
      break;
    word_in >> time >> delim >> word;
  }
  if(time <= last_time + window_size_in_seconds){
    ros::shutdown();
  }

  vector <int> pose;
  for (size_t i = 0; i < words.size(); i++){
    pose.push_back(i);
  }
  vector <int> scale(words.size(), 1);
  if (words.size() > 0){
    rost_common::WordObservation words_msg;
    words_msg.words = words;
    words_msg.word_pose = pose;
    words_msg.word_scale = scale;
    words_msg.source = "audio";
    words_msg.vocabulary_begin = 0;
    words_msg.vocabulary_size = vocab_size;
    seq++;
    words_msg.seq = seq;
    words_msg.header.seq = seq;
    words_msg.header.stamp = ros::Time(last_time);
    words_msg.observation_pose.push_back(seq);
    words_pub.publish(words_msg);
  }

  last_time += window_size_in_seconds ;
}

int main(int argc, char *argv[]){
  ros::init(argc, argv, "csv_words");
  ros::NodeHandle nh("");
  ros::NodeHandle nhp("~");
  
  

  //nhp.param<double>("num_words_in_msg",num_words_in_msg, 100);
  nhp.param<double>("window_size",window_size_in_seconds, 1.0); //time in seconds
  nhp.param<int>("V",vocab_size, 2000); 
  nhp.param<double>("rate",rate, 1.0); //speed of playing back. 1.0 is normal, 2.0 is 2 times the speed. 0.5 is half the speed.
  string filename;
  nhp.param<string>("file",filename,"/dev/stdin"); //time in seconds

  word_in.open(filename.c_str());
  ROS_ASSERT(word_in);    

  words_pub = nh.advertise<rost_common::WordObservation>("words", 1);
  ros::Timer timer = nh.createTimer(ros::Duration(window_size_in_seconds/rate), publish_words);
  ros::spin();
}
