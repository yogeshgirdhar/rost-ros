
#include <ros/ros.h>
#include <ros/topic.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>

#include "rost_common/Summary.h"
#include "rost_common/GetImage.h"
#include <map>
#include <vector>
using namespace std;

map<int, sensor_msgs::Image> cache;
vector<int> save_list;
void image_callback(sensor_msgs::Image::ConstPtr msg)
{
  cache[msg->header.seq]=*msg;
  
  map<int, sensor_msgs::Image>::iterator i;
  for(i=cache.begin(); i!=cache.end(); ++i){
    if(msg->header.seq - i->first > 100 && 
       find(save_list.begin(), save_list.end(), i->first) == save_list.end()){
      cache.erase(i);
    }
  }
}

void summary_callback(const rost_common::Summary::ConstPtr&  msg){
  save_list = msg->summary;
}

bool get_image(rost_common::GetImage::Request& request, rost_common::GetImage::Response& response){
  map<int, sensor_msgs::Image>::iterator i;
  i = cache.find(request.seq);
  if(i != cache.end()){
    response.image = (i->second);
  }
  return true;
}

int main(int argc, char**argv){


  ros::init(argc, argv, "image_cache");
  ros::NodeHandle nh("~");
  std::string image_topic_name;

  nh.param<string>("image",image_topic_name, "/image");

  cerr<<"image source: "<<image_topic_name<<endl;

  


  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe(image_topic_name, 60, image_callback);
  ros::Subscriber sum_sub = nh.subscribe("/summary", 60, summary_callback);
  ros::ServiceServer request_image = nh.advertiseService("get_image", get_image);
  ros::spin();
}
