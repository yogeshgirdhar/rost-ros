#include <ros/ros.h>
#include <rost_common/WordObservation.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
using namespace std;
namespace enc = sensor_msgs::image_encodings;
map<unsigned, sensor_msgs::ImageConstPtr> image_cache;

void image_callback(const sensor_msgs::ImageConstPtr& msg){
  image_cache[msg->header.seq]=msg;
  if(image_cache.size()>10)
    image_cache.erase(image_cache.begin());
}

void words_callback(const rost_common::WordObservation::ConstPtr&  z){
  if(z->word_pose.size() == 2* z->words.size()){
    vector<cv::KeyPoint> keypoints(z->words.size());
    for(size_t i=0; i< z->words.size(); ++i){
      keypoints[i].pt.x = static_cast<float>(z->word_pose[i*2]);
      keypoints[i].pt.y = static_cast<float>(z->word_pose[i*2+1]);
      keypoints[i].size = static_cast<float>(z->word_scale[i]);
    }
    sensor_msgs::ImageConstPtr img_msg = image_cache[z->seq];
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(img_msg, enc::BGR8);
    cv::Mat out_img;
    cv::drawKeypoints(cv_ptr->image, keypoints, out_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG + cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(z->source, out_img);
    cv::waitKey(1);
  }

}


int main(int argc, char**argv){
  ros::init(argc, argv, "viewer");
  ros::NodeHandle *nh = new ros::NodeHandle("~");


  image_transport::ImageTransport it(*nh);
  image_transport::Subscriber image_sub = it.subscribe("/image", 1, image_callback);
  ros::Subscriber word_sub = nh->subscribe("/topics", 1, words_callback);

  ros::spin();

  return 0;
}
