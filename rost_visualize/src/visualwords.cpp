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


cv::Mat get_colors(int n_colors=32){
  cv::Mat bgr_colors, hsv_colors;
  hsv_colors.create(n_colors,1,CV_8UC3);
  bgr_colors.create(n_colors,1,CV_8UC3);
  for(int i=0;i<n_colors; ++i){
    hsv_colors.at<cv::Vec3b>(i,0) = cv::Vec3b(160/n_colors*i,255,255);
  }
  cv::cvtColor(hsv_colors, bgr_colors, CV_HSV2BGR);
  return bgr_colors;
}

void image_callback(const sensor_msgs::ImageConstPtr& msg){
  image_cache[msg->header.seq]=msg;
  if(image_cache.size()>100)
    image_cache.erase(image_cache.begin());
}

void words_callback(const rost_common::WordObservation::ConstPtr&  z){
  int n_colors = min<int>(16, z->vocabulary_size);
  cv::Mat bgr_colors = get_colors(n_colors);
  vector<vector<cv::KeyPoint> > keypoints(n_colors);
  if(z->word_pose.size() == 2* z->words.size()){
    //    vector<cv::KeyPoint> keypoints(z->words.size());
    for(size_t i=0; i< z->words.size(); ++i){
      size_t color = (z->words[i] - z->vocabulary_begin)%n_colors;
      keypoints[color].
	push_back(cv::KeyPoint(static_cast<float>(z->word_pose[i*2]),
			       static_cast<float>(z->word_pose[i*2+1]),
			       static_cast<float>(z->word_scale[i])));
			       
    }
    sensor_msgs::ImageConstPtr img_msg = image_cache[z->seq];
    if(!img_msg) return;
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(img_msg, enc::BGR8);
    cv::Mat out_img = cv_ptr->image.clone();
    for(size_t i=0;i<keypoints.size(); ++i){
      cv::Scalar color(bgr_colors.at<cv::Vec3b>(i,0)[0],
		       bgr_colors.at<cv::Vec3b>(i,0)[1],
		       bgr_colors.at<cv::Vec3b>(i,0)[2]);
      cv::drawKeypoints(out_img, keypoints[i], out_img, color, cv::DrawMatchesFlags::DRAW_OVER_OUTIMG + cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }
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
