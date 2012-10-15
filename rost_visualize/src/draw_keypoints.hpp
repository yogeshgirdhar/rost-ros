#ifndef ROST_VISUALIZE_DRAW_KEYPOINTS
#define ROST_VISUALIZE_DRAW_KEYPOINTS

using namespace std;
cv::Mat get_colors(int n_colors=32, int value=255){
  cv::Mat bgr_colors, hsv_colors;
  hsv_colors.create(n_colors,1,CV_8UC3);
  bgr_colors.create(n_colors,1,CV_8UC3);
  for(int i=0;i<n_colors; ++i){
    hsv_colors.at<cv::Vec3b>(i,0) = cv::Vec3b(160/n_colors*i,255,value);
  }
  cv::cvtColor(hsv_colors, bgr_colors, CV_HSV2BGR);
  return bgr_colors;
}
cv::Scalar get_color(int i){
  static cv::Mat bgr_colors_high = get_colors(16,255);
  static cv::Mat bgr_colors_low = get_colors(16,127);
  i = i%32;
  if(i<16){
    return cv::Scalar (bgr_colors_high.at<cv::Vec3b>(i,0)[0],
		       bgr_colors_high.at<cv::Vec3b>(i,0)[1],
		       bgr_colors_high.at<cv::Vec3b>(i,0)[2]);
  }
  else{
    return cv::Scalar (bgr_colors_low.at<cv::Vec3b>(i-16,0)[0],
		       bgr_colors_low.at<cv::Vec3b>(i-16,0)[1],
		       bgr_colors_low.at<cv::Vec3b>(i-16,0)[2]);
  }
}

cv::Mat draw_keypoints(const rost_common::WordObservation::ConstPtr&  z, cv::Mat img){
  int n_colors = min<int>(16, z->vocabulary_size);
  cv::Mat bgr_colors = get_colors(n_colors);
  vector<vector<cv::KeyPoint> > keypoints(n_colors);
  cv::Mat out_img = img.clone();
  if(z->word_pose.size() == 2* z->words.size()){
    //    vector<cv::KeyPoint> keypoints(z->words.size());
    for(size_t i=0; i< z->words.size(); ++i){
      size_t color = (z->words[i] - z->vocabulary_begin)%n_colors;
      keypoints[color].
	push_back(cv::KeyPoint(static_cast<float>(z->word_pose[i*2]),
			       static_cast<float>(z->word_pose[i*2+1]),
			       static_cast<float>(z->word_scale[i])));
			       
    }

    for(size_t i=0;i<keypoints.size(); ++i){
      cv::drawKeypoints(out_img, keypoints[i], out_img, get_color(i), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG + cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    }
  }
  return out_img;

}

#endif
