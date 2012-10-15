#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/CvBridge.h>
#include "rost_common/WordObservation.h"
#include "rost_common/Summary.h"
#include "rost_common/SummaryObservations.h"
#include "draw_keypoints.hpp"
#include <map>
#include<iostream>
#include<algorithm>
using namespace std;

map<unsigned int, cv::Mat> images;
map<unsigned int, vector<int> > scales;
map<unsigned int, vector<int> > coordinates;
map<unsigned int, vector<int> > summary_topics;
set<unsigned int> summary;
int window_w(1600), window_h(1200);
cv::Mat window, summary_image, score_image, live_image, topics_raw_image, topics_image, topics_hist_image;
cv::Scalar background;
cv::VideoWriter video_writer;
bool topics_mode; // true if we are showing topics
//summarizer::LocalSurprise::ConstPtr  local_surprise;

int colors_bgr[24][3]={{  7, 41,165}, {  0,155,220}, {  0,255,255}, {  0,202,132},
		       {198,189,123}, {255,  0,  0}, {171, 42, 97}, {194,156,223},
		       {120, 87, 13}, {153,153,153}, {  64,  64,  64}, {200,200,200},
                       {208,205,137}, {  0,241,249}, { 38,153,223}, {246,206,238},
		       {255,172,163}, {114,196,159}, {184,223,229}, { 36, 49,141},
		       {220, 66, 53}, {  0,104, 70}, { 18, 77,112}, {197, 33,105}};

ros::ServiceClient image_cache_client;

cv::Mat fit(cv::Mat img, int w, int h){
  float a = float(w) / float(h);
  float ai = float(img.cols) / float(img.rows);
  int rw, rh;
  if(ai > a){ // scale to fit the with
    rw = w;
    rh = float(rw)/ai;
  }
  else{
    rh = h;
    rw = float(rh)*ai;
  }
  cv::Mat r;
  cv::resize(img,r,cv::Size(rw,rh));
  return r;
}

class ScorePlot{
public:
  int m_win_length, m_win_height;
  float x_scale;
  cv::Mat plot;
  std::vector<cv::Point> score_pts, threshold_pts;
  std::vector<double> scores, thresholds;

  ScorePlot(int win_length, int win_height):
    m_win_length(win_length),
    m_win_height(win_height),
    x_scale(4.0) // each time step is 4 pixels apart
  {

  }
  void push(double score, double threshold){
    scores.push_back(score);
    thresholds.push_back(threshold);
    int num_vis_points=std::min<size_t>(
				(size_t)((float)m_win_length/x_scale),
				scores.size());

    score_pts.clear();
    threshold_pts.clear();  
    float y_scale=max(*max_element(scores.end()-num_vis_points, scores.end()),
		      *max_element(thresholds.end()-num_vis_points, thresholds.end()));
    
    for(size_t i=scores.size()-num_vis_points, j=0; i<scores.size(); ++i, ++j){
      score_pts.push_back(cv::Point(j*x_scale,m_win_height-scores[i]/y_scale*m_win_height*0.9));
      threshold_pts.push_back(cv::Point(j*x_scale,m_win_height-thresholds[i]/y_scale*m_win_height*0.9));
    }
    
    
    /*
    for(size_t i=0, j=0; j<num_vis_points; ++i, ++j){
      i = std::pow(float(j)/num_vis_points,3) * scores.size() - 1 ;
      score_pts.push_back(cv::Point(j*x_scale,m_win_height-scores[i]/y_scale*m_win_height*0.9));
      threshold_pts.push_back(cv::Point(j*x_scale,m_win_height-thresholds[i]/y_scale*m_win_height*0.9));
    }
    */
    if(plot.empty())
      plot = cv::Mat(m_win_height,m_win_length,CV_8UC3);
    plot = background;
    //cerr<<"Offset: "<<offset<<endl;
    const cv::Point* curveArr[1]={&(score_pts[0])}; 
    int      nCurvePts[1]={num_vis_points}; 
    int      nCurves=1; 
    int      isCurveClosed=0; 
    int      lineWidth=2; 
    cv::polylines(plot, curveArr, nCurvePts, nCurves,isCurveClosed,cv::Scalar(255,102,51),lineWidth,CV_AA); 

    curveArr[0]=&(threshold_pts[0]); 
    cv::polylines(plot, curveArr, nCurvePts, nCurves,isCurveClosed,cv::Scalar(51,102,255),lineWidth,CV_AA); 


    //cv::imshow("Score", plot);
    score_image = plot;
  }
};

//std::map<int, cv::Scalar> colors;
std::vector<cv::Scalar> colors;
cv::RNG rng;
int K;
using namespace std;
using namespace cv;

//delete images and other data not needed anymore
void trim(){
  map<unsigned int, cv::Mat>::iterator img_it = images.begin();
  //cerr<<"TRIM: ";
  while(images.size()>(30+summary.size()) && img_it !=images.end()){
    unsigned int imgid = img_it->first;
    img_it++;
    //    cerr<<imgid<<"?:";
    if(summary.find(imgid) == summary.end()){ //if not part of the summary
      images.erase(imgid);
      scales.erase(imgid);
      coordinates.erase(imgid);
      //cerr<<"Y";
    }
    //else{cerr<<"N";}
    //cerr<<endl;
  }

}
ScorePlot *scoreplot;
//ScorePlot scoreplot(window_w/4*3,window_h/4);


void update_window(){
  window = background;
  //cerr<<"summary img: "<<summary_image.cols<<" "<<summary_image.rows<<endl;
  cv::Mat w_sum(window,cv::Rect((window_w/4*3 - summary_image.cols)/2,
				(window_h/4*3 - summary_image.rows)/2,
				summary_image.cols,
				summary_image.rows));
  cv::Mat w_live(window,cv::Rect(window_w/4*3 + (window_w/4 - live_image.cols)/2,
				 (window_h/4 - live_image.rows)/2,
				 live_image.cols,
				 live_image.rows));
  cv::Mat w_topics(window,cv::Rect(window_w/4*3 + (window_w/4 - topics_image.cols)/2,
				   window_h/4 + (window_h/4 - topics_image.rows)/2,
				   topics_image.cols,
				   topics_image.rows));
  cv::Mat w_topics_hist(window,cv::Rect(window_w/4*3 + (window_w/4 - topics_hist_image.cols)/2,
					window_h/4*2 + (window_h/4 - topics_hist_image.rows)/2,
					topics_hist_image.cols,
					topics_hist_image.rows));
  cv::Mat w_score(window,cv::Rect(0 + (window_w - score_image.cols)/2,
				  window_h/4*3 + (window_h/4 - score_image.rows)/2,
				  score_image.cols,
				  score_image.rows));
  summary_image.copyTo(w_sum);
  //  if(topics_mode)
      topics_raw_image.copyTo(w_live);
      //  else
      //    live_image.copyTo(w_live);

  topics_image.copyTo(w_topics);
  topics_hist_image.copyTo(w_topics_hist);
  score_image.copyTo(w_score);
  cv::imshow("Summarizer",window);
  if(video_writer.isOpened()){
    video_writer << window;
  }
}

void image_callback(const sensor_msgs::ImageConstPtr& msg)
{
  //std::cerr<<"Received image: "<<msg->header.seq<<endl;  
  sensor_msgs::CvBridge bridge;   
  cv::Mat img = bridge.imgMsgToCv(msg, "bgr8");  
  live_image = fit(img,window_w/4, window_h/4);
  //cv::imshow("live", live_image);

  images[msg->header.seq]=img.clone();
  //cerr<<"#image in buffer: "<<images.size()<<endl;
  trim();
  update_window();
}

void words_callback(const rost_common::WordObservation::ConstPtr&  msg){
  scales[msg->seq]=msg->word_scale;
  coordinates[msg->seq]=msg->word_pose;
  if(!topics_mode){
    topics_raw_image = images[msg->seq];
    topics_raw_image = fit(topics_raw_image,window_w/4, window_h/4);
    
    vector<int>&tscales = scales[msg->seq];
    vector<int>&tcoordinates = coordinates[msg->seq];
    int dim = tcoordinates.size()/msg->words.size();

    cv::Mat img = images[msg->seq].clone();
    //randomly show maximum of 500 features
    vector<unsigned int> showlist;
    for(size_t i=0;i<msg->words.size();++i)
      showlist.push_back(i);
    std::random_shuffle(showlist.begin(), showlist.end()); 
    for(size_t j=0;j<std::min<size_t>(500ul,msg->words.size()); ++j){
      size_t i = showlist[j];
      int z = msg->words[i];
      cv::circle(img,
		 cv::Point(tcoordinates[i*dim], tcoordinates[i*dim+1]),
		 tscales[i]/2,
		 colors[z%colors.size()],2,CV_AA);
    }
    topics_image = fit(img,window_w/4, window_h/4);
  }
}

//void local_surprise_callback(summarizer::LocalSurprise::ConstPtr  msg){
//  local_surprise=msg;
//}

vector<float> calc_hist(const vector<int>& values){
  if(!values.empty())
    K = max(*(max_element(values.begin(), values.end())),K);


  vector<int> hist(K+1,1);
  int total=K+1;
  for(size_t i=0;i<values.size();++i){
    hist[values[i]]++;
    total++;
  }


  vector<float> histf(K+1,0);
  for(size_t i=0;i<hist.size(); ++i){
    histf[i]=(float)hist[i]/(float)total;
  }
  return histf;
}


void draw_topics_hist(const vector<int>& topics, cv::Mat& hist_img){
    vector<float> hist = calc_hist(topics);
    int hist_img_w = hist_img.cols; int hist_img_h=hist_img.rows;
    int bin_width = hist_img_w/(hist.size());
    for(size_t i=0;i<hist.size();++i){
      rectangle(hist_img,Point(i*bin_width,hist_img_h), Point((i+1)*bin_width-1,hist_img_h - hist_img_h*hist[i]),get_color(i),-1,CV_AA);
      rectangle(hist_img,Point(i*bin_width,hist_img_h), Point((i+1)*bin_width-1,hist_img_h - hist_img_h*hist[i]),cv::Scalar(0,0,0),1,CV_AA);
    }
}

cv::Mat draw_topics_hist(const vector<int>& topics, int hist_img_w, int hist_img_h){
    vector<float> hist = calc_hist(topics);
    cv::Mat hist_img(hist_img_h,hist_img_w,CV_8UC3, background);
    int bin_width = hist_img_w/(hist.size());
    for(size_t i=0;i<hist.size();++i){
      rectangle(hist_img,Point(i*bin_width,hist_img_h), Point((i+1)*bin_width-1,hist_img_h - hist_img_h*hist[i]),get_color(i),-1,CV_AA);
      rectangle(hist_img,Point(i*bin_width,hist_img_h), Point((i+1)*bin_width-1,hist_img_h - hist_img_h*hist[i]),cv::Scalar(0,0,0),1,CV_AA);
    }
    return hist_img;
}

void topics_callback(const rost_common::WordObservation::ConstPtr&  msg){
  //cerr<<"Received topic for image:"<<msg->image_seq<<endl;
  if(images.find(msg->seq)!=images.end() && 
     coordinates.find(msg->seq) != coordinates.end()){
    topics_raw_image = images[msg->seq];
    cv::Mat img = draw_keypoints(msg,topics_raw_image);

    /*    cv::Mat img = images[msg->seq].clone();
    vector<int>&tscales = scales[msg->seq];
    vector<int>&tcoordinates = coordinates[msg->seq];

    int dim = tcoordinates.size()/msg->words.size();

    //randomly show maximum of 500 features
    vector<unsigned int> showlist;
    for(size_t i=0;i<msg->words.size();++i)
      showlist.push_back(i);
    std::random_shuffle(showlist.begin(), showlist.end()); 
    for(size_t j=0;j<std::min<size_t>(500ul,msg->words.size()); ++j){
      size_t i = showlist[j];
      int z = msg->words[i];
      cv::circle(img,
		 cv::Point(tcoordinates[i*dim], tcoordinates[i*dim+1]),
		 tscales[i]/2,
		 colors[z%colors.size()],2,CV_AA);
		 }*/

    topics_image = fit(img,window_w/4, window_h/4);
    topics_raw_image = fit(topics_raw_image,window_w/4, window_h/4);

    //Draw the histogram
    topics_hist_image = draw_topics_hist(msg->words, window_w/4, window_h/4);

  }
  else
    cerr<<"Warning: corresponding image was not seen!! "<< msg->seq<<endl;

}


void show_summary(unsigned int mark){
  if(summary.size()==0) return;
  int nrows=std::ceil(std::sqrt(((float)summary.size())*3.0/4.0));
  int ncols=std::ceil((float)summary.size() / nrows);

  //cerr<<"Show sum: ";copy(summary.begin(), summary.end(), ostream_iterator<int>(cerr," ")); cerr<<endl;
  if(images.find(*summary.begin())==images.end()){
    cerr<<"Could not find image: "<<*(summary.begin())<<endl;
    //return;
  }
  cv::Mat img0 = images[* (-- (summary.end()))];

  int width=ncols*img0.cols;
  int width1=img0.cols;
  int height=nrows*img0.rows;
  int height1=img0.rows;
  int type=img0.type();
  cv::Mat simage(height,width,type);
  simage=background;
  set<unsigned int>::iterator sit=summary.begin();
  for(size_t i=0;i<summary.size();i++){
    int r=i/ncols;
    int c=i-r*ncols;
    cv::Mat img=simage(cv::Range(r*height1,(r+1)*height1),cv::Range(c*width1,(c+1)*width1));
    img=background;
    if(images.find(*sit)==images.end()){
      images[*sit]=cv::Mat(height1,width1,type,cv::Scalar(128,128,128))	;
    }
    images[*sit].copyTo(img);

    if(summary_topics.find(*sit)!=summary_topics.end()){
      cv::Mat topics_hist=img(cv::Range(img.rows/2, img.rows-img.rows/40), cv::Range(img.cols/40, img.cols - img.cols/40));
      draw_topics_hist(summary_topics[*sit], topics_hist);
    }
    
    //show the marked summary image with a frame around it
    if(mark==*sit)
      cv::rectangle(img,cv::Point(0,0),Point(width1, height1),cv::Scalar(colors_bgr[1][0],colors_bgr[1][1],colors_bgr[1][2]),height1/20);
    ++sit;
  }
  simage = fit(simage,window_w/4*3, window_h/4*3);

  summary_image = simage;
}

void summary_observations_callback(const rost_common::SummaryObservations::ConstPtr&  msg){
  summary_topics.clear();
  for(size_t i=0;i<msg->summary.size(); ++i){
    summary_topics[msg->summary[i].seq] = msg->summary[i].words;
  }  
}

void summary_callback(const rost_common::Summary::ConstPtr&  msg){
  summary.clear();
  summary.insert(msg->summary.begin(), msg->summary.end());
  show_summary(msg->closest_seq);
  scoreplot->push(msg->surprise, msg->threshold);
}



int main(int argc, char**argv){


  ros::init(argc, argv, "viewer");
  ros::NodeHandle nh("~");
  std::string image_topic_name, video_out;
  bool bg_black;
  nh.param<string>("image",image_topic_name, "/image");
  nh.param<int>("width",window_w, 1024);
  nh.param<int>("height",window_h, 768);
  nh.param<string>("vout",video_out, "");
  nh.param<bool>("black",bg_black, false);
  nh.param<bool>("topics",topics_mode, true);
  cerr<<"image source: "<<image_topic_name<<endl;
  //cv::namedWindow("live");
  //cv::namedWindow("topics");

  if(bg_black)
    background = cv::Scalar(0,0,0);    
  else
    background = cv::Scalar(255,255,255);

  cerr<<"Creating window: "<<window_w<<" "<< window_h<<endl;
  window.create(window_h,window_w,CV_8UC3);
  cv::namedWindow("Summarizer");
  
  if(!video_out.empty()){    
    video_writer.open(video_out,CV_FOURCC('M','J','P','G'), 10,cv::Size(window_w, window_h));
  }
  update_window();

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe(image_topic_name, 60, image_callback);
  ros::Subscriber topic_sub = nh.subscribe("/topics", 10, topics_callback);
  ros::Subscriber word_sub = nh.subscribe("/words", 10, words_callback);
  ros::Subscriber sum_sub = nh.subscribe("/summary", 60, summary_callback);
  //  ros::Subscriber local_surp_sub = nh.subscribe("/summarizer/local_surprise", 10, local_surprise_callback);
  ros::Subscriber sum_obz_sub = nh.subscribe("/summary_observations", 10, summary_observations_callback);

  //  image_cache_client =  nh.serviceClient<rost_common::GetImage>("image_cache/get_image");

  scoreplot = new ScorePlot(window_w-10,window_h/4);
  /*
  int c[]={255,170,85,0};
  int nc=4;
  for(int i=0;i<nc;i++)
    for(int j=0;j<nc; j++)
      for(int k=0;k<nc;k++){
	if( (i==0 && j==0 && k==0)||
	    (i==(nc-1) && j==(nc-1) && k==(nc-1)))
	  continue;
	colors.push_back(cv::Scalar(c[i],c[j],c[k]));
      }
      random_shuffle(colors.begin(), colors.end());*/
  for(int i=0;i<24;i++){
    colors.push_back(cv::Scalar(colors_bgr[i][0],colors_bgr[i][1],colors_bgr[i][2]));
  }

  K=1; //#topics
  int ch;
  while(ros::ok()){
    ros::spinOnce();
    ch=cv::waitKey(10);
    if(ch=='f'){
      if(cv::getWindowProperty("Summarizer",CV_WND_PROP_FULLSCREEN)==CV_WINDOW_FULLSCREEN)
	cv::setWindowProperty("Summarizer",CV_WND_PROP_FULLSCREEN,CV_WINDOW_NORMAL);
      else
	 cv::setWindowProperty("Summarizer",CV_WND_PROP_FULLSCREEN,CV_WINDOW_FULLSCREEN);
    }
    else if(ch == 'q'){
      break;
    }
  }
  cvDestroyAllWindows();
  return 0;
}
