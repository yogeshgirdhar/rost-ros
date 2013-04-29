#ifndef ROST_DRAW_TOPIC_HIST
#define ROST_DRAW_TOPIC_HIST

vector<float> calc_hist(const vector<int>& values, int K=0, int alpha=1){
  if(!values.empty() && K==0)
    K = max(*(max_element(values.begin(), values.end())),K);

  vector<int> hist(K+1,alpha);
  int total=(K+1)*alpha;
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

template<typename T>
cv::Mat draw_barchart(const vector<T>& hist, int hist_img_w, int hist_img_h, cv::Scalar background){
    cv::Mat hist_img(hist_img_h,hist_img_w,CV_8UC3, background);
    int bin_width = hist_img_w/(hist.size());
    T max_val = *(max_element(hist.begin(), hist.end()));
    for(size_t i=0;i<hist.size();++i){
      rectangle(hist_img,
		cv::Point(i*bin_width,hist_img_h), 
		cv::Point((i+1)*bin_width-1,hist_img_h - hist_img_h*static_cast<float>(hist[i])/max_val),
		get_color(i),-1,CV_AA);
      rectangle(hist_img,
		cv::Point(i*bin_width,hist_img_h), 
		cv::Point((i+1)*bin_width-1,hist_img_h - hist_img_h*static_cast<float>(hist[i])/max_val),
		cv::Scalar(0,0,0),1,CV_AA);
    }
    return hist_img;
}
template<typename T>
cv::Mat draw_barchart_sparse(const vector<T>& hist, int hist_img_w, int hist_img_h, cv::Scalar background, cv::Scalar foreground){
    cv::Mat hist_img(hist_img_h,hist_img_w,CV_8UC3, background);
    T max_val = *(max_element(hist.begin(), hist.end()));
    for(size_t i=0;i<hist.size();++i){
      if(hist[i]>0){
	int x = min((int)round(i/static_cast<float>(hist.size()) * hist_img_w), hist_img_w-1);      
	line(hist_img, cv::Point(x,hist_img_h),  cv::Point(x,hist_img_h - hist_img_h*static_cast<float>(hist[i])/max_val),foreground,1,CV_AA);
      }
    }
    return hist_img;
}

template<typename T>
cv::Mat draw_log_barchart(const vector<T>& hist, int hist_img_w, int hist_img_h, cv::Scalar background){
    cv::Mat hist_img(hist_img_h,hist_img_w,CV_8UC3, background);
    int bin_width = hist_img_w/(hist.size());
    float max_val = log(*(max_element(hist.begin(), hist.end())) + 1 );
    for(size_t i=0;i<hist.size();++i){
      rectangle(hist_img,
		cv::Point(i*bin_width,hist_img_h), 
		cv::Point((i+1)*bin_width-1,hist_img_h - hist_img_h*log(hist[i]+1.0)/max_val),
		get_color(i),-1,CV_AA);
      rectangle(hist_img,
		cv::Point(i*bin_width,hist_img_h), 
		cv::Point((i+1)*bin_width-1,hist_img_h - hist_img_h*log(hist[i]+1.0)/max_val),
		cv::Scalar(0,0,0),1,CV_AA);
    }
    return hist_img;
}


cv::Mat draw_topic_hist(const rost_common::WordObservation::ConstPtr&  z, int w, int h, cv::Scalar bg=cv::Scalar(255,255,255)){
  vector<float> hist = calc_hist(z->words);
  return draw_barchart(hist, w, h,bg);
}

#endif
