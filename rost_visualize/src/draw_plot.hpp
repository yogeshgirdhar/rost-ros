#ifndef ROST_SCORE_PLOT
#define ROST_SCORE_PLOT
class ScorePlot{
public:
  int m_win_length, m_win_height;
  float x_scale;
  cv::Mat plot;
  std::vector<cv::Point> score_pts, threshold_pts;
  std::vector<double> scores, thresholds;
  cv::Scalar background;
  ScorePlot(int win_length=800, int win_height=100):
    m_win_length(win_length),
    m_win_height(win_height),
    x_scale(4.0) // each time step is 4 pixels apart
  {
    background = cv::Scalar(255,255,255);    
  }
  cv::Mat push(double score){
    scores.push_back(score);
    int num_vis_points=std::min<size_t>(
				(size_t)((float)m_win_length/x_scale),
				scores.size());

    score_pts.clear();
    float y_scale=*max_element(scores.end()-num_vis_points, scores.end());
    
    for(size_t i=scores.size()-num_vis_points, j=0; i<scores.size(); ++i, ++j){
      score_pts.push_back(cv::Point(j*x_scale,m_win_height-scores[i]/y_scale*m_win_height*0.9));
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

    return plot;
  }
};

#endif
