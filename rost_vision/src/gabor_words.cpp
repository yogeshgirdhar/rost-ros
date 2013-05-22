#include "gabor_words.hpp"
#include <bitset>
namespace rost{
  using namespace std;
  using namespace rost_common;
  using namespace cv;
  
  GaborBOW::GaborBOW(int vocabulary_begin_, int size_cols_, double img_scale_):
    BOW("Gabor",vocabulary_begin_,0),
    size_cols(size_cols_),
    img_scale(img_scale_)
  {
    BOW::vocabulary_size = 1<<16; //16 bits
    thetas = {{0.0, 45.0, 90.0, 135.0}};
    lambdas = {{2.0, 5.0, 11.0, 17.0}};
    size_t n=thetas.size()*lambdas.size();
    sin_gabor.resize(n);
    cos_gabor.resize(n);
    filter_response.resize(n);

    double gamma = 1.0;
    //initialize the filter bank
    int i=0;
    for(double theta: thetas){
      for(double lambda: lambdas){
	double sigma = lambda*0.5;
	sin_gabor[i] = getGaborKernel( Size(), sigma, theta/180.0*M_PI, lambda,  gamma, M_PI/2.0, CV_32F );
	cos_gabor[i] = getGaborKernel( Size(), sigma, theta/180.0*M_PI, lambda,  gamma, 0.0, CV_32F );
	i++;
      }
    }
  }


  vector<unsigned int> GaborBOW::make_words(vector<Mat_<float> >& response){
    vector<bitset<16> > words(response[0].rows * response[0].cols);
    for(size_t b=0; b< 16; ++b){
      for(int i = 0; i< response[0].rows; ++i){
	for(int j = 0; j< response[0].cols; ++j){
	  words[i*response[b].cols + j][b] = response[b].at<float>(i,j)>0.5;
	}
      }
    }
    vector<unsigned int> out(words.size());
    for(size_t i=0;i<words.size();++i){
      out[i]=words[i].to_ulong();
    }
    return out;
  }

  WordObservation::Ptr GaborBOW::operator()(cv::Mat& imgs, unsigned image_seq, const vector<int>& pose){

    Mat  thumb;
    int size_rows = size_cols*static_cast<float>(imgs.rows)/imgs.cols;
    cv::resize(imgs,thumb,cv::Size(size_cols,size_rows));

    Mat img_h, img_s, img_l;
    Mat_<float> img_l_f, img_gabor_cos, img_gabor_sin, img_gabor;
    cv::Mat hls(thumb.size(), CV_8UC3);

    //width of each pixel in the original image
    float word_scale = static_cast<float>(imgs.cols)/size_cols/img_scale;

    img_h.create(thumb.size(), CV_8U);
    img_l.create(thumb.size(), CV_8U);
    img_s.create(thumb.size(), CV_8U);
    cv::Mat splitchannels[]={img_h,img_l,img_s};
    cvtColor(thumb, hls, CV_BGR2HLS);
    split(hls, splitchannels);

    //convert to floating point
    img_l.convertTo(img_l_f, CV_32F, 1.0/256.0, -0.5); //img_f is between -0.5 to 0.5      
    //imshow("float", img_l_f+0.5);



    //apply the filter bank
    int i=0;
    for(double theta: thetas){
      for(double lambda: lambdas){

	filter2D(img_l_f, img_gabor_cos, -1, cos_gabor[i], Point(-1,-1));
	filter2D(img_l_f, img_gabor_sin, -1, sin_gabor[i], Point(-1,-1));
	multiply(img_gabor_cos, img_gabor_cos, img_gabor_cos);
	multiply(img_gabor_sin, img_gabor_sin, img_gabor_sin);
	filter_response[i] = img_gabor_cos + img_gabor_sin;

	//imshow((string("sin+cos gabor  lambda:")+to_string(lambda)+" theta:"+to_string(theta)).c_str(), filter_response[i]*0.5);
	i++;
      }
    }

    //convert the filter respone to words
    auto words = make_words(filter_response);
    //copy(words.begin(), words.end(), ostream_iterator<unsigned int>(cout,",")); cout<<endl;

      
    WordObservation::Ptr z(new rost_common::WordObservation);
    z->source=name;
    z->seq = image_seq;
    z->observation_pose=pose;
    z->vocabulary_begin=vocabulary_begin;
    z->vocabulary_size=vocabulary_size;


    for(int i=0;i<thumb.rows; ++i) // y
      for(int j=0;j<thumb.cols; ++j){ //x
	z->words.push_back(ivocab0 + words[i*thumb.rows + j]);
	z->word_pose.push_back(j*word_scale + word_scale/2);
	z->word_pose.push_back(i*word_scale + word_scale/2);
	z->word_scale.push_back(word_scale/2);	  
      }
    //cerr<<"#color-words: "<<z->words.size();
    return z;
  }

}

