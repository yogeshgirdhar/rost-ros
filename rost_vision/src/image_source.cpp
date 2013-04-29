#include "image_source.hpp"
#include<iostream>

using namespace std;

  int ImagePipe::last_id(0);

/*  ImagePipe::ImagePipe():
    id(last_id++)
  {    
    name=std::string("Pipe:");
  }
*/
  bool ImagePipe::grab(){
    return source->grab();
  }

  bool ImagePipe::retrieve(cv::Mat&img){
    //    std::cerr<<name<<" retrieve\n";
    return source->retrieve(img);
  }


  ImagePipe& operator>(ImagePipe&source, ImagePipe& sink){
    sink.source = &source; 
    std::cerr<<"Connected "<<source.name<<" >> "<<sink.name<<std::endl;
    return sink;
  }
  
  ImageSource* ImageSource::camera(int device){
    return new CVImageSource(device);
  }
  ImageSource* ImageSource::videofile(const string& filename){
    return new CVImageSource(filename);
  }
  ImageSource* ImageSource::imagefile(const string& filename){
    return new ImageFilenameSource(filename);
  }


  

CVImageSource::CVImageSource(int device):cap(device){name="camera";}
CVImageSource::CVImageSource(const string& filename):cap(filename){name="videofile";}
  bool CVImageSource::grab(){
    //std::cerr<<"ImageSource::Grab()"<<std::endl;
    return cap.grab();
  }
  bool CVImageSource::retrieve(cv::Mat& image){
    return cap.retrieve(image);
  }
  

  ImageFilenameSource::ImageFilenameSource(const string& f)
    :count(0), is_multi(false), filename(f)
  {
  }
  bool ImageFilenameSource::grab(){
    if(!is_multi && count>0)
      return false;

    count++;
    return true;
  }
  bool ImageFilenameSource::retrieve(cv::Mat& image){
    image=cv::imread(this->filename);
    return true;
  }


  //////////////////////////////////////////////////////////

Subsample::Subsample(int r):ImagePipe("subsample"),rate(r){}

  bool Subsample::grab(){
    assert(rate>=0);
    if(rate==0)
      return false;
    for(int i=0;i<rate-1;++i){
      source->grab();
    }
    bool r=source->grab();
    return r;
  }

Scale::Scale(double s):ImagePipe("scale"),scale(s){}

  bool Scale::retrieve(cv::Mat& img){
    if(source->retrieve(img)){
      if(scale!=1.0){
	cv::Mat imgs;
	cv::resize(img,imgs,cv::Size(),scale,scale);
	img=imgs;
      }
      return true;
    }
    return false;
  }
  


  


