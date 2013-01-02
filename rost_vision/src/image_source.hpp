#ifndef IMAGE_SOURCE_HPP
#define IMAGE_SOURCE_HPP

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <vector>
#include <string>


  class ImagePipe
  {
  public:
    //grab moves the pipe forward
    virtual bool grab();
    //retrieve actually does the work if needed and retrieves data
    virtual bool retrieve(cv::Mat& image);
    //set the data source of this node
    //    void set_source(ImagePipe&);
    ImagePipe* source;
    std::string name;
    int id;
    static int last_id;
    ImagePipe(){}
    ImagePipe(std::string name_):name(name_){}
  };
  ImagePipe& operator>(ImagePipe&source, ImagePipe& sink);


  class ImageSource: public ImagePipe
  {
  public:
    //    ImageSource():ImagePipe("ImageSource"){}
    static ImageSource* camera(int device);
    static ImageSource* videofile(const std::string& filename);
    static ImageSource* imagefile(const std::string& filename);
    virtual bool grab()=0;
    virtual bool retrieve(cv::Mat& image)=0;
  };

  class CVImageSource:public ImageSource{
  public:
    CVImageSource(int);
    CVImageSource(const std::string&);
    virtual bool grab();
    virtual bool retrieve(cv::Mat& image);
  protected:
    cv::VideoCapture cap;
  };

  class ImageFilenameSource:public ImageSource{
  public:
    ImageFilenameSource(const std::string& filename);
    virtual bool grab();
    virtual bool retrieve(cv::Mat& image);
  protected:
    int count;
    bool is_multi;
    std::string filename;
  };


  class Subsample: public ImagePipe{
  public:
    Subsample(int rate);
    bool grab();
  protected:
    int rate;
  };

  class Scale: public ImagePipe{
  public:
    Scale(double scale);
    bool retrieve(cv::Mat& image);
  protected:
    double scale;
  };


#endif
