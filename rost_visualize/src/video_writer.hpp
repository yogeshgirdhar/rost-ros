#include <opencv2/opencv.hpp>
#include "boost/filesystem.hpp"
#include <sstream>
using namespace std;
class ImageSeqVideoWriter{
public:
  string name;
  bool opened;
  bool output_imageseq; //if true output imageseq, else a video file.
  int count;
  cv::VideoWriter vwriter;

  ImageSeqVideoWriter():
    opened(false),
    count(0)
  {}
  bool isOpened(){
    return opened;
  }
  void open(string filename)
  {    
    name=filename;
    boost::filesystem::path path(name); 
    cerr<<"Opening" <<name<<" with Extension: "<<path.extension()<<endl;
    if(!path.has_extension() || path.extension()==".seq" || path.extension()==".SEQ"){
      cerr<<"Output format: image sequence"<<endl;
      output_imageseq=true;
      if ( boost::filesystem::exists(path) ){
	assert(boost::filesystem::is_directory(path));
	cerr<<"WARNING: "<<name<<" already exists"<<endl;
      }
      else{
	create_directories(path);
      }
    }
    else{
      output_imageseq=false;
    }
    opened=true;
  }
  /*void push(cv::Mat& img, int seq=-1){
    if(opened){

      if(output_imageseq){
	//cerr<<"Writing img"<<endl;
	stringstream s;
	if(seq>=0)
	  s<<w.name<<"/"<<"img_"<<seq<<".jpg";
	else
	  s<<name<<"/"<<"img_"<<count++<<".jpg";
	
	cv::imwrite(s.str(), img);
      }
      else{
	if(!w.vwriter.isOpened()){
	  w.vwriter.open(w.name,CV_FOURCC('M','J','P','G'), 5, img.size());
	}
	w.vwriter<<img;
      }
      }
      }*/

};


ImageSeqVideoWriter& operator<<(ImageSeqVideoWriter& w, cv::Mat& img){
  if(w.opened){

    if(w.output_imageseq){
      //cerr<<"Writing img"<<endl;
      stringstream s;
      s<<w.name<<"/"<<"img_"<<w.count++<<".jpg";
      cv::imwrite(s.str(), img);
    }
    else{
      if(!w.vwriter.isOpened()){
	w.vwriter.open(w.name,CV_FOURCC('M','J','P','G'), 5, img.size());
      }
      w.vwriter<<img;
    }
  }
  return w;
}
