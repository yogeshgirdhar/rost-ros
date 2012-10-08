//#include "epitomizer2.hpp"
//#include "descriptor.hpp"
//#include "densesurf.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "image_source.hpp"
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
namespace po = boost::program_options;


///process program options..
///output is variables_map args
int read_program_options(int argc, char*argv[], po::variables_map& args){
  po::options_description desc("Bag of Words toolkit, used to compute vocabulary and BOW representation of each image.");
  desc.add_options()
    ("help", "help")
    ("task", po::value<string>()->default_value("describe"),"\"train\" or \"describe\"")
    ("video", po::value<string>(),"Video file")
    ("camera", "Use Camera")
    ("image", po::value<string>(),"Image file")

    ("subsample", po::value<int>()->default_value(1),"Subsampling rate for input sequence of images and video.")
    //("fps",po::value<double>()->default_value(-1),"fps for input")
    ("numframe,N", po::value<int>()->default_value(0)," Number of input images (0=no limit)")
    ("scale",po::value<float>()->default_value(1.0),"Scale image")
    ("nwords",po::value<int>()->default_value(1000), "Vocabulary size.(Only needed for training task)")
    
    ("fdetector,f",po::value<string>()->default_value("ORB"),"Specifies the feature detector: \"FAST\",\"STAR\", \"SIFT\", \"SURF\", \"ORB\", \"MSER\", \"GFTT\", \"HARRIS\", \"Dense\",\"SimpleBlob\".\n Also a combined format is supported: feature detector adapter name ( \"Grid\", \"Pyramid\") + feature detector name (see above), for example: \"GridFAST\", \"PyramidSTAR\" .)")
    ("fdescriptor,r",po::value<string>()->default_value("ORB"), "Feature description technique. Can be one of the following: \"ORB\", \"BRIEF\" + optional modifiers: \"Opponent\"")

    ("vocabulary,v", po::value<string>()->default_value("vocab.yml"),"Vocabulary file. In training this specifies the outfile, whereas in description mode, it specifies the input vocabulary")
    ("words,w", po::value<string>()->default_value("words.txt"),"Words file output. ")
    ("descriptors,d", po::value<string>()->default_value("descriptors.txt"),"normalized descriptors output")

    //    ("dense","Use Dense SURF for all 3 RGB channels, using multi-resolution overlapping windows(instead of the default sparse SURF)")
    ("keypoints,k", po::value<string>()->default_value("keypoints.txt"),"File which saves the location and size of each keypoint in the same order as the word out")
    ("wformat",po::value<string>()->default_value("words"), "words/counts: word matrix output is word counts or just words?")
    ;

  po::positional_options_description pos_desc;
  pos_desc.add("task", -1);
  

  po::store(po::command_line_parser(argc, argv)
	    .style(po::command_line_style::default_style ^ po::command_line_style::allow_guessing)
	    .options(desc)
	    .positional(pos_desc)
	    .run(), 
	    args);
  //po::store(po::command_line_parser(argc, argv).options(desc).run(), args);
  po::notify(args);    

  if (args.count("help")) {
    cerr << desc << "\n";
    exit(0);
  }
  return 0;
}

enum {Train, Describe, TrainDescribe};

int main(int argc, char*argv[]){
  po::variables_map ARGS;
  read_program_options(argc,argv, ARGS);
  cv::initModule_nonfree();


  int task;
  if(ARGS["task"].as<string>()=="train")
    task=Train;
  else if(ARGS["task"].as<string>()=="describe")
    task=Describe;
  else
    task=TrainDescribe;

  if(task==Train){
    std::ofstream metaout("bow_train_meta.txt");
    metaout<<"nwords: "<<ARGS["nwords"].as<int>()<<endl
	   <<"scale: "<<ARGS["scale"].as<float>()<<endl
	   <<"subsample: "<<ARGS["subsample"].as<int>()<<endl
	   <<"fdetector: "<<ARGS["fdetector"].as<string>()<<endl
	   <<"fdescriptor: "<<ARGS["fdescriptor"].as<string>()<<endl
      //	   <<"thresholdsurf: "<<ARGS["thresholdsurf"].as<double>()<<endl
      ;
    metaout.close();
  }
  else if(task==Describe || task == TrainDescribe){
    std::ofstream metaout("bow_describe_meta.txt");
    metaout<<"nwords: "<<ARGS["nwords"].as<int>()<<endl
	   <<"scale: "<<ARGS["scale"].as<float>()<<endl
	   <<"subsample :"<<ARGS["subsample"].as<int>()<<endl
      //   <<"thresholdsurf: "<<ARGS["thresholdsurf"].as<double>()<<endl
	   <<"vocab: "<<ARGS["vocabulary"].as<string>()<<endl
	   <<"fdetector: "<<ARGS["fdetector"].as<string>()<<endl
	   <<"fdescriptor: "<<ARGS["fdescriptor"].as<string>()<<endl
      ;
    metaout.close();    
  }

  ImageSource * img_source;

  if(ARGS.count("video"))
    img_source = ImageSource::videofile(ARGS["video"].as<string>());
  else if(ARGS.count("image"))
    img_source = ImageSource::imagefile(ARGS["image"].as<string>());
  else
    img_source = ImageSource::camera(1);


  std::ostream *wcout;
  if(ARGS["words"].as<string>()=="-" || 
     ARGS["words"].as<string>()=="STDOUT"){
    wcout = &std::cout;
  }
  else
    wcout = new std::ofstream(ARGS["words"].as<string>().c_str());

  cv::Mat img,imgs, gray;

  bool densewout=false;
  if(ARGS["wformat"].as<string>()=="words")
    densewout=true;
  std::cerr<<"Output format is words? "<<densewout<<"("<<ARGS["wformat"].as<string>()<<")" <<endl;
  bool done=false;
  int imgid=0;
  int numframe=0;
  Subsample subsample(ARGS["subsample"].as<int>());
  Scale scale(ARGS["scale"].as<float>());
  //pipeline
  *img_source > subsample > scale;

  std::vector<cv::KeyPoint> keypoints;

  //choose and configure the right feature detector
  cv::Ptr<cv::FeatureDetector> feature_detector;

  feature_detector = cv::FeatureDetector::create(ARGS["fdetector"].as<string>());

  cv::Ptr<cv::DescriptorExtractor> desc_extractor = cv::DescriptorExtractor::create(ARGS["fdescriptor"].as<string>());
  cv::Ptr<cv::DescriptorMatcher> desc_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  //  cv::BOWKMeansTrainer bow_trainer(ARGS["nwords"].as<int>());

  cv::Mat show_keypoints_img;

  if(task==Train){
    cerr<<"Starting.."<<endl;
    while(!done && (ARGS["numframe"].as<int>()==0 || numframe < ARGS["numframe"].as<int>())){
      done=!scale.grab();
      
      if(done)
	{std::cerr<<"Done reading files.\n"; continue;}

      scale.retrieve(imgs);
      
      cv::Mat descriptors;
      
      feature_detector->detect(imgs, keypoints);
      cerr<<"Img "<<imgid<<"  #keypoints: "<<keypoints.size()<<endl;

      if(keypoints.size()>0){
	desc_extractor->compute(imgs,keypoints,descriptors);
	//bow_trainer.add(descriptors);
      }
      cerr<<"Img "<<imgid<<"  #descriptors: "<<keypoints.size()<<endl;
      
      show_keypoints_img = imgs.clone();
      cv::drawKeypoints(imgs, keypoints, show_keypoints_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG + cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      cv::imshow("keypoints", show_keypoints_img);

      imgid+=ARGS["subsample"].as<int>();  
      numframe++;
      cv::waitKey(1);
    }

  }

  return 0;
}
