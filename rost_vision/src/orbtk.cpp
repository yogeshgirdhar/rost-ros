//#include "epitomizer2.hpp"
//#include "descriptor.hpp"
//#include "densesurf.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "image_source.hpp"
#include "binary_features.hpp"
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
namespace po = boost::program_options;

template<typename T>
void write_table(ostream& out, cv::Mat& m){
  for(int i=0;i<m.rows;++i){
    cv::Mat row = m.row(i);
    copy(row.begin<T>(), row.end<T>(), ostream_iterator<T>(out," "));
    out<<endl;
  }
}
template<>
void write_table<unsigned char>(ostream& out, cv::Mat& m){
  for(int i=0;i<m.rows;++i){
    cv::Mat row = m.row(i);
    copy(row.begin<unsigned char>(), row.end<unsigned char>(), ostream_iterator<int>(out," "));
    out<<endl;
  }
}

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
    
    ("fdetector,f",po::value<vector<string> >()->multitoken(),"Specifies the feature detector: \"FAST\",\"STAR\", \"SIFT\", \"SURF\", \"ORB\", \"MSER\", \"GFTT\", \"HARRIS\", \"Dense\",\"SimpleBlob\".\n Also a combined format is supported: feature detector adapter name ( \"Grid\", \"Pyramid\") + feature detector name (see above), for example: \"GridFAST\", \"PyramidSTAR\" .)")
    ("nfeatures",po::value<int>()->default_value(1000),"Number of features for each descriptor in an image")
    ("fdescriptor,r",po::value<string>()->default_value("ORB"), "Feature description technique. Can be one of the following: \"ORB\", \"BRIEF\" + optional modifiers: \"Opponent\"")

    ("vocabulary,v", po::value<string>()->default_value("vocabulary.yml"),"Vocabulary file. In training this specifies the outfile, whereas in description mode, it specifies the input vocabulary")
    ("words,w", po::value<string>()->default_value("words.txt"),"Words file output. ")
    ("descriptors,d", po::value<string>()->default_value("descriptors.txt"),"normalized descriptors output")
    ("show.keypoints", po::value<bool>()->default_value(true),"Show keypoints")
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

cv::Ptr<cv::FeatureDetector> get_feature_detector(const string& name, int num_features){
  cv::Ptr<cv::FeatureDetector> fd;
  if(name=="Grid3ORB"){
    cv::Ptr<cv::FeatureDetector> orb(new cv::OrbFeatureDetector(num_features/9));
    fd = cv::Ptr<cv::FeatureDetector>(new cv::GridAdaptedFeatureDetector(orb,num_features,3,3));;
  }
  else if(name=="Grid4ORB"){
    cv::Ptr<cv::FeatureDetector> orb(new cv::OrbFeatureDetector(num_features/16));
    fd = cv::Ptr<cv::FeatureDetector>(new cv::GridAdaptedFeatureDetector(orb,num_features,4,4));;
  }
  else if(name=="Dense"){
    fd = cv::Ptr<cv::FeatureDetector>(new cv::DenseFeatureDetector(32, 1, 0.1, 16, 32));
  }
  else 
    fd = cv::FeatureDetector::create(name);
  return fd;
}

namespace cv{
struct orient_keypoints{
  float IC_Angle(const Mat& image, const int half_k, Point2f pt,
		 const vector<int> & u_max)
  {
    int m_01 = 0, m_10 = 0;
    
    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));
    // Treat the center line differently, v=0                                                                                                               
    for (int u = -half_k; u <= half_k; ++u)
      m_10 += u * center[u];
    
    // Go line by line in the circular patch                                                                                                                
    int step = (int)image.step1();
    for (int v = 1; v <= half_k; ++v)
      {
	// Proceed over the two lines                                                                                                                       
	int v_sum = 0;
	int d = u_max[v];
	for (int u = -d; u <= d; ++u)
	  {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
	  }
        m_01 += v * v_sum;
      }
    
    return fastAtan2((float)m_01, (float)m_10);
  }
  void operator()(const Mat& image, vector<KeyPoint>& keypoints,  int patchSize)
  {
    int halfPatchSize = patchSize / 2;
    vector<int> umax(halfPatchSize + 1);
    
    int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v)
      umax[v] = cvRound(sqrt((double)halfPatchSize * halfPatchSize - v * v));

    // Make sure we are symmetric      
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
       while (umax[v0] == umax[v0 + 1])
	 ++v0;
       umax[v] = v0;
       ++v0;
    }
    
    // Process each keypoint 
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
	   keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
      {
        keypoint->angle = IC_Angle(image, halfPatchSize, keypoint->pt, umax);
      }
  }
  
};
}


void get_keypoints(cv::Mat& imgs, vector<string>& feature_detector_names, vector<cv::Ptr<cv::FeatureDetector> >& feature_detectors, vector<cv::KeyPoint>& keypoints)
{
  static cv::orient_keypoints keypoint_orienter;
  keypoints.clear();
  for(size_t i=0;i<feature_detectors.size(); ++i){
    std::vector<cv::KeyPoint> keypoints_i;
    feature_detectors[i]->detect(imgs, keypoints_i);
    if(feature_detector_names[i]=="Dense"){
      cv::Mat img_gray;
      cvtColor(imgs,img_gray,CV_RGB2GRAY);
      keypoint_orienter(img_gray, keypoints_i, 31);
    }
    keypoints.insert(keypoints.end(), keypoints_i.begin(), keypoints_i.end());
  }  
}


int main(int argc, char*argv[]){
  po::variables_map ARGS;
  read_program_options(argc,argv, ARGS);
  cv::initModule_nonfree();


  bool do_train=false, do_describe=false;
  if(ARGS["task"].as<string>()=="train")
    do_train=true;
  else if(ARGS["task"].as<string>()=="describe")
    do_describe=true;
  else{
    do_train=true;
    do_describe=true;
  }

  bool show_keypoints = ARGS["show.keypoints"].as<bool>();

  std::ofstream metaout("orbtk_meta.txt");
  metaout<<"nwords: "<<ARGS["nwords"].as<int>()<<endl
	 <<"scale: "<<ARGS["scale"].as<float>()<<endl
	 <<"subsample :"<<ARGS["subsample"].as<int>()<<endl
    //   <<"thresholdsurf: "<<ARGS["thresholdsurf"].as<double>()<<endl
	 <<"vocab: "<<ARGS["vocabulary"].as<string>()<<endl
    //	   <<"fdetector: "<<ARGS["fdetector"].as<string>()<<endl
	 <<"fdescriptor: "<<ARGS["fdescriptor"].as<string>()<<endl
    ;
  metaout.close();    
  

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

  vector<string> feature_detector_names;
  vector<cv::Ptr<cv::FeatureDetector> > feature_detectors;
  if(ARGS.count("fdetector")==0){
    feature_detector_names.push_back("Grid3ORB");
    feature_detector_names.push_back("Grid4ORB");
    cerr<<"Using default feature detector"<<endl;
    //    exit(0);
  }
  else{
    feature_detector_names=ARGS["fdetector"].as<vector<string> >();
  }
  size_t num_feature_detectors = feature_detector_names.size();

  cerr<<"Using "<<feature_detector_names.size()<<" feature detectors: ";
  for(size_t i=0;i<num_feature_detectors; ++i){    
    cerr<<feature_detector_names[i]<<", ";
    cv::Ptr<cv::FeatureDetector> feature_detector = get_feature_detector(feature_detector_names[i],1000);
    feature_detectors.push_back(feature_detector);
  }
  cerr<<endl;

  cv::Ptr<cv::DescriptorExtractor> desc_extractor = cv::DescriptorExtractor::create(ARGS["fdescriptor"].as<string>());

  cv::BOWKMeansTrainer bow_trainer(ARGS["nwords"].as<int>());

  cv::Mat show_keypoints_img, img_gray;


  cv::Mat binary_vocabulary;
  if(do_train){
    cerr<<"Starting.."<<endl;
    while(!done && (ARGS["numframe"].as<int>()==0 || numframe < ARGS["numframe"].as<int>())){
      done=!scale.grab();
      
      if(done)
	{std::cerr<<"Done reading files.\n"; continue;}

      scale.retrieve(imgs);
      get_keypoints(imgs,feature_detector_names, feature_detectors, keypoints);
      cerr<<"Img "<<imgid<<"  #keypoints: "<<keypoints.size()<<endl;

      cv::Mat descriptors;
      
      if(keypoints.size()>0){
	desc_extractor->compute(imgs,keypoints,descriptors);
	bow_trainer.add(binary_to_float(descriptors));
      }
      if(show_keypoints){
	show_keypoints_img = imgs.clone();
	cv::drawKeypoints(imgs, keypoints, show_keypoints_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG + cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("keypoints", show_keypoints_img);
      }

      imgid+=ARGS["subsample"].as<int>();  
      numframe++;
      cv::waitKey(1);
    }
    
    std::cerr<<"Clustering "<<bow_trainer.descripotorsCount()<<" descriptors to generate a vocabulary of size "<<ARGS["nwords"].as<int>()<<std::endl;
    cv::Mat vocabulary = bow_trainer.cluster();
    binary_vocabulary = float_to_binary(vocabulary);
    std::cerr<<"Writing vocabulary to file: "<<ARGS["vocabulary"].as<string>()<<endl;
    cv::FileStorage fs(ARGS["vocabulary"].as<string>(), cv::FileStorage::WRITE);
    fs<<"vocabulary"<<binary_vocabulary;
    fs.release();
    ofstream out_vocab_bin("vocab_bin.txt");
    write_table<unsigned char>(out_vocab_bin,binary_vocabulary);
    
  }//end train


  if(do_describe){
    std::cerr<<"Reading vocabulary file: "<<ARGS["vocabulary"].as<string>()<<endl;
    cv::FileStorage fs(ARGS["vocabulary"].as<string>(), cv::FileStorage::READ);
    fs["vocabulary"]>>binary_vocabulary;
    fs.release();
    std::cerr<<"Read vocabulary with "<<binary_vocabulary.rows<<" words.\n";
    cv::Ptr<cv::DescriptorMatcher> desc_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    cv::BOWImgDescriptorExtractor bow_extractor(desc_extractor, desc_matcher);
    bow_extractor.setVocabulary(binary_vocabulary);

    vector<cv::DMatch> matches;
    while(!done && (ARGS["numframe"].as<int>()==0 || numframe < ARGS["numframe"].as<int>())){
      done=!scale.grab();
      
      if(done)
	{std::cerr<<"Done reading files.\n"; continue;}

      scale.retrieve(imgs);
      get_keypoints(imgs,feature_detector_names, feature_detectors, keypoints);
      cerr<<"Img "<<imgid<<"  #keypoints: "<<keypoints.size()<<endl;

      cv::Mat descriptors;
      
      if(keypoints.size()>0){
	matches.clear();
	desc_extractor->compute(imgs,keypoints,descriptors);
	desc_matcher->match(descriptors,binary_vocabulary,matches);
	
	assert(matches.size()==descriptors.rows);
	vector<int> words(matches.size(),0);
	for(size_t i=0;i<matches.size(); ++i){
	  words[matches[i].queryIdx] = matches[i].trainIdx;
	}
	copy(words.begin(), words.end(), ostream_iterator<int>(cout," "));
	cout<<endl;
      }

      if(show_keypoints){
	show_keypoints_img = imgs.clone();
	cv::drawKeypoints(imgs, keypoints, show_keypoints_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG + cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("keypoints", show_keypoints_img);
      }

      imgid+=ARGS["subsample"].as<int>();  
      numframe++;
      cv::waitKey(1);
    }

  }
  return 0;
}

